"""Disk-backed working state for the native group loop on a shared host.

What a step carries to the next lives in private files in the run's
working-files folder and is in this process one tensor or chunk at a time
(host_pages), the rest of it file cache the kernel keeps while memory is free:
the native model file, the model's FP32 weights and the dense rounding
remainders. An install sends to those files only what it changes: a step moves
few stored values, so the model file and the weights take the pages and the
elements that differ, and only the remainders change whole. The joint direction
and the rounded displacement are made again by every step and stay in RAM: in
files, each pass over them was a read and a write of the disk (through the
fourth streamed start, November 19, 2026, about 30 of the 59 GB a step wrote).
A member's fresh native trace has no second copy: each merge is read from the
capture's own stream when it is asked for. What the host still holds whole is
what the measurement itself forms from one member's trace (its correspondence
and registration arrays). Every value is the one the in-memory loop computes:
paging moves bytes between RAM and a file and never changes them.
"""
from collections.abc import Sequence
from contextlib import ExitStack, contextmanager
from functools import wraps
import hashlib
import json
import math
import mmap
import os
from unittest.mock import patch
import weakref

import numpy as np
import torch

from . import full_parameters, full_state, gap_state, host_pages, turn_pair, working_identity

MODEL = 'final_experiment_cpu_model'   # the frozen controller's name for its host model module


def file_limits(tensors, architecture, model_bytes):
    """The most bytes each working file holds on disk, from the stored tensors' shapes.

    model: the native file. leaves: FP32 of every tensor the frozen forward keeps on the host (it reads
    the two tables by row and never reads a late layer's own keys and values). remainder: FP32 of every
    eligible dense tensor. Every array takes whole pages (host_pages.file_bytes).
    """
    late = architecture['layers'] - architecture['layers_sharing_keys']

    def kept(name):
        part = name.split('.')
        return name not in full_parameters.TABLES and not (
            part[0] == 'blk' and int(part[1]) >= late and part[2] in ('attn_k', 'attn_v', 'attn_k_norm'))

    count = lambda name: math.prod(tensors[name]['shape'])
    dense = [count(row.name) * 4 for row in full_parameters.inventory(tensors, architecture)
             if row.eligible and row.name not in full_parameters.TABLES]
    return dict(model=model_bytes, leaves=host_pages.file_bytes(count(name) * 4 for name in tensors if kept(name)),
                remainder=host_pages.file_bytes(dense))


def total(limits):
    return sum(limits.values())


def disk_model(path, seed, expected_sha256):
    before = working_identity.fingerprint(seed)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        digest = hashlib.sha256()
        # An ordinary copy: the kernel writes the working file out in the background and keeps or
        # drops the cached pages of both files as the machine's memory allows.
        with open(seed, 'rb') as source, os.fdopen(os.dup(fd), 'wb') as destination:
            while block := source.read(gap_state.BLOCK):
                digest.update(block)
                destination.write(block)
        os.fdatasync(fd)
        copied = os.fstat(fd)
        if (digest.hexdigest() != expected_sha256 or before != working_identity.fingerprint(seed)
                or (copied.st_dev, copied.st_ino) == tuple(before[:2]) or copied.st_size != before[2]):
            raise ValueError('Disk working copy differs from its declared seed')
        return fd
    except BaseException:
        os.close(fd)
        raise


class Delegate:
    """A module as frozen code sees it, with the named attributes replaced and every other one its own."""

    def __init__(self, module, **replaced):
        self.__dict__.update(replaced, _module=module)

    def __getattr__(self, name):
        return getattr(self._module, name)


class Stream(Sequence):
    """One side of a member's native trace, each merge read from the capture's stream when it is asked for.

    A cell is what the frozen reader took a merge with: the stream, the label's rows and the positions. Taking
    it again gives the same bytes, so the trace needs no copy of its own in RAM or in a file.
    """

    def __init__(self, take, cells):
        self.take, self.cells = take, tuple(cells)

    def __len__(self):
        return len(self.cells)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self.take(*cell) for cell in self.cells[index]]
        return self.take(*self.cells[index])


class Remainders(dict):
    """A state's rounding remainders, each dense array in a Store and one of them in RAM at a time.

    Sparse tensors keep their in-memory row dictionaries. A dense array handed
    out is let go when the next one is asked for.
    """

    def __init__(self, store, changed=lambda: None):
        super().__init__()
        self.store, self.changed, self.held = store, changed, None

    def _hand(self, value):
        if isinstance(value, np.ndarray) and value is not self.held:
            host_pages.release(self.held)
            self.held = value
        return value

    def __getitem__(self, name):
        return self._hand(super().__getitem__(name))

    def get(self, name, default=None):
        return self._hand(super().get(name, default))

    def __setitem__(self, name, value):
        fresh = isinstance(value, np.ndarray) and value.size and host_pages.location(value) is None
        if fresh:
            paged = self.store.array(value.shape, value.dtype)
            np.copyto(paged, value)
            host_pages.release(paged)
            value = paged
        super().__setitem__(name, value)
        if fresh:
            self.changed()

    def settle(self):
        host_pages.release(self.held)
        self.held = None

    def layout(self):
        """Where each dense remainder lives in the working file."""
        rows = {}
        for name, value in self.items():
            if isinstance(value, np.ndarray) and value.size:
                file, offset = host_pages.location(value)
                rows[name] = dict(file=file, offset_bytes=offset, shape=list(value.shape), dtype=value.dtype.str)
        return rows


def page_state(state, store, *, changed=lambda: None):
    """Move a restored state's dense remainders into a Store and let each of its passes go as it ends."""
    remainders = Remainders(store, changed)
    for name, value in state.remainders.items():
        remainders[name] = value
    state.remainders = remainders
    changed()
    preview, checkpoint = state.preview, state.checkpoint

    @wraps(preview)
    def paged_preview(name, direction, length, **options):
        try:
            yield from preview(name, direction, length, **options)
        finally:
            host_pages.release(direction)
            remainders.settle()

    @wraps(checkpoint)
    def paged_checkpoint(folder):
        try:
            return checkpoint(folder)
        finally:
            store.release()

    state.preview, state.checkpoint = paged_preview, paged_checkpoint
    return state


class Working:
    """The run's working files and the hooks that keep the frozen controller's arrays in them."""

    def __init__(self, app, run, limits=None):
        self.app, self.folder, self.limits = app, run / 'working-files', limits
        self.folder.mkdir(mode=0o700)
        self.model = self.folder / 'model.gguf'
        self.leaves, self.remainder = (host_pages.Store(self.folder, stem) for stem in ('leaves', 'remainder'))
        self.remainders = None

    def settle(self):
        """Take every page of the working files out of this process: a phase has ended.

        The pages stay cached. What the phase changed is on the disk when this returns, or the failed write is raised.
        """
        for store in (self.leaves, self.remainder):
            store.release()
            store.sync()
        if self.model.exists():
            fd = os.open(self.model, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
            try:
                os.fdatasync(fd)
            finally:
                os.close(fd)

    def pending(self):
        """Bytes the working files may still take on disk, given what they hold now."""
        held = dict(model=self.model.stat().st_size if self.model.exists() else 0, leaves=self.leaves.size,
                    remainder=self.remainder.size)
        return sum(max(0, self.limits[name] - size) for name, size in held.items())

    def create(self, seed, expected_sha256):
        return disk_model(self.model, seed, expected_sha256)

    def load(self, original):
        @wraps(original)
        def load_module(name, path):
            module = original(name, path)
            if name == MODEL:
                self._page_weights(module)
            return module
        return load_module

    def _page_weights(self, module):
        """The host model keeps each stored weight as FP32 in the leaves file: the values it would hold in RAM."""
        model, stored = module.Model, module.stored

        def paged(file, tensor):
            value = stored(file, tensor)
            leaf = self.leaves.tensor(value.shape)
            leaf.copy_(value)   # BF16 to FP32 is exact
            host_pages.release(leaf)
            return leaf

        def build(*args, **options):
            if options.get('device') != 'cpu':
                return model(*args, **options)
            with patch.object(module, 'stored', paged):
                return model(*args, **options)

        module.Model = build

    def restore(self, original):
        @wraps(original)
        def restore(*args, **options):
            last = [None]

            def allocate(shape, dtype='<f4'):
                # Each dense remainder is read from the checkpoint straight into the working file.
                host_pages.release(last[0])
                last[0] = self.remainder.array(shape, dtype)
                return last[0]

            with patch.object(full_state, 'dense_allocator', allocate):
                state = original(*args, **options)
            host_pages.release(last[0])
            page_state(state, self.remainder, changed=lambda: self._layout(state))
            self.remainders = state.remainders
            return state
        return restore

    def _layout(self, state):
        if isinstance(state.remainders, Remainders):
            path = self.folder / 'remainder-layout.json'
            scratch = path.with_suffix('.partial')
            scratch.write_text(json.dumps(state.remainders.layout(), indent=1) + '\n')
            scratch.replace(path)

    def sync(self, original):
        """An installed span reaches the host model as the elements it changes, not as the span whole.

        The frozen copy stores every element of the span, which dirties every page of the weights' file at
        each step for the few values a step moves. Storing the ones whose bits differ leaves the same array.
        A table's retained rows are RAM tensors and keep the frozen copy.
        """
        @wraps(original)
        def sync_parameter(model, state, name, start, values):
            try:
                if state.specs[name].sparse_rows:
                    return original(model, state, name, start, values)
                with torch.no_grad():
                    target = model.weights[name].view(-1)[start:start + len(values)]
                    installed = torch.from_numpy(values)
                    differs = target.view(torch.int32) != installed.view(torch.int32)   # by bits: -0.0 is stored too
                    target[differs] = installed[differs]
                    if not np.array_equal(target.numpy(), values):
                        raise ValueError('derivative state differs from installed native values')
            finally:
                host_pages.release(model.weights[name])
        return sync_parameter

    @contextmanager
    def native_traces(self):
        """A fresh native trace stays in its capture's stream, with no copy of it held in this process.

        The frozen reader takes and checks every merge as before. What it returns is then replaced by the
        cells it took, so no array of the trace outlives the read. Entered after every other replacement of
        the native reader, so it wraps the reader the run uses.
        """
        held = self.app.held
        read, taken = held.raw_forward, held.states_reading.taken

        def take(values, entry, positions):
            value = torch.from_numpy(taken(values, entry, positions))
            mapping = getattr(values, '_mmap', None)
            if mapping is not None:
                mapping.madvise(mmap.MADV_DONTNEED)   # the stream's pages are the kernel's cache, not held here
            return value

        @wraps(read)
        def streamed_read(raw, *args, **options):
            cells = {}   # where each array the reader holds came from, by the array's address

            def noted(values, entry, positions):
                value = taken(values, entry, positions)
                address = value.ctypes.data
                cells[address] = (values, entry, positions)
                weakref.finalize(value, cells.pop, address, None)   # a freed array's address names nothing
                return value

            with patch.object(held.states_reading, 'taken', noted):
                native = read(raw, *args, **options)
            try:
                sides = {key: Stream(take, [cells[value.data_ptr()] for value in native[key]])
                         for key in ('incoming', 'merged')}
            except KeyError:
                raise ValueError('a native merge is not an array the reader took from the capture stream') from None
            return {**native, **sides}

        with patch.object(held, 'raw_forward', streamed_read):
            yield

    def close_scratch(self):
        """Derived arrays are made again by a restart; only the installed state is worth keeping."""
        self.leaves.close()

    def remove(self):
        self.remainder.close()
        (self.folder / 'remainder-layout.json').unlink(missing_ok=True)
        self.model.unlink()
        self.folder.rmdir()


@contextmanager
def connected(app, run, limits=None):
    working = Working(app, run, limits)
    lazy = Delegate(app.turn_pair, turns=lambda forward, *, expected_merges: turn_pair.Turns(forward, expected_merges))
    try:
        with ExitStack() as stack:
            for target, name, value in (
                    (app.factory, 'memory_model', working.create),
                    (app.continuation, 'restore', working.restore(app.continuation.restore)),
                    (app, 'load_module', working.load(app.load_module)),
                    (app, 'sync_parameter', working.sync(app.sync_parameter)),
                    (app, 'turn_pair', lazy)):
                stack.enter_context(patch.object(target, name, value))
            yield working
    finally:
        working.close_scratch()
    # The checkpoint writer publishes its hashed manifest only after both payloads.
    # Preserve the installed model and remainders on every failed or incomplete exit.
    final = app.held.checked(app.ROOT, app.binding(run / 'final-state.json'))
    exited = app.held.checked(app.ROOT, app.binding(run / 'launcher-exit.json'))
    if exited['exit_code'] == 0 and final['status'] == 'complete':
        manifest = app.held.checked(app.ROOT, final['checkpoint'])
        if manifest['complete'] and manifest['model_identity'] == final['model_identity']:
            working.remove()
