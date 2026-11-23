"""Host memory that follows the piece being worked on.

A Store hands out arrays backed by a private file in a run's working folder.
To numpy and torch such an array is ordinary CPU memory. release() takes its
pages out of this process and leaves them to the kernel's page cache, so a set
of tensors far larger than is wanted in RAM is worked one tensor at a time; a
released array comes back the next time it is touched, from the cache while
the machine has the memory free and from the file once it has not. Releasing
never changes a value, and releasing memory no Store handed out does nothing.

The cache is the kernel's to keep or drop. A released page is file cache: the
kernel takes it back the moment something else needs the memory, and writes a
changed one to the file in the background. So every page a phase changes still
reaches the disk once: working arrays in files cost that write, and only arrays
kept in RAM would not. What is gone is the forcing. Flushing each release and
dropping it from the cache, with the files other code reads dropped once a
second, made every later use a fresh read of the disk (November 19, 2026:
2.4 TB read and 258 GB written in four and a half hours of one run, with half
the machine's memory free). Store.sync() waits for a file's background writes
at the end of a phase, so a failed write stops the run instead of passing.
"""
import bisect
import math
import mmap
import os
from pathlib import Path
import resource
import threading

import numpy as np
import torch
from torch.autograd import forward_ad

PAGE = mmap.PAGESIZE
_FLAGS = os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC

_lock = threading.Lock()
_starts = []    # mapped array addresses, sorted
_regions = {}   # address -> (mapping, bytes, fd, file offset, path)


def _region(address):
    with _lock:
        index = bisect.bisect_right(_starts, address) - 1
        if index < 0:
            return None
        start = _starts[index]
        region = _regions[start]
    return (start, *region) if address < start + region[1] else None


def _spans(value):
    if isinstance(value, np.ndarray):
        if value.size:
            low, high = np.lib.array_utils.byte_bounds(value)
            yield low, high - low
    elif isinstance(value, torch.Tensor):
        # A forward-mode dual carries two host tensors: its value and its tangent.
        for part in forward_ad.unpack_dual(value):
            if (part is not None and part.device.type == 'cpu' and part.layout == torch.strided
                    and part.numel()):
                storage = part.untyped_storage()
                yield storage.data_ptr(), storage.nbytes()


def _pages(value):
    for address, nbytes in _spans(value):
        found = _region(address)
        if found is None:
            continue
        start, mapping, length, fd, offset, path = found
        low = (address - start) // PAGE * PAGE
        high = min(length, -(-(address - start + nbytes) // PAGE) * PAGE)
        yield mapping, low, high - low, fd, offset + low, path


def _return(mapping, low, size):
    # The mapping is shared with its file: a page this process lets go of stays in the page cache,
    # a changed one still marked to be written, and reads back the same bytes.
    mapping.madvise(mmap.MADV_DONTNEED, low, size)


def release(*values):
    """Let go of each Store-backed array's or tensor's pages: they stay the kernel's file cache."""
    for value in values:
        for mapping, low, size, *_ in _pages(value):
            _return(mapping, low, size)


def prefetch(*values):
    """Ask for a Store-backed array's pages ahead of a whole read."""
    for value in values:
        for mapping, low, size, *_ in _pages(value):
            mapping.madvise(mmap.MADV_WILLNEED, low, size)


def location(value):
    """Where a Store-backed array lives, as (file name, byte offset), or None for ordinary memory."""
    for address, _ in _spans(value):
        found = _region(address)
        if found is not None:
            start, _, _, _, offset, path = found
            return path.name, offset + address - start
    return None


def file_bytes(sizes):
    """The size of a Store's file once it has handed out arrays of these byte sizes: each takes whole pages."""
    return sum(-(-int(size) // PAGE) * PAGE for size in sizes)


class Store:
    """Arrays in one private file, each on its own pages. reset() starts a new file and lets the old one go."""

    def __init__(self, folder, stem):
        self.folder, self.stem, self.generation = Path(folder), stem, 0
        # Every array is its own mapping and Python keeps a descriptor open for each: the limit this
        # process may raise for itself is raised, so thousands of arrays fit.
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft != hard:
            resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        self._open()

    def _open(self):
        self.path = self.folder / f'{self.stem}-{self.generation}.bin'
        self.fd = os.open(self.path, _FLAGS, 0o600)
        self.size, self.addresses = 0, []

    def array(self, shape, dtype='<f4'):
        dtype, shape = np.dtype(dtype), tuple(int(n) for n in shape)
        count = math.prod(shape)
        if not count:
            return np.empty(shape, dtype=dtype)
        length = file_bytes([count * dtype.itemsize])
        # Allocated, not sparse: a full disk fails here, never as a fault inside a later write.
        os.posix_fallocate(self.fd, self.size, length)
        mapping = mmap.mmap(self.fd, length, offset=self.size)
        value = np.frombuffer(mapping, dtype=dtype, count=count).reshape(shape)
        address = value.__array_interface__['data'][0]
        with _lock:
            bisect.insort(_starts, address)
            _regions[address] = (mapping, length, self.fd, self.size, self.path)
        self.addresses.append(address)
        self.size += length
        return value

    def tensor(self, shape, dtype='<f4'):
        return torch.from_numpy(self.array(shape, dtype))

    def release(self):
        """Let go of every page of this file, whichever array holds it."""
        for address in self.addresses:
            mapping, length, *_ = _regions[address]
            _return(mapping, 0, length)

    def sync(self):
        """Wait until the file's changed pages are written: a write the disk refused is raised here."""
        os.fdatasync(self.fd)

    def _retire(self):
        with _lock:
            for address in self.addresses:
                del _regions[address]
                _starts.remove(address)
        self.addresses = []
        os.close(self.fd)
        self.path.unlink()

    def reset(self):
        """Begin a new empty file. Arrays already handed out stay valid until dropped, but are no longer releasable."""
        self._retire()
        self.generation += 1
        self._open()

    def close(self):
        self._retire()

