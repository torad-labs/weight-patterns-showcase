"""Typed native state for the Final Experiment; no controller or automatic resume.

The caller owns a writable working GGUF/memfd, preloaded from its declared base.
Shapes are resident/output-major C order (GGUF dimensions reversed after dropping
trailing singleton dimensions). Embedding directions use sorted unique physical
row IDs. Preview is read-only and can feed the coefficient-one installed JVP;
installation recomputes/streams the same chunks, without a per-step model copy.

A transaction cannot supply a fresh model until every planned native span and its
derivative-model callback completed. Failure preserves the actual partial state,
marked incomplete; recovery is explicit from a complete exit checkpoint only.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import mmap
import os
from pathlib import Path

import numpy as np

FORMAT = 'full-native-state/1'


dense_allocator = None  # optional callable(shape, dtype=...) -> writable array


def _identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _write(fd, offset, raw):
    view = memoryview(raw).cast('B')
    while view:
        n = os.pwrite(fd, view, offset)
        if n <= 0:
            raise OSError('native write made no progress')
        offset += n
        view = view[n:]


def _write_changed(fd, offset, before, raw):
    """Write the pages of raw that differ from before, the bytes the file holds at offset.

    A step changes few stored values (about 22,000 of 1.9 billion on November 19, 2026), so most
    pages of a span already hold their new bytes, and writing the span whole sent the model file
    to the disk again at every step.
    """
    differing = np.flatnonzero(np.frombuffer(before, dtype=np.uint8) != np.frombuffer(raw, dtype=np.uint8))
    if not differing.size:
        return
    pages = np.unique((offset + differing) // mmap.PAGESIZE)
    for run in np.split(pages, np.flatnonzero(np.diff(pages) != 1) + 1):
        low = max(offset, int(run[0]) * mmap.PAGESIZE)
        high = min(offset + len(raw), (int(run[-1]) + 1) * mmap.PAGESIZE)
        _write(fd, low, memoryview(raw)[low - offset:high - offset])


def _read(fd, offset, size):
    pieces = []
    while size:
        part = os.pread(fd, size, offset)
        if not part:
            raise ValueError('native payload is truncated')
        pieces.append(part)
        size -= len(part)
        offset += len(part)
    return b''.join(pieces)


def _digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def _empty_receipt(spec):
    return dict(name=spec.name, native_type=spec.native_type, shape=spec.shape,
                complete=False, chunks=0, elements=0, changed_entries=0,
                requested_nonzero_entries=0, fresh_direction_nonzero_entries=0,
                displacement_squared=0., remainder_squared=0., remainder_before_squared=0.,
                remainder_after_squared=0.,
                displacement_dtype='float64', exact_in_f32=True)


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple
    native_type: str
    offset_bytes: int
    sparse_rows: bool = False
    aliases: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, 'shape', tuple(self.shape))
        object.__setattr__(self, 'aliases', tuple(self.aliases))
        if (not self.name or not self.shape or any(type(n) is not int or n <= 0 for n in self.shape)
                or self.native_type not in ('bf16', 'f32') or type(self.offset_bytes) is not int
                or self.offset_bytes < 0 or (self.sparse_rows and len(self.shape) != 2)):
            raise ValueError('invalid typed native tensor specification')

    @classmethod
    def from_binding(cls, binding, *, sparse_rows=False, aliases=()):
        dimensions = list(binding['shape'])
        while len(dimensions) > 1 and dimensions[-1] == 1:
            dimensions.pop()
        result = cls(binding['name'], tuple(reversed(dimensions)), binding['native_type'],
                     binding['offset_bytes'], sparse_rows, tuple(aliases))
        if binding['bytes'] != result.nbytes:
            raise ValueError('native tensor byte count differs from dtype/shape')
        return result

    @property
    def size(self):
        return math.prod(self.shape)

    @property
    def itemsize(self):
        return 2 if self.native_type == 'bf16' else 4

    @property
    def nbytes(self):
        return self.size * self.itemsize


def decode(raw, native_type):
    if native_type == 'bf16':
        bits = np.frombuffer(raw, dtype='<u2').astype('<u4')
        np.left_shift(bits, 16, out=bits)
        return bits.view('<f4')
    if native_type == 'f32':
        return np.frombuffer(raw, dtype='<f4').copy()
    raise ValueError('unsupported native type')


def _bf16_bytes(values):
    """Same ggml RNE/flush bits as a2.to_bf16, without U64 array temporaries.

    The final uint16 result depends only on the low32 bits of the rounded sum:
    a uint32 carry discarded here would also be discarded by the old uint16 cast.
    Neither the caller's F32 values nor their bit view is modified.
    """
    bits = np.ascontiguousarray(values, dtype='<f4').view('<u4')
    rounded = bits.copy()
    tie = np.right_shift(bits, 16)
    np.bitwise_and(tie, 1, out=tie)
    np.add(rounded, tie, out=rounded)
    del tie
    np.add(rounded, np.uint32(0x7fff), out=rounded)
    np.right_shift(rounded, 16, out=rounded)
    subnormal = (bits & np.uint32(0x7f800000)) == 0
    np.bitwise_and(rounded, np.uint32(0x8000), out=rounded, where=subnormal)
    del subnormal
    return rounded.astype('<u2').tobytes(order='C')


def accumulate(base, remainder, direction, length, native_type, *, receipt=None):
    """Specified F64 temporaries, F32 remainder, exact native endpoint bytes.

    Direction/base/remainder are F32 arrays. BF16 deliberately retains the old
    wanted32 conversion; F32 forms its residual before that conversion loses it.
    Optional receipt counts the ordered U64-minus-base request before storage
    conversions, including E; fresh_direction_nonzero_entries counts S separately.
    """
    if (not math.isfinite(length) or length < 0 or native_type not in ('bf16', 'f32')
            or any(not isinstance(v, np.ndarray) or v.dtype != np.dtype('<f4')
                   or v.shape != base.shape or not np.isfinite(v).all()
                   for v in (base, remainder, direction))):
        raise ValueError('finite matching F32 operands and nonnegative finite length required')
    with np.errstate(over='ignore', invalid='ignore'):
        wanted = base.astype(np.float64)
        wanted += remainder
        scaled = direction.astype(np.float64)
        scaled *= length
        wanted += scaled
        del scaled
        if not np.isfinite(wanted).all():
            raise ValueError('nonfinite native endpoint or remainder')
        if receipt is not None:
            # Count before reusing/freeing the ordered wanted accumulator.
            receipt.update(requested_nonzero_entries=int(np.count_nonzero(wanted != base)),
                           fresh_direction_nonzero_entries=int(np.count_nonzero(direction)),
                           remainder_before_squared=float(np.einsum('i,i->', remainder.reshape(-1),
                               remainder.reshape(-1), dtype=np.float64)))
        wanted32 = wanted.astype('<f4')
        if native_type == 'bf16':
            del wanted
            raw = _bf16_bytes(wanted32)
            stored = decode(raw, 'bf16').reshape(base.shape)
            residual = wanted32 - stored
            del wanted32
        else:
            stored = wanted32
            raw = stored.tobytes(order='C')
            # NumPy promotes the F32 operand to the F64 accumulator dtype;
            # subtraction order is unchanged, without another full F64 array.
            np.subtract(wanted, stored, out=wanted)
            residual = wanted.astype('<f4')
            del wanted
    if not all(np.isfinite(v).all() for v in (stored, residual)):
        raise ValueError('nonfinite native endpoint or remainder')
    delta64 = stored.astype(np.float64)
    np.subtract(delta64, base, out=delta64)
    with np.errstate(over='ignore', invalid='ignore'):
        delta32 = delta64.astype('<f4')
    # array_equal compares F32/F64 arrays in the promoted F64 dtype; no full
    # materialized upcast is needed to test the same displacement fidelity.
    return raw, stored, residual, delta64, bool(np.array_equal(delta32, delta64))


@dataclass
class Chunk:
    name: str
    start: int                 # flattened physical native tensor index
    before: bytes
    raw: bytes
    stored: np.ndarray
    remainder: np.ndarray
    displacement: np.ndarray   # F64 endpoint subtraction, not a fitted direction
    exact_in_f32: bool
    source_identity: str
    requested_nonzero_entries: int
    fresh_direction_nonzero_entries: int
    remainder_before_squared: float


class FullState:
    """One owner of native state, sparse/dense remainders and transition receipts.

    Native fd remains caller-owned. chunk_elements is an operational memory bound,
    supplied by the caller, never a training parameter. sync(name,start,values)
    must copy each installed flattened physical span into the derivative model.
    Use complete=False while bootstrapping a seed before restore_legacy; naming
    the saved317 identity must not prematurely claim its overrides are installed.
    """
    def __init__(self, fd, specs, *, model_identity, base_identity, iteration=0,
                 chunk_elements, journal=None, complete=True):
        self.fd, self.chunk_elements = fd, chunk_elements
        if type(chunk_elements) is not int or chunk_elements <= 0:
            raise ValueError('positive chunk capacity required')
        self.specs, self.names = {}, {}
        for spec in specs:
            if spec.name in self.specs:
                raise ValueError('duplicate physical tensor')
            self.specs[spec.name] = spec
            for name in (spec.name, *spec.aliases):
                if name in self.names:
                    raise ValueError('duplicate alias or physical name')
                self.names[name] = spec.name
        spans = sorted((s.offset_bytes, s.offset_bytes + s.nbytes) for s in self.specs.values())
        if (not spans or spans[-1][1] > os.fstat(fd).st_size
                or any(a[1] > b[0] for a, b in zip(spans, spans[1:]))):
            raise ValueError('canonical native spans must be disjoint and inside working model')
        self.inventory = [asdict(s) for s in self.specs.values()]
        self.inventory_identity = _identity(self.inventory)
        self.model_identity, self.base_identity = model_identity, base_identity
        self.iteration, self.status, self.transition = iteration, 'complete' if complete else 'uninitialized', None
        self.remainders = {}     # absent dense or absent row means exactly zero
        self.journal = None if journal is None else Path(journal)
        if self.journal is not None and self.journal.exists():
            raise ValueError('existing transition journal requires explicit recovery, never overwrite')
        self._persist()

    def receipt(self):
        return dict(format=FORMAT, status=self.status, iteration=self.iteration,
                    model_identity=self.model_identity, base_identity=self.base_identity,
                    inventory_identity=self.inventory_identity, transition=self.transition)

    def rebind_complete_identity(self, next_identity, *, iteration=0, provenance):
        """Name a new controller baseline without changing native/remainder bytes."""
        if (self.status != 'complete' or type(iteration) is not int or iteration < 0
                or next_identity == self.model_identity or not isinstance(provenance, dict)
                or not provenance):
            raise ValueError('identity handoff requires a complete state and explicit provenance')
        _identity(next_identity)
        _identity(provenance)
        self.transition = dict(kind='identity-handoff', source_identity=self.model_identity,
                               source_iteration=self.iteration, next_identity=next_identity,
                               iteration=iteration, provenance=provenance,
                               native_and_remainder_bytes_unchanged=True)
        self.model_identity, self.iteration = next_identity, iteration
        self._persist()
        return self.receipt()

    def _persist(self):
        if self.journal is not None:
            temporary = self.journal.with_name(self.journal.name + '.next')
            temporary.write_text(json.dumps(self.receipt(), indent=2) + '\n')
            os.replace(temporary, self.journal)

    def _rows(self, spec, rows):
        if not spec.sparse_rows:
            if rows is not None:
                raise ValueError('dense tensor does not accept row selection')
            return None
        values = np.asarray(rows)
        if (values.ndim != 1 or (len(values) and values.dtype.kind not in 'iu') or np.any(values < 0)
                or np.any(values >= spec.shape[0]) or np.any(values[1:] <= values[:-1])):
            raise ValueError('sparse rows must be sorted unique in-range integer IDs')
        return tuple(int(v) for v in values)

    def required_rows(self, name, current_rows):
        """Current gradient rows plus retained rows needing their zero-S recurrence."""
        spec = self.specs[self.names[name]]
        if not spec.sparse_rows:
            raise ValueError('row inventory is only for sparse tables')
        rows = self._rows(spec, current_rows)
        return tuple(sorted(set(rows) | set(self.remainders.get(spec.name, {}))))

    def expand_rows(self, name, direction, current_rows):
        """Expand compact graph directions; dormant retained rows have S=0."""
        spec = self.specs[self.names[name]]
        current = self._rows(spec, current_rows)
        if (not isinstance(direction, np.ndarray) or direction.dtype != np.dtype('<f4')
                or direction.shape != (len(current), spec.shape[1])):
            raise ValueError('compact table direction differs from current graph rows')
        rows = self.required_rows(name, current)
        expanded = np.zeros((len(rows), spec.shape[1]), dtype='<f4')
        positions = {r: i for i, r in enumerate(rows)}
        for i, r in enumerate(current):
            expanded[positions[r]] = direction[i]
        return expanded, rows

    def _ranges(self, spec, rows):
        if rows is None:
            yield 0, spec.size
        else:
            for row in rows:
                yield row * spec.shape[1], (row + 1) * spec.shape[1]

    def remainder(self, name, rows=None):
        spec = self.specs[self.names[name]]
        rows = self._rows(spec, rows)
        value = self.remainders.get(spec.name)
        if rows is None:
            return np.zeros(spec.shape, dtype='<f4') if value is None else value.copy()
        return np.stack([np.zeros(spec.shape[1], dtype='<f4') if value is None or r not in value
                         else value[r].copy() for r in rows]) if rows else np.empty((0, spec.shape[1]), dtype='<f4')

    def _remainder_span(self, spec, start, count):
        value = self.remainders.get(spec.name)
        if value is None:
            return np.zeros(count, dtype='<f4')
        if not spec.sparse_rows:
            return value.reshape(-1)[start:start + count]
        row, column = divmod(start, spec.shape[1])
        if column + count > spec.shape[1]:
            raise ValueError('sparse chunk crosses physical row')
        return np.zeros(count, dtype='<f4') if row not in value else value[row][column:column + count]

    def preview(self, name, direction, length, *, rows=None):
        """Stream read-only proposals at source identity, including zero directions.

        Iterate once for the installed-displacement JVP and again to install, or
        consume immediately. Caller must finish all source-state derivative work
        before installing the first chunk. No other writer may use the descriptor.
        """
        if self.status not in ('complete', 'installing'):
            raise ValueError('preview requires a complete source state')
        spec = self.specs[self.names[name]]
        selected = self._rows(spec, rows)
        expected = spec.shape if selected is None else (len(selected), spec.shape[1])
        if (not isinstance(direction, np.ndarray) or direction.dtype != np.dtype('<f4')
                or direction.shape != expected):
            raise ValueError('direction differs from declared dense/compact-row layout')
        source = _identity(self.model_identity)
        flat, cursor = direction.reshape(-1), 0
        for begin, end in self._ranges(spec, selected):
            for start in range(begin, end, self.chunk_elements):
                count = min(self.chunk_elements, end - start)
                before = _read(self.fd, spec.offset_bytes + start * spec.itemsize, count * spec.itemsize)
                base = decode(before, spec.native_type)
                part = flat[cursor:cursor + count]
                numbers = {}
                result = accumulate(base, self._remainder_span(spec, start, count),
                                    part, length, spec.native_type, receipt=numbers)
                cursor += count
                chunk = Chunk(spec.name, start, before, *result, source, **numbers)
                del base, result
                yield chunk
                del chunk

    def begin(self, selections, *, expected_identity, next_identity, sync):
        """Selections must name every canonical tensor; sparse zero rows may be [].

        Aliases cannot form a second update. An explicit empty sparse selection is
        an acknowledgement of no consumed rows, not omission of that parameter.
        """
        if (self.status != 'complete' or expected_identity != self.model_identity
                or set(selections) != set(self.specs) or not callable(sync)
                or next_identity == expected_identity):
            raise ValueError('complete explicit inventory, source identity and synchronizer required')
        plans = {name: self._rows(spec, selections[name]) for name, spec in self.specs.items()}
        if any(spec.sparse_rows and set(self.remainders.get(name, {})) - set(plans[name])
               for name, spec in self.specs.items()):
            raise ValueError('sparse installation must include retained rows with zero direction if dormant')
        self._plans, self._seen, self._sync = plans, {n: [] for n in self.specs}, sync
        self._next_identity = next_identity
        self.transition = dict(source_identity=self.model_identity, next_identity=next_identity,
                               iteration=self.iteration + 1, selections=plans,
                               committed={n: _empty_receipt(s) for n, s in self.specs.items()}, in_progress=None)
        self.status = 'installing'
        self._persist()

    def install(self, chunk):
        """Install one previewed chunk; any exception keeps the state incomplete."""
        if self.status != 'installing' or chunk.source_identity != _identity(self.model_identity):
            raise ValueError('chunk belongs to another or incomplete source state')
        spec = self.specs[chunk.name]
        start, end = chunk.start, chunk.start + len(chunk.stored)
        ranges = list(self._ranges(spec, self._plans[spec.name]))
        if (not any(a <= start < end <= b for a, b in ranges)
                or any(start < b and end > a for a, b in self._seen[spec.name])
                or chunk.stored.dtype != np.dtype('<f4') or chunk.remainder.dtype != np.dtype('<f4')
                or chunk.stored.ndim != 1 or chunk.remainder.shape != chunk.stored.shape
                or len(chunk.raw) != len(chunk.stored) * spec.itemsize
                or len(chunk.before) != len(chunk.raw)
                or not np.isfinite(chunk.stored).all() or not np.isfinite(chunk.remainder).all()
                or decode(chunk.raw, spec.native_type).tobytes() != chunk.stored.tobytes()
                or not np.array_equal(chunk.displacement, chunk.stored.astype(np.float64)
                                      - decode(chunk.before, spec.native_type).astype(np.float64))):
            raise ValueError('invalid, overlapping or undeclared installation chunk')
        offset = spec.offset_bytes + start * spec.itemsize
        if _read(self.fd, offset, len(chunk.before)) != chunk.before:
            raise ValueError('native source span changed after preview')
        self.transition['in_progress'] = dict(name=spec.name, start=start, end=end)
        self._persist()
        try:
            _write_changed(self.fd, offset, chunk.before, chunk.raw)   # the file holds chunk.before: checked above
            if _read(self.fd, offset, len(chunk.raw)) != chunk.raw:
                raise OSError('native span readback differs')
            self._sync(spec.name, start, chunk.stored)
            if spec.sparse_rows:
                rows = self.remainders.setdefault(spec.name, {})
                row, column = divmod(start, spec.shape[1])
                rows.setdefault(row, np.zeros(spec.shape[1], dtype='<f4'))[column:column + len(chunk.stored)] = chunk.remainder
            else:
                if spec.name not in self.remainders:
                    self.remainders[spec.name] = np.zeros(spec.shape, dtype='<f4')
                value = self.remainders[spec.name]
                value.reshape(-1)[start:end] = chunk.remainder
            self._seen[spec.name].append((start, end))
            # One bounded scalar receipt per canonical tensor, not an ever-growing
            # per-chunk journal or a second copy of model payloads.
            receipt = self.transition['committed'][spec.name]
            receipt['chunks'] += 1
            receipt['elements'] += len(chunk.stored)
            receipt['changed_entries'] += int(np.count_nonzero(chunk.displacement))
            receipt['requested_nonzero_entries'] += chunk.requested_nonzero_entries
            receipt['fresh_direction_nonzero_entries'] += chunk.fresh_direction_nonzero_entries
            receipt['remainder_before_squared'] += chunk.remainder_before_squared
            receipt['displacement_squared'] += float(np.dot(chunk.displacement, chunk.displacement))
            receipt['remainder_squared'] += float(np.einsum('i,i->', chunk.remainder, chunk.remainder, dtype=np.float64))
            receipt['remainder_after_squared'] = receipt['remainder_squared']
            receipt['exact_in_f32'] &= chunk.exact_in_f32
            self.transition['in_progress'] = None
            self._persist()
        except BaseException:
            self.status = 'incomplete'
            self._persist()
            raise

    def finish(self):
        if self.status != 'installing':
            raise ValueError('no finishable installation')
        for name, spec in self.specs.items():
            actual = sorted(self._seen[name])
            expected = list(self._ranges(spec, self._plans[name]))
            for begin, end in expected:
                at = begin
                for a, b in actual:
                    if begin <= a < end:
                        if a != at:
                            raise ValueError('installation has a missing span')
                        at = b
                if at != end:
                    raise ValueError('installation has a missing span')
        self.status = 'complete'
        for receipt in self.transition['committed'].values():
            receipt['complete'] = True
        self.iteration += 1
        self.model_identity = self._next_identity
        self._persist()
        return self.receipt()

    def import_legacy_remainders(self, record, folder, *, _restoring=False):
        """Extend a complete saved317 BF16/F32 remainder state without resetting it.

        Caller first restores the matching native overrides/base seed and supplies
        that exact legacy model_identity. The old layout, payload hash and shape
        are checked; only those saved slots are populated, all new slots stay zero.
        """
        saved = record['accumulator']
        if (self.status != ('restoring' if _restoring else 'complete') or self.remainders or record.get('weight_state') != 'complete'
                or record.get('pending_install') is not None or not saved.get('complete')
                or saved.get('format') != 'bf16-fp32-remainder/1' or saved.get('dtype') != '<f4'
                or saved['model_identity'] != record['model_identity']
                or record['model_identity'] != self.model_identity or saved['iteration'] != record['update']):
            raise ValueError('legacy accumulator is incomplete or belongs to another state')
        path = _payload(folder, saved['file'])
        if path.stat().st_size != saved['bytes'] or _digest(path) != saved['sha256']:
            raise ValueError('legacy remainder payload hash/size differs')
        cursor, seen, loaded = 0, set(), {}
        with path.open('rb') as stream:
            for row in saved['layout']:
                spec = self.specs[row['name']]
                if (spec.name in seen or spec.sparse_rows or spec.native_type != 'bf16'
                        or list(spec.shape) != row['shape'] or row['offset_bytes'] != cursor
                        or row['bytes'] != spec.size * 4):
                    raise ValueError('legacy remainder layout differs')
                value = np.frombuffer(stream.read(row['bytes']), dtype='<f4').copy().reshape(spec.shape)
                if not np.isfinite(value).all():
                    raise ValueError('legacy remainder is nonfinite')
                loaded[spec.name] = value
                cursor += row['bytes']
                seen.add(spec.name)
        if cursor != saved['bytes'] or seen != {v['name'] for v in record['inventory']}:
            raise ValueError('legacy remainder inventory is incomplete')
        self.remainders = loaded

    def restore_legacy(self, record, folder, *, expected_base_seed, sync):
        """Apply the original saved317 sparse BF16 overrides to its verified seed.

        The caller supplies a native memfd already loaded from expected_base_seed.
        The saved override payload is raw native uint16, in physical flat order;
        matching remainders are validated before any native write. New slots keep
        the native seed bytes and zero remainder. This is explicit, never resume.
        """
        if record['base_seed'] != expected_base_seed or self.base_identity != expected_base_seed or not callable(sync):
            raise ValueError('legacy override seed differs from the declared native seed')
        if self.iteration != record['update']:
            raise ValueError('legacy local iteration differs')
        if self.status not in ('complete', 'uninitialized'):
            raise ValueError('legacy restoration requires an explicit clean bootstrap state')
        self.status = 'restoring'
        self._persist()
        try:
            return self._restore_legacy(record, folder, sync=sync)
        except BaseException:
            self.status = 'incomplete'
            self._persist()
            raise

    def _restore_legacy(self, record, folder, *, sync):
        self.import_legacy_remainders(record, folder, _restoring=True)
        saved = record['overrides']
        path = _payload(folder, saved['file'])
        if _digest(path) != saved['sha256']:
            raise ValueError('legacy native override hash differs')
        with np.load(path, allow_pickle=False) as archive:
            expected_keys = {f'{kind}_{i}' for i in range(len(record['inventory'])) for kind in ('index', 'value')}
            if set(archive.files) != expected_keys:
                raise ValueError('legacy override array inventory differs')
            # Validate every array before the first write. Do not retain all70.
            for i, row in enumerate(record['inventory']):
                spec = self.specs[row['name']]
                indexes, values = archive[f'index_{i}'], archive[f'value_{i}']
                if (spec.native_type != 'bf16' or indexes.dtype != np.dtype('<i8')
                        or values.dtype != np.dtype('<u2') or indexes.ndim != 1
                        or values.shape != indexes.shape or len(indexes) != row['entries']
                        or np.any(indexes < 0) or np.any(indexes >= spec.size)
                        or np.any(indexes[1:] <= indexes[:-1])
                        or not np.isfinite(decode(values.tobytes(), 'bf16')).all()):
                    raise ValueError('invalid legacy sparse native overrides')
            self.status = 'restoring'
            self._persist()
            try:
                for i, row in enumerate(record['inventory']):
                    spec = self.specs[row['name']]
                    indexes, values = archive[f'index_{i}'], archive[f'value_{i}']
                    # Coalesce sorted index runs; bounded by chunk_elements.
                    cursor = 0
                    while cursor < len(indexes):
                        end = cursor + 1
                        while (end < len(indexes) and end - cursor < self.chunk_elements
                               and indexes[end] == indexes[end - 1] + 1):
                            end += 1
                        start = int(indexes[cursor])
                        raw = values[cursor:end].tobytes()
                        offset = spec.offset_bytes + start * spec.itemsize
                        _write(self.fd, offset, raw)
                        if _read(self.fd, offset, len(raw)) != raw:
                            raise OSError('legacy native override readback differs')
                        sync(spec.name, start, decode(raw, spec.native_type))
                        cursor = end
                self.status = 'complete'
                self._persist()
            except BaseException:
                self.status = 'incomplete'
                self._persist()
                raise

    @classmethod
    def restore(cls, fd, folder, *, specs, expected_base_identity, expected_model_identity,
                chunk_elements, sync, journal=None):
        """Explicit restore of a complete typed checkpoint onto its declared base.

        Validates all payload hashes/layout/finite values before the first native
        write. A failing native write or derivative callback leaves journal state
        incomplete. The supplied descriptor's bytes outside eligible spans must
        already be the verified declared base, established by the controller.
        """
        folder, specs = Path(folder), tuple(specs)
        record = json.loads((folder / 'state.json').read_text())
        inventory = [asdict(s) for s in specs]
        if (record.get('format') != FORMAT or not record.get('complete')
                or record.get('status') != 'complete' or not callable(sync)
                or record['base_identity'] != expected_base_identity
                or record['model_identity'] != expected_model_identity
                or _identity(record['inventory']) != _identity(inventory)
                or record['inventory_identity'] != _identity(inventory)):
            raise ValueError('checkpoint identity/inventory is incomplete or differs')
        self = cls(fd, specs, model_identity=expected_model_identity,
                   base_identity=expected_base_identity, iteration=record['iteration'],
                   chunk_elements=chunk_elements)  # no ready journal before validation
        if journal is not None and Path(journal).exists():
            raise ValueError('restore requires a new journal; preserve the existing partial record')
        paths = {}
        for kind in ('native', 'remainder'):
            info = record[kind]
            path = _payload(folder, info['file'])
            if path.stat().st_size != info['bytes'] or _digest(path) != info['sha256']:
                raise ValueError('checkpoint payload hash/size differs')
            paths[kind] = path
        if [r['name'] for r in record['layout']] != list(self.specs):
            raise ValueError('checkpoint physical layout inventory differs')
        at, rat = 0, 0
        with paths['native'].open('rb') as native, paths['remainder'].open('rb') as remaining:
            for row in record['layout']:
                spec = self.specs[row['name']]
                if row['offset_bytes'] != at or row['bytes'] != spec.nbytes:
                    raise ValueError('checkpoint native layout differs')
                for begin in range(0, spec.size, chunk_elements):
                    count = min(chunk_elements, spec.size - begin)
                    if not np.isfinite(decode(native.read(count * spec.itemsize), spec.native_type)).all():
                        raise ValueError('checkpoint native tensor is nonfinite')
                at += spec.nbytes
                info = row['remainder']
                if info['kind'] == 'zero':
                    continue
                ids = self._rows(spec, info['rows']) if spec.sparse_rows else None
                count = spec.size if ids is None else len(ids) * spec.shape[1]
                if (info['kind'] != ('rows' if spec.sparse_rows else 'dense')
                        or info['offset_bytes'] != rat or info['bytes'] != count * 4):
                    raise ValueError('checkpoint remainder layout differs')
                if ids is None:
                    # A controller may supply file-backed storage so a restored dense
                    # remainder is read once, straight into place, never held in RAM.
                    value = (dense_allocator or np.empty)(spec.shape, dtype='<f4')
                    view = memoryview(value).cast('B')
                    if remaining.readinto(view) != len(view) or not np.isfinite(value).all():
                        raise ValueError('invalid dense remainder')
                    self.remainders[spec.name] = value
                else:
                    values = {}
                    for r in ids:
                        value = np.frombuffer(remaining.read(spec.shape[1] * 4), dtype='<f4').copy()
                        if len(value) != spec.shape[1] or not np.isfinite(value).all():
                            raise ValueError('invalid sparse remainder')
                        values[r] = value
                    self.remainders[spec.name] = values
                rat += info['bytes']
        if at != record['native']['bytes'] or rat != record['remainder']['bytes']:
            raise ValueError('checkpoint has unaccounted payload bytes')
        self.journal = None if journal is None else Path(journal)
        self.status = 'restoring'
        self.transition = dict(restored_from=str(folder), in_progress=None)
        self._persist()
        try:
            with paths['native'].open('rb') as native:
                for row in record['layout']:
                    spec = self.specs[row['name']]
                    for begin in range(0, spec.size, chunk_elements):
                        count = min(chunk_elements, spec.size - begin)
                        raw = native.read(count * spec.itemsize)
                        offset = spec.offset_bytes + begin * spec.itemsize
                        self.transition['in_progress'] = dict(name=spec.name, start=begin)
                        self._persist()
                        _write(fd, offset, raw)
                        if _read(fd, offset, len(raw)) != raw:
                            raise OSError('restored native span differs')
                        sync(spec.name, begin, decode(raw, spec.native_type))
            self.status, self.transition = 'complete', record.get('transition')
            self._persist()
            return self
        except BaseException:
            self.status = 'incomplete'
            self._persist()
            raise

    def checkpoint(self, folder):
        """Explicit exit/handoff only. Saves all eligible native bytes, no whole GGUF.

        Outside bytes are supplied by the declared base_identity on recovery.
        Incomplete checkpoints preserve evidence but are refused for recovery.
        A manifest is published last; an interrupted payload write is not complete.
        """
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=False)
        native_path, remainder_path = folder / 'native.bin', folder / 'remainder.f32'
        layout, at, rat = [], 0, 0
        with native_path.open('xb') as native, remainder_path.open('xb') as remaining:
            for name, spec in self.specs.items():
                row = dict(name=name, offset_bytes=at, bytes=spec.nbytes)
                for begin in range(0, spec.size, self.chunk_elements):
                    count = min(self.chunk_elements, spec.size - begin)
                    native.write(_read(self.fd, spec.offset_bytes + begin * spec.itemsize, count * spec.itemsize))
                at += spec.nbytes
                value = self.remainders.get(name)
                row['remainder'] = dict(kind='zero')
                if value is not None:
                    ids = sorted(value) if spec.sparse_rows else None
                    row['remainder'] = dict(kind='rows' if spec.sparse_rows else 'dense',
                                            offset_bytes=rat, rows=ids)
                    arrays = (value[r] for r in ids) if ids is not None else (value,)
                    for array in arrays:
                        remaining.write(memoryview(np.ascontiguousarray(array, dtype='<f4')).cast('B'))
                        rat += array.nbytes
                    row['remainder']['bytes'] = rat - row['remainder']['offset_bytes']
                layout.append(row)
            native.flush(); os.fsync(native.fileno())
            remaining.flush(); os.fsync(remaining.fileno())
        result = dict(self.receipt(), inventory=self.inventory, layout=layout,
                      complete=self.status == 'complete', native=dict(file=native_path.name, bytes=at, sha256=_digest(native_path)),
                      remainder=dict(file=remainder_path.name, bytes=rat, sha256=_digest(remainder_path)))
        path = folder / 'state.json'
        with path.open('x') as f:
            json.dump(result, f, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
        return result


def _payload(folder, name):
    if Path(name).name != name:
        raise ValueError('payload must be beside its checkpoint')
    return Path(folder) / name
