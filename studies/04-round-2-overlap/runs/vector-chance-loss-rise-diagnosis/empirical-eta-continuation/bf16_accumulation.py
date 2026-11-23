"""FP32 error-feedback arithmetic and one bound exit-only remainder checkpoint."""
import hashlib
import math
from pathlib import Path

import numpy as np

from pipeline.weight_edit.a2 import from_bf16, to_bf16

FORMAT = 'bf16-fp32-remainder/1'


def layout(tensors, names):
    rows, offset = [], 0
    for name in names:
        tensor = tensors[name]
        shape = [int(tensor['shape'][1]), int(tensor['shape'][0])]
        count = math.prod(shape)
        if (tensor['native_type'] != 'bf16' or tensor['shape'][2:] != [1, 1]
                or tensor['bytes'] != count * 2):
            raise ValueError('accumulation requires the declared complete native matrices')
        rows.append(dict(name=name, shape=shape, offset_bytes=offset, bytes=count * 4))
        offset += count * 4
    return rows


def squared(value):
    flat = value.reshape(-1)
    return float(np.einsum('i,i->', flat, flat, dtype=np.float64, optimize=False))


def step(base, remainder, direction, length):
    """Use the declared arithmetic order, rounding the native endpoint once."""
    if (not math.isfinite(length) or length < 0
            or any(x.dtype != np.float32 or x.shape != base.shape
                   or not np.isfinite(x).all() for x in (base, remainder, direction))):
        raise ValueError('native weights, FP32 remainder and direction must match and be finite')
    # Preserve the prescribed addition order without retaining several full
    # float64 matrix temporaries at once.
    wanted = base.astype(np.float64)
    np.add(wanted, remainder, out=wanted)
    scaled = direction.astype(np.float64)
    scaled *= length
    wanted += scaled
    del scaled
    wanted = wanted.astype(np.float32)
    stored = from_bf16(to_bf16(wanted)).reshape(base.shape)
    np.subtract(wanted, stored, out=wanted)
    if not np.isfinite(stored).all() or not np.isfinite(wanted).all():
        raise ValueError('accumulated endpoint or retained remainder is nonfinite')
    return stored, wanted


def actual_delta(base, stored):
    delta = stored - base
    changed = stored != base
    exact = stored[changed].astype(np.float64) - base[changed].astype(np.float64)
    if not np.array_equal(delta[changed].astype(np.float64), exact):
        raise ValueError('the total installed displacement is not exact in the JVP dtype')
    return delta


def validate_arrays(remainders, rows):
    if set(remainders) != {row['name'] for row in rows}:
        raise ValueError('the remainder must contain every declared matrix')
    for row in rows:
        value = remainders[row['name']]
        if (value.dtype != np.dtype('<f4') or list(value.shape) != row['shape']
                or not value.flags.c_contiguous or not np.isfinite(value).all()):
            raise ValueError('remainder layout or finite FP32 values differ')


def load(root, candidate_binding, candidate, mode, rows):
    """A saved accumulator is mandatory on resume and cannot silently reset."""
    saved = candidate.get('accumulator')
    if mode == 'zero':
        if saved is not None:
            raise ValueError('zero initialization cannot discard a checkpoint accumulator')
        return {row['name']: np.zeros(row['shape'], dtype='<f4') for row in rows}
    if mode != 'resume' or saved is None:
        raise ValueError('declare zero initialization or resume a saved accumulator')
    expected_bytes = sum(row['bytes'] for row in rows)
    if (saved.get('format') != FORMAT or not saved.get('complete')
            or candidate['weight_state'] != 'complete' or candidate['pending_install'] is not None
            or saved['model_identity'] != candidate['model_identity']
            or saved['iteration'] != candidate['update'] or saved['layout'] != rows
            or saved['dtype'] != '<f4' or saved['bytes'] != expected_bytes):
        raise ValueError('remainder checkpoint is incomplete or belongs to another native state')
    filename = saved['file']
    if Path(filename).name != filename:
        raise ValueError('remainder payload must be beside its checkpoint')
    path = Path(root) / Path(candidate_binding['record']).parent / filename
    if path.stat().st_size != expected_bytes:
        raise ValueError('remainder payload has the wrong exact size')
    digest, result = hashlib.sha256(), {}
    with path.open('rb') as stream:
        for row in rows:
            value = np.empty(row['shape'], dtype='<f4')
            view = memoryview(value).cast('B')
            at = 0
            while at < len(view):
                count = stream.readinto(view[at:])
                if not count:
                    raise ValueError('truncated remainder payload')
                at += count
            digest.update(view)
            result[row['name']] = value
        if stream.read(1):
            raise ValueError('remainder payload has trailing bytes')
    if digest.hexdigest() != saved['sha256']:
        raise ValueError('remainder payload differs from its bound SHA256')
    validate_arrays(result, rows)
    return result


def save(run, remainders, rows, *, model_identity, iteration, complete, installation):
    """One full raw FP32 payload at exit; metadata is bound in final-state.json."""
    validate_arrays(remainders, rows)
    path = Path(run) / 'final-remainder.f32'
    digest = hashlib.sha256()
    with path.open('xb') as stream:
        for row in rows:
            view = memoryview(remainders[row['name']]).cast('B')
            at = 0
            while at < len(view):
                written = stream.write(view[at:])
                if not written:
                    raise OSError('remainder checkpoint write made no progress')
                at += written
            digest.update(view)
    expected_bytes = sum(row['bytes'] for row in rows)
    if path.stat().st_size != expected_bytes:
        raise OSError('remainder checkpoint did not reach its exact required size')
    return dict(format=FORMAT, file=path.name, sha256=digest.hexdigest(),
                dtype='<f4', bytes=expected_bytes, layout=rows,
                model_identity=model_identity, iteration=iteration, complete=complete,
                installation=installation,
                meaning='FP32 wanted-minus-stored remainder, resident(output,input) C order')
