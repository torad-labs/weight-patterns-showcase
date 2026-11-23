"""Read the recorded native values and actual connections without inventing edges.

The observation must retain tensor layout, native precision, token/execution
identity and source links. Numeric summaries are views over this record, never
substitutes for it. The reader never writes an activation or a checkpoint.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


DTYPES={'f32':'<f4','f16':'<f2','bf16':'<u2','i32':'<i4','i64':'<i8'}


def _dtype(kind):
    if kind.lower() not in DTYPES:
        raise ValueError(f'Unsupported native type {kind}; no implicit conversion')
    return np.dtype(DTYPES[kind.lower()])


def _values(raw,kind,shape,strides=None):
    dtype=_dtype(kind)
    shape=tuple(int(x) for x in shape)
    if any(x<0 for x in shape):
        raise ValueError('Invalid native tensor shape')
    if 0 in shape:
        if len(raw):
            raise ValueError('Empty native tensor has stored values')
        return np.empty(0,dtype=np.float32 if kind.lower() in ('f16','bf16') else dtype)
    if strides is None:
        flat=np.frombuffer(raw,dtype=dtype)
        if flat.size!=int(np.prod(shape)):
            raise ValueError('Native weight byte count does not match shape')
    else:
        strides=tuple(int(x) for x in strides)
        if len(strides)!=len(shape) or any(x<1 for x in strides):
            raise ValueError('Invalid native tensor strides')
        required=dtype.itemsize+sum((n-1)*s for n,s in zip(shape,strides))
        if required>len(raw):
            raise ValueError('Native tensor layout exceeds stored bytes')
        flat=np.ndarray(shape=shape,dtype=dtype,buffer=raw,strides=strides).ravel(order='F')
    if kind.lower()=='bf16':
        return (flat.astype(np.uint32)<<16).view(np.float32)
    return flat.astype(np.float32) if kind.lower()=='f16' else flat.copy()


def snapshot_values(path, snapshot, verify=True):
    """Reconstruct native chunks and strides as logical x-fast tensor values."""
    chunks=[]
    with Path(path).open('rb') as f:
        for c in snapshot['chunks']:
            f.seek(c['offset']);raw=f.read(c['bytes'])
            if len(raw)!=c['bytes'] or (verify and hashlib.sha256(raw).hexdigest()!=c['sha256']):
                raise ValueError('Native chunk is truncated or has a different hash')
            chunks.append(raw)
    raw=b''.join(chunks)
    if len(raw)!=snapshot['bytes']:
        raise ValueError('Native chunk lengths do not match tensor bytes')
    return _values(raw,snapshot['type'],snapshot['shape'],snapshot['strides'])


def checkpoint_values(path, weight):
    """Read the weight's declared native storage, refusing unknown formats."""
    _dtype(weight['native_type'])
    with Path(path).open('rb') as f:
        f.seek(weight['offset_bytes']);raw=f.read(weight['bytes'])
    if len(raw)!=weight['bytes']:
        raise ValueError('Native weight storage is truncated')
    return _values(raw,weight['native_type'],weight['shape'])


def linked_inputs(event):
    """Distinguish observed producers, versioned cache reads, weights and gaps."""
    links=[]
    for port,x in enumerate(event['inputs']):
        if x is None:
            continue
        link={'port':port,'name':x['layout']['name']}
        if 'weight' in x:
            link.update(kind='weight',weight=x['weight'])
        elif x.get('cache_state') is not None:
            link.update(kind='cache',event=x['cache_state'],storage=x.get('cache_storage'))
        elif x.get('producer_event') is not None:
            link.update(kind='producer',event=x['producer_event'])
        else:
            link.update(kind='unresolved',reason=x.get('source_status','No producer occurrence recorded'))
        links.append(link)
    return links


def events(path):
    """Stream original event records; caller chooses their evidence view."""
    with Path(path).open() as f:
        for line in f:
            yield json.loads(line)
