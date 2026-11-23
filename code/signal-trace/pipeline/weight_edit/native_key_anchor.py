"""Rebase post-RoPE keys at a measured native origin, before the original cache cast."""
from inspect import unwrap
import json
from pathlib import Path
from types import MethodType

import numpy as np
import torch


def load_native_keys(raw, model, events):
    """Read packed native [position, KV head, head coordinate] with exact call coverage."""
    raw = Path(raw)
    values = np.memmap(raw/'stream.f32', mode='r', dtype='<f4')
    layers = range(model.keys_from)
    expected = {layer: {(row['call'],row['microbatch']): tuple(row['positions'])
                        for row in model.attention_calls[layer]} for layer in layers}
    keys, seen = {}, {layer:set() for layer in layers}
    with (raw/'stream.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            name = row['label']
            if not name.startswith('Kcur_pos-'):
                continue
            layer = int(name.removeprefix('Kcur_pos-'))
            call = (row['call'],row['microbatch'])
            ids = tuple(row['positions'])
            if (layer not in expected or call in seen[layer] or expected[layer].get(call) != ids
                    or not ids or ids != tuple(range(row['microbatch_start'],row['microbatch_start']+row['microbatch_count']))
                    or row['position_basis']!='materialized microbatch order' or row['name'] != name
                    or row['operation']!='ROPE' or row['native_type']!='f32' or row['token_axis']!=2):
                raise ValueError(f'Unexpected, repeated or mismatched native K call: {name}, {call}')
            width, heads, count, outer = row['shape']
            if (width != model.head(layer) or heads < 1 or count != len(ids) or outer != 1
                    or model.tensors[f'blk.{layer}.attn_k.weight']['shape'][1] != heads*width
                    or row['values'] != count*heads*width or row['offset_bytes']%4 or row['offset_bytes']<0):
                raise ValueError(f'Invalid native K head or payload layout: {name}, {call}')
            start = row['offset_bytes']//4
            block = values[start:start+row['values']]
            if len(block)!=row['values'] or not np.isfinite(block).all():
                raise ValueError(f'Missing or nonfinite native K payload: {name}, {call}')
            if layer not in keys:
                keys[layer] = np.empty((events,heads,width),dtype='<f4')
            if keys[layer].shape != (events,heads,width) or min(ids)<0 or max(ids)>=events:
                raise ValueError(f'Native K coordinates outside architecture: {name}, {call}')
            keys[layer][list(ids)] = block.reshape(count,heads,width)
            seen[layer].add(call)
    for layer in layers:
        if (layer not in keys or seen[layer] != set(expected[layer])
                or sorted(i for ids in expected[layer].values() for i in ids) != list(range(events))):
            raise ValueError(f'Missing K-producing calls or positions at layer {layer}')
    return {layer:torch.from_numpy(a) for layer,a in keys.items()}


class _PrecacheParameters:
    # The historical Gemma producer branches only on this field, after positional
    # rotation. All operations and weights still resolve to the live parameter model.
    cache = None

    def __init__(self, parameters):
        self.parameters = parameters

    def __getattr__(self, name):
        return getattr(self.parameters,name)


class AnchoredKeys:
    def __init__(self, model, history, native, measured):
        self.model = model
        self.measured = measured
        self.origin = {}
        if model.cache != 'float16':
            raise ValueError('This source-bound producer is qualified for its original float16 cache')
        producer = model.parameter_model.keys.__func__
        self.uncached = MethodType(producer,_PrecacheParameters(model.parameter_model))
        self.cache_value = unwrap(producer).__globals__['half']
        self.positions = torch.arange(len(history),device=model.device)
        if len(native['incoming'])!=3*model.layers or set(measured)!=set(range(model.keys_from)):
            raise ValueError('Native merges and K producers must cover this architecture')
        with torch.no_grad():
            for layer, saved in measured.items():
                x = native['incoming'][3*layer].to(device=model.device)
                k, _ = self.raw(layer,x,self.positions)
                if k.ndim==2 and saved.shape[1]==1:
                    saved = saved[:,0]
                if k.shape!=saved.shape or k.dtype!=torch.float32 or not torch.isfinite(k).all():
                    raise ValueError(f'Native and adapter post-RoPE K layouts disagree at layer {layer}')
                self.measured[layer] = saved
                self.origin[layer] = k.detach().cpu()

    def __getattr__(self, name):
        return getattr(self.model,name)

    @property
    def w(self):
        return self.model.w

    @w.setter
    def w(self, values):
        self.model.w = values

    def raw(self, layer, x, positions):
        # Same grouped native-shape QKV product policy as the historical projection adapter.
        return self.model._products(layer,self.model.attention_calls[layer],self.model._key_parts(layer),
                                    self.uncached,layer,x,positions)

    def keys(self, layer, x, positions):
        if layer not in self.origin or not torch.equal(positions,self.positions):
            raise ValueError('Native key anchor belongs to one producer and its bound history')
        k,v = self.raw(layer,x,positions)
        corrected = self.measured[layer].to(k.device) + (k-self.origin[layer].to(k.device))
        return self.cache_value(corrected), self.cache_value(v)


def wrap(model, history, native, projection_raw):
    """Wrap the fully constructed projection adapter before building full_graph.Graph."""
    return AnchoredKeys(model, history, native, load_native_keys(projection_raw,model,len(history)))
