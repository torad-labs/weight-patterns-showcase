"""Experimental projection STE: finite native-call primals, F32 VJP and JVP.

No loss, optimizer, acquisition or installation is defined here. A synthetic CPU
fixture may supply a primal backend explicitly; production defaults to CUDA.
"""
from contextlib import contextmanager
from unittest.mock import patch

import torch
import torch.nn.functional as F

# These primitives do not recurse through the scoped F.linear interception.
_linear = torch._C._nn.linear


def cuda_product(xq, wq):
    if xq.device.type != 'cuda' or wq.device != xq.device:
        raise ValueError('production BF16 primal requires the declared CUDA device')
    return torch.mm(xq, wq.T, out_dtype=torch.float32)


class _Product(torch.autograd.Function):
    """Discrete primal; specified identity-through-rounding derivative, both modes.

Returning the primal directly also preserves signed zero; no rounded primal is
reconstructed by adding/subtracting a surrogate. Higher-order AD is not claimed.
"""
    @staticmethod
    def forward(ctx, x, w, product):
        xq, wq = x.to(torch.bfloat16), w.to(torch.bfloat16)
        xf, wf = xq.float(), wq.float()
        ctx.save_for_backward(xf, wf)
        ctx.save_for_forward(xf, wf)
        return product(xq, wq)

    @staticmethod
    def backward(ctx, adjoint):
        xq, wq = ctx.saved_tensors
        return (torch.mm(adjoint, wq) if ctx.needs_input_grad[0] else None,
                torch.mm(adjoint.T, xq) if ctx.needs_input_grad[1] else None, None)

    @staticmethod
    def jvp(ctx, dx, dw, _product):
        xq, wq = ctx.saved_tensors
        out = None if dx is None else torch.mm(dx, wq.T)
        if dw is not None:
            term = torch.mm(xq, dw.T)
            out = term if out is None else out + term
        return out


def grouped_linear(value, weight, rows, *, product=cuda_product, factor=False):
    """One logical projection, preserving native group order and F32 derivatives."""
    x = torch.autograd.forward_ad.unpack_dual(value).primal.detach()
    w = torch.autograd.forward_ad.unpack_dual(weight).primal.detach()
    if (x.ndim != 2 or w.ndim != 2 or x.shape[1] != w.shape[1]
            or x.dtype != torch.float32 or w.dtype != torch.float32 or x.device != w.device
            or not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(w).all())
            or not torch.equal(w, w.to(torch.bfloat16).float())):
        raise ValueError('finite F32 inputs and exact native BF16 weights required')
    order = [i for row in rows for i in row['positions']]
    if not rows or any(not row['positions'] for row in rows) or sorted(order) != list(range(len(x))):
        raise ValueError('native calls must cover each event exactly once')
    outputs, factors = [], []
    for row in rows:
        part = value[list(row['positions'])]
        multi = len(part) != 1
        outputs.append(_Product.apply(part, weight, product) if multi else _linear(part, weight))
        if factor:
            # Factor is detached because it describes the weight leg, not a
            # replacement for the connected input used by the actual product.
            h = part.detach()
            factors.append(h.to(torch.bfloat16).float() if multi else h)
    inverse = torch.argsort(torch.tensor(order, device=x.device))
    output = torch.cat(outputs)[inverse]
    return (output, torch.cat(factors)[inverse]) if factor else output


class Model:
    """Wrap only Q/K/V/output and third projections; delegate everything else."""
    def __init__(self, source, attention_calls, third_calls, *, product=cuda_product):
        self.source, self.attention_calls, self.third_calls = source, attention_calls, third_calls
        self.product, self.projection_tap = product, None
        if source.cache != 'float16':
            raise ValueError('the existing FP16 cache policy is required')
        expected = set(range(source.layers))
        if set(attention_calls) != expected or set(third_calls) != expected:
            raise ValueError('every attention and third layer needs recorded geometry')
        for layer in expected:
            parts = ['attn_q', 'attn_output', 'inp_gate', 'proj']
            if layer < source.keys_from:
                parts += list(self._key_parts(layer))
            for part in parts:
                t = source.tensors[f'blk.{layer}.{part}.weight']
                inner, outer, d2, d3 = t['shape']
                rows = third_calls[layer] if part in ('inp_gate', 'proj') else attention_calls[layer]
                if (t['native_type'] != 'bf16' or d2 != 1 or d3 != 1 or inner % 64 or outer % 32
                        or any(1 < len(r['positions']) <= 8 and outer*len(r['positions']) <= 1024 for r in rows)):
                    raise ValueError('projection geometry is outside the reviewed native operand policy')
            if not 0 <= source.keys_of(layer) < source.keys_from:
                raise ValueError('missing original key producer')

    @property
    def w(self):
        return self.source.w

    @w.setter
    def w(self, value):
        self.source.w = value

    def __getattr__(self, name):
        return getattr(self.source, name)

    def _key_parts(self, layer):
        return ('attn_k', 'attn_v') if f'blk.{layer}.attn_v.weight' in self.w else ('attn_k',)

    @contextmanager
    def tapping(self, callback):
        old = self.projection_tap
        self.projection_tap = callback
        try:
            yield
        finally:
            self.projection_tap = old

    def _products(self, layer, rows, parts, function, *args):
        seen = []
        def apply(value, weight, bias=None):
            if bias is not None or len(seen) >= len(parts):
                raise ValueError('unexpected projection call')
            name = f'blk.{layer}.{parts[len(seen)]}.weight'
            shape = tuple(reversed(self.tensors[name]['shape'][:2]))
            if tuple(weight.shape) != shape:
                raise ValueError('projection call shape changed')
            seen.append(name)
            tap = self.projection_tap if name == 'blk.0.attn_output.weight' else None
            if tap is None:
                return grouped_linear(value, weight, rows, product=self.product)
            output, h = grouped_linear(value, weight, rows, product=self.product, factor=True)
            tap(name, weight, value, h, output)
            return output
        with patch.object(F, 'linear', apply):
            result = function(*args)
        if len(seen) != len(parts):
            raise ValueError('a declared projection was omitted')
        return result

    def keys(self, layer, x, positions):
        return self._products(layer, self.attention_calls[layer], self._key_parts(layer),
                              self.source.keys, layer, x, positions)

    def attention(self, layer, x, positions, k, v):
        return self._products(layer, self.attention_calls[layer], ('attn_q', 'attn_output'),
                              self.source.attention, layer, x, positions, k, v)

    def third(self, layer, x, auxiliary):
        return self._products(layer, self.third_calls[layer], ('inp_gate', 'proj'),
                              self.source.third, layer, x, auxiliary)
