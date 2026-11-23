"""Prospective FFN operand precision with an EXPLICIT straight-through surrogate.

The discrete BF16 forward is not differentiable as the surrogate below claims.
AD uses identity through operand rounding, evaluated at rounded operands; native
finite/fresh measurements remain necessary. No attention/third-merge override.
"""
import json

import torch
import torch.nn.functional as F
from torch.autograd import forward_ad


def calls_from_index(path, *, events, layers, width):
    """Keep every recorded FFN call and position; labels identify operators only."""
    labels = {f'ffn_post_norm-{layer}': layer for layer in range(layers)}
    groups = {layer: [] for layer in range(layers)}
    with open(path) as file:
        for line in file:
            row = json.loads(line)
            if row.get('label') not in labels:
                continue
            positions = tuple(row['positions'])
            columns = row['microbatch_count']
            if row['native_type'] != 'f32' or row['shape'] != [width, columns, 1, 1] or len(positions) != columns:
                raise ValueError('native FFN call shape/type does not match its positions')
            groups[labels[row['label']]].append(dict(call=row['call'], microbatch=row['microbatch'], positions=positions))
    for layer, calls in groups.items():
        if sorted(position for row in calls for position in row['positions']) != list(range(events)):
            raise ValueError(f'layer {layer} has missing or repeated computational positions')
    return groups


def bf16_linear_ste(x, weight):
    """BF16 operands/F32 output; AD is the stated continuous STE, not native AD.

    At rounded operands Xq,Wq the surrogate differential is
    dY = dX Wq.T + Xq dW.T. Both dX and dW remain float32.
    """
    xq = x.detach().to(torch.bfloat16)
    wq = weight.detach().to(torch.bfloat16)
    exact = torch.mm(xq, wq.T, out_dtype=torch.float32)
    reverse = torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad)
    tangent = forward_ad.unpack_dual(x).tangent is not None or forward_ad.unpack_dual(weight).tangent is not None
    if not reverse and not tangent:
        return exact
    # Explicit identity-through-rounding surrogate. Exact + (s-stopgrad(s))
    # keeps the finite primal bitwise exact instead of subtracting an offset.
    xs = xq.float() + (x - x.detach())
    ws = wq.float() + (weight - weight.detach())
    surrogate = F.linear(xs, ws)
    return exact + (surrogate - surrogate.detach())


class Model:
    """Wrap a reached model BEFORE constructing its native-anchored Graph.

    Build fresh calls from each new native capture. Its Graph's frozen computed
    side and differentiated side must both use this wrapper. The owner binds
    this prospective source; this module never patches a live model or file.
    """
    def __init__(self, source, calls, *, rms, gelu, batch_single_column=False):
        self.source, self.calls, self.rms, self.gelu = source, calls, rms, gelu
        if set(calls) != set(range(source.layers)):
            raise ValueError('every feed-forward layer needs its actual native calls')
        self.events = {}
        self.order = {}
        self.compute_calls = {}
        for layer, rows in calls.items():
            positions = [position for row in rows for position in row['positions']]
            if not positions or sorted(positions) != list(range(len(positions))):
                raise ValueError('FFN call geometry must cover every position once')
            self.events[layer] = len(positions)
            compute = [dict(positions=row['positions'], bf16=len(row['positions']) > 1) for row in rows]
            if batch_single_column:
                # Native metadata still chooses operand precision for every row.
                # One F32 matrix call collects independent one-column operators;
                # this changes reduction kernels, not their precision policy.
                singles = tuple(position for row in rows if len(row['positions']) == 1 for position in row['positions'])
                compute = [row for row in compute if row['bf16']]
                if singles:
                    compute.append(dict(positions=singles, bf16=False))
            self.compute_calls[layer] = compute
            order = [position for row in compute for position in row['positions']]
            self.order[layer] = torch.argsort(torch.tensor(order, device=source.device))
            for part in ('ffn_gate', 'ffn_up', 'ffn_down'):
                tensor = source.tensors[f'blk.{layer}.{part}.weight']
                inner, outer, dim2, dim3 = tensor['shape']
                # These are the retained native source's shape/type conditions,
                # not experimentally chosen caps. These FFNs never take the
                # tiny-output MMVF fallback for multicolumn calls.
                if tensor['native_type'] != 'bf16' or dim2 != 1 or dim3 != 1 or inner % 64 or outer % 32:
                    raise ValueError('FFN matrix is outside the reviewed BF16 source geometry')
                if any(len(row['positions']) > 1 and outer * len(row['positions']) <= 1024 for row in rows):
                    raise ValueError('small-output native vector fallback needs separate qualification')
        if len(set(self.events.values())) != 1:
            raise ValueError('native FFN layers cover different event inventories')

    @property
    def w(self):
        return self.source.w

    @w.setter
    def w(self, weights):
        self.source.w = weights

    def __getattr__(self, name):
        return getattr(self.source, name)

    def feed_forward(self, layer, h):
        if h.shape != (self.events[layer], self.source.width) or h.dtype != torch.float32:
            raise ValueError('FFN input does not match the bound native event inventory')
        name = lambda part: f'blk.{layer}.{part}.weight'
        gate, up, down = [self.source.weight(name(part)) for part in ('ffn_gate', 'ffn_up', 'ffn_down')]
        values = []
        for row in self.compute_calls[layer]:
            positions = list(row['positions'])
            product = bf16_linear_ste if row['bf16'] else F.linear
            f = self.rms(h[positions], self.w[name('ffn_norm')])
            f = self.gelu(product(f, gate)) * product(f, up)
            values.append(self.rms(product(f, down), self.w[name('post_ffw_norm')]))
        return torch.cat(values, dim=0)[self.order[layer]]
