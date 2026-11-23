"""Diagnostic attribution for the persistent merge-0 cells; never an update rule.

The selection comes from retained local9 loss maps and the independent
2026-11-17-remaining-loss-review. A changed correspondence may make it absent.
Only scalar moments leave this function. It does not modify the full direction,
calibration, correspondence, native state, or parameter inventory.
"""
import math

import torch


CELLS = ((373, 374), (374, 374))
CAUSAL_NAMES = (
    'token_embd.weight', 'blk.0.attn_norm.weight',
    'blk.0.attn_q.weight', 'blk.0.attn_q_norm.weight',
    'blk.0.attn_k.weight', 'blk.0.attn_k_norm.weight',
    'blk.0.attn_v.weight', 'blk.0.attn_output.weight',
    'blk.0.post_attention_norm.weight',
)


def selection(pairs, cells=CELLS):
    rows = [tuple(map(int, row)) for row in pairs]
    cells = tuple(tuple(cell) for cell in cells)
    if len(set(cells)) != len(cells) or not cells:
        raise ValueError('diagnostic cells must be distinct and nonempty')
    missing = [cell for cell in cells if cell not in rows]
    if missing:
        return dict(status='selected_pairs_absent', merge=0, cells=cells,
                    missing=missing, indices=[])
    if any(rows.count(cell) != 1 for cell in cells):
        raise ValueError('a selected computational pair occurs more than once')
    return dict(status='selected', merge=0, cells=cells,
                indices=[rows.index(cell) for cell in cells])


def attribute(graph, history, reference, pairs, full_direction, *,
              cells=CELLS, names=CAUSAL_NAMES):
    """One local VJP, compared to the unchanged full gradient g=-direction.

    The complement g-local is all other objective contributions in these same
    coordinates. Its dot products distinguish local attenuation from opposing
    contributions; neither norms nor cosines establish native reachability.
    """
    chosen = selection(pairs, cells)
    if chosen['status'] != 'selected':
        return chosen
    original = graph.w
    names = tuple(names)
    if not names or len(set(names)) != len(names):
        raise ValueError('causal parameter names must be distinct and nonempty')
    if any(name not in graph.leaves or name not in full_direction for name in names):
        raise ValueError('diagnostic causal parameter is absent from the eligible gradient')
    weights = tuple(original[name].detach().requires_grad_(True) for name in names)
    try:
        with torch.enable_grad():
            graph.w = original | dict(zip(names, weights))
            forward = graph.forward(history)
            observations = reference.measure(forward, pairs)
            try:
                residual = next(observations).residual
                scalar = residual[chosen['indices']].double().square().sum()
                if not bool(torch.isfinite(scalar)):
                    raise ValueError('nonfinite selected-cell loss')
                gradients = (torch.autograd.grad(scalar, weights, allow_unused=True)
                             if scalar.requires_grad else (None,) * len(weights))
            finally:
                observations.close()
    finally:
        graph.w = original
    fields = ('local_norm_squared', 'full_norm_squared', 'complement_norm_squared',
              'local_dot_full', 'local_dot_complement')
    per_tensor = {}
    for name, weight, local in zip(names, weights, gradients):
        disconnected = local is None
        local = torch.zeros_like(weight) if disconnected else local.detach()
        full = -full_direction[name].detach()
        if local.shape != full.shape or not bool(torch.isfinite(local).all()):
            raise ValueError('invalid local derivative shape or values')
        if not bool(torch.isfinite(full).all()):
            raise ValueError('nonfinite full derivative')
        local, full = local.double(), full.double()
        complement = full - local
        per_tensor[name] = dict(
            gradient_was_none=disconnected,
            local_norm_squared=float(local.square().sum()),
            full_norm_squared=float(full.square().sum()),
            complement_norm_squared=float(complement.square().sum()),
            local_dot_full=float((local * full).sum()),
            local_dot_complement=float((local * complement).sum()),
        )
    totals = {key: math.fsum(row[key] for row in per_tensor.values()) for key in fields}
    denominator = math.sqrt(totals['local_norm_squared'] * totals['full_norm_squared'])
    totals['local_full_cosine'] = totals['local_dot_full'] / denominator if denominator else None
    return dict(chosen, status='measured', selected_objective=float(scalar.detach()),
                per_tensor=per_tensor, totals=totals,
                scope='Held-history estimator attribution only; full training objective and direction unchanged')


def moments(residual, chosen):
    """Read selected moments from an existing dual; never replace that dual."""
    if chosen['status'] != 'selected':
        return chosen
    primal, tangent = torch.autograd.forward_ad.unpack_dual(residual)
    r = primal.detach()[chosen['indices']].double()
    q = None if tangent is None else tangent.detach()[chosen['indices']].double()
    return dict(chosen, status='measured', baseline=float(r.square().sum()),
                loss_cross=0. if q is None else 2 * float((r * q).sum()),
                response_norm_squared=0. if q is None else float(q.square().sum()))
