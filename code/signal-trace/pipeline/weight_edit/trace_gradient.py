"""One complete weight derivative, one measured response, one native BF16 step.

For the fixed complete residual r, S = -d||r||²/dW and q = J_r S.
alpha = -<r,q>/<q,q> minimizes the local quadratic along S. It is a
response-scaled weight-gradient step, not an inverse solve. There is no
candidate search or claim that the fresh native objective will decrease.
"""
from dataclasses import dataclass
import math
import time
from typing import Any

import numpy as np
import torch

from . import bf16_direction, model_trace_response, trace_response


@dataclass
class Gradient:
    direction: dict[str, torch.Tensor]
    before: dict[str, Any]
    norm_squared: float
    seconds: float


@dataclass
class Proposal:
    status: str
    before: dict[str, Any]
    response: trace_response.Step | None
    gradient_norm_squared: float
    weights: dict[str, np.ndarray] | None
    changed_values: int
    timings: dict[str, float]


def weight_gradient(graph, history, reference, *, pairs) -> Gradient:
    """Differentiate the tensor objective once through every declared matrix."""
    started = time.monotonic()
    names = tuple(graph.leaves)
    original = graph.w
    if not names or len(set(names)) != len(names) or any(name not in original for name in names):
        raise ValueError('the weight derivative requires every distinct declared matrix')
    if any(original[name].dtype != torch.float32 for name in names):
        raise ValueError('weight derivatives use float32 leaves of native BF16 values')
    with torch.enable_grad():
        weights = tuple(original[name].detach().requires_grad_(True) for name in names)
        graph.w = original | dict(zip(names, weights))
        try:
            forward = graph.forward(history)
            terms, values, outside = [], [], 0
            for observed in reference.measure(forward, pairs):
                term = observed.residual.double().square().sum()
                terms.append(term)
                values.append(float(term.detach()))
                outside += int(observed.outside.sum())
            if len(terms) != reference.merges or not all(math.isfinite(x) for x in values):
                raise ValueError('the complete finite residual at every merge is required')
            # The residual already contains its integration weights. Do not
            # average, normalize, detach it or differentiate a reporting float.
            objective = torch.stack(terms).sum()
            if objective.requires_grad:
                gradients = torch.autograd.grad(objective, weights, allow_unused=True)
            else:
                gradients = (None,) * len(weights)
        finally:
            graph.w = original
    direction, norms = {}, []
    for name, weight, gradient in zip(names, weights, gradients):
        value = torch.zeros_like(weight) if gradient is None else gradient.detach()
        if not bool(torch.isfinite(value).all()):
            raise ValueError('the complete weight gradient must be finite')
        norms.append(float(value.double().square().sum()))
        # These are private autograd.grad outputs, never model weights/.grad.
        direction[name] = value.neg_()
    norm = math.fsum(norms)
    if not math.isfinite(norm):
        raise ValueError('the joint weight-gradient norm must be finite')
    before = dict(objective=math.fsum(values), outside=outside,
                  pair_merge_cells=len(pairs) * reference.merges)
    return Gradient(direction, before, norm, time.monotonic() - started)


def propose(graph, history, reference, *, pairs, progress=None) -> Proposal:
    """Construct exactly one rounded update; no weights are installed here."""
    def tell(phase, **details):
        if progress is not None:
            progress(phase, details)

    tell('weight_derivative_started')
    gradient = weight_gradient(graph, history, reference, pairs=pairs)
    timings = dict(weight_derivative_seconds=gradient.seconds)
    tell('weight_derivative_complete', gradient_norm_squared=gradient.norm_squared,
         objective=gradient.before['objective'], **timings)
    if gradient.before['outside'] == 0:
        return Proposal('inside_tolerance', gradient.before, None, gradient.norm_squared, None, 0, timings)
    if gradient.norm_squared == 0:
        return Proposal('zero_weight_gradient', gradient.before, None, 0., None, 0, timings)

    def residual(forward):
        return (observed.residual for observed in reference.measure(forward, pairs))

    started = time.monotonic()
    response = model_trace_response.measure_model(graph, tuple(graph.leaves), history,
                                                  gradient.direction, residual)
    timings['directional_response_seconds'] = time.monotonic() - started
    tell('directional_response_complete', slope=response.slope, response_norm_squared=response.curvature,
         length=response.length, linear_prediction=response.predicted_objective, **timings)
    if response.status != 'proposal' or response.length is None:
        return Proposal(response.status, gradient.before, response, gradient.norm_squared, None, 0, timings)

    started = time.monotonic()
    weights, changed = {}, 0
    # Keep the direction on the graph GPU for the one JVP. Transfer only one
    # matrix at a time for the existing engine-compatible native rounding.
    for name in graph.leaves:
        base = graph.w[name].detach().cpu().numpy()
        direction = gradient.direction.pop(name).cpu().numpy()
        stored = bf16_direction.rounded({name: base}, {name: direction}, response.length)[name]
        changed += int(np.count_nonzero(stored != base))
        weights[name] = stored
    timings['native_rounding_seconds'] = time.monotonic() - started
    tell('native_rounding_complete', changed_values=changed, **timings)
    if not changed:
        return Proposal('native_rounding_stall', gradient.before, response, gradient.norm_squared, None, 0, timings)
    return Proposal('ready', gradient.before, response, gradient.norm_squared, weights, changed, timings)
