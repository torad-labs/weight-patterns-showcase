"""Joint weight-to-trace response for a squared-residual proposal.

The residual callback owns the observable, fixed ORIGINAL reference, correspondence
and square-root integration weights. This module selects none of those. For
L = ||r||² and a weight direction S, propagate q = J_theta(r) S through the
WHOLE current calculation, then minimize ||r + alpha q||². Contributions from
different matrices meet in q before its norm is taken.

This is the undamped directional version of the guide's Section 5, Eq. 6/8.
It gives a local proposal, never native acceptance, closure or a learning rate.
No weights or .grad fields are written. Forward AD needs no dense Jacobian or
second derivatives. A callback may stream residual chunks to bound memory.
"""
import math
from dataclasses import dataclass
from typing import Callable, Iterable, Literal, Sequence

import torch
from torch.autograd import forward_ad

from . import host_pages

Residual = torch.Tensor | Iterable[torch.Tensor]
Status = Literal['proposal', 'zero_residual', 'zero_response', 'non_descent']


@dataclass(frozen=True)
class Step:
    objective: float
    slope: float
    curvature: float
    length: float | None
    predicted_objective: float
    status: Status


def _finite(value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        raise ValueError('weights, direction, residual and response must be finite')


def measure(residual: Callable[..., Residual], parameters: Sequence[torch.Tensor],
            direction: Sequence[torch.Tensor]) -> Step:
    """Read one joint directional response on held histories and correspondence.

    Each returned tensor is ALREADY a weighted residual: its squared norm is
    its contribution to the one objective. Unresolved coverage is the caller's
    separate account and cannot be certified by a zero residual here.
    """
    if not parameters or len(parameters) != len(direction):
        raise ValueError('one direction tensor is required for each parameter')
    for weight, delta in zip(parameters, direction):
        if weight.shape != delta.shape:
            raise ValueError('parameter and direction shapes differ')
        if weight.dtype != delta.dtype or weight.device != delta.device:
            raise ValueError('parameter and direction need the same dtype and device')
        if not weight.is_floating_point():
            raise ValueError('the response requires floating-point parameters')
        _finite(weight)
        _finite(delta)
        host_pages.release(weight, delta)   # a file-backed pair is read back when its dual is used
    norms, products, responses = [], [], []
    with torch.no_grad(), forward_ad.dual_level():
        duals = tuple(forward_ad.make_dual(w.detach(), s.detach())
                      for w, s in zip(parameters, direction))
        result = residual(*duals)
        chunks = (result,) if isinstance(result, torch.Tensor) else result
        for chunk in chunks:
            if not isinstance(chunk, torch.Tensor) or chunk.numel() == 0:
                raise ValueError('an empty or nontensor residual has no response measure')
            r, q = forward_ad.unpack_dual(chunk)
            _finite(r)
            r = r.double()
            norms.append(float(r.square().sum()))
            if q is None:
                products.append(0.)
                responses.append(0.)
            else:
                _finite(q)
                q = q.double()
                products.append(float((r * q).sum()))
                responses.append(float(q.square().sum()))
    if not norms:
        raise ValueError('an empty residual has no response measure')
    objective, rq, qq = math.fsum(norms), math.fsum(products), math.fsum(responses)
    slope = 2 * rq
    if not all(math.isfinite(x) for x in (objective, slope, qq)):
        raise ValueError('the summed response moments must be finite')
    if objective == 0:
        return Step(objective, slope, qq, None, objective, 'zero_residual')
    if qq == 0:
        return Step(objective, slope, qq, None, objective, 'zero_response')
    if rq >= 0:
        return Step(objective, slope, qq, None, objective, 'non_descent')
    length = -rq / qq
    if not math.isfinite(length):
        raise ValueError('the measured response gives a nonfinite length')
    # Nonnegative by Cauchy-Schwarz; eliminate only summation roundoff below zero.
    predicted = max(0., objective + length * rq)
    return Step(objective, slope, qq, length, predicted, 'proposal')
