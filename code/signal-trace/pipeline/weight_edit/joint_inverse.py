"""Matrix-free joint least squares for a held complete trace request.

Solve min_delta ||r(theta) + J(theta) delta||² from zero by CGLS. J contains
ALL declared matrices together, including cross-layer effects. No ridge,
learning rate, preconditioner, iteration cap or model write is selected here.
Solver vectors live on the CPU; model products use the parameters' devices.

A strict decrease in the float32 linear objective is required at every accepted
iteration. Stationarity or representational stagnation ends this INNER solve;
neither certifies native nonlinear loss closure. In exact arithmetic the zero
start gives the minimum-Euclidean-norm linear solution, not a minimum-damage edit.
"""
from dataclasses import dataclass
import math
from typing import Callable, Iterable, Literal, Sequence

import torch
from torch.autograd import forward_ad

Blocks = tuple[torch.Tensor, ...]
Residual = torch.Tensor | Iterable[torch.Tensor]


def _chunks(value: Residual) -> Blocks:
    result = (value,) if isinstance(value, torch.Tensor) else tuple(value)
    if not result or any(not item.numel() for item in result):
        raise ValueError('the complete residual must contain nonempty chunks')
    if any(not bool(torch.isfinite(item).all()) for item in result):
        raise ValueError('the complete residual and products must be finite')
    return result


def _cpu(values: Sequence[torch.Tensor]) -> Blocks:
    return tuple(value.detach().to(device='cpu', dtype=torch.float32, copy=True)
                 for value in values)


def _same(values: Sequence[torch.Tensor], template: Sequence[torch.Tensor]) -> None:
    if len(values) != len(template) or any(a.shape != b.shape for a, b in zip(values, template)):
        raise ValueError('the complete linearized residual and parameter shapes must stay fixed')
    _chunks(tuple(values))


def _dot(left: Blocks, right: Blocks) -> float:
    _same(left, right)
    result = math.fsum(float((a.double() * b.double()).sum()) for a, b in zip(left, right))
    if not math.isfinite(result):
        raise ValueError('the summed joint product must be finite')
    return result


def _objective(values: Blocks) -> float:
    # Products use a stable reduction; the acceptance comparison is represented
    # in the declared solver precision, not an arbitrary small-change threshold.
    result = float(torch.tensor(_dot(values, values), dtype=torch.float32))
    if not math.isfinite(result):
        raise ValueError('the float32 linear objective must be finite')
    return result


def _add(left: Blocks, right: Blocks, scale: float) -> Blocks:
    _same(left, right)
    return _chunks(tuple(a.add(b, alpha=scale) for a, b in zip(left, right)))


class Linearization:
    """Fresh J-vector and transposed-J-vector products at immutable current weights."""
    def __init__(self, residual: Callable[..., Residual], parameters: Sequence[torch.Tensor]):
        if not parameters or any(w.dtype != torch.float32 for w in parameters):
            raise ValueError('the joint solver requires declared float32 parameter tensors')
        _chunks(tuple(parameters))
        self.function = residual
        # Detached views are read only throughout; no parameter or .grad is written.
        self.parameters = tuple(w.detach() for w in parameters)
        with torch.no_grad():
            self.residual = _cpu(_chunks(residual(*self.parameters)))

    def push(self, direction: Blocks) -> Blocks:
        _same(direction, self.parameters)
        with torch.no_grad(), forward_ad.dual_level():
            duals = tuple(forward_ad.make_dual(w, d.to(w))
                          for w, d in zip(self.parameters, direction))
            response = []
            for chunk in _chunks(self.function(*duals)):
                value, tangent = forward_ad.unpack_dual(chunk)
                response.append(torch.zeros_like(value) if tangent is None else tangent)
            result = _cpu(response)
        _same(result, self.residual)
        return result

    def pull(self, cotangent: Blocks) -> Blocks:
        _same(cotangent, self.residual)
        with torch.enable_grad():
            weights = tuple(w.detach().requires_grad_(True) for w in self.parameters)
            chunks = _chunks(self.function(*weights))
            _same(chunks, self.residual)
            active = [(value, supplied.to(value)) for value, supplied in zip(chunks, cotangent)
                      if value.requires_grad]
            if not active:
                return tuple(torch.zeros_like(w, device='cpu') for w in weights)
            gradient = torch.autograd.grad(tuple(pair[0] for pair in active), weights,
                                           grad_outputs=tuple(pair[1] for pair in active),
                                           allow_unused=True)
            result = _cpu(tuple(torch.zeros_like(w) if g is None else g
                                for w, g in zip(weights, gradient)))
        _same(result, self.parameters)
        return result


@dataclass
class Proposal:
    changes: Blocks
    objectives: list[float]
    iterations: int
    status: Literal['zero_linear_residual', 'stationary', 'float32_stall']


def solve(residual: Callable[..., Residual], parameters: Sequence[torch.Tensor], *,
          progress: Callable[[int, float], None] | None = None) -> Proposal:
    linear = Linearization(residual, parameters)
    change = tuple(torch.zeros_like(w, device='cpu') for w in linear.parameters)
    error = linear.residual
    objectives = [_objective(error)]
    if progress is not None:
        progress(0, objectives[0])
    if objectives[0] == 0:
        return Proposal(change, objectives, 0, 'zero_linear_residual')
    gradient = linear.pull(error)
    gamma = _dot(gradient, gradient)
    search = tuple(-g for g in gradient)
    restarted = False
    while gamma > 0:
        response = linear.push(search)
        curvature = _dot(response, response)
        if curvature == 0:
            return Proposal(change, objectives, len(objectives) - 1, 'stationary')
        candidate = _add(change, search, -_dot(gradient, search) / curvature)
        # Recompute the complete linear prediction, rather than letting recursive
        # residual drift decide whether an update has improved it.
        candidate_error = _add(linear.residual, linear.push(candidate), 1.)
        objective = _objective(candidate_error)
        if objective >= objectives[-1]:
            if restarted:
                return Proposal(change, objectives, len(objectives) - 1, 'float32_stall')
            # Conjugacy can be lost in finite arithmetic. One fresh steepest
            # linear direction tests whether the current inverse really stalled.
            search, restarted = tuple(-g for g in gradient), True
            continue
        change, error = candidate, candidate_error
        objectives.append(objective)
        if progress is not None:
            progress(len(objectives) - 1, objective)
        if objective == 0:
            return Proposal(change, objectives, len(objectives) - 1, 'zero_linear_residual')
        next_gradient = linear.pull(error)
        next_gamma = _dot(next_gradient, next_gradient)
        search = _add(tuple(-g for g in next_gradient), search, next_gamma / gamma)
        gradient, gamma, restarted = next_gradient, next_gamma, False
    return Proposal(change, objectives, len(objectives) - 1, 'stationary')
