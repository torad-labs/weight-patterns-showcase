"""Prospective device-resident CGLS; the running solver is not modified.

The residual/Jacobian graph stays on its declared parameter device. Solver
vectors stay on vector_device, and the final correction returns to CPU for the
existing BF16 adapter. Arithmetic, zero start and stopping rules are retained.
Explicit lifetimes avoid keeping unused full parameter vectors on the card.
"""
from typing import Callable, Sequence

import torch
from torch.autograd import forward_ad

from pipeline.weight_edit.joint_inverse import (
    Blocks, Proposal, Residual, _add, _chunks, _dot, _objective, _same,
)


class Linearization:
    def __init__(self, residual: Callable[..., Residual], parameters: Sequence[torch.Tensor],
                 vector_device):
        if not parameters or any(w.dtype != torch.float32 for w in parameters):
            raise ValueError('the joint solver requires declared float32 parameter tensors')
        _chunks(tuple(parameters))
        self.function = residual
        self.parameters = tuple(w.detach() for w in parameters)
        self.vector_device = torch.device(vector_device)
        with torch.no_grad():
            self.residual = self.held(_chunks(residual(*self.parameters)))

    def held(self, values: Sequence[torch.Tensor]) -> Blocks:
        return tuple(value.detach().to(device=self.vector_device, dtype=torch.float32, copy=True)
                     for value in values)

    def push(self, direction: Blocks) -> Blocks:
        _same(direction, self.parameters)
        with torch.no_grad(), forward_ad.dual_level():
            duals = tuple(forward_ad.make_dual(w, d.to(w))
                          for w, d in zip(self.parameters, direction))
            response = []
            for chunk in _chunks(self.function(*duals)):
                value, tangent = forward_ad.unpack_dual(chunk)
                response.append(torch.zeros_like(value) if tangent is None else tangent)
            result = self.held(response)
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
                return tuple(torch.zeros_like(w, device=self.vector_device) for w in weights)
            gradient = torch.autograd.grad(tuple(pair[0] for pair in active), weights,
                                           grad_outputs=tuple(pair[1] for pair in active),
                                           allow_unused=True)
            result = self.held(tuple(torch.zeros_like(w) if g is None else g
                                     for w, g in zip(weights, gradient)))
        _same(result, self.parameters)
        return result


def next_search(gradient: Blocks, previous: Blocks, ratio: float) -> Blocks:
    # Same negate-then-add expression as the original, one block at a time.
    # Materializing every negated block first would retain another full vector.
    _same(gradient, previous)
    return _chunks(tuple(g.neg().add(s, alpha=ratio) for g, s in zip(gradient, previous)))


def solve(residual: Callable[..., Residual], parameters: Sequence[torch.Tensor], *,
          vector_device, progress: Callable[[int, float], None] | None = None) -> Proposal:
    linear = Linearization(residual, parameters, vector_device)
    change = tuple(torch.zeros_like(w, device=linear.vector_device) for w in linear.parameters)
    error = linear.residual
    objectives = [_objective(error)]

    def finish(status):
        # Preserve the public result expected by turn_correction's numpy adapter.
        cpu = tuple(w.detach().to(device='cpu', dtype=torch.float32, copy=True) for w in change)
        return Proposal(cpu, objectives, len(objectives) - 1, status)

    if progress is not None:
        progress(0, objectives[0])
    if objectives[0] == 0:
        return finish('zero_linear_residual')
    gradient = linear.pull(error)
    gamma = _dot(gradient, gradient)
    search = tuple(-g for g in gradient)
    restarted = False
    while gamma > 0:
        response = linear.push(search)
        curvature = _dot(response, response)
        if curvature == 0:
            return finish('stationary')
        scale = -_dot(gradient, search) / curvature
        del response
        candidate = _add(change, search, scale)
        candidate_response = linear.push(candidate)
        candidate_error = _add(linear.residual, candidate_response, 1.)
        del candidate_response
        objective = _objective(candidate_error)
        if objective >= objectives[-1]:
            del candidate, candidate_error
            if restarted:
                return finish('float32_stall')
            search, restarted = tuple(-g for g in gradient), True
            continue
        change, error = candidate, candidate_error
        del candidate, candidate_error
        objectives.append(objective)
        if progress is not None:
            progress(len(objectives) - 1, objective)
        if objective == 0:
            return finish('zero_linear_residual')
        # The old gradient is no longer used; gamma holds its scalar norm.
        del gradient
        gradient = linear.pull(error)
        next_gamma = _dot(gradient, gradient)
        search = next_search(gradient, search, next_gamma / gamma)
        gamma, restarted = next_gamma, False
    return finish('stationary')
