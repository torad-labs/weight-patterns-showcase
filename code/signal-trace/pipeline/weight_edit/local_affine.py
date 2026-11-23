"""Evaluate a local affine FUNCTION at an explicitly shared input.

For F and execution input c, this reads T_c(u) = F(c) + DF(c)[u-c].
It does not compare F(c) with another execution's output value. In particular,
an affine F has the same T at every c, including its offset.

This is a mathematical primitive for a candidate observable, not a selected loss,
reference, correspondence or full-function equivalence test. A common probe domain
is the caller's obligation. Attention must include all causal memory inputs.
Input probes are fixed by default; execution points and weights remain differentiable.
A fixed reference rule evaluated on current data uses fixed_probe=False so its
input argument remains in the current graph; that does not change the rule.
"""
from typing import Callable, Sequence

import torch


def action(function: Callable[..., torch.Tensor], inputs: Sequence[torch.Tensor],
           probe: Sequence[torch.Tensor], *, fixed_probe: bool = True) -> torch.Tensor:
    """Read a local affine map without forming its dense input Jacobian.

    A reverse-over-reverse Jacobian action preserves the mixed derivatives needed
    to optimize a derivative-dependent observable. It also composes with the
    joint forward-mode weight response. Neither the execution input nor weights
    are detached. Local differentiation is enabled even during a response-only
    outer pass; the caller's original tensors and .grad fields are unchanged.
    """
    if not inputs or len(inputs) != len(probe):
        raise ValueError('a probe must supply every causal input')
    for at, point in zip(inputs, probe):
        if at.shape != point.shape or at.device != point.device or at.dtype != point.dtype:
            raise ValueError('execution and probe inputs must share shapes, devices and dtypes')
        if not at.is_floating_point() or not bool(torch.isfinite(at).all() & torch.isfinite(point).all()):
            raise ValueError('local function inputs must be finite floating-point tensors')
    with torch.enable_grad():
        current = tuple(at.clone().requires_grad_(True) for at in inputs)
        value = function(*current)
        if not isinstance(value, torch.Tensor) or not value.numel() or not bool(torch.isfinite(value).all()):
            raise ValueError('the local function must return a nonempty finite tensor')
        if not value.requires_grad:
            return value
        cotangent = torch.zeros_like(value, requires_grad=True)
        pulled = torch.autograd.grad(value, current, cotangent, create_graph=True, allow_unused=True)
        terms = [(gradient * ((point.detach() if fixed_probe else point) - at)).sum()
                 for gradient, point, at in zip(pulled, probe, current) if gradient is not None]
        if not terms:
            return value
        directed = terms[0]
        for term in terms[1:]:
            directed = directed + term
        pushed = torch.autograd.grad(directed, cotangent, create_graph=True, allow_unused=True)[0]
        return value if pushed is None else value + pushed
