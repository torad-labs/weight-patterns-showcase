"""Frozen convex requests with exact symbolic feasibility and bounded evaluation.

The target is (sum a_i v_i)/(sum a_i) for the represented, nonnegative a_i.
That EXACT ratio is a convex combination; a rounded projected point is not
silently declared feasible. Binary64 outward enclosures account for evaluating
the ratio. The differentiable residual uses the recorded approximation; actual
fixed-request decrease can be established by comparing distance enclosures.

Assumes ordinary IEEE binary64 operations, including subnormals and no fast-math
reassociation outside the stated reduction error bound. These are execution
bindings to qualify, not a theorem about every native arithmetic backend.
"""
from dataclasses import dataclass

import torch


def _up(x: torch.Tensor) -> torch.Tensor:
    return torch.nextafter(x, torch.full_like(x, float('inf')))


def _down(x: torch.Tensor) -> torch.Tensor:
    return torch.nextafter(x, torch.full_like(x, -float('inf')))


def _finite(*values: torch.Tensor) -> None:
    if not all(bool(torch.isfinite(value).all()) for value in values):
        raise ValueError('the request and its evaluation must be finite')


def _sum_bounds(lower: torch.Tensor, upper: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # A reduction uses at most n-1 additions. eps (rather than eps/2) is
    # conservative; n and eps come from this represented reduction's size/dtype.
    n = lower.shape[-1]
    error = torch.tensor(n * torch.finfo(torch.float64).eps, dtype=torch.float64, device=lower.device)
    if not bool(error < 1):
        raise ValueError('the binary64 reduction bound cannot resolve this dimension')
    gamma = _up(error / _down(1 - error))
    if not bool(gamma < 1):
        raise ValueError('the reduction error leaves no positive upper-bound denominator')
    lo = _down(lower.sum(-1) / _up(1 + gamma)).clamp_min(0)
    hi = _up(upper.sum(-1) / _down(1 - gamma))
    _finite(lo, hi)
    return lo, hi


@dataclass(frozen=True)
class Request:
    coefficients: torch.Tensor
    approximation: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor

    def residual(self, current: torch.Tensor) -> torch.Tensor:
        """Connected approximation to the frozen request, never a moving projection."""
        if current.shape != self.approximation.shape:
            raise ValueError('current values and frozen request have different shapes')
        return current.double() - self.approximation

    @torch.no_grad()
    def distance_bounds(self, current: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Enclose squared distance to the EXACT feasible target, one value per row.

        These bounds do not certify optimal projection or hull membership.
        A trial upper bound below the start lower bound proves decrease for
        this same frozen request. Overlap of bounds proves neither direction.
        """
        if current.shape != self.approximation.shape or current.device != self.lower.device:
            raise ValueError('current values and frozen request have different shapes or devices')
        current = current.detach().double()
        _finite(current)
        distance_low = torch.maximum(_down(self.lower - current), _down(current - self.upper)).clamp_min(0)
        distance_high = torch.maximum(_up((current - self.lower).abs()), _up((current - self.upper).abs()))
        squared_low = _down(distance_low.square()).clamp_min(0)
        squared_high = _up(distance_high.square())
        _finite(squared_low, squared_high)
        return _sum_bounds(squared_low, squared_high)


@torch.no_grad()
def freeze(vertices: torch.Tensor, coefficients: torch.Tensor) -> Request:
    """Bind a proposed nonnegative mixture and enclose its exactly normalized value."""
    vertices, coefficients = vertices.detach().double(), coefficients.detach().double()
    if (vertices.ndim != 3 or coefficients.shape != vertices.shape[:2]
            or min(vertices.shape) < 1 or coefficients.device != vertices.device):
        raise ValueError('a request needs row x witness x coordinate vertices and matching weights')
    _finite(vertices, coefficients)
    if bool((coefficients < 0).any()) or bool((coefficients.sum(1) == 0).any()):
        raise ValueError('a feasible request needs nonnegative weights with positive mass')
    rows, witnesses, width = vertices.shape
    low = torch.zeros((rows, width), dtype=torch.float64, device=vertices.device)
    high = torch.zeros_like(low)
    mass_low = torch.zeros((rows, 1), dtype=torch.float64, device=vertices.device)
    mass_high = torch.zeros_like(mass_low)
    for i in range(witnesses):
        weight = coefficients[:, i, None]
        product = weight * vertices[:, i]
        low, high = _down(low + _down(product)), _up(high + _up(product))
        mass_low, mass_high = _down(mass_low + weight), _up(mass_high + weight)
    if bool((mass_low <= 0).any()):
        raise ValueError('the exact positive weight sum is unresolved in this arithmetic')
    quotients = torch.stack((low / mass_low, low / mass_high, high / mass_low, high / mass_high))
    lower, upper = _down(quotients.amin(0)), _up(quotients.amax(0))
    approximation = (vertices * coefficients[:, :, None]).sum(1) / coefficients.sum(1, keepdim=True)
    _finite(lower, upper, approximation)
    return Request(coefficients.clone(), approximation, lower, upper)
