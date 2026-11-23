"""Generic signed interval excess; no computational observable is defined here.

The withdrawn Taylor-remainder discrepancy is preserved in the single-reference-
original run's retired-sources.json. These bounds do not validate a caller's loss.
"""
from dataclasses import dataclass

import torch


@dataclass
class Outside:
    residual: torch.Tensor
    outside: torch.Tensor


def outside(residual: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor,
            mass: torch.Tensor) -> Outside:
    """Return weighted excess and support for the same fixed interval."""
    residual = residual.double()
    lower, upper, mass = lower.detach().double(), upper.detach().double(), mass.detach().double()
    if (residual.ndim != 2 or lower.shape != residual.shape or upper.shape != residual.shape
            or mass.shape != residual.shape[:1]):
        raise ValueError('each complete relation row needs its original bounds and mass')
    if not all(bool(torch.isfinite(value).all()) for value in (residual, lower, upper, mass)):
        raise ValueError('a nonfinite relation or reference makes the reading unavailable')
    if bool((lower > upper).any()) or bool((mass <= 0).any()):
        raise ValueError('bounds must be ordered and every represented row has positive mass')
    below, above = residual < lower, residual > upper
    excess = torch.where(below, residual - lower,
                         torch.where(above, residual - upper, torch.zeros_like(residual)))
    # A collapsed interval is a point: the residual is identity minus that
    # point even at equality, so the joint inverse must retain its Jacobian.
    excess = torch.where(lower == upper, residual - lower, excess)
    return Outside(excess * mass.sqrt()[:, None], (below | above).any(1))
