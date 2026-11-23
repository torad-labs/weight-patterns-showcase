"""Reuse ORIGINAL reference products, not a current-by-current eigenproblem.

The simplex solver consumes only reference Gram and reference/current cross
products. Every reported distance and gap is checked in the full original
function coordinates. All probes remain; no basis rank or direction is selected.
"""
from dataclasses import dataclass
from typing import Any

import torch

from . import set_points


@dataclass
class Geometry:
    origin: torch.Tensor
    vectors: torch.Tensor
    gram: torch.Tensor
    cross: torch.Tensor
    norms: torch.Tensor
    reference_slot: torch.Tensor


@torch.no_grad()
def geometry(values: torch.Tensor, references=None) -> Geometry:
    if values.ndim != 3 or min(values.shape) < 1 or not bool(torch.isfinite(values).all()):
        raise ValueError('geometry needs finite event x function x coordinate values')
    values = values.double()
    batch, functions, _ = values.shape
    rows = torch.arange(batch, device=values.device)
    refs = (torch.arange(functions, device=values.device)[None, :].expand(batch, -1)
            if references is None else torch.as_tensor(references, dtype=torch.long, device=values.device))
    if refs.ndim != 2 or refs.shape[0] != batch or not refs.shape[1]:
        raise ValueError('one nonempty original reference inventory is needed per event')
    origin = values[rows[:, None], refs].mean(1, keepdim=True)
    about = values - origin
    original = about[rows[:, None], refs]
    gram = original @ original.transpose(1, 2)
    cross = about @ original.transpose(1, 2)
    norms = about.square().sum(2)
    slots = torch.full((batch, functions), -1, dtype=torch.long, device=values.device)
    slots.scatter_(1, refs, torch.arange(refs.shape[1], device=values.device)[None, :].expand_as(refs))
    return Geometry(origin, about, gram, cross, norms, slots)


@torch.no_grad()
def nearest(shared: Geometry, references, standing, cells=None) -> dict[str, Any]:
    device = shared.vectors.device
    rows = torch.arange(len(shared.vectors), device=device) if cells is None else torch.as_tensor(cells, device=device).long()
    refs = torch.as_tensor(references, dtype=torch.long, device=device)
    refs = refs[None, :].expand(len(rows), -1) if refs.ndim == 1 else refs
    at = torch.as_tensor(standing, dtype=torch.long, device=device)
    at = at.expand(len(rows)) if at.ndim == 0 else at
    if refs.ndim != 2 or refs.shape[0] != len(rows) or at.shape != rows.shape or refs.shape[1] < 1:
        raise ValueError('each current function needs a nonempty reference set')
    slots = shared.reference_slot[rows[:, None], refs]
    if bool((slots < 0).any()):
        raise ValueError('a current function cannot become an original reference')
    gram = shared.gram[rows[:, None, None], slots[:, :, None], slots[:, None, :]]
    cross = shared.cross[rows[:, None], at[:, None], slots]
    spread = gram.diagonal(dim1=1, dim2=2).amax(1).clamp_min(0)
    unit = torch.where(spread > 0, spread, torch.ones_like(spread))
    count, width = refs.shape[1], shared.vectors.shape[-1]
    bound = 2 * (width + 2 * count) * torch.finfo(torch.float64).eps * (
        1 + (shared.norms[rows, at].clamp_min(0) / unit).sqrt())
    solved: Any = set_points.from_products(gram / unit[:, None, None], cross / unit[:, None], bound)
    coefficients = solved['coefficients'].clamp_min(0)
    coefficients = coefficients / coefficients.sum(1, keepdim=True)
    own = shared.vectors[rows, at]
    target, centre = torch.zeros_like(own), torch.zeros_like(own)
    # A repeated current event never duplicates an entire reference tensor.
    for column in range(count):
        point = shared.vectors[rows, refs[:, column]]
        target.add_(coefficients[:, column, None] * point)
        centre.add_(point / count)
    residual = own - target
    gap = torch.zeros(len(rows), dtype=torch.float64, device=device)
    spread = torch.zeros_like(gap)
    for column in range(count):
        point = shared.vectors[rows, refs[:, column]]
        gap = torch.maximum(gap, (residual * (point - target)).sum(1))
        spread = torch.maximum(spread, (point - centre).square().sum(1))
    unit = torch.where(spread > 0, spread, torch.ones_like(spread))
    full_bound = 2 * (width + 2 * count) * torch.finfo(torch.float64).eps * (
        1 + (own - centre).norm(dim=1) / unit.sqrt()) * unit
    return dict(point=shared.origin[rows, 0] + target, coefficients=coefficients,
                squared=residual.square().sum(1), gap=gap, bound=full_bound,
                rounds=solved['rounds'])
