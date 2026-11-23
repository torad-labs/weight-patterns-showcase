"""Whole-vector predictive chance: one boundary for excess and complete closure."""
from collections.abc import Iterator, Sequence
from itertools import chain

import torch

from pipeline.overlap.complete_overlap import validate_pairs

from .reference_rule import Outside
from .turn_pair import Turns


def outside(delta: torch.Tensor, chance: torch.Tensor, mass: torch.Tensor) -> Outside:
    """Weighted distance to the original ball of radius sqrt(2 * chance).

    At zero radius the residual is delta itself, including its Jacobian at zero.
    Positive-radius interiors have zero request and zero local response.
    """
    delta = delta.double()
    chance, mass = chance.detach().double(), mass.detach().double()
    if delta.ndim != 2 or chance.shape != delta.shape[:1] or mass.shape != delta.shape[:1]:
        raise ValueError('every full vector needs its original chance and positive mass')
    if not all(bool(torch.isfinite(value).all()) for value in (delta, chance, mass)):
        raise ValueError('a nonfinite vector or original chance is not measurable')
    if bool((chance < 0).any()) or bool((mass <= 0).any()):
        raise ValueError('original chance is nonnegative and represented mass is positive')
    distance = torch.linalg.vector_norm(delta, dim=-1)
    radius = (2 * chance).sqrt()
    active = distance > radius
    denominator = torch.where(distance > 0, distance, torch.ones_like(distance))
    shrink = torch.where(active, 1 - radius / denominator, torch.zeros_like(distance))
    shrink = torch.where(chance == 0, torch.ones_like(shrink), shrink)
    return Outside(delta * shrink[:, None] * mass.sqrt()[:, None], active)


def turns(forward, *, expected_merges: int) -> Turns:
    """A trace's turns for one measurement, each formed when the measurement reads it."""
    return Turns(forward, expected_merges)


class Reference:
    """One immutable original turn field and its original per-merge vector chance."""

    def __init__(self, original: Sequence[torch.Tensor], chance: Sequence[torch.Tensor],
                 *, expected_merges: int):
        if (type(expected_merges) is not int or expected_merges < 1
                or len(original) != expected_merges or len(chance) != expected_merges):
            raise ValueError('the reference must supply every declared merge')
        shape = original[0].shape
        if (len(shape) != 2 or min(shape) < 1 or any(value.shape != shape
                or not bool(torch.isfinite(value).all()) for value in original)):
            raise ValueError('original turns need complete finite event-by-coordinate arrays')
        if any(value.shape != shape[:1] or not bool(torch.isfinite(value).all())
               or bool((value < 0).any()) for value in chance):
            raise ValueError('original chance needs every event, finite and nonnegative')
        self._reference = tuple(value.detach().double().cpu().clone() for value in original)
        self._chance = tuple(value.detach().double().cpu().clone() for value in chance)
        self.rows, self.width = shape
        self.merges = expected_merges

    def measure(self, forward, pairs) -> Iterator[Outside]:
        stream = iter(turns(forward, expected_merges=self.merges))
        first = next(stream)
        current = chain([first], stream)
        if first.shape[1] != self.width:
            raise ValueError('the current coordinates differ from the original')
        index = torch.as_tensor(pairs).detach().cpu()
        if index.dtype not in (torch.int32, torch.int64):
            raise ValueError('correspondence needs integer event pairs')
        validate_pairs(index.numpy(), self.rows, len(first))
        original, reached = index[:, 0].long(), index[:, 1].long()
        multiplicity = torch.bincount(original, minlength=self.rows)
        mass = 1. / (self.rows * self.merges * multiplicity[original].double())
        for value, reference, chance in zip(current, self._reference, self._chance):
            device = value.device
            delta = value[reached.to(device)] - reference[original].to(device)
            yield outside(delta, chance[original].to(device), mass.to(device))
