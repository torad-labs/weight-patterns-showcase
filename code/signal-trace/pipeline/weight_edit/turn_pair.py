"""The supplied guide's full turn trace against one immutable original trace.

This binds the observable to an explicit original component boundary. It does not
select a reference, estimate that boundary, choose correspondence, or admit a run.
No token, answer label, raw-state target or input-Jacobian comparison enters here.
"""
from collections.abc import Iterator, Sequence
from itertools import chain

import torch

from pipeline.overlap.complete_overlap import validate_pairs

from .reference_rule import Outside, outside


def turn_stream(forward, *, expected_merges: int) -> Iterator[torch.Tensor]:
    """Every merge's unit(output)-unit(input), one merge at a time, retaining its amount and gradient."""
    incoming, merged = forward['incoming'], forward['merged']
    if (type(expected_merges) is not int or expected_merges < 1
            or len(incoming) != expected_merges or len(merged) != expected_merges):
        raise ValueError('the trace must supply every declared merge')
    shape = incoming[0].shape
    if len(shape) != 2 or min(shape) < 1:
        raise ValueError('each merge needs a nonempty event-by-coordinate array')
    for before, after in zip(incoming, merged):
        if before.shape != shape or after.shape != shape or before.device != after.device:
            raise ValueError('all native merges must have the same complete event inventory')
        # Original calibration and fresh native measurements use one canonical
        # CPU float64 reduction. Transfers retain both reverse and forward AD;
        # device-specific normalization must not create excess at zero chance.
        before = before.to(device='cpu', dtype=torch.float64)
        after = after.to(device='cpu', dtype=torch.float64)
        if not bool(torch.isfinite(before).all() & torch.isfinite(after).all()):
            raise ValueError('a nonfinite native edge has no defined turn')
        before_norm = torch.linalg.vector_norm(before, dim=-1, keepdim=True)
        after_norm = torch.linalg.vector_norm(after, dim=-1, keepdim=True)
        if bool((before_norm == 0).any() | (after_norm == 0).any()):
            raise ValueError('a zero native edge has no defined direction')
        yield after / after_norm - before / before_norm


def turns(forward, *, expected_merges: int) -> tuple[torch.Tensor, ...]:
    """Every merge's turn at once; use turn_stream where one merge at a time is enough."""
    return tuple(turn_stream(forward, expected_merges=expected_merges))


class Turns(Sequence):
    """Every merge's turn, formed when it is asked for, so a trace read from a stream is in RAM one merge at a time."""

    def __init__(self, forward, expected_merges):
        if (type(expected_merges) is not int or expected_merges < 1
                or any(len(forward[key]) != expected_merges for key in ('incoming', 'merged'))):
            raise ValueError('the trace must supply every declared merge')
        self.forward, self.merges, self.shape = forward, expected_merges, None

    def __len__(self):
        return self.merges

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self[i] for i in range(self.merges)[index])
        at = range(self.merges)[index]
        one = {key: self.forward[key][at:at + 1] for key in ('incoming', 'merged')}
        if self.shape is None:
            self.shape = one['incoming'][0].shape if at == 0 else self.forward['incoming'][0].shape
        if one['incoming'][0].shape != self.shape:
            raise ValueError('all native merges must have the same complete event inventory')
        return next(turn_stream(one, expected_merges=1))

    def __iter__(self):
        return turn_stream(self.forward, expected_merges=self.merges)


class Reference:
    """One frozen original trace; tolerance is supplied, never fitted here.

    The component boundary is in full turn units. A caller must justify and bind
    its original-only source. The same boundary supplies direct excess and the
    outside indicator. Reading an inside indicator alone is not a repair verdict.
    """

    def __init__(self, native, tolerance: Sequence[torch.Tensor], *, expected_merges: int):
        self._bind(turns(native, expected_merges=expected_merges), tolerance, expected_merges)

    @classmethod
    def from_turns(cls, original: Sequence[torch.Tensor], tolerance: Sequence[torch.Tensor],
                   *, expected_merges: int):
        """Use already compiled turns without inventing input/output state targets."""
        reference = cls.__new__(cls)
        reference._bind(original, tolerance, expected_merges)
        return reference

    def _bind(self, original, tolerance, expected_merges):
        if (type(expected_merges) is not int or expected_merges < 1
                or len(original) != expected_merges):
            raise ValueError('the original turn trace must supply every declared merge')
        shape = original[0].shape
        if (len(shape) != 2 or min(shape) < 1
                or any(value.shape != shape or not bool(torch.isfinite(value).all()) for value in original)):
            raise ValueError('the original turns need complete finite event-by-coordinate arrays')
        self._reference = tuple(value.detach().double().cpu().clone() for value in original)
        self.rows, self.width = shape
        self.merges = expected_merges
        if len(tolerance) != self.merges:
            raise ValueError('the original tolerance must cover every merge')
        self._tolerance = tuple(value.detach().double().cpu().clone() for value in tolerance)
        if any(value.shape != (self.rows, self.width)
               or not bool(torch.isfinite(value).all()) or bool((value < 0).any())
               for value in self._tolerance):
            raise ValueError('the original component tolerance must be complete, finite and nonnegative')

    def measure(self, forward, pairs) -> Iterator[Outside]:
        """Weighted excess at every pair and merge, with fixed original-event mass."""
        stream = turn_stream(forward, expected_merges=self.merges)
        first = next(stream)
        current = chain([first], stream)
        if first.shape[1] != self.width:
            raise ValueError('the current turn coordinates differ from the original')
        index = torch.as_tensor(pairs)
        if index.dtype not in (torch.int32, torch.int64):
            raise ValueError('correspondence must give integer event pairs')
        index = index.detach().cpu()
        validate_pairs(index.numpy(), self.rows, len(first))
        original, reached = index[:, 0].long(), index[:, 1].long()
        multiplicity = torch.bincount(original, minlength=self.rows)
        mass = 1. / (self.rows * self.merges * multiplicity[original].double())
        for value, reference, tolerance in zip(current, self._reference, self._tolerance):
            device = value.device
            delta = value[reached.to(device)] - reference[original].to(device)
            bound = tolerance[original].to(device)
            yield outside(delta, -bound, bound, mass.to(device))
