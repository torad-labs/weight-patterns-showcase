"""One calculation-trace functional on fixed ORIGINAL reference coordinates.

No token IDs, token targets, output probabilities, or group labels enter this
module. Local terms decompose a whole-trace loss; correspondence redistributes
reference-event mass without changing it when a current trace has more rows.
This is generic interval arithmetic, not a selected computational observable.
The turn/log-gain helper was withdrawn and removed before any model run.
See records/retrospectives/2026-11-15-original-agreement-loop-astra.md.
"""
from dataclasses import dataclass
from typing import Sequence, TypedDict

import torch


class AnchorReading(TypedDict):
    numerator: torch.Tensor
    supported_mass: float
    reference_events: int
    current_events: int
    missing_reference_events: int
    missing_current_events: int
    outside_reference_events: int
    outside_components: int
    largest_excess: float
    agrees: bool


def finite(*values):
    if not all(bool(torch.isfinite(value).all()) for value in values):
        raise ValueError('the computation and original reference must be finite')


@dataclass
class Bounds:
    lower: torch.Tensor
    upper: torch.Tensor
    lower_owner: torch.Tensor
    upper_owner: torch.Tensor
    witnesses: int

    def excluding(self, witness=None):
        """Exact interval after excluding at most one original recipient."""
        if witness is not None and not 0 <= witness < self.witnesses:
            raise ValueError('excluded witness is not in this original bank')
        lower, upper = self.lower[0], self.upper[0]
        if witness is not None:
            lower = torch.where(self.lower_owner[0] == witness, self.lower[1], lower)
            upper = torch.where(self.upper_owner[0] == witness, self.upper[1], upper)
        supported = torch.full(lower.shape[:-1], self.witnesses - int(witness is not None) >= 2,
                               dtype=torch.bool, device=lower.device)
        return lower, upper, supported


@torch.no_grad()
def original_bounds(witnesses, upper_witnesses=None):
    """Two original extrema per component permit exact leave-recipient-out use.

    Input is witness x original-event x merge x observable-coordinate. Each
    witness is an independently named original success; the axis is already out.
    With multiplicity, supply that witness's lower and upper observed values
    separately: its mapped events are never averaged into a synthetic point.
    No candidate observation is permitted as input by the reference compiler.
    """
    if witnesses.ndim != 4 or witnesses.shape[0] < 2 or min(witnesses.shape[1:]) < 1:
        raise ValueError('at least two witnesses with event x merge x coordinates are required')
    upper_witnesses = witnesses if upper_witnesses is None else upper_witnesses
    if witnesses.shape != upper_witnesses.shape:
        raise ValueError('each original witness needs matching lower and upper arrays')
    finite(witnesses, upper_witnesses)
    if bool((witnesses > upper_witnesses).any()):
        raise ValueError('an original witness lower value exceeds its upper value')
    lower, low_owner = torch.topk(witnesses, 2, dim=0, largest=False, sorted=True)
    upper, high_owner = torch.topk(upper_witnesses, 2, dim=0, largest=True, sorted=True)
    return Bounds(lower, upper, low_owner, high_owner, witnesses.shape[0])


def native_anchor(current, baseline, native):
    """Native value at the origin, with the current graph's full derivative."""
    if current.shape != baseline.shape or current.shape != native.shape:
        raise ValueError('current, baseline and native relations must have equal shapes')
    return native.detach() + (current - baseline.detach())


def anchor_loss(current, lower, upper, pairs, supported=None, *, chunk: int) -> AnchorReading:
    """One original-success axis's contribution and its complete coverage account.

    `current` is current-event x merge x relation-coordinate. `lower`/`upper`
    are ORIGINAL-event x merge x relation-coordinate. `pairs` names
    [original-event, current-event]. Each original event has one unit of mass,
    split over its mapped current events, and all original events/merges have
    equal mass. Missing mass is reported, not zero-valued evidence.

    Returns a graph-connected numerator and known reference mass. trace_loss
    combines complete axes with their fixed original mass. An incomplete reading
    stays an explicit partial account, not a comparable whole-trace objective.
    `agrees` is the local full-coverage predicate, not native verification or
    a claim about a capability. Native fresh-generation confirmation is the
    loop's separate responsibility.
    """
    if (current.ndim != 3 or lower.ndim != 3 or lower.shape != upper.shape
            or current.shape[1:] != lower.shape[1:] or min(current.shape) < 1
            or min(lower.shape) < 1):
        raise ValueError('current and reference need event x merge x observable coordinates')
    if not isinstance(chunk, int) or chunk < 1:
        raise ValueError('chunk partitions work and must be a positive integer')
    if lower.device != current.device or upper.device != current.device:
        raise ValueError('current and reference must be on the same device')
    finite(current, lower, upper)
    if bool((lower > upper).any()):
        raise ValueError('an original lower bound exceeds its upper bound')
    refs, merges, _ = lower.shape
    own = len(current)
    pairs = torch.as_tensor(pairs, device=current.device)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype not in (torch.int32, torch.int64):
        raise ValueError('pairs must be an integer array of [reference, current] events')
    pairs = pairs.long()
    if len(pairs) and (bool((pairs < 0).any()) or bool((pairs[:, 0] >= refs).any()) or bool((pairs[:, 1] >= own).any())):
        raise ValueError('a correspondence event is outside its trace')
    supported = (torch.ones((refs, merges), dtype=torch.bool, device=current.device) if supported is None
                 else supported.to(device=current.device))
    if supported.shape != (refs, merges) or supported.dtype != torch.bool:
        raise ValueError('supported must mark each original event and merge')
    counts = torch.bincount(pairs[:, 0], minlength=refs)
    known = supported & (counts[:, None] > 0)
    current_known = torch.zeros((own, merges), dtype=torch.int64, device=current.device)
    outside_ref = torch.zeros((refs, merges), dtype=torch.int64, device=current.device)
    numerator = current.reshape(-1)[0].double() * 0.0
    outside_components, peak = 0, 0.0
    total_events = refs * merges
    for start in range(0, len(pairs), chunk):
        ref, here = pairs[start:start + chunk].unbind(-1)
        valid = supported[ref]
        z = current[here].double()
        excess = torch.relu(lower[ref].double() - z) + torch.relu(z - upper[ref].double())
        weights = valid.double() / counts[ref, None].double() / total_events
        numerator = numerator + (excess.square().sum(-1) * weights).sum()
        with torch.no_grad():
            outside = (excess > 0) & valid[..., None]
            outside_ref.index_add_(0, ref, outside.any(-1).long())
            current_known.index_add_(0, here, valid.long())
            outside_components += int(outside.sum())
            peak = max(peak, float(torch.where(valid[..., None], excess.detach(), 0.0).max()))
    missing_reference = total_events - int(known.sum())
    missing_current = own * merges - int((current_known > 0).sum())
    outside_events = int((outside_ref > 0).sum())
    return AnchorReading(numerator=numerator,
                supported_mass=float(known.sum()) / total_events,
                reference_events=total_events, current_events=own * merges,
                missing_reference_events=missing_reference, missing_current_events=missing_current,
                outside_reference_events=outside_events, outside_components=outside_components,
                largest_excess=peak,
                agrees=missing_reference == 0 and missing_current == 0 and outside_events == 0)


def trace_loss(anchors: Sequence[AnchorReading]) -> torch.Tensor:
    """One COMPLETE trace's objective, with fixed equal original-axis mass.

    A partial reading has an unknown contribution. Renormalizing its surviving
    support or filling that contribution with zero would change the objective.
    The caller may inspect partial AnchorReadings but cannot optimize this scalar
    or compare it across states until current and original coverage are complete.
    """
    if not anchors or not any(item['supported_mass'] for item in anchors):
        raise ValueError('the trace has no supported original reference mass')
    if any(item['missing_reference_events'] or item['missing_current_events']
           or item['supported_mass'] != 1 for item in anchors):
        raise ValueError('an incomplete trace has no comparable whole-trace objective')
    numerator = anchors[0]['numerator']
    for item in anchors[1:]:
        numerator = numerator + item['numerator']
    return numerator / len(anchors)
