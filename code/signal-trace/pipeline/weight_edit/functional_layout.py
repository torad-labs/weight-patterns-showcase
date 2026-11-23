"""Fixed original event coordinates for the conditional-function reference.

Paths associate computational events, never token identities. Every raw mapped
event remains distinct; a duplicate path entry is not another observation.
"""
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class Event:
    case: str
    position: int


@dataclass(frozen=True)
class Probe(Event):
    denominator: int


class Plan:
    def __init__(self, members: Mapping[str, Mapping[str, int]],
                 successes: Sequence[str], paths: Mapping, *, merges: int):
        self.axes = tuple(successes)
        if (len(self.axes) < 3 or len(set(self.axes)) != len(self.axes)
                or any(axis not in members for axis in self.axes)):
            raise ValueError('the original inventory needs distinct successes and a nonself witness')
        if type(merges) is not int or merges < 1:
            raise ValueError('the merge inventory must be a positive integer')
        self.merges = merges
        copied = {}
        for name, entry in members.items():
            if not isinstance(name, str) or not name:
                raise ValueError('original members need explicit names')
            counts = {field: entry[field] for field in ('question', 'steps')}
            if any(type(count) is not int or count < 1 for count in counts.values()):
                raise ValueError('original domains need positive integer event counts')
            copied[name] = MappingProxyType(counts)
        self.members = MappingProxyType(copied)
        self._pairs: dict[tuple[str, str, str], tuple[tuple[int, int], ...]] = {}
        self._mapped: dict[tuple[str, str, str], tuple[tuple[int, ...], ...]] = {}
        for axis in self.axes:
            for donor in self.members:
                for domain, field in (('question', 'question'), ('answer', 'steps')):
                    n, m = self.members[donor][field], self.members[axis][field]
                    if donor == axis:
                        pairs = tuple((row, row) for row in range(n))
                    else:
                        try:
                            raw = np.asarray(paths[axis][donor][domain])
                        except (KeyError, TypeError) as error:
                            raise ValueError('original correspondence must be complete') from error
                        if (raw.ndim != 2 or raw.shape[1] != 2
                                or not np.issubdtype(raw.dtype, np.integer)):
                            raise ValueError('original correspondence needs complete integer pairs')
                        pairs = tuple(sorted(set((int(i), int(j)) for i, j in raw)))
                    if ({i for i, _ in pairs} != set(range(n))
                            or {j for _, j in pairs} != set(range(m))):
                        raise ValueError('original correspondence must cover both complete domains')
                    key = axis, donor, domain
                    self._pairs[key] = pairs
                    grouped: list[list[int]] = [[] for _ in range(m)]
                    for i, j in pairs:
                        grouped[j].append(i)
                    self._mapped[key] = tuple(tuple(group) for group in grouped)

    def path(self, axis: str, donor: str, domain: str) -> np.ndarray:
        """Return complete original [donor row, axis row] correspondence."""
        try:
            pairs = self._pairs[axis, donor, domain]
        except KeyError as error:
            raise ValueError('unknown original axis, donor or domain') from error
        result = np.asarray(pairs, dtype=np.int64)
        result.setflags(write=False)
        return result

    def _positions(self, axis: str, donor: str, domain: str, row: int) -> tuple[int, ...]:
        try:
            mapped = self._mapped[axis, donor, domain]
        except KeyError as error:
            raise ValueError('unknown original axis, donor or domain') from error
        if type(row) is not int or not 0 <= row < len(mapped):
            raise ValueError('axis event is outside its original domain')
        offset = self.members[donor]['question'] if domain == 'answer' else 0
        return tuple(offset + position for position in mapped[row])

    def probes(self, axis: str, domain: str, row: int) -> tuple[Probe, ...]:
        """Every original success has equal mass, split over its raw events."""
        result = []
        for donor in self.axes:
            positions = self._positions(axis, donor, domain, row)
            denominator = len(self.axes) * len(positions)
            result.extend(Probe(donor, position, denominator) for position in positions)
        return tuple(result)

    def witnesses(self, axis: str, domain: str, row: int, *, recipient: str) -> tuple[Event, ...]:
        """Exclude original identities, without pooling the surviving functions."""
        # Validate the axis/event even if the caller names an excluded identity.
        self._positions(axis, axis, domain, row)
        return tuple(Event(donor, position) for donor in self.axes
                     if donor not in (axis, recipient)
                     for position in self._positions(axis, donor, domain, row))

    def statistics(self) -> dict[str, int]:
        events = sum(sum(self.members[axis].values()) for axis in self.axes)
        slots = sum(len(self._pairs[axis, donor, domain])
                    for axis in self.axes for donor in self.axes
                    for domain in ('question', 'answer'))
        return dict(axis_events=events, axis_merge_events=events * self.merges,
                    probe_slots=slots, probe_merge_slots=slots * self.merges)
