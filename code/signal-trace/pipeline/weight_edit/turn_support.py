"""One fixed original trace's signed observed-support interval and coverage."""
from dataclasses import dataclass
from typing import Any

import numpy as np

from pipeline.overlap.complete_overlap import validate_pairs
from .turn_reference import _trace


@dataclass
class Support:
    lower: np.ndarray
    upper: np.ndarray
    coverage: dict[str, Any]


def compile(reference, names, read, paths, *, progress=None) -> Support:
    names = tuple(names)
    witnesses = tuple(name for name in names if name != reference)
    if (reference not in names or len(set(names)) != len(names) or not witnesses
            or set(paths) != set(witnesses)):
        raise ValueError('support needs the complete distinct original inventory and correspondence')
    original = _trace(read(reference))
    lower, upper = np.zeros_like(original), np.zeros_like(original)
    traces, ranges, records, events = {}, {}, {}, {}
    for name in witnesses:
        trace = traces[name] = _trace(read(name), original.shape[1:])
        validate_pairs(paths[name], len(original), len(trace))
        ranges[name] = np.searchsorted(paths[name][:, 0], np.arange(len(original) + 1))
        records[name] = {}
        for kind in ('frozen', 'leave_one_witness_out'):
            records[name][kind] = dict(components=0, outside_components=0, pairs=0,
                                       outside_pairs=0, cells=0, outside_cells=0,
                                       current_events=len(trace), objective=0.)
            events[name, kind] = np.zeros(len(trace), dtype=bool)
    for row in range(len(original)):
        difference, mapped = {}, {}
        for name in witnesses:
            begin, end = ranges[name][row:row + 2]
            mapped[name] = paths[name][begin:end, 1]
            difference[name] = traces[name][mapped[name]] - original[row]
        minima = np.stack([difference[name].min(axis=0) for name in witnesses])
        maxima = np.stack([difference[name].max(axis=0) for name in witnesses])
        zero = np.zeros_like(minima[:1])
        low_prefix = np.minimum.accumulate(np.concatenate((zero, minima)), axis=0)
        low_suffix = np.minimum.accumulate(np.concatenate((zero, minima[::-1])), axis=0)[::-1]
        high_prefix = np.maximum.accumulate(np.concatenate((zero, maxima)), axis=0)
        high_suffix = np.maximum.accumulate(np.concatenate((zero, maxima[::-1])), axis=0)[::-1]
        lower[row], upper[row] = low_prefix[-1], high_prefix[-1]
        for index, name in enumerate(witnesses):
            bounds = {'frozen': (lower[row], upper[row])}
            if len(witnesses) > 1:
                bounds['leave_one_witness_out'] = (
                    np.minimum(low_prefix[index], low_suffix[index + 1]),
                    np.maximum(high_prefix[index], high_suffix[index + 1]))
            for kind, (low, high) in bounds.items():
                delta = difference[name]
                outside = (delta < low) | (delta > high)
                excess = np.where(delta < low, delta - low, np.where(delta > high, delta - high, 0.))
                record = records[name][kind]
                record['components'] += outside.size
                record['outside_components'] += int(outside.sum())
                record['pairs'] += len(outside)
                record['outside_pairs'] += int(outside.any(axis=(1, 2)).sum())
                record['cells'] += outside.shape[0] * outside.shape[1]
                record['outside_cells'] += int(outside.any(axis=2).sum())
                record['objective'] += float(np.sum(excess ** 2) / (
                    len(original) * original.shape[1] * len(outside)))
                events[name, kind][mapped[name]] |= outside.any(axis=(1, 2))
        if progress is not None:
            progress(row + 1, len(original))
    for name in witnesses:
        for kind in records[name]:
            records[name][kind]['outside_current_events'] = int(events[name, kind].sum())
        if len(witnesses) == 1:
            records[name]['leave_one_witness_out'] = None
    covered = {kind: [name for name in witnesses
                      if records[name][kind] is not None
                      and records[name][kind]['outside_current_events'] == 0]
               for kind in ('frozen', 'leave_one_witness_out')}
    return Support(lower, upper, dict(reference=reference, witnesses=records,
        complete_witnesses_inside=covered, reference_components=original.size,
        zero_width_components=int(((lower == 0) & (upper == 0)).sum()),
        independent_validation=False, retuned=False,
        all_mapped_events_define_support=True,
        prospective_weakening_condition_met=not covered['leave_one_witness_out']))
