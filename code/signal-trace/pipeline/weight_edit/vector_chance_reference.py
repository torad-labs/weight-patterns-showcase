"""Predictive whole-vector variation around one fixed original turn trace.

C is the empirical expected squared distance of ONE execution from the fixed
reference. It is not the standard error of a population mean.
"""
from dataclasses import dataclass
from typing import Any

import numpy as np

from pipeline.overlap.complete_overlap import validate_pairs
from .turn_reference import _trace


@dataclass
class Calibration:
    chance: np.ndarray
    within: np.ndarray
    between: np.ndarray
    offset: np.ndarray
    coverage: dict[str, Any]


def compile(reference, names, read, paths, *, progress=None) -> Calibration:
    names = tuple(names)
    witnesses = tuple(name for name in names if name != reference)
    if (reference not in names or len(set(names)) != len(names) or not witnesses
            or set(paths) != set(witnesses)):
        raise ValueError('chance needs the complete distinct original inventory and correspondence')
    original = _trace(read(reference))
    shape = original.shape[:2]
    chance, within, between, offset = (np.zeros(shape, dtype=np.float64) for _ in range(4))
    traces, ranges, records, events = {}, {}, {}, {}
    for name in witnesses:
        trace = traces[name] = _trace(read(name), original.shape[1:])
        validate_pairs(paths[name], len(original), len(trace))
        ranges[name] = np.searchsorted(paths[name][:, 0], np.arange(len(original) + 1))
        records[name] = {}
        for kind in ('frozen', 'leave_one_witness_out'):
            records[name][kind] = dict(cells=0, outside_cells=0, pairs=0, outside_pairs=0,
                                       current_events=len(trace), objective=0., largest_excess=0.)
            events[name, kind] = np.zeros(len(trace), dtype=bool)
    for row in range(len(original)):
        differences, mapped, squares, means, internal = {}, {}, [], [], []
        for name in witnesses:
            begin, end = ranges[name][row:row + 2]
            mapped[name] = paths[name][begin:end, 1]
            delta = differences[name] = traces[name][mapped[name]] - original[row]
            mean = delta.mean(axis=0)
            means.append(mean)
            squares.append(np.mean(np.sum(delta ** 2, axis=-1), axis=0))
            internal.append(np.mean(np.sum((delta - mean) ** 2, axis=-1), axis=0))
        mean_array, moments = np.stack(means), np.stack(squares)
        centre = mean_array.mean(axis=0)
        chance[row] = moments.mean(axis=0)
        within[row] = np.stack(internal).mean(axis=0)
        between[row] = np.mean(np.sum((mean_array - centre) ** 2, axis=-1), axis=0)
        offset[row] = np.sum(centre ** 2, axis=-1)
        zero = np.zeros_like(moments[:1])
        prefix = np.concatenate((zero, np.cumsum(moments, axis=0)))
        suffix = np.concatenate((np.cumsum(moments[::-1], axis=0)[::-1], zero))
        for index, name in enumerate(witnesses):
            levels = {'frozen': chance[row]}
            if len(witnesses) > 1:
                levels['leave_one_witness_out'] = (prefix[index] + suffix[index + 1]) / (len(witnesses) - 1)
            distance = np.linalg.norm(differences[name], axis=-1)
            for kind, level in levels.items():
                radius = np.sqrt(2 * level)
                active = distance > radius
                excess = np.maximum(distance - radius, 0.)
                record = records[name][kind]
                record['cells'] += active.size
                record['outside_cells'] += int(active.sum())
                record['pairs'] += len(active)
                record['outside_pairs'] += int(active.any(axis=1).sum())
                record['objective'] += float(np.sum(excess ** 2) / (
                    len(original) * original.shape[1] * len(active)))
                record['largest_excess'] = max(record['largest_excess'], float(excess.max()))
                events[name, kind][mapped[name]] |= active.any(axis=1)
        if progress is not None:
            progress(row + 1, len(original))
    for name in witnesses:
        for kind in records[name]:
            records[name][kind]['outside_current_events'] = int(events[name, kind].sum())
        if len(witnesses) == 1:
            records[name]['leave_one_witness_out'] = None
    covered = {kind: [name for name in witnesses if records[name][kind] is not None
                      and records[name][kind]['outside_current_events'] == 0]
               for kind in ('frozen', 'leave_one_witness_out')}
    coverage = dict(reference=reference, witnesses=records, complete_witnesses_inside=covered,
        zero_chance_cells=int((chance == 0).sum()), reference_cells=chance.size,
        decomposition_max_absolute_error=float(np.max(np.abs(chance - within - between - offset))),
        independent_validation=False, retuned=False,
        whole_trace_coverage_is_descriptive_not_an_admission_gate=True)
    return Calibration(chance, within, between, offset, coverage)
