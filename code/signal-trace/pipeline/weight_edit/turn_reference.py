"""Choose one original turn trace and measure its nonself component variation.

Inputs must be the declared ORIGINAL fitting-success inventory, hash-bound by the
caller. This pure arithmetic never reads labels, tokens, current-model outcomes,
or an edited trace. It does not admit a run or claim that outcome labels certify
every computation in the original witnesses.
"""
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from pipeline.overlap.complete_overlap import distance_path, unit_steps, validate_pairs


@dataclass
class Selection:
    reference: str
    tolerance: np.ndarray
    scores: dict[str, float]
    paths: dict[str, np.ndarray]


def _trace(value, shape=None):
    array = np.asarray(value, dtype=np.float64)
    if (array.ndim != 3 or min(array.shape) < 1 or not np.isfinite(array).all()
            or (shape is not None and array.shape[1:] != shape)):
        raise ValueError('every original trace needs complete finite event-by-merge-by-coordinate data')
    return array


def select(names: Sequence[str], read: Callable[[str], np.ndarray], *, progress=None) -> Selection:
    """Select the observed medoid under complete turn-distance correspondence.

    For each axis event and component, epsilon squared is mean nonself squared
    departure: equal original-witness mass, and equal mass among that witness's
    events mapped to this axis event. Score averages the full squared vector
    departure over axis events and merges. Strict score ties retain source order.
    """
    names = tuple(names)
    if len(names) < 2 or len(set(names)) != len(names) or any(not name for name in names):
        raise ValueError('reference selection needs distinct originals and a nonself witness')
    scores: dict[str, float] = {}
    chosen = None
    shape = None
    for axis in names:
        original = _trace(read(axis), shape)
        shape = original.shape[1:]
        direction = unit_steps(original)
        squared = np.zeros_like(original)
        paths = {}
        for name in names:
            if name == axis:
                continue
            witness = _trace(read(name), shape)
            path = distance_path(direction @ unit_steps(witness).T)
            paths[name] = path
            boundaries = np.searchsorted(path[:, 0], np.arange(len(original) + 1))
            for row in range(len(original)):
                mapped = path[boundaries[row]:boundaries[row + 1], 1]
                difference = witness[mapped] - original[row]
                squared[row] += np.einsum('ijk,ijk->jk', difference, difference) / (
                    len(mapped) * (len(names) - 1))
        score = float(squared.sum(dtype=np.float64) / (original.shape[0] * original.shape[1]))
        if not np.isfinite(score):
            raise ValueError('the original reference score is nonfinite')
        scores[axis] = score
        if chosen is None or score < scores[chosen.reference]:
            np.sqrt(squared, out=squared)
            chosen = Selection(axis, squared, scores, paths)
        if progress is not None:
            progress(axis, score)
    assert chosen is not None
    return chosen


def coverage(selected: Selection, read: Callable[[str], np.ndarray], *, progress=None) -> dict[str, Any]:
    """Describe original coverage without changing selection or the frozen boundary.

    Leave-one-witness-out omits that witness only from the RMS calculation. The
    reference and paths still came from the full original inventory: this is not
    independent validation, a new threshold, or a separate stopping condition.
    """
    original = _trace(read(selected.reference))
    names = tuple(selected.paths)
    if set(names) != set(selected.scores) - {selected.reference} or not names:
        raise ValueError('coverage needs the complete selected nonself witness inventory')
    traces, ranges, records, outside_events = {}, {}, {}, {}
    for name in names:
        trace = traces[name] = _trace(read(name), original.shape[1:])
        path = selected.paths[name]
        validate_pairs(path, len(original), len(trace))
        ranges[name] = np.searchsorted(path[:, 0], np.arange(len(original) + 1))
        records[name] = {}
        for kind in ('frozen', 'leave_one_witness_out'):
            records[name][kind] = dict(components=0, outside_components=0, pairs=0,
                                       outside_pairs=0, cells=0, outside_cells=0,
                                       current_events=len(trace), objective=0.)
            outside_events[name, kind] = np.zeros(len(trace), dtype=bool)
    for row in range(len(original)):
        differences, mapped = {}, {}
        for name in names:
            start, end = ranges[name][row:row + 2]
            mapped[name] = selected.paths[name][start:end, 1]
            differences[name] = traces[name][mapped[name]] - original[row]
        squares = np.stack([np.mean(differences[name] ** 2, axis=0) for name in names])
        # Prefix/suffix sums omit a witness without subtractive cancellation.
        prefix = np.concatenate((np.zeros_like(squares[:1]), np.cumsum(squares, axis=0)))
        suffix = np.concatenate((np.cumsum(squares[::-1], axis=0)[::-1], np.zeros_like(squares[:1])))
        for index, name in enumerate(names):
            bounds = {'frozen': selected.tolerance[row]}
            if len(names) > 1:
                bounds['leave_one_witness_out'] = np.sqrt(
                    (prefix[index] + suffix[index + 1]) / (len(names) - 1))
            for kind, bound in bounds.items():
                amount = np.abs(differences[name])
                outside = amount > bound
                excess = np.maximum(amount - bound, 0.)
                record = records[name][kind]
                record['components'] += outside.size
                record['outside_components'] += int(outside.sum())
                record['pairs'] += len(outside)
                record['outside_pairs'] += int(outside.any(axis=(1, 2)).sum())
                record['cells'] += outside.shape[0] * outside.shape[1]
                record['outside_cells'] += int(outside.any(axis=2).sum())
                record['objective'] += float(np.sum(excess ** 2) / (
                    len(original) * original.shape[1] * len(outside)))
                outside_events[name, kind][mapped[name]] |= outside.any(axis=(1, 2))
        if progress is not None:
            progress(row + 1, len(original))
    for name in names:
        for kind in records[name]:
            records[name][kind]['outside_current_events'] = int(outside_events[name, kind].sum())
        if len(names) == 1:
            records[name]['leave_one_witness_out'] = None
    return dict(reference=selected.reference, witnesses=records,
                reference_components=selected.tolerance.size,
                zero_tolerance_components=int((selected.tolerance == 0).sum()),
                retuned=False, independent_validation=False)
