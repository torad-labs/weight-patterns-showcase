"""Connect one complete turn request to the joint inverse and native BF16 search.

The caller supplies the reached state's native-anchored graph. Correspondence is
made from complete turns once, then held through this proposal. Neither a surrogate
decrease nor inner stationarity is fresh-generation closure. Nothing is installed.
"""
from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np
import torch

from pipeline.overlap.complete_overlap import distance_path, unit_steps

from . import bf16_direction, joint_functional_step as joint, joint_inverse, trace_response, turn_pair, vector_chance

Reference = turn_pair.Reference | vector_chance.Reference


@dataclass
class Proposal:
    pairs: np.ndarray
    before: dict[str, Any]
    inverse: joint_inverse.Proposal | None
    response: trace_response.Step | None
    native: bf16_direction.Result


def correspondence(reference: Reference, native) -> np.ndarray:
    # This package consumes the exact frozen trace used by Reference.measure;
    # accepting a separate caller-provided array could silently bind other pairs.
    original = np.stack([value.numpy() for value in reference._reference], axis=1)
    current = None
    for merge, value in enumerate(turn_pair.turn_stream(native, expected_merges=reference.merges)):
        turn = value.detach().cpu().numpy()
        if current is None:  # one merge in flight; the stacked array is the only full copy
            current = np.empty((turn.shape[0], reference.merges, turn.shape[1]), dtype=turn.dtype)
        current[:, merge] = turn
    return distance_path(unit_steps(original) @ unit_steps(current).T)


def measurement(reference: Reference, forward, pairs) -> dict[str, Any]:
    values, outside = [], 0
    for observed in reference.measure(forward, pairs):
        values.append(float(observed.residual.double().square().sum()))
        outside += int(observed.outside.sum())
    return dict(objective=math.fsum(values), outside=outside, pair_merge_cells=len(pairs) * reference.merges)


def propose(graph, history: Sequence[int], reference: Reference, *, member: str,
            pairs=None, progress=None) -> Proposal:
    names = tuple(graph.leaves)
    if not member or not names or len(set(names)) != len(names):
        raise ValueError('one member and every distinct editable matrix must be declared')
    with torch.no_grad():
        current = graph.forward(history)
        if pairs is None:
            pairs = correspondence(reference, current)
        before = measurement(reference, current, pairs)
    origin = bf16_direction.Value(before['objective'], before['objective'], before['objective'])
    if before['outside'] == 0:
        return Proposal(pairs, before, None, None,
                        bf16_direction.Result('inside_tolerance', None, origin, None, []))

    def residual(forward):
        return (reading.residual for reading in reference.measure(forward, pairs))
    traces = {member: joint.Trace(graph, history, residual)}
    changes, solved = joint.inverse(graph, names, traces, expected_members=[member], progress=progress)
    direction_on_device = {name: changes[name].to(graph.w[name]) for name in names}
    response = joint.response(graph, names, traces, direction_on_device, expected_members=[member])
    del direction_on_device
    if response.status != 'proposal' or response.length is None:
        return Proposal(pairs, before, solved, response,
                        bf16_direction.Result(response.status, None, origin, None, []))

    original = graph.w
    base = {name: original[name].detach().cpu().numpy() for name in names}
    direction = {name: changes[name].numpy() for name in names}

    def evaluate(weights):
        graph.w = original | {name: torch.from_numpy(weights[name]).to(original[name]) for name in names}
        try:
            with torch.no_grad():
                value = measurement(reference, graph.forward(history), pairs)['objective']
            return bf16_direction.Value(value, value, value)
        finally:
            graph.w = original

    native = bf16_direction.search(base, direction, length=response.length,
                                   slope=response.slope, evaluate=evaluate)
    return Proposal(pairs, before, solved, response, native)
