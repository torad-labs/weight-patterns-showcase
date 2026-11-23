"""One joint inverse and one matrix-response proposal for all bound native traces.

Requests and integration weights belong to the caller and stay fixed throughout
this proposal. No target is selected here. Omitting a declared member is an error,
not permission to renormalize the remaining histories.
"""
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import torch

from . import joint_inverse, trace_response
from .model_trace_response import Model


@dataclass
class Trace:
    graph: Model
    history: Sequence[int]
    residual: Callable


def _check(model, leaves, traces, expected_members):
    names, members = tuple(leaves), tuple(expected_members)
    if (not members or len(set(members)) != len(members)
            or set(traces) != set(members)):
        raise ValueError('the complete declared population must be represented exactly once')
    if (not names or len(set(names)) != len(names) or any(name not in model.w for name in names)):
        raise ValueError('every declared matrix must be represented exactly once')
    for trace in traces.values():
        if trace.graph.w is not model.w:
            raise ValueError('every trace must share the proposal model weight mapping')
    return names, members


def _chunks(trace):
    values = trace.residual(trace.graph.forward(trace.history))
    yield from (values,) if isinstance(values, torch.Tensor) else values


def inverse(model, leaves: Sequence[str], traces: Mapping[str, Trace], *, expected_members, progress=None):
    names, members = _check(model, leaves, traces, expected_members)
    original = model.w

    def residual(*weights):
        model.w = original | dict(zip(names, weights))
        try:
            for member in members:
                count = 0
                for chunk in _chunks(traces[member]):
                    count += 1
                    yield chunk
                if not count:
                    raise ValueError('a declared member has no complete functional residual')
        finally:
            model.w = original

    solved = joint_inverse.solve(residual, tuple(original[name] for name in names), progress=progress)
    return dict(zip(names, solved.changes)), solved


def response(model, leaves: Sequence[str], traces: Mapping[str, Trace],
             changes: Mapping[str, torch.Tensor], *, expected_members) -> trace_response.Step:
    names, members = _check(model, leaves, traces, expected_members)
    if set(changes) != set(names):
        raise ValueError('one joint direction must cover every declared matrix')
    original = model.w

    def residual(*weights):
        model.w = original | dict(zip(names, weights))
        try:
            for member in members:
                count = 0
                for chunk in _chunks(traces[member]):
                    count += 1
                    yield chunk
                if not count:
                    raise ValueError('a declared member has no complete functional residual')
        finally:
            model.w = original

    return trace_response.measure(residual, tuple(original[name] for name in names),
                                  tuple(changes[name] for name in names))
