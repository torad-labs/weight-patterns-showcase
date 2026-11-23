"""Bind joint forward-mode responses to the existing complete model graph.

This adapter neither chooses an observable nor generates/grades text. Histories,
declared leaves, residual construction and direction are supplied by the caller.
It restores the model's weight mapping even if the measurement fails.
"""
from typing import Any, Callable, Mapping, Protocol, Sequence

import torch

from pipeline.weight_edit.trace_response import Residual, Step, measure


class Model(Protocol):
    @property
    def w(self) -> dict[str, torch.Tensor]: ...

    @w.setter
    def w(self, weights: dict[str, torch.Tensor]) -> None: ...

    def forward(self, tokens: Sequence[int]) -> Mapping[str, Any]: ...


def measure_model(model: Model, leaves: Sequence[str], history: Sequence[int],
                  direction: Mapping[str, torch.Tensor],
                  residual: Callable[[Mapping[str, Any]], Residual]) -> Step:
    """Propagate ALL declared matrix directions through one held complete trace.

    The history is acquisition input, not a comparison, eligibility filter or
    denominator. Model.forward retains shared K/V and earlier-layer dependencies.
    measure() disables the reverse graph, so the model's reverse checkpoint path
    is not used and forward tangents traverse the ordinary full forward.
    """
    names = tuple(leaves)
    if not names or len(set(names)) != len(names) or set(direction) != set(names):
        raise ValueError('the direction must name every declared matrix exactly once')
    original = model.w
    if any(name not in original for name in names):
        raise ValueError('a declared matrix is absent from the model')
    parameters = tuple(original[name] for name in names)

    def current(*weights: torch.Tensor):
        model.w = original | dict(zip(names, weights))
        try:
            chunks = residual(model.forward(history))
            if isinstance(chunks, torch.Tensor):
                yield chunks
            else:
                yield from chunks
        finally:
            model.w = original

    generators = []
    def tracked(*weights):
        generator = current(*weights)
        generators.append(generator)
        return generator

    try:
        return measure(tracked, parameters, tuple(direction[name] for name in names))
    finally:
        # Reducers can retain current() while it is suspended at yield. Close
        # deterministically so a later finalizer cannot overwrite another call's
        # candidate mapping or retain its graph; restore even if close raises.
        try:
            for generator in generators:
                generator.close()
        finally:
            model.w = original
