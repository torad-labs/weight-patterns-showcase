"""Numerical conditional query functions at fixed original native observations.

This reader reconstructs contexts with the bound Torch model, including its
declared cache rounding. It does NOT claim to observe native cache bytes or a
native Jacobian. Every intercept is its observed native merge; its derivative is
the explicit conditional Torch function's input derivative.
"""
from functools import partial
from typing import Callable

import torch


class OriginalFunctions:
    def __init__(self, reading):
        self.reading, self.model = reading, reading.model
        self.histories = {}
        self.acquisition_limits: list[str] = []
        for name in reading.members:
            history, _, complete, ended = reading.acquisition(name)
            if complete is not True:
                raise ValueError('original functional measurement requires complete own acquisitions')
            if not ended:
                self.acquisition_limits.append(name)
            self.histories[name] = history
        self.contexts: dict[str, tuple[torch.Tensor, torch.Tensor, Callable]] = {}
        self.merge: int | None = None

    @torch.no_grad()
    def prepare(self, merge: int) -> None:
        model, reading = self.model, self.reading
        if type(merge) is not int or not 0 <= merge < 3 * model.layers:
            raise ValueError('unknown conditional merge')
        self.contexts.clear()
        layer, kind = divmod(merge, 3)
        for name, history in self.histories.items():
            x, observed = reading.incoming(name, merge), reading.state(name, merge)
            positions = torch.arange(len(history), device=model.device)
            if kind == 0:
                key_layer = model.keys_of(layer)
                key_input = x if key_layer == layer else reading.incoming(name, 3 * key_layer)
                keys, values = model.keys(key_layer, key_input, positions)
                raw = partial(self._attention, layer, positions, keys, values)
            elif kind == 1:
                raw = partial(self._feed_forward, layer)
            else:
                first = reading.incoming(name, 0)
                auxiliary = model.per_layer_input(first, history)[:, layer].contiguous()
                raw = partial(self._third, layer, auxiliary)
            self.contexts[name] = x.detach(), observed.detach(), raw
        self.merge = merge

    def _attention(self, layer, positions, keys, values, rows, query):
        return query + self.model.attention(layer, query, positions[rows], keys, values, positions)

    def _feed_forward(self, layer, rows, query):
        return query + self.model.feed_forward(layer, query)

    def _third(self, layer, auxiliary, rows, query):
        return query + self.model.third(layer, query, auxiliary[rows])

    def query(self, name: str) -> torch.Tensor:
        return self.contexts[name][0]

    @torch.no_grad()
    def evaluate(self, name: str, rows: torch.Tensor, probes: torch.Tensor) -> torch.Tensor:
        x, observed, raw = self.contexts[name]
        if rows.ndim != 1 or rows.dtype not in (torch.int32, torch.int64) or not rows.numel():
            raise ValueError('conditional events need a nonempty integer position vector')
        if bool((rows < 0).any() | (rows >= len(x)).any()):
            raise ValueError('conditional event is outside its own history')
        current = x[rows]
        # Reference measurement needs only the input action, not a reverse
        # weight graph. Direct forward AD avoids keeping a second-order reverse
        # graph for every Cartesian-product query. The differentiated update
        # continues to use local_affine.action for mixed derivatives.
        from torch.autograd import forward_ad
        if probes.shape != current.shape or probes.dtype != current.dtype or probes.device != current.device:
            raise ValueError('common probes must share the conditional input shape, dtype and device')
        with forward_ad.dual_level():
            dual = forward_ad.make_dual(current, probes - current)
            _, tangent = forward_ad.unpack_dual(raw(rows, dual))
            result = observed[rows] if tangent is None else observed[rows] + tangent
        if not bool(torch.isfinite(result).all()):
            raise ValueError('a nonfinite query action leaves the original function unavailable')
        return result.detach()
