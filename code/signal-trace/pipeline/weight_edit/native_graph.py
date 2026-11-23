"""A complete differentiable graph rebased at an observed native trace.

Native observations and the Torch computations at the reached checkpoint are
frozen once. Each merge is native + (current computation - frozen computation).
Thus the origin reproduces every observed merge exactly without detaching the
current graph. Recomputing the frozen side during trials would hide movement;
this class never does that.

Keys/values and auxiliary inputs are reconstructed by the declared Torch model
at native layer inputs. They are NOT claimed to be captured native cache bytes.
The reader must bind and qualify that reconstruction separately.
"""
from itertools import chain
import math
from typing import Any, Mapping, Sequence

import torch
from torch.utils.checkpoint import checkpoint


class Graph:
    def __init__(self, model, history: Sequence[int], native: Mapping[str, Any]):
        self.model = model
        self.history = tuple(int(token) for token in history)
        count = 3 * model.layers
        if len(native['incoming']) != count or len(native['merged']) != count:
            raise ValueError('a native trace must supply every merge')
        shape = (len(self.history), model.width)
        for value in chain(native['incoming'], native['merged']):   # a streamed trace is read one merge at a time
            if value.shape != shape or value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
                raise ValueError('native observations must be finite position x width float32 arrays')
        self.native_in = tuple(value.detach().to(model.device).clone() for value in native['incoming'])
        self.native_out = tuple(value.detach().to(model.device).clone() for value in native['merged'])
        self.computed: list[torch.Tensor] = []
        self.exits: list[torch.Tensor] = []
        with torch.no_grad():
            positions = torch.arange(len(self.history), device=model.device)
            self.start = (model.rows('token_embd.weight', self.history) * math.sqrt(model.width)).detach()
            per = model.per_layer_input(self.native_in[0], self.history)
            cache = {}
            for layer in range(model.layers):
                i = 3 * layer
                x = self.native_in[i]
                if layer < model.keys_from:
                    cache[layer] = model.keys(layer, x, positions)
                keys, values = cache[model.keys_of(layer)]
                self.computed.extend((
                    (x + model.attention(layer, x, positions, keys, values)).detach(),
                    (self.native_in[i + 1] + model.feed_forward(layer, self.native_in[i + 1])).detach(),
                    (self.native_in[i + 2] + model.third(layer, self.native_in[i + 2], per[:, layer])).detach(),
                ))
                self.exits.append((self.native_out[i + 2] * model.w[f'blk.{layer}.layer_output_scale.weight']).detach())

    @property
    def w(self) -> dict[str, torch.Tensor]:
        return self.model.w

    @w.setter
    def w(self, weights: dict[str, torch.Tensor]) -> None:
        self.model.w = weights

    def __getattr__(self, name):
        return getattr(self.model, name)

    def _layer(self, layer, x, per, positions, keys, values):
        i = 3 * layer
        a = self.native_out[i] + (x + self.model.attention(layer, x, positions, keys, values) - self.computed[i])
        before_b = self.native_in[i + 1] + (a - self.native_out[i])
        b = self.native_out[i + 1] + (before_b + self.model.feed_forward(layer, before_b) - self.computed[i + 1])
        before_c = self.native_in[i + 2] + (b - self.native_out[i + 1])
        c = self.native_out[i + 2] + (before_c + self.model.third(layer, before_c, per) - self.computed[i + 2])
        out = c * self.model.w[f'blk.{layer}.layer_output_scale.weight']
        return a, before_b, b, before_c, c, out

    def forward(self, tokens: Sequence[int], again=True) -> dict[str, Any]:
        if tuple(int(token) for token in tokens) != self.history:
            raise ValueError('a native-anchored graph belongs to its bound acquisition history')
        positions = torch.arange(len(self.history), device=self.model.device)
        start = self.model.rows('token_embd.weight', self.history) * math.sqrt(self.model.width)
        x = self.native_in[0] + (start - self.start)
        per = self.model.per_layer_input(x, self.history)
        incoming, merged, cache = [], [], {}
        run = (lambda function, *args: checkpoint(function, *args, use_reentrant=False)) if again and torch.is_grad_enabled() else (lambda function, *args: function(*args))
        for layer in range(self.model.layers):
            if layer:
                x = self.native_in[3 * layer] + (x - self.exits[layer - 1])
            if layer < self.model.keys_from:
                cache[layer] = run(lambda given, layer=layer: self.model.keys(layer, given, positions), x)
            keys, values = cache[self.model.keys_of(layer)]
            a, before_b, b, before_c, c, out = run(
                lambda given, auxiliary, k, v, layer=layer: self._layer(layer, given, auxiliary, positions, k, v),
                x, per[:, layer], keys, values)
            incoming.extend((x, before_b, before_c))
            merged.extend((a, b, c))
            x = out
        return dict(incoming=incoming, merged=merged, exit=x, keys=cache, per_layer_input=per,
                    anchors=dict(native=self.native_out, computed=self.computed))
