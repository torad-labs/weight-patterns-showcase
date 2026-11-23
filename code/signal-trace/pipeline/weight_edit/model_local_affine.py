"""Expose all causal arguments of each existing model merge as a function.

No reference, probes, alignment, score or closure is selected here. The caller
must supply a common input domain before comparing local affine maps. Functions
are live views of the model's current weights, not frozen checkpoint artifacts.
"""
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Mapping, Sequence

import torch


@dataclass(frozen=True)
class Merge:
    index: int
    kind: str
    function: Callable[..., torch.Tensor]
    inputs: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class ConditionalMerge:
    index: int
    kind: str
    function: Callable[[torch.Tensor], torch.Tensor]
    query: torch.Tensor
    condition: tuple[torch.Tensor, ...]


def _anchored(function, index, forward, rows=None):
    anchors = forward.get('anchors')
    if anchors is None:
        return function
    native, computed = anchors['native'][index], anchors['computed'][index]
    if rows is not None:
        native, computed = native[rows], computed[rows]
    def anchored(*args):
        return native.detach() + (function(*args) - computed.detach())
    return anchored


def _conditional_attention(model, layer, positions, keys_at, keys, values, x):
    return x + model.attention(layer, x, positions, keys, values, keys_at)


def _conditional_third(model, layer, per, x):
    return x + model.third(layer, x, per)


def conditional_merges(model, history: Sequence[int], forward: Mapping[str, Any],
                       rows: torch.Tensor | None = None, *,
                       merge_indices: Sequence[int] | None = None) -> list[ConditionalMerge]:
    """Local query-response functions conditioned on the actual causal context.

    Memory is NOT transported, averaged, reindexed or duplicated by correspondence.
    Repeated output queries see the same actual cache, mask and RoPE coordinates.
    Conditioning restricts the counterfactual input domain; it does NOT detach
    memory or per-layer signals from the weight derivative. This is not a claim
    of equality as functions of arbitrary alternative memory states.
    """
    if len(forward['incoming']) != 3 * model.layers:
        raise ValueError('the forward must contain every layer and merge')
    positions = torch.arange(len(history), device=model.device)
    rows = positions if rows is None else rows.to(device=model.device)
    if rows.ndim != 1 or rows.dtype not in (torch.int32, torch.int64) or not len(rows):
        raise ValueError('queried rows must be a nonempty integer vector')
    if bool((rows < 0).any() | (rows >= len(history)).any()):
        raise ValueError('a queried row is outside the actual history')
    indices = tuple(range(3 * model.layers)) if merge_indices is None else tuple(merge_indices)
    if (not indices or len(set(indices)) != len(indices)
            or any(type(index) is not int or not 0 <= index < 3 * model.layers for index in indices)):
        raise ValueError('streamed merge indices must be distinct model merges')
    per = forward.get('per_layer_input')
    if any(index % 3 == 2 for index in indices) and per is None:
        per = model.per_layer_input(forward['incoming'][0], history)
    result = []
    for index in indices:
        layer, kind = divmod(index, 3)
        query = forward['incoming'][index][rows]
        if kind == 0:
            keys, values = forward['keys'][model.keys_of(layer)]
            result.append(ConditionalMerge(index, 'attention',
                partial(_conditional_attention, model, layer, positions[rows], positions, keys, values),
                query, (keys, values)))
        elif kind == 1:
            result.append(ConditionalMerge(index, 'feed_forward', partial(_feed_forward, model, layer), query, ()))
        else:
            assert per is not None
            auxiliary = per[rows, layer]
            result.append(ConditionalMerge(index, 'third', partial(_conditional_third, model, layer, auxiliary),
                                          query, (auxiliary,)))
    return [ConditionalMerge(item.index, item.kind, _anchored(item.function, item.index, forward, rows),
                             item.query, item.condition) for item in result]


def _attention(model, layer, positions, x, keys, values):
    return x + model.attention(layer, x, positions, keys, values)


def _feed_forward(model, layer, x):
    return x + model.feed_forward(layer, x)


def _third(model, layer, x, per):
    return x + model.third(layer, x, per)


def merges(model, history: Sequence[int], forward: Mapping[str, Any]) -> list[Merge]:
    """Expose all three merges per layer from a complete current forward.

    The history is used only to reconstruct positions and the per-layer input
    acquisition. No token comparison selects events, probes or reference support.
    Attention includes its entire shared key/value cache. Both its own position
    inputs and earlier key-making layers retain their graph connections.
    """
    if len(forward['incoming']) != 3 * model.layers:
        raise ValueError('the forward must contain every layer and merge')
    positions = torch.arange(len(history), device=model.device)
    per = model.per_layer_input(forward['incoming'][0], history)
    result = []
    for layer in range(model.layers):
        index = 3 * layer
        keys, values = forward['keys'][model.keys_of(layer)]
        result.append(Merge(index, 'attention', partial(_attention, model, layer, positions),
                            (forward['incoming'][index], keys, values)))
        result.append(Merge(index + 1, 'feed_forward', partial(_feed_forward, model, layer),
                            (forward['incoming'][index + 1],)))
        result.append(Merge(index + 2, 'third', partial(_third, model, layer),
                            (forward['incoming'][index + 2], per[:, layer])))
    return [Merge(item.index, item.kind, _anchored(item.function, item.index, forward), item.inputs)
            for item in result]
