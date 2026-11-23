"""Search a joint direction on the engine's native BF16 grid.

The first length comes from the whole matrix response. Subsequent proposals use
measured objective curvature or a distinct native rounding boundary, never a
borrowed decay factor. This module writes no model and makes no closure claim.
"""
from dataclasses import dataclass
import hashlib
import math
from typing import Callable, Mapping

import numpy as np

from .a2 import from_bf16, to_bf16

Weights = Mapping[str, np.ndarray]


@dataclass(frozen=True)
class Value:
    estimate: float
    lower: float
    upper: float

    def __post_init__(self):
        if (not all(math.isfinite(value) for value in (self.estimate, self.lower, self.upper))
                or self.estimate < 0 or not 0 <= self.lower <= self.upper):
            raise ValueError('a fixed-request reading needs finite nonnegative distance bounds')


@dataclass
class Result:
    status: str
    weights: dict[str, np.ndarray] | None
    value: Value
    length: float | None
    trials: list[dict]
    closure: bool = False


def _validate(base: Weights, direction: Weights) -> None:
    if not base or set(base) != set(direction):
        raise ValueError('one direction must name every native matrix')
    for name, value in base.items():
        delta = direction[name]
        if (value.dtype != np.float32 or delta.dtype != np.float32 or value.shape != delta.shape
                or not np.isfinite(value).all() or not np.isfinite(delta).all()):
            raise ValueError('native weights and directions need matching finite float32 arrays')
        if not np.array_equal(value, from_bf16(to_bf16(value)).reshape(value.shape)):
            raise ValueError('the starting weights are not the engine native values')


def rounded(base: Weights, direction: Weights, length: float) -> dict[str, np.ndarray]:
    if not math.isfinite(length) or length < 0 or set(base) != set(direction):
        raise ValueError('a native trial needs a finite nonnegative joint length')
    result = {}
    for name, value in base.items():
        if value.shape != direction[name].shape:
            raise ValueError('native trial direction shape differs from its weight')
        wanted = (value.astype(np.float64) + length * direction[name].astype(np.float64)).astype(np.float32)
        stored = from_bf16(to_bf16(wanted)).reshape(value.shape)
        if not np.isfinite(stored).all():
            raise ValueError('a native trial produced a nonfinite stored weight')
        result[name] = stored
    return result


def _neighbor(values: np.ndarray, sign: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32).astype(np.int64) >> 16
    positive = values > 0
    goes_up = sign > 0
    step = np.where(positive == goes_up, 1, -1)
    moved = from_bf16((bits + step).astype(np.uint16)).reshape(values.shape)
    tiny = np.finfo(np.float32).tiny
    moved = np.where(np.abs(moved) < tiny, np.copysign(np.float32(0), moved), moved)
    moved = np.where(values == 0, np.copysign(tiny, sign), moved)
    return moved.astype(np.float32)


def _across_boundary(one, other, toward):
    boundary = (one.astype(np.float64) + other.astype(np.float64)) / 2
    # ggml flushes every float32 subnormal before BF16 rounding: the native
    # zero/minimum-normal boundary is at minimum-normal, not their midpoint.
    zero_edge = (one == 0) | (other == 0)
    boundary = np.where(zero_edge, np.where(one == 0, other, one), boundary).astype(np.float32)
    return np.nextafter(boundary, np.copysign(np.float32(np.inf), toward), dtype=np.float32)


def first_change(base: Weights, direction: Weights) -> float | None:
    _validate(base, direction)
    lengths = []
    for name, value in base.items():
        delta = direction[name]
        active = delta != 0
        if not active.any():
            continue
        w, s = value[active], delta[active]
        neighbor = _neighbor(w, s)
        beyond = _across_boundary(w, neighbor, s)
        candidates = (beyond.astype(np.float64) - w.astype(np.float64)) / s.astype(np.float64)
        candidates = candidates[np.isfinite(candidates) & (candidates > 0)]
        if candidates.size:
            lengths.append(float(np.nextafter(candidates.min(), np.inf)))
    return min(lengths) if lengths else None


def previous_change(base: Weights, direction: Weights, trial: Weights, length: float) -> float | None:
    lengths = []
    for name, value in base.items():
        delta, there = direction[name], trial[name]
        active = (delta != 0) & (there != value)
        if not active.any():
            continue
        w, s, v = value[active], delta[active], there[active]
        neighbor = _neighbor(v, -s)
        inside = _across_boundary(v, neighbor, -s)
        candidates = (inside.astype(np.float64) - w.astype(np.float64)) / s.astype(np.float64)
        candidates = candidates[np.isfinite(candidates) & (candidates > 0) & (candidates < length)]
        if candidates.size:
            lengths.append(float(np.nextafter(candidates.max(), 0)))
    return max(lengths) if lengths else None


def _same(one: Weights, two: Weights) -> bool:
    return all(np.array_equal(value, two[name]) for name, value in one.items())


def _identity(weights: Weights) -> str:
    sha = hashlib.sha256()
    for name in sorted(weights):
        sha.update(name.encode())
        sha.update(to_bf16(weights[name]).tobytes())
    return sha.hexdigest()


def search(base: Weights, direction: Weights, *, length: float, slope: float,
           evaluate: Callable[[Weights], Value]) -> Result:
    _validate(base, direction)
    if not math.isfinite(length) or length <= 0 or not math.isfinite(slope) or slope >= 0:
        raise ValueError('native search starts from a finite positive response length and descent slope')
    origin = evaluate(base)
    first = first_change(base, direction)
    if first is None:
        return Result('stalled', None, origin, None, [])
    alpha = length
    seen, trials = set(), []
    while alpha is not None:
        trial = rounded(base, direction, alpha)
        if _same(trial, base):
            alpha = first
            trial = rounded(base, direction, alpha)
        key = _identity(trial)
        if _same(trial, base) or key in seen:
            break
        seen.add(key)
        value = evaluate(trial)
        trials.append(dict(length=alpha, estimate=value.estimate, lower=value.lower, upper=value.upper,
                           native_sha256=key,
                           changed_values=sum(int(np.count_nonzero(trial[name] != base[name])) for name in base)))
        if value.upper < origin.lower:
            return Result('accepted', trial, value, alpha, trials)
        curvature = 2 * (value.estimate - origin.estimate - slope * alpha)
        interpolated = -slope * alpha * alpha / curvature if curvature > 0 else None
        if interpolated is not None and math.isfinite(interpolated) and 0 < interpolated < alpha:
            candidate = rounded(base, direction, interpolated)
            if not _same(candidate, base) and _identity(candidate) not in seen:
                alpha = interpolated
                continue
        alpha = previous_change(base, direction, trial, alpha)
    return Result('stalled', None, origin, None, trials)
