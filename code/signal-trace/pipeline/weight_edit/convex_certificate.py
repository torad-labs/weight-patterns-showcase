"""Exact membership witnesses for finite represented floating-point arrays.

An optimizer can propose coefficients, but only exact rational verification
certifies a zero here. Failure to certify is UNKNOWN membership, not proof that
the point is outside the hull. This proves no scientific meaning for the points.
"""
import hashlib
from fractions import Fraction
from typing import TypedDict

import numpy as np


class Certificate(TypedDict):
    method: str
    coordinates: int
    weights: list[list[str]]
    point_sha256: str
    vertices_sha256: str


def _digest(array: np.ndarray) -> str:
    canonical = np.asarray(array, dtype='<f8', order='C')
    payload = str(canonical.shape).encode('ascii') + b':' + canonical.tobytes()
    return hashlib.sha256(payload).hexdigest()


def certify(point, vertices, proposed_weights) -> Certificate | None:
    """Verify one nonnegative joint mixture on EVERY represented coordinate.

    Binary32 and binary64 values are exact rational numbers here, not decimals
    rounded to a tolerance. Proposed nonnegative weights are normalized exactly;
    the resulting point is then checked, never presumed unchanged. The witness
    and hashes use canonical binary64 arrays, an exact widening of binary32.
    """
    point, vertices, weights = (np.asarray(value) for value in (point, vertices, proposed_weights))
    if (point.ndim != 1 or vertices.ndim != 2 or weights.ndim != 1
            or vertices.shape != (len(weights), len(point)) or not len(point) or not len(weights)):
        raise ValueError('membership needs a point, witness x coordinate vertices, and one weight per witness')
    if any(value.dtype.kind != 'f' or value.dtype.itemsize > 8 for value in (point, vertices, weights)):
        raise ValueError('certification requires represented binary floating-point arrays up to binary64')
    if not all(bool(np.isfinite(value).all()) for value in (point, vertices, weights)) or bool((weights < 0).any()):
        return None
    exact = [Fraction(float(value)) for value in weights]
    total = sum(exact, Fraction())
    if total == 0:
        return None
    exact = [value / total for value in exact]
    active = [(row, value) for row, value in enumerate(exact) if value]
    for coordinate, target in enumerate(point):
        actual = sum((weight * Fraction(float(vertices[row, coordinate])) for row, weight in active), Fraction())
        if actual != Fraction(float(target)):
            return None
    return Certificate(method='exact-binary-rational-convex-combination/1',
                       coordinates=len(point),
                       weights=[[str(value.numerator), str(value.denominator)] for value in exact],
                       point_sha256=_digest(point), vertices_sha256=_digest(vertices))
