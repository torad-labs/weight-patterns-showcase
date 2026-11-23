"""A conservative common displacement in the model's recorded coordinates.

Each group is summarized independently. An interval gap exists only where
EVERY success is above EVERY failure, or EVERY success is below EVERY failure.
This is descriptive unanimous support, not causal or statistical significance.
It can miss common directions that rotate or vary across overlapping contexts.
"""
import numpy as np


def group_envelope(values):
    x = np.asarray(values, dtype="f8")
    if x.ndim < 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Expected finite group members with at least one coordinate")
    return dict(count=len(x), minimum=x.min(axis=0), maximum=x.max(axis=0), mean=x.mean(axis=0))


def common_difference(failure, success):
    positive = success["minimum"] - failure["maximum"]
    negative = success["maximum"] - failure["minimum"]
    gap = np.maximum(positive, 0) + np.minimum(negative, 0)
    multi = min(failure["count"], success["count"]) > 1
    if not multi: gap = np.zeros_like(gap)
    return dict(gap=gap, raw_difference=success["mean"] - failure["mean"], multi_member=multi)


def omission_cosines(failure_values, success_values, gap):
    """How the common-gap VECTOR changes when one member is omitted.

Unlike class margins, this measures changes to the vector itself. No cosine
cutoff is used. An undefined direction is left undefined, never counted positive.
"""
    f, s = np.asarray(failure_values, dtype="f8"), np.asarray(success_values, dtype="f8")
    fe, se = group_envelope(f), group_envelope(s)
    original_norm = np.linalg.norm(gap, axis=-1)
    results = []
    for values, own, other, is_failure in ((f, fe, se, True), (s, se, fe, False)):
        if len(values) <= 2:
            results.extend([np.full(gap.shape[:-1], np.nan)] * len(values)); continue
        ordered = np.sort(values, axis=0)
        for member in values:
            omitted = dict(count=len(values)-1,
                minimum=np.where(member == ordered[0], ordered[1], ordered[0]),
                maximum=np.where(member == ordered[-1], ordered[-2], ordered[-1]),
                mean=(own["mean"]*len(values)-member)/(len(values)-1))
            difference = common_difference(omitted, other) if is_failure else common_difference(other, omitted)
            g = difference["gap"]; denom = original_norm * np.linalg.norm(g, axis=-1)
            results.append(np.divide(np.sum(g*gap, axis=-1), denom,
                out=np.full_like(denom, np.nan), where=denom>0))
    return np.array(results)
