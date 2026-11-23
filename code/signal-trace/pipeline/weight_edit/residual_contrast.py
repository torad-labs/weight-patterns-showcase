"""Signed, full-field diagnostics; these arrays are not fitting targets."""
from typing import Any

import numpy as np

from pipeline.overlap.complete_overlap import member_means, native_targets, validate_pairs
from pipeline.weight_edit.context_targets import features


def held_sets(members, folds):
    """Each original fitting member is held once; both outcomes share donors."""
    ids = [m['id'] for m in members]
    flat = [key for fold in folds for key in fold]
    if len(set(ids)) != len(ids) or sorted(flat) != sorted(ids):
        raise ValueError('Folds must partition the fitting source members exactly')
    result = []
    for fold in folds:
        recipients = [i for i, key in enumerate(ids) if key in fold]
        donors = [i for i, m in enumerate(members) if m['correct'] and m['id'] not in fold]
        if not recipients or not donors or {bool(members[i]['correct']) for i in recipients} != {False, True}:
            raise ValueError('Each fold needs both recipient groups and outside-fold successes')
        result.append((recipients, donors))
    return result


def axis_mass(pairs, steps, places, total_rows):
    """Push native row mass onto the axis, splitting a row's repeated matches."""
    pairs = np.asarray(pairs)
    validate_pairs(pairs, steps, places)
    if total_rows < steps or total_rows <= 0:
        raise ValueError('Native denominator must include every segment row')
    counts = np.bincount(pairs[:, 0], minlength=steps)
    return np.bincount(pairs[:, 1], weights=1/(total_rows*counts[pairs[:, 0]]), minlength=places)


def complete_sum(values, weights):
    """Keep partial observed sums, but never call missing mass a complete score."""
    values, weights = np.asarray(values), np.asarray(weights)
    if values.shape != weights.shape or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError('A finite nonnegative weight is required for each value')
    known = np.isfinite(values)
    observed = float(np.sum(values[known]*weights[known]))
    return dict(observed=observed, complete=observed if known.all() else None,
                unknown=int((~known).sum()), unknown_mass=float(weights[~known].sum()))


def field_statistics(values, correct):
    """Cellwise signed mean gap and ordered distinct-member cross products."""
    values, correct = np.asarray(values, dtype=float), np.asarray(correct, dtype=bool)
    if values.ndim != 3 or correct.shape != (len(values),):
        raise ValueError('Expected members by places by coordinates and one label per member')
    wrong, right = values[~correct], values[correct]
    if min(len(wrong), len(right)) < 2:
        raise ValueError('Distinct-member recurrence needs two members per group')
    mw, mc = wrong.mean(0), right.mean(0)
    gap = mw - mc
    cross = np.sum(gap*gap, axis=-1)
    for group, mean in ((wrong, mw), (right, mc)):
        cross -= np.sum((group-mean)**2, axis=(0, 2))/(len(group)*(len(group)-1))
    member_dot = np.empty(values.shape[:2])
    for i, row in enumerate(values):
        if correct[i]:
            own = mw-row
            other = mw-(right.sum(0)-row)/(len(right)-1)
        else:
            own = row-mc
            other = (wrong.sum(0)-row)/(len(wrong)-1)-mc
        member_dot[i] = np.sum(own*other, axis=-1)
    return dict(gap=gap, cross_member=cross, member_dot=member_dot,
                gap_squared=np.sum(gap*gap, axis=-1))


def residual_fields(unit, log_size, predicted_unit, predicted_size, pairs, places):
    """Native residuals, separately retained round trip, and their signed axes."""
    unit, log_size, _ = features(unit, log_size)
    steps = len(unit)
    own_unit = member_means(unit, pairs, places)
    own_size = member_means(log_size, pairs, places)
    self_u, self_s, _ = features(native_targets(own_unit, pairs, steps), native_targets(own_size, pairs, steps))
    pred_u, pred_s, _ = features(native_targets(predicted_unit, pairs, steps), native_targets(predicted_size, pairs, steps))
    native = dict(raw_direction=unit-pred_u, raw_size=log_size-pred_s,
                  registration_direction=unit-self_u, registration_size=log_size-self_s)
    result: dict[str, Any] = {key: member_means(value, pairs, places) for key, value in native.items()}
    for channel in ('direction', 'size'):
        result['adjusted_'+channel] = result['raw_'+channel]-result['registration_'+channel]
    result['native'] = {}
    for key, value in native.items():
        squared = np.sum(value*value, axis=-1) if value.ndim == 2 else value*value
        result['native'][key] = complete_sum(squared, np.ones(steps))
    return result
