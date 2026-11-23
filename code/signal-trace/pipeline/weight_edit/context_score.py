"""Full-history pairwise loss under the frozen contextual objective.

Each original reference frame keeps its original coefficient, split among its
matched current frames. Averaging references and dropping their dispersion is
not this loss. This is scoring arithmetic, not a native capture producer or a
claim that computational correspondence establishes semantic identity.
"""
import numpy as np

from pipeline.overlap.complete_overlap import validate_pairs
from pipeline.weight_edit.context_targets import features


def history_pairs(current_tokens, original_tokens, pairs=None):
    """Token-identical histories use identity; changed histories need a full map.

    The caller supplies a source-bound map from CURRENT frame to ORIGINAL frame,
    built against original computation, never by minimizing error to the target.
    This helper validates coverage/order, not how a map was obtained.
    """
    current, original = list(current_tokens), list(original_tokens)
    if not current or not original:
        raise ValueError("A complete answer history cannot be empty")
    if current == original:
        identity = np.column_stack([np.arange(len(current))]*2)
        if pairs is not None and not np.array_equal(np.asarray(pairs), identity):
            raise ValueError("Identical token histories require identity correspondence")
        return identity
    if pairs is None:
        raise ValueError("Changed history needs original-computation correspondence")
    result = np.asarray(pairs)
    validate_pairs(result, len(current), len(original))
    return result


def own_answer(record, limit):
    """Require complete recorded evidence, including the answer's ending.

    Matches the evidence produced by whole_reading.keep/ending. A token-perfect
    teacher replay with an unobserved ending is not an own complete answer.
    """
    needed = ("tokens", "steps", "first_step_not_its_own", "steps_not_its_own",
              "ended_with_an_end_token")
    if any(key not in record for key in needed):
        raise ValueError("Capture lacks complete own-history evidence")
    steps = record["steps"]
    if (not isinstance(steps, int) or isinstance(steps, bool) or steps < 1
            or steps != len(record["tokens"])):
        raise ValueError("Captured step count differs from the token history")
    if record["first_step_not_its_own"] is not None or record["steps_not_its_own"] != 0:
        raise ValueError("Given text is not this model's own answer")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < steps:
        raise ValueError("Capture has no valid declared generation limit")
    if record["ended_with_an_end_token"] is not True and steps != limit:
        raise ValueError("Own-history ending was not established")
    return True


def paired_loss(current_unit, current_size, target_unit, target_size,
                loss_weight, status, pairs):
    """One member/segment/merge, using actual loss coefficients, not roots.

    Targets are used exactly as serialized. Only current native directions are
    re-expressed as unit vectors in float64, as the target factory does for its
    originals. Undefined target or current channels produce explicit incomplete
    scores; observed partial sums are diagnostic and cannot masquerade as a
    complete lower loss. No domain or member normalization is added.
    """
    target_u = np.asarray(target_unit, dtype=np.float64)
    target_s = np.asarray(target_size, dtype=np.float64)
    current_u = np.asarray(current_unit, dtype=np.float64)
    current_s = np.asarray(current_size, dtype=np.float64)
    weight, status, pairs = np.asarray(loss_weight, dtype=np.float64), np.asarray(status), np.asarray(pairs)
    if (target_u.ndim != 2 or current_u.ndim != 2 or target_u.shape[1] != current_u.shape[1]
            or target_u.shape[1] < 1 or target_s.shape != (len(target_u),)
            or current_s.shape != (len(current_u),) or weight.shape != (len(target_u), 2)
            or status.shape != weight.shape):
        raise ValueError("Current/reference feature or channel shapes disagree")
    if not np.issubdtype(status.dtype, np.integer) or np.any(status < 0):
        raise ValueError("Status must contain nonnegative integer codes")
    validate_pairs(pairs, len(current_u), len(target_u))
    known = status == 0
    target_finite = np.column_stack([np.isfinite(target_u).all(1), np.isfinite(target_s)])
    target_missing = np.column_stack([np.isnan(target_u).all(1), np.isnan(target_s)])
    if not np.isfinite(weight).all() or np.any(known & (~target_finite | (weight <= 0))):
        raise ValueError("Resolved targets must be finite with positive coefficients")
    if np.any(~known & (~target_missing | (weight != 0))):
        raise ValueError("Unknown targets must be NaN with zero effective coefficients")
    current_u, current_s, current_status = features(current_u, current_s)
    current, original = pairs.T
    counts = np.bincount(original, minlength=len(target_u))
    pair_weight = weight[original] / counts[original, None]
    required = known[original]
    available = current_status[current] == 0
    usable = required & available
    missing = required & ~available

    direction_rows = usable[:, 0]
    delta = current_u[current[direction_rows]] - target_u[original[direction_rows]]
    direction = float(np.sum(pair_weight[direction_rows, 0]*np.sum(delta*delta, axis=1)))
    size_rows = usable[:, 1]
    size_delta = current_s[current[size_rows]] - target_s[original[size_rows]]
    size = float(np.sum(pair_weight[size_rows, 1]*size_delta**2))
    if not np.isfinite(direction) or not np.isfinite(size):
        raise ValueError("Native comparison arithmetic is nonfinite")
    unknown_reference = (~known).sum(0)
    missing_pairs = missing.sum(0)
    missing_mass = np.where(missing, pair_weight, 0).sum(0)
    complete = (unknown_reference == 0) & (missing_pairs == 0)
    return dict(
        direction=direction if complete[0] else None,
        size=size if complete[1] else None,
        observed_direction=direction, observed_size=size,
        complete=bool(complete.all()), complete_channels=complete.tolist(),
        reference_rows=len(target_u), current_rows=len(current_u), matched_pairs=len(pairs),
        known_reference_mass=weight.sum(0).tolist(),
        unknown_reference_rows=unknown_reference.tolist(),
        unresolved_current_pairs=missing_pairs.tolist(),
        unresolved_current_mass=missing_mass.tolist(),
    )
