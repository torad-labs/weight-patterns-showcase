"""Diagnostic query changes in the recorded query/key coordinate system.

These functions never touch a checkpoint or an inference execution. They expose
the local input-dependent transformation to inspect before deriving any edit.
"""
import numpy as np


def query(raw, gain, epsilon, rotation):
    normalized = raw * gain / np.sqrt(np.mean(raw * raw) + epsilon)
    cosine, sine = rotation
    half = len(raw) // 2
    return np.concatenate([cosine * normalized[:half] - sine * normalized[half:],
                           sine * normalized[:half] + cosine * normalized[half:]])


def source_ratio(q, keys, descent, climb, scale):
    if not np.any(descent) or not np.any(climb) or np.any(descent & climb):
        raise ValueError('Source groups must be present and disjoint')
    logits = keys @ q * scale
    def group(mask):
        z = logits[mask]
        exponent = np.exp(z - z.max())
        return float(z.max() + np.log(exponent.sum())), exponent @ keys[mask] / exponent.sum()
    a, ka = group(descent)
    b, kb = group(climb)
    return a - b, scale * (ka - kb)


def solve_query(raw, gain, epsilon, rotation, keys, descent, climb, scale, target):
    original = np.asarray(raw, dtype='f8')
    changed = original.copy()
    length = np.linalg.norm(original)
    if not np.isfinite(length) or length == 0:
        raise ValueError('Query has no finite nonzero direction')
    cosine, sine = rotation
    half = len(raw) // 2
    start, _ = source_ratio(query(changed, gain, epsilon, rotation), keys, descent, climb, scale)
    for iteration in range(64):
        actual, output_gradient = source_ratio(query(changed, gain, epsilon, rotation), keys, descent, climb, scale)
        residual = target - actual
        if abs(residual) < 1e-9:
            return changed, dict(iterations=iteration, original_ratio=start, target_ratio=target,
                                 achieved_ratio=actual, relative_raw_change=float(np.linalg.norm(changed - original) / length))
        rotated_gradient = np.concatenate([cosine * output_gradient[:half] + sine * output_gradient[half:],
                                           -sine * output_gradient[:half] + cosine * output_gradient[half:]])
        rms = np.sqrt(np.mean(changed * changed) + epsilon)
        weighted = rotated_gradient * gain
        gradient = (weighted - changed * np.mean(weighted * changed) / (rms * rms)) / rms
        gradient -= changed * np.dot(gradient, changed) / (length * length)
        square = np.dot(gradient, gradient)
        if square < 1e-24:
            raise ValueError('Requested source ratio is locally unreachable at this query length')
        step = residual * gradient / square
        for power in range(24):
            candidate = changed + step * (2. ** -power)
            candidate *= length / np.linalg.norm(candidate)
            score, _ = source_ratio(query(candidate, gain, epsilon, rotation), keys, descent, climb, scale)
            if abs(score - target) < abs(residual):
                changed = candidate
                break
        else:
            raise ValueError('Requested source ratio is locally unreachable: no improving direction')
    raise ValueError('Requested source ratio is locally unreachable within numerical convergence')
