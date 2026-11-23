"""Computation-only correspondence retaining every native step.

The axis is a coordinate, not a replacement response. Each saved pair is
(native_step, axis_place). Overlap gives every member equal weight at a place;
construction averages a place target over all matches of each native step.

python3 -m pipeline.overlap.complete_overlap align|overlap
"""
import json
import sys

import numpy as np

from pipeline.native_records.run_declaration import check_native_run
from pipeline.paths import ROUND_TWO_RUNS

SOURCE = ROUND_TWO_RUNS / "wide-overlap"
PREVIOUS = ROUND_TWO_RUNS / "trace-overlap"
RUN = ROUND_TWO_RUNS / "trace-overlap-complete-registered"


def distance_path(likeness):
    """Minimum summed cosine distance; diagonal, then vertical, then horizontal ties.

    The totals are filled one anti-diagonal at a time: a cell reads the cell diagonally before it (two anti-diagonals back) and the
    cells above and to its left (one back), never a cell of its own. Each cell is made by the comparisons and the one float64
    addition of the cell-by-cell loop this replaced on November 15, 2026, so every total, every move and the path are that loop's
    bit for bit (tests/test_complete_overlap.py holds it to the loop, ties included). The loop took about 1.4 s for an answer of
    400 steps laid on another, 43.1 s of a capture of thirty answers."""
    likeness = np.asarray(likeness, dtype=np.float64)
    if likeness.ndim != 2 or min(likeness.shape) == 0 or not np.isfinite(likeness).all():
        raise ValueError("Correspondence needs a nonempty finite similarity matrix")
    cost = 1 - np.clip(likeness, -1, 1)
    n, m = cost.shape
    total = np.full((n + 1, m + 1), np.inf)
    total[0, 0] = 0
    move = np.zeros((n, m), dtype=np.int8)
    for k in range(n + m - 1):
        i = np.arange(max(0, k - m + 1), min(n, k + 1))
        j = k - i
        diagonal, vertical, horizontal = total[i, j], total[i, j + 1], total[i + 1, j]
        first = (diagonal <= vertical) & (diagonal <= horizontal)
        second = ~first & (vertical <= horizontal)
        total[i + 1, j + 1] = np.where(first, diagonal, np.where(second, vertical, horizontal)) + cost[i, j]
        move[i, j] = np.where(first, 0, np.where(second, 1, 2))
    pairs = []
    i, j = n - 1, m - 1
    while i >= 0 and j >= 0:
        pairs.append((i, j))
        direction = move[i, j]
        if direction == 0:
            i, j = i - 1, j - 1
        elif direction == 1:
            i -= 1
        else:
            j -= 1
    pairs = np.asarray(pairs[::-1], dtype=np.int32)
    validate_pairs(pairs, n, m)
    return pairs


def validate_pairs(pairs, steps, places):
    pairs = np.asarray(pairs)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or not np.issubdtype(pairs.dtype, np.integer):
        raise ValueError("Pairs must be integer (native_step, axis_place) rows")
    if (len(pairs) == 0 or not np.array_equal(pairs[0], [0, 0])
            or not np.array_equal(pairs[-1], [steps - 1, places - 1])):
        raise ValueError("Correspondence does not cover both complete traces")
    increments = np.diff(pairs, axis=0)
    if np.any((increments < 0) | (increments > 1)) or np.any(increments.sum(1) == 0):
        raise ValueError("Correspondence is not a monotonic unit-step path")
    if (not np.array_equal(np.unique(pairs[:, 0]), np.arange(steps))
            or not np.array_equal(np.unique(pairs[:, 1]), np.arange(places))):
        raise ValueError("Correspondence lost a native step or axis place")


def member_means(values, pairs, places):
    """One member's mean at each place, retaining all its matched native steps."""
    values, pairs = np.asarray(values), np.asarray(pairs)
    validate_pairs(pairs, len(values), places)
    result = np.zeros((places,) + values.shape[1:], dtype=np.float64)
    np.add.at(result, pairs[:, 1], values[pairs[:, 0]])
    counts = np.bincount(pairs[:, 1], minlength=places)
    return result / counts.reshape((-1,) + (1,) * (values.ndim - 1))


def native_targets(place_values, pairs, steps):
    """One target per native step; repeated axis matches do not duplicate fit rows."""
    place_values, pairs = np.asarray(place_values), np.asarray(pairs)
    validate_pairs(pairs, steps, len(place_values))
    result = np.zeros((steps,) + place_values.shape[1:], dtype=np.float64)
    np.add.at(result, pairs[:, 0], place_values[pairs[:, 1]])
    counts = np.bincount(pairs[:, 0], minlength=steps)
    return result / counts.reshape((-1,) + (1,) * (place_values.ndim - 1))


def unit_steps(turns):
    flat = np.asarray(turns, dtype=np.float32).reshape(len(turns), -1)
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms == 0):
        raise ValueError("Undefined computation direction")
    return flat / norms


def source_trace(member):
    return np.load(SOURCE / "exported" / member["id"] / "turns.npy", mmap_mode="r")


def align():
    if (RUN / "pairs").exists() or (RUN / "alignment.json").exists():
        raise ValueError("Inspect existing correspondence; never overwrite it")
    old = json.loads((PREVIOUS / "members.json").read_text())
    members = old["members"]
    if len({m["id"] for m in members}) != len(members):
        raise ValueError("Duplicate member")
    axis = source_trace(next(m for m in members if m["id"] == old["axis"]))
    grid = unit_steps(axis)
    old_aligned = np.load(PREVIOUS / "aligned.npy")
    (RUN / "pairs").mkdir()
    rows = []
    for i, member in enumerate(members):
        turns = source_trace(member)
        likeness = unit_steps(turns) @ grid.T
        pairs = distance_path(likeness)
        np.save(RUN / "pairs" / f"{member['id']}.npy", pairs)
        # A native step has weight one in this diagnostic, regardless of match count.
        counts = np.bincount(pairs[:, 0], minlength=len(turns))
        similarities = np.bincount(pairs[:, 0], weights=likeness[pairs[:, 0], pairs[:, 1]], minlength=len(turns)) / counts
        rows.append(dict(**member, steps=len(turns), pairs=len(pairs),
                         represented_steps=len(np.unique(pairs[:, 0])),
                         previous_represented_steps=len(np.unique(old_aligned[i])),
                         mean_native_step_likeness=float(similarities.mean()),
                         maximum_axis_places_per_step=int(counts.max())))
        print(json.dumps(rows[-1]), flush=True)
    (RUN / "members.json").write_text(json.dumps(dict(
        axis=old["axis"], axis_steps=len(axis), members=members,
        pair_columns=["native_step", "axis_place"], pair_files="pairs/<id>.npy"
    ), indent=2) + "\n")
    (RUN / "alignment.json").write_text(json.dumps(dict(
        axis=old["axis"], axis_steps=len(axis), members=rows,
        pair_columns=["native_step", "axis_place"],
        objective="Minimum sum(1 - cosine of turns at all 105 merges); no answer text or group label enters correspondence",
        overlap_weighting="Mean of each member's matched native steps at a place, then mean over members",
        construction_weighting="Mean of all matched place targets for each native step; one fit row per native step",
        limits="Global DTW represents all steps but does not establish causal correspondence or that a shortened response executed missing computations"
    ), indent=2) + "\n")


def overlap():
    if (RUN / "correct-pattern.npy").exists() or (RUN / "recurrence.npz").exists():
        raise ValueError("Inspect existing overlap; never overwrite it")
    record = json.loads((RUN / "alignment.json").read_text())
    members, places = record["members"], record["axis_steps"]
    traces = [source_trace(m) for m in members]
    correct = np.array([m["correct"] for m in members])
    if min(correct.sum(), (~correct).sum()) < 2:
        raise ValueError("Group spread requires multiple members")
    matches = []
    for member, trace in zip(members, traces):
        pairs = np.load(RUN / "pairs" / f"{member['id']}.npy")
        validate_pairs(pairs, len(trace), places)
        matches.append([pairs[pairs[:, 1] == p, 0] for p in range(places)])
    merges, width = traces[0].shape[1:]
    patterns = {name: np.lib.format.open_memmap(RUN / f"{name}-pattern.npy", mode="w+", dtype="f4", shape=(places, merges, width))
                for name in ("correct", "wrong")}
    gap = np.zeros((places, merges))
    chance, halves = np.zeros_like(gap), np.zeros_like(gap)
    within = np.zeros((2, places, merges))
    even = np.zeros(len(members), dtype=bool)
    for selected in (correct, ~correct):
        even[np.flatnonzero(selected)[::2]] = True
    for p in range(places):
        x = np.stack([np.asarray(trace[matches[i][p]], dtype="f8").mean(0)
                      for i, trace in enumerate(traces)])
        a, b = x[correct], x[~correct]
        ca, cb = a.mean(0), b.mean(0)
        patterns["correct"][p], patterns["wrong"][p] = ca, cb
        d = ca - cb
        gap[p] = np.sum(d * d, axis=-1)
        chance[p] = sum(np.var(g, axis=0, ddof=1).sum(-1) / len(g) for g in (a, b))
        d1 = x[correct & even].mean(0) - x[~correct & even].mean(0)
        d2 = x[correct & ~even].mean(0) - x[~correct & ~even].mean(0)
        denominator = np.linalg.norm(d1, axis=-1) * np.linalg.norm(d2, axis=-1)
        np.divide((d1 * d2).sum(-1), denominator, out=halves[p], where=denominator > 0)
        for k, g in enumerate((a, b)):
            denominator = (g * g).sum(-1).mean(0)
            np.divide((g.mean(0) ** 2).sum(-1), denominator, out=within[k, p], where=denominator > 0)
    for pattern in patterns.values():
        pattern.flush()
    share = np.zeros_like(gap)
    np.divide(gap - chance, gap, out=share, where=gap > 0)
    share = np.clip(share, 0, 1)
    np.savez(RUN / "recurrence.npz", gap=gap, chance=chance, halves=halves, within=within, shared_part=share)
    print(json.dumps(dict(members=len(members), places=places, merges=merges,
                         native_steps=sum(m["steps"] for m in members),
                         represented_native_steps=sum(m["represented_steps"] for m in members),
                         previous_represented_native_steps=sum(m["previous_represented_steps"] for m in members),
                         gap_squared=float(gap.sum()), sampling_variation_estimate=float(chance.sum()),
                         gap_to_variation_ratio=float(gap.sum() / chance.sum()),
                         median_split_half_cosine=float(np.median(halves)))), flush=True)


if __name__ == "__main__":
    check_native_run(RUN, RUN / "declaration.json")
    dict(align=align, overlap=overlap)[sys.argv[1]]()
