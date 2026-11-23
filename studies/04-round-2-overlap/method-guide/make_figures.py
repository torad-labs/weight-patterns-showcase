"""Figures and checked numbers for the round-two report on the path edit.

Every number is read from the run records under ../runs, not typed in. Besides the figures, the script
writes numbers.json: the values the guide's text quotes, so the text can be checked against the records.
Run:
    uv run --with matplotlib --with numpy python make_figures.py
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

HERE = Path(__file__).parent
RUNS = HERE.parent / "runs"
A2 = RUNS / "a2-path"
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ORIG = "#a3a29c"
BLUE, GREEN, GOLD, RED = "#2a78d6", "#1baf7a", "#eda100", "#d03b3b"
REGION = (142, 313)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK2, "axes.titlesize": 9,
    "axes.titleweight": "bold", "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "savefig.dpi": 200, "savefig.bbox": "tight",
})
numbers = {}


def save(fig, name):
    fig.savefig(OUT / f"{name}.svg")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)


def places():
    """Where the two patterns part: the gap of the states against chance, place by place (state-overlap)."""
    z = np.load(RUNS / "state-overlap" / "recurrence.npz")
    gap, chance = z["gap"], z["chance"]
    ratio = gap.sum(1) / chance.sum(1)
    diverged = gap > 2 * chance
    lo, hi = REGION
    inside = slice(lo, hi + 1)
    outside = np.r_[0:lo, hi + 1:gap.shape[0]]
    per_matrix = gap[inside].sum(0) / chance[inside].sum(0)
    numbers["places"] = dict(
        places=int(gap.shape[0]), matrices=int(gap.shape[1]),
        gap_over_chance_inside=float(gap[inside].sum() / chance[inside].sum()),
        gap_over_chance_outside=float(gap[outside].sum() / chance[outside].sum()),
        per_matrix_inside_least=float(per_matrix.min()), per_matrix_inside_most=float(per_matrix.max()),
        per_matrix_inside_median=float(np.median(per_matrix)),
        halves_alike_inside_median=float(np.median(z["halves"][inside])),
        cells_diverged_inside=float(diverged[inside].mean()), cells_diverged_outside=float(diverged[outside].mean()),
        cells_diverged_all=float(diverged.mean()))
    x = np.arange(gap.shape[0])
    fig, (top, low) = plt.subplots(2, 1, figsize=(6.8, 3.0), sharex=True, gridspec_kw=dict(height_ratios=[2.0, 1], hspace=0.14))
    for ax in (top, low):
        ax.axvspan(lo, hi, color=BLUE, alpha=0.08, lw=0)
    top.plot(x, ratio, color=BLUE, lw=1.1)
    top.axhline(2, color=INK2, lw=0.8, ls="--", label="twice chance")
    top.legend(frameon=False, fontsize=7.5, loc="upper right", bbox_to_anchor=(0.985, 0.98))
    top.set_title(f"places {lo} to {hi}", color=BLUE, fontsize=8, fontweight="normal",
                  x=(lo + hi) / 2 / (gap.shape[0] - 1), ha="center", pad=3)
    top.set_ylabel("gap over chance")
    top.set_ylim(0, ratio.max() * 1.05)
    low.fill_between(x, diverged.mean(1), color=GREEN, alpha=0.55, lw=0)
    low.set_ylabel("share of matrices\ndiverged")
    low.set_ylim(0, 1.02)
    low.set_xlabel("place")
    low.set_xlim(0, gap.shape[0] - 1)
    save(fig, "fig1-places")


def build_log():
    return [json.loads(line) for line in (A2 / "build.jsonl").read_text().splitlines()]


def ridge(log):
    """The search for one matrix's ridge: every candidate tried for the first matrix on the path."""
    first = log[0]
    tried = sorted(first["candidates"], key=lambda c: c["order"])
    k = np.array([-np.log2(c["ridge_over_strongest_direction"]) for c in tried])
    left = np.array([c["other_half_left_over_asked"] for c in tried])
    size = np.array([100 * c["change_over_matrix"] for c in tried])
    walk = np.isclose(k, np.round(k))
    best = int(np.argmin(left))
    numbers["ridge_search_first_matrix"] = dict(
        tensor=first["tensor"], candidates=len(tried), walked=int(walk.sum()), narrowed=int((~walk).sum()),
        chosen_ridge_over_strongest=tried[best]["ridge_over_strongest_direction"], chosen_halvings=float(k[best]),
        left_on_other_half=float(left[best]), change_percent=float(size[best]), halves_agree=tried[best]["halves_agree"],
        left_at_start=float(left[0]), left_one_step_weaker_than_best_walk=float(left[walk][-1]))
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.7), gridspec_kw=dict(wspace=0.38))
    for ax, values in ((a, left), (b, size)):
        order = np.argsort(k)
        ax.plot(k[order], values[order], color=GRID, lw=1.2, zorder=1)
        ax.scatter(k[walk], values[walk], s=16, color=BLUE, zorder=2, label="halving")
        ax.scatter(k[~walk], values[~walk], s=22, color=GOLD, zorder=3, label="golden-section search")
        ax.scatter(k[best], values[best], s=70, facecolor="none", edgecolor=GREEN, lw=1.6, zorder=4, label="chosen")
        ax.set_xlabel("ridge, in halvings from its starting value", fontsize=7.6)
    a.set_ylabel("error left on the held-out half\n(1 = no edit)")
    b.set_ylabel("size of the change (% of the matrix norm)")
    b.legend(frameon=False, fontsize=7.2, loc="upper left")
    save(fig, "fig3-ridge")


def build(log):
    """What each matrix on the path was asked and what it reached."""
    names = [f"{r['layer']}{'a' if r['kind'] == 'attention' else 'f'}" for r in log]
    moving = [r["readings"]["wrong answers, where the patterns diverge"] for r in log]
    rest = [r["readings"]["wrong answers, the rest of their path"] for r in log]
    other = [r["readings"]["correct answers"] for r in log]
    reached = np.array([m["share_of_ask_reached"] for m in moving])
    change = np.array([100 * r["change_over_matrix"] for r in log])
    move = np.array([r["where_a_move_is_asked"]["move_rms"] for r in log])
    drift = np.array([m["asked_rms"] for m in rest])
    drift_correct = np.array([m["asked_rms"] for m in other])
    numbers["build"] = dict(
        matrices=len(log), written=int(sum(r["written"] for r in log)),
        change_percent_least=float(change.min()), change_percent_most=float(change.max()),
        reached_least=float(reached.min()), reached_most=float(reached.max()), reached_median=float(np.median(reached)),
        moving_rows_least=int(min(m["rows"] for m in moving)), moving_rows_most=int(max(m["rows"] for m in moving)),
        fitted_rows=int(moving[0]["rows"] + rest[0]["rows"]), correct_rows=int(other[0]["rows"]),
        candidates_least=int(min(len(r["candidates"]) for r in log)), candidates_most=int(max(len(r["candidates"]) for r in log)),
        steps_least=int(min(r["steps"] for r in log)), steps_most=int(max(r["steps"] for r in log)),
        shortened_most=int(max(r["steps_shortened"] for r in log)), restarts_most=int(max(r["restarts"] for r in log)),
        ridge_over_strongest_least=float(min(r["ridge_over_strongest_direction"] for r in log)),
        ridge_over_strongest_most=float(max(r["ridge_over_strongest_direction"] for r in log)),
        other_half_left_least=float(min(r["other_half_left_over_asked"] for r in log)),
        other_half_left_most=float(max(r["other_half_left_over_asked"] for r in log)),
        halves_agree_least=float(min(r["halves_agree"] for r in log)), halves_agree_most=float(max(r["halves_agree"] for r in log)),
        minutes=float(sum(r["seconds"] for r in log) / 60),
        move_rms_first=float(move[0]), move_rms_last=float(move[-1]),
        drift_of_rows_asked_to_stay_before_last=float(drift[-1]), drift_of_correct_rows_before_last=float(drift_correct[-1]),
        stay_rows_moved_over_moving_rows_moved_median=float(np.median([r_["moved_rms"] / m["moved_rms"] for r_, m in zip(rest, moving)])))
    x = np.arange(len(log))
    colors = [BLUE if r["kind"] == "attention" else GREEN for r in log]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw=dict(wspace=0.34))
    a.bar(x, reached, color=colors, width=0.72)
    a.set_ylabel("share of the requested move\nreached, moving rows")
    a.set_ylim(0, 1)
    a.legend(handles=[Patch(color=BLUE, label="attention output"), Patch(color=GREEN, label="feed-forward down")],
             frameon=False, fontsize=7.2, loc="upper right")
    b.plot(x, move, color=BLUE, marker="o", ms=3, lw=1.1, label="requested move")
    b.plot(x, drift_correct, color=ORIG, marker="o", ms=3, lw=1.1, label="correct answers' rows: distance from original")
    b.plot(x, drift, color=RED, marker="o", ms=3, lw=1.1, label="staying rows: distance from original")
    b.set_ylabel("distance in the state (rms)")
    b.legend(frameon=False, fontsize=6.6, loc="upper left")
    b.set_ylim(0, max(move.max(), drift.max(), drift_correct.max()) * 1.45)
    for ax in (a, b):
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=6.4)
        ax.set_xlabel("matrix, in path order")
    save(fig, "fig4-build")


def score_of(run):
    rows = {}
    for line in (RUNS / run / "score.out").read_text().splitlines():
        m = re.match(r"(overlap|round one)\s+(wrong|correct)\s+answers\s+(\d+) \| complete before\s+(\d+) after\s+(\d+) \| available right before\s+(\d+) after\s+(\d+)", line)
        if m:
            rows[f"{m[1]}, {m[2]}"] = dict(answers=int(m[3]), complete_before=int(m[4]), complete_after=int(m[5]),
                                           available_before=int(m[6]), available_after=int(m[7]))
    return rows


def answers(result):
    score = score_of("a2-path")
    groups = result["answers_on_its_own"]["groups"]
    numbers["a2_answers"] = dict(score=score, groups={name: {k: v for k, v in g.items() if not isinstance(v, dict)} for name, g in groups.items()},
                                 first_errors={name: g["first_error_of_those_not_complete"] for name, g in groups.items()})
    labels = ["overlap, wrong\n(fitted)", "overlap, correct\n(not fitted)", "round one, wrong\n(outside the fit)", "round one, correct\n(outside the fit)"]
    keys = ["overlap, wrong", "overlap, correct", "round one, wrong", "round one, correct"]
    before = [score[k]["complete_before"] for k in keys]
    after = [score[k]["complete_after"] for k in keys]
    total = [score[k]["answers"] for k in keys]
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(6.4, 2.7))
    ax.bar(x - 0.19, before, 0.36, color=ORIG, label="original model")
    ax.bar(x + 0.19, after, 0.36, color=BLUE, label="edited model")
    for i in range(4):
        ax.text(x[i] - 0.19, before[i] + 0.5, f"{before[i]} of {total[i]}", ha="center", fontsize=7.5, color=INK2)
        ax.text(x[i] + 0.19, after[i] + 0.5, f"{after[i]} of {total[i]}", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.4)
    ax.set_ylabel("complete answers")
    ax.set_ylim(0, 35)
    ax.legend(frameon=False, fontsize=7.4, loc="upper right")
    save(fig, "fig5-answers")


def traces(result):
    rows = [a for a in result["failures_only"]["answers"] if a["set"] == "overlap"]
    sections = result["own_traces"]["sections"]
    numbers["a2_traces"] = dict(states=sections["states"]["places 142-313"], turns=sections["turns"]["places 142-313"],
                                states_before_142=sections["states"]["before 142"],
                                each_answer=result["own_traces"]["each_answer_places_142_313_states"],
                                trace_and_calculation={k: v for k, v in result["own_traces"]["trace_and_calculation"].items() if k not in ("what", "limit")},
                                size=result["own_traces"]["size_of_the_state_edited_over_original"])
    fig, ax = plt.subplots(figsize=(6.6, 2.5))
    rng = np.random.default_rng(0)
    lanes = [("wrong", "original", "were wrong, original model"), ("wrong", "edited", "were wrong, edited model"),
             ("correct", "original", "were correct, original model"), ("correct", "edited", "were correct, edited model")]
    for lane, (group, when, label) in enumerate(lanes):
        members = [a for a in rows if a["group"] == group]
        values = np.array([a["own_trace_between_the_patterns_places_142_313"]["states"][when] for a in members])
        y = len(lanes) - 1 - lane + rng.uniform(-0.16, 0.16, len(values))
        if when == "original":
            ax.scatter(values, y, s=14, color=ORIG, zorder=2)
        else:
            done = np.array([a["edited"]["complete"] for a in members])
            ax.scatter(values[done], y[done], s=16, color=BLUE, zorder=3)
            ax.scatter(values[~done], y[~done], s=16, facecolor="white", edgecolor=BLUE, lw=0.9, zorder=3)
        ax.plot([np.median(values)] * 2, [len(lanes) - 1 - lane - 0.3, len(lanes) - 1 - lane + 0.3], color=INK, lw=1.2, zorder=4)
    ax.axvline(0, color=RED, lw=0.9, ls="--")
    ax.axvline(1, color=GREEN, lw=0.9, ls="--")
    ax.text(0, 3.62, "wrong pattern", color=RED, fontsize=7.4, ha="center")
    ax.text(1, 3.62, "correct pattern", color=GREEN, fontsize=7.4, ha="center")
    ax.set_yticks(range(len(lanes)))
    ax.set_yticklabels([label for *_, label in lanes][::-1], fontsize=7.6)
    ax.set_ylim(-0.5, 3.9)
    ax.set_xlabel("position between the two patterns, places 142 to 313")
    ax.grid(axis="y", visible=False)
    save(fig, "fig6-traces")


def ladder():
    """The rungs side by side, from each run's own score and result."""
    out = {}
    for run in ("a2-path", "b2-path", "c2-path", "c2-protected", "d2-path"):
        result = json.loads((RUNS / run / "result.json").read_text())
        score = score_of(run)
        states = result["own_traces"]["sections"]["states"]["places 142-313"]
        out[run] = dict(wrong_complete=score["overlap, wrong"]["complete_after"], correct_complete=score["overlap, correct"]["complete_after"],
                        wrong_available_right=score["overlap, wrong"]["available_after"], correct_available_right=score["overlap, correct"]["available_after"],
                        round_one_wrong_complete=score["round one, wrong"]["complete_after"], round_one_correct_complete=score["round one, correct"]["complete_after"],
                        towards=states["wrong_towards_the_correct_pattern"], sideways=states["wrong_sideways_in_gaps"],
                        correct_travelled=states["correct_travelled_in_gaps"], groups_apart=states["gap_between_the_groups_after_over_before"])
    steps = json.loads((RUNS / "a2-path-steps" / "result.json").read_text())
    out["a2-path-steps"] = {size: {group: dict(complete=row["complete"], subtract_the_descent=row["subtract_the_descent"])
                                       for group, row in curve.items() if group.startswith("overlap")}
                            for size, curve in steps["curves"].items() if isinstance(curve, dict)}
    numbers["ladder"] = out


def cohort():
    index = json.loads((RUNS / "input-overlap" / "inputs" / "index.json").read_text())
    members = json.loads((RUNS / "state-overlap" / "members.json").read_text())["members"]
    wrong = [a for m, a in zip(members, index["answers"]) if not m["correct"]]
    right = [a for m, a in zip(members, index["answers"]) if m["correct"]]
    numbers["cohort"] = dict(
        answers=len(members), wrong=len(wrong), correct=len(right),
        steps_all=sum(a["steps"] for a in index["answers"]), question_positions_all=sum(a["question_positions"] for a in index["answers"]),
        wrong_steps=sum(a["steps"] for a in wrong), wrong_question_positions=sum(a["question_positions"] for a in wrong),
        correct_steps=sum(a["steps"] for a in right), correct_question_positions=sum(a["question_positions"] for a in right),
        attention_inputs=index["width"]["attention-0"], feedforward_inputs=index["width"]["feedforward-0"], layers=len(index["layers"]))


if __name__ == "__main__":
    result = json.loads((A2 / "result.json").read_text())
    log = build_log()
    cohort()
    places()
    ridge(log)
    build(log)
    answers(result)
    traces(result)
    ladder()
    (HERE / "numbers.json").write_text(json.dumps(numbers, indent=1) + "\n")
    print(json.dumps(numbers, indent=1))
