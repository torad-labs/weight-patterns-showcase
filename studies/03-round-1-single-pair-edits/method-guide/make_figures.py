"""Figures for the internal method guide.

Numbers are read from the run records named beside each figure, not typed in,
except the ordinary-generation time (GPU1 baseline average: 18 generations in
218.90 s, A.md), the full-recorder time (375.31 s, expanded-043-h3 capture
audit) and step 3's rejected objective (A-CONSTRUCTION-02.md). Run:
    uv run --with matplotlib --with numpy python make_figures.py
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
RUN = ROOT / "runs/legs/gemma4-e2b/signal-trace-A-2026-11-07"
C02 = RUN / "trace-construction-02"
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ORIG = "#a3a29c"
BLUE, GREEN, GOLD, RED = "#2a78d6", "#1baf7a", "#eda100", "#d03b3b"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK2, "axes.titlesize": 9,
    "axes.titleweight": "bold", "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "savefig.dpi": 200, "savefig.bbox": "tight",
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.svg")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)


def load(path):
    return json.loads(Path(path).read_text())


def box(ax, x, y, w, h, text, color="#ffffff", edge=INK2, size=7.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=0.8, edgecolor=edge, facecolor=color))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK)


def arrow(ax, a, b, color=INK2, lw=0.9):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=8, color=color, linewidth=lw))


# Figure 1: one decoder layer, its three residual merges, and where A2's edits landed.
fig, ax = plt.subplots(figsize=(6.9, 3.2))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.plot([0.03, 0.97], [0.5, 0.5], color=INK, linewidth=2.2, zorder=1)
ax.text(0.03, 0.455, "residual stream:\n1,536 numbers per token", fontsize=6.8, color=INK2, va="top")
for x, label in [(0.36, "merge 1\nattn_out"), (0.58, "merge 2\npe_in"), (0.82, "merge 3\npe_out")]:
    ax.add_patch(Circle((x, 0.5), 0.022, facecolor="#ffffff", edgecolor=INK, linewidth=1.1, zorder=3))
    ax.text(x, 0.5, "+", ha="center", va="center", fontsize=9, zorder=4)
    ax.text(x, 0.43, label, ha="center", va="top", fontsize=6.8, color=INK2)
box(ax, 0.02, 0.70, 0.15, 0.13, "RMS norm, then\nQ, K, V", "#eef4fc", size=6.9)
box(ax, 0.20, 0.70, 0.13, 0.13, "attention\n(reads cache)", "#eef4fc", size=6.9)
box(ax, 0.36, 0.70, 0.15, 0.13, "output proj.\n+ post-norm", "#eef4fc", size=6.9)
arrow(ax, (0.06, 0.51), (0.08, 0.70)); arrow(ax, (0.17, 0.765), (0.20, 0.765)); arrow(ax, (0.33, 0.765), (0.36, 0.765))
arrow(ax, (0.42, 0.70), (0.365, 0.525))
box(ax, 0.38, 0.08, 0.26, 0.13, "RMS norm, gate and up (GeGLU),\ndown projection, post-norm", "#eefaf4")
arrow(ax, (0.39, 0.49), (0.44, 0.21)); arrow(ax, (0.62, 0.21), (0.585, 0.478))
box(ax, 0.66, 0.70, 0.17, 0.12, "per-layer\nembedding branch", "#fdf6e9")
arrow(ax, (0.62, 0.51), (0.68, 0.70)); arrow(ax, (0.79, 0.70), (0.815, 0.525))
ax.text(0.865, 0.5, "x scale\n= l_out", fontsize=6.8, color=INK2, va="center")
ax.text(0.02, 0.95, "Each of the 35 layers adds three branches to the stream. The trace is read at the three '+' points.",
        fontsize=7.4, color=INK)
ax.text(0.38, 0.015, "A2 step 1 edited layer 11's down projection", fontsize=6.9, color=GREEN, fontweight="bold")
ax.text(0.02, 0.645, "A2 step 2 edited layer 14's K and V", fontsize=6.9, color=BLUE, fontweight="bold")
save(fig, "fig1-layer")

# Figure 2: the turn at one merge.
fig, ax = plt.subplots(figsize=(3.4, 3.0))
ax.set_aspect("equal"); ax.axis("off"); ax.grid(False)
theta = [i / 100 * math.pi for i in range(101)]
ax.plot([math.cos(t) for t in theta], [math.sin(t) for t in theta], color=GRID, linewidth=1)
h = (1.55, 0.55); b = (-0.75, 0.85); m = (h[0] + b[0], h[1] + b[1])
uh = (h[0] / math.hypot(*h), h[1] / math.hypot(*h)); um = (m[0] / math.hypot(*m), m[1] / math.hypot(*m))
arrow(ax, (0, 0), h, ORIG, lw=1.4); ax.text(h[0] + 0.03, h[1] - 0.12, "incoming h", fontsize=7.2, color=INK2)
arrow(ax, h, m, GOLD, lw=1.2); ax.text(h[0] - 0.15, h[1] + 0.52, "branch b", fontsize=7.2, color=GOLD)
arrow(ax, (0, 0), m, INK, lw=1.4); ax.text(m[0] - 0.62, m[1] + 0.07, "merged h' = h + b", fontsize=7.2)
ax.plot(*uh, "o", color=ORIG, ms=4); ax.plot(*um, "o", color=INK, ms=4)
arrow(ax, uh, um, RED, lw=1.6)
ax.text((uh[0] + um[0]) / 2 + 0.07, (uh[1] + um[1]) / 2 - 0.02, "T", fontsize=10, color=RED, fontweight="bold")
ax.text(-1.05, -0.33, "T = h'/|h'| - h/|h|: how far this merge turned\nthe stream's direction. Grey arc: unit circle.",
        fontsize=7.0, color=INK2)
ax.set_xlim(-1.1, 2.0); ax.set_ylim(-0.48, 1.6)
save(fig, "fig2-turn")

# Figure 3: recorder cost per response.
timing = {r["scope"]: r["seconds"] for r in
          map(json.loads, (RUN / "stream-qualification-2026-11-08/timing.jsonl").read_text().splitlines())}
labels = ["Full recorder\n(every operation)", "Stream recorder,\nprojection scope",
          "Stream + full detail\nat 5 positions", "Ordinary generation\n(no recording)"]
values = [375.31, timing["projection"], timing["hybrid"], 218.90 / 18]
fig, ax = plt.subplots(figsize=(5.6, 2.3))
bars = ax.barh(labels[::-1], values[::-1], color=[ORIG, GREEN, GREEN, INK2][::-1])
for bar, v in zip(bars, values[::-1]):
    ax.text(v + 4, bar.get_y() + bar.get_height() / 2, f"{v:.0f} s", va="center", fontsize=7.4)
ax.set_xlabel("seconds per response of about 480 tokens"); ax.set_xlim(0, 420)
ax.grid(axis="y", visible=False)
save(fig, "fig3-recorders")

# Figure 4: A2 construction steps on the fixed trace objective.
steps = [("step 1\n(full size)", C02 / "step-01/absolute-comparison/index.json"),
         ("step 1\n(scaled 0.54)", C02 / "step-01-scaled/absolute-comparison/index.json"),
         ("steps 1+2\n(+ K/V)", C02 / "step-02/absolute-comparison/index.json")]
rows = [(name, 100 * load(p)["fraction_closed"], load(p)["failures_closer"]) for name, p in steps]
original_objective = load(steps[0][1])["original_objective"]
names = [r[0] for r in rows] + ["steps 1+2+3\n(rejected)"]
closed = [r[1] for r in rows] + [100 * (1 - 2.0476949452 / original_objective)]
closer = [r[2] for r in rows] + [6]
colors = [ORIG, GREEN, BLUE, RED]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.5))
a1.bar(names, closed, color=colors); a1.set_title("Share of the trace gap closed (%)")
for i, v in enumerate(closed): a1.text(i, v + 0.15, f"{v:.1f}", ha="center", fontsize=7.2)
a2.bar(names, closer, color=colors); a2.set_title("Failures moved closer (of 14)"); a2.set_ylim(0, 15.5)
for i, v in enumerate(closer): a2.text(i, v + 0.3, str(v), ha="center", fontsize=7.2)
for a in (a1, a2):
    a.grid(axis="x", visible=False); a.tick_params(axis="x", labelsize=6.8)
fig.tight_layout()
save(fig, "fig4-steps")

# Figure 5: natural answers, original vs edited checkpoints.
c1 = load(C02 / "step-01-scaled/exploratory-natural-comparison.json")["summary"]
c2 = load(C02 / "step-02/exploratory-natural-comparison.json")["summary"]
groups = [("original model", ORIG, [c2["original_numbers_correct"], c2["original_verdicts_correct"], c2["original_capped"]]),
          ("step 1 only", GREEN, [c1["edited_numbers_correct"], c1["edited_verdicts_correct"], c1["edited_capped"]]),
          ("steps 1 + 2 (frozen A2)", BLUE, [c2["edited_numbers_correct"], c2["edited_verdicts_correct"], c2["edited_capped"]])]
fig, ax = plt.subplots(figsize=(5.8, 2.5))
width = 0.26
for k, (label, color, vals) in enumerate(groups):
    pos = [x + (k - 1) * width for x in range(3)]
    ax.bar(pos, vals, width, color=color, label=label)
    for p, v in zip(pos, vals): ax.text(p, v + 0.8, str(v), ha="center", fontsize=7)
ax.set_xticks(range(3)); ax.set_xticklabels(["Both energies right", "Verdict right", "Hit the 768-token cap"])
ax.set_ylabel(f"questions (of {c2['questions']})"); ax.legend(frameon=False, fontsize=7.2)
ax.grid(axis="x", visible=False)
save(fig, "fig5-answers")

# Figure 6: what happened to available energy after step 1 (physics questions only).
d = load(C02 / "step-01-scaled/exploratory-natural-comparison.json")
pts = []
for r in d["rows"]:
    e = r["expected"] or {}; got = r["edited"]["available"]
    if got is None or "availableJ" not in e or r["group"] != "physics":
        continue
    pts.append((e["availableJ"], got, e["initialKineticJ"] - e["descentGainJ"] - e["lossJ"]))
right = [p for p in pts if p[1] == p[0]]
sign = [p for p in pts if p[1] != p[0] and abs(p[1] - p[2]) < 1e-6]
other = [p for p in pts if p[1] != p[0] and abs(p[1] - p[2]) >= 1e-6]
lim = max(max(p[0] for p in pts), max(p[1] for p in pts)) + 20
fig, ax = plt.subplots(figsize=(3.7, 3.3))
ax.plot([-40, lim], [-40, lim], color=GRID, linewidth=1)
ax.scatter([p[0] for p in right], [p[1] for p in right], s=14, color=ORIG, label=f"correct ({len(right)})")
ax.scatter([p[0] for p in sign], [p[1] for p in sign], s=16, color=RED, label=f"exactly KE - descent - loss ({len(sign)})")
ax.scatter([p[0] for p in other], [p[1] for p in other], s=14, facecolor="none", edgecolor=INK2, label=f"other wrong value ({len(other)})")
ax.set_xlabel("correct available energy (J)"); ax.set_ylabel("step-1 model's available energy (J)")
ax.set_xlim(-40, lim); ax.set_ylim(-40, lim); ax.legend(frameon=False, fontsize=6.6, loc="upper left")
save(fig, "fig6-descent")
print("ok", sorted(p.name for p in OUT.glob("*.svg")), "points", len(pts), len(right), len(sign), len(other))
