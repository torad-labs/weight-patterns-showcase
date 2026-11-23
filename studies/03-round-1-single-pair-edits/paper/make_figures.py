"""Figures for the A-D paper draft.

Every number here is copied from the study reports cited beside it; nothing is
recomputed from captures. Run:  uv run --with matplotlib python make_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
ORIG = "#a3a29c"  # neutral: the unedited model is a reference, not a series hue
EDIT = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "D": "#eda100"}
GOOD, CRIT = "#0ca30c", "#d03b3b"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK2,
        "axes.titlesize": 9,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "legend.frameon": False,
        "svg.fonttype": "none",
    }
)


def save(fig, name):
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def bars(ax, xs, ys, color, width, labels=True, fmt="{:g}", dy=0.0):
    b = ax.bar(xs, ys, width=width, color=color, edgecolor="white", linewidth=1.0, zorder=3)
    if labels:
        for x, y in zip(xs, ys):
            va = "bottom" if y >= 0 else "top"
            ax.text(x, y + (dy if y >= 0 else -dy), fmt.format(y), ha="center", va=va, fontsize=7, color=INK)
    return b


# Figure 2. FF14/FF15 relative-separation ratios (Q4 follow-up, PHYSICS-TRACE-FOLLOWUP.md).
pairs = [
    ("cart", "physics", 94, 0.559, 1.506),
    ("sled", "physics", 104, 0.567, 1.515),
    ("bead", "physics", 4, 0.506, 1.510),
    ("coaster", "physics", 223, 0.556, 1.497),
    ("block", "physics", 4, 0.493, 1.405),
    ("trolley", "physics", 251, 0.574, 1.513),
    ("cart", "paraphrase", 17, 0.535, 1.504),
    ("coaster", "paraphrase", 16, 0.552, 1.530),
    ("block", "paraphrase", 40, 0.545, 1.531),
    ("cart", "arithmetic", 25, 0.563, 1.625),
    ("coaster", "arithmetic", 34, 0.556, 1.690),
    ("block", "arithmetic", 34, 0.568, 1.682),
]
fig, ax = plt.subplots(figsize=(6.4, 3.4))
ys = list(range(len(pairs)))[::-1]
for y, (name, kind, n, f14, f15) in zip(ys, pairs):
    ax.plot([f14, f15], [y, y], color=GRID, lw=2, zorder=2)
    ax.scatter([f14], [y], s=34, color=EDIT["A"], zorder=3, edgecolor="white", linewidth=1)
    ax.scatter([f15], [y], s=34, color=EDIT["B"], zorder=3, edgecolor="white", linewidth=1)
    ax.text(2.02, y, f"n={n}", va="center", fontsize=7, color=INK2)
ax.set_yticks(ys)
ax.set_yticklabels([f"{name} ({kind})" for name, kind, *_ in pairs])
ax.axvline(1.0, color=INK2, lw=0.8, ls="--", zorder=1)
ax.text(1.01, -1.0, "no change", ha="left", fontsize=7, color=INK2)
ax.set_xlim(0.4, 2.15)
ax.set_xlabel("relative gap after the step ÷ before it (median)")
ax.grid(axis="y", visible=False)
ax.scatter([], [], color=EDIT["A"], label="layer-14 feed-forward step")
ax.scatter([], [], color=EDIT["B"], label="layer-15 feed-forward step")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7)
ax.set_ylim(-1.3, 11.6)
save(fig, "fig2-recurrence")


# Figure 3. Outcome grid on the four tracked cases (FINDINGS.md, B/C/D reports).
cases = [
    ("Target block\n3 kg, 3 m climb", "60 / 90 / NO", ["60/120/NO", "60/90/NO", "60/120/NO", "60/90/NO", "60/90/NO"]),
    ("Sled\n1 m climb", "80 / 40 / YES", ["0/40/NO", "80/40/YES", "0/40/NO", "80/40/YES", "80/40/YES"]),
    ("Paired block (right run)\n3 kg, 1 m climb", "60 / 30 / YES", ["60/30/YES"] * 5),
    ("4 kg block\n2 m climb", "60 / 80 / NO", ["60/80/NO", "60/40/YES", "60/80/NO", "60/80/NO", "60/80/NO"]),
]
cols = ["Original", "A", "B", "C", "D"]
fig, ax = plt.subplots(figsize=(6.6, 2.75))
ax.set_xlim(0, 6.6)
ax.set_ylim(0, len(cases) + 0.75)
ax.axis("off")
x0, cw = 1.6, 1.0
for j, c in enumerate(cols):
    ax.text(x0 + j * cw + cw / 2, len(cases) + 0.4, c, ha="center", va="center", fontweight="bold",
            color=INK)
for i, (label, expected, vals) in enumerate(cases):
    y = len(cases) - 1 - i
    ax.text(0.0, y + 0.55, label, va="center", fontsize=7.5, color=INK)
    ax.text(0.0, y + 0.12, f"expected {expected}", va="center", fontsize=6.5, color=INK2)
    exp = expected.replace(" ", "")
    for j, v in enumerate(vals):
        ok = v == exp
        col = GOOD if ok else CRIT
        changed = j > 0 and v != vals[0]
        ax.add_patch(FancyBboxPatch((x0 + j * cw + 0.05, y + 0.08), cw - 0.1, 0.8,
                                    boxstyle="round,pad=0,rounding_size=0.06",
                                    facecolor=col + "22", edgecolor=col if changed else col + "55",
                                    linewidth=1.6 if changed else 0.8))
        ax.text(x0 + j * cw + cw / 2, y + 0.6, ("✓ " if ok else "✗ ") + v, ha="center", va="center",
                fontsize=7, color=INK)
        ax.text(x0 + j * cw + cw / 2, y + 0.28, "correct" if ok else "wrong", ha="center", va="center",
                fontsize=6, color=col)
save(fig, "fig3-cases")


# Figure 4. Aggregates on the 28-question development suite + stored matrix change.
models = ["Original", "A", "B", "C", "D"]
chains = [7, 9, 7, 9, 9]
verdicts = [16, 16, 16, 17, 17]
norm = {"A": 0.029656, "B": 0.0445366, "C": 0.0948835, "D": 0.0452194}
changed_vals = {"A": 200965, "B": 312037, "C": 523778, "D": 331670}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.6), gridspec_kw={"width_ratios": [1.5, 1]})
colors = [ORIG] + [EDIT[m] for m in "ABCD"]
w = 0.38
for k, (m, c) in enumerate(zip(models, colors)):
    bars(ax1, [k - w / 2], [chains[k]], c, w, dy=0.3)
    b = ax1.bar([k + w / 2], [verdicts[k]], width=w, color=c, alpha=0.45, edgecolor="white", zorder=3,
                hatch="////")
    ax1.text(k + w / 2, verdicts[k] + 0.3, str(verdicts[k]), ha="center", va="bottom", fontsize=7, color=INK)
ax1.set_xticks(range(5))
ax1.set_xticklabels(models)
ax1.set_ylim(0, 28)
ax1.set_ylabel("answers out of 28")
ax1.set_title("Main questions (28)", loc="left")
ax1.legend(handles=[Patch(facecolor=INK2, label="complete correct calculation"),
                    Patch(facecolor=INK2, alpha=0.45, hatch="////", label="correct YES/NO line")],
           loc="upper left", fontsize=6.5)
ax1.grid(axis="x", visible=False)
ks = list("ABCD")
for k, m in enumerate(ks):
    bars(ax2, [k], [norm[m]], EDIT[m], 0.6, fmt="{:.4f}", dy=0.002)
ax2.set_xticks(range(4))
ax2.set_xticklabels(ks)
ax2.set_ylim(0, 0.115)
ax2.set_ylabel("change size vs. matrix (%)")
ax2.set_title("Size of the change", loc="left")
ax2.grid(axis="x", visible=False)
fig.tight_layout()
save(fig, "fig4-aggregates")


# Figure 5. Native relative change in projection output at three shared-history decisions.
decisions = ["target block, step 235\n(designed for)", "target block, step 269\n(where A's fix shows)",
             "4 kg block, step 438\n(where A broke it)"]
proj = {
    "A": [0.9191, 0.7212, 0.8609],
    "B": [0.9548, 0.06969, 0.08501],
    "C": [0.8858, 0.6623, 0.2132],
    "D": [0.91851, 0.45033, 0.61852],
}
fig, ax = plt.subplots(figsize=(6.6, 2.7))
w = 0.2
for k, m in enumerate("ABCD"):
    xs = [i + (k - 1.5) * w for i in range(3)]
    bars(ax, xs, proj[m], EDIT[m], w, fmt="{:.2f}", dy=0.012)

ax.set_xticks(range(3))
ax.set_xticklabels(decisions)
ax.set_ylabel("change in W's output (%)")
ax.set_ylim(0, 1.12)
ax.grid(axis="x", visible=False)
ax.legend(handles=[Patch(facecolor=EDIT[m], label=f"edit {m}") for m in "ABCD"], ncol=4, loc="upper right",
          fontsize=7)
save(fig, "fig5-projection")


# Figure 6. Propagation of A's change at prediction 235 (FINDINGS.md).
bnd = ["input\nto W", "W's\noutput", "after\nnormalizing", "end of\nlayer 16", "end of\nlayer 34",
       "final\nnormalization"]
vals = [None, 0.9191, 1.0273, 0.3570, 0.02357, 0.05797]
fig, ax = plt.subplots(figsize=(6.0, 2.5))
xs = list(range(1, 6))
ax.plot(xs, vals[1:], color=EDIT["A"], lw=2, marker="o", ms=6, markeredgecolor="white", zorder=3)
for x, v in zip(xs, vals[1:]):
    below = x == 4
    ax.text(x, v / 1.6 if below else v * 1.35, f"{v:g}%", ha="center", va="top" if below else "bottom",
            fontsize=7, color=INK)
ax.scatter([0], [0.006], marker="v", s=40, color=EDIT["A"], zorder=3)
ax.text(0, 0.009, "exactly 0", ha="center", fontsize=7, color=INK)
ax.set_yscale("log")
ax.set_ylim(0.005, 3)
ax.set_xticks(range(6))
ax.set_xticklabels(bnd)
ax.set_ylabel("change (%)")
ax.grid(axis="x", visible=False)
save(fig, "fig6-propagation")


# Figure 7. D's fixed-input action relative to A (background-sensitivity report).
pops = ["4 held-out ordinary requests", "6 language checks", "6 arithmetic checks",
        "4 physics answers"]
ratio = [0.14497, 0.30348, 0.65523, 0.76460]
fig, ax = plt.subplots(figsize=(6.0, 1.9))
ys = list(range(4))[::-1]
ax.barh(ys, ratio, color=EDIT["D"], height=0.55, edgecolor="white", zorder=3)
for y, r in zip(ys, ratio):
    ax.text(r + 0.015, y, f"{r:.3f}×  ({(1 - r) * 100:.1f}% lower)", va="center", fontsize=7, color=INK)
ax.axvline(1.0, color=INK2, lw=0.8, ls="--")
ax.text(1.0, 3.55, "A", ha="center", fontsize=7, color=INK2)
ax.set_yticks(ys)
ax.set_yticklabels(pops)
ax.set_xlim(0, 1.15)
ax.set_xlabel("size of D's effect ÷ size of A's effect (same inputs)")
ax.grid(axis="y", visible=False)
save(fig, "fig7-background")


# Figure 8. Logit margins at the near-tie branch decisions (connected-edit report).
dec = ["target, step 269\n\"gained\" vs \"reached\"", "sled, step 33\nmath vs new line", "4 kg, step 438\n\"energy\" vs \"total\""]
margins = {
    "Original": [-0.006393, -0.000462, 0.022707],
    "A": [0.008394, 0.003975, -0.007727],
    "B": [-0.006218, -0.001232, 0.021198],
    "C": [0.008017, 0.004982, 0.015263],
}
fig, ax = plt.subplots(figsize=(6.4, 2.6))
w = 0.2
for k, m in enumerate(margins):
    c = ORIG if m == "Original" else EDIT[m]
    xs = [i + (k - 1.5) * w for i in range(3)]
    bars(ax, xs, [v * 1000 for v in margins[m]], c, w, fmt="{:.1f}", dy=0.6)
ax.axhline(0, color=INK2, lw=0.8)
ax.set_xticks(range(3))
ax.set_xticklabels(dec)
ax.set_ylabel("score gap (×10⁻³)")
ax.set_ylim(-12, 27)
ax.grid(axis="x", visible=False)
ax.legend(handles=[Patch(facecolor=ORIG if m == "Original" else EDIT[m], label=m) for m in margins], ncol=4,
          loc="upper left", fontsize=7)
save(fig, "fig8-margins")

print("ok", sorted(p.name for p in OUT.glob("*.svg")))
