#!/usr/bin/env python3
"""Deterministic generator for studies/04-round-2-overlap/RUN-CATALOG.md.

Reads prospective hypotheses and method steps from runs/*/declaration.json
and empirical outcomes from runs/*/result.json (plus tracked READMEs where
applicable) to produce a structured two-part guide:
  1. Landmark Runs (Start Here) — 16 pivotal runs across Round 2.
  2. Complete Thematic Index — every run directory in runs/, grouped into
     7 scientific families, plus Nov 17 transplant/context diagnostics
     documented in ROUND-2.md whose raw folders were excluded by .gitignore.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
RUNS_DIR = HERE / "runs"
OUT_PATH = HERE / "RUN-CATALOG.md"


# Curated 16 Landmark Runs (in scientific & chronological reading order)
LANDMARKS: list[dict[str, str]] = [
    {
        "name": "wide-overlap",
        "step": "1–3",
        "hypothesis": "Answers that strongly exercise the energy-accounting capability share a computational trace, and the recurring discrepancy between the 30 correct and 30 downhill-sign-error traces isolates the capability gap rather than prompt numbers.",
        "outcome": "Verified on 60 native GPU0 BF16 generations across all 35 layers (105 merges): prompt-number positions do not separate outcome groups; strong multi-times-chance recurring divergence emerges across generated calculation steps (142–313), with only a weak pre-error lean (~0.56).",
        "extra_link": "runs/wide-overlap/result.json",
    },
    {
        "name": "trace-overlap-complete-registered",
        "step": "1–3",
        "hypothesis": "Recurring discrepancy between complete correct and wrong trajectories remains sharply measurable under pure computational-distance alignment (chronological DTW on unit turns) with zero discarded native steps.",
        "outcome": "Registered all 30,546 captured native steps across all 60 cohort members and 105 merges with zero dropped events; recurring separation survived complete-path registration without text or token-position anchors.",
        "extra_link": "runs/trace-overlap-complete-registered/result.json",
    },
    {
        "name": "state-overlap",
        "step": "1–3",
        "hypothesis": "Capability divergence recurs not only in each merge's directional turn but also in post-merge residual states across all 70 attention-output and FFN-down matrices.",
        "outcome": "Post-merge state gap exceeded split-half chance across steps 142–313 at all 70 matrices; motivated targeting post-merge states in the initial `a2-path` ladder (later refined on Nov 15 to input-output transformation relations).",
        "extra_link": "runs/state-overlap/result.json",
    },
    {
        "name": "a2-path",
        "step": "5",
        "hypothesis": "Closing the recurring multi-layer post-merge state gap across the 18 diverging matrices repairs downhill energy addition on free native BF16 generation and transfers to held-out failures.",
        "outcome": "**First multi-trace repair & transfer:** Flipped downhill subtraction to addition in **29/30** originally wrong fitting-cohort answers (**17/30** complete valid chains) and repaired **7/21** held-out failures outside construction—while breaking **12/30** originally correct controls.",
        "extra_link": "runs/a2-path/result.json",
    },
    {
        "name": "d2-path",
        "step": "6",
        "hypothesis": "Weighting `a2-path`'s input updates by background second-moment statistics on general text preserves the multi-layer repair while quieting off-task movement and sparing originally correct physics answers.",
        "outcome": "**Falsified at multi-layer scale:** Held-out ordinary-text movement fell only to 0.60–0.74x of `a2-path` (missing the <0.5x prediction at all 18 matrices) and attenuated the active task perturbation enough to erase `a2-path`'s repairs.",
        "extra_link": "runs/d2-path/result.json",
    },
    {
        "name": "controlled-bf16-4",
        "step": "5",
        "hypothesis": "Splitting preservation into separate group-mean (M <= V/N) and average row-displacement (Q <= T, 29x wider than group-mean variance) bounds allows sequential BF16 sweep updates to clear native rounding.",
        "outcome": "Verified exact original-only reference identities across 105 merges, but sequential matrix fitting stalled early at layer-9 FFN under strict BF16 feasibility—motivating the Nov 12 shift away from separate protection tubes to a unified correction objective.",
        "extra_link": "runs/controlled-bf16-4/result.json",
    },
    {
        "name": "vector-chance-original",
        "step": "3",
        "hypothesis": "Predictive whole-vector variation of the 14 nonself original successes around the medoid correct reference (`expanded-004-h1`, 560 events x 105 merges x 1,536 dims) defines a principled tolerance scale.",
        "outcome": "Compiled complete predictive whole-vector chance field C (within/between/offset decomposition) in 44.2 s on CPU, establishing the fixed baseline loss `0.00029234` (`4,386` outside cells across `61,320` aligned pair-merge cells) for `discovery-05`.",
        "extra_link": "runs/vector-chance-original/README.md",
    },
    {
        "name": "vector-chance-final-experiment-installed-348-recovery",
        "step": "5",
        "hypothesis": "Differentiating full-path unit-turn discrepancy through the 538-tensor computation graph (continuing the 70-matrix `0 -> 317` lineage) into native BF16 weights with FP32 rounding remainders iteratively closes computational loss and repairs free generation.",
        "outcome": "**Landmark checkpoint `390`:** Reduced fresh native computational loss **78.09%** (`2.9234e-4 -> 6.4059e-5`) and **repaired free BF16 generation on `discovery-05`** (`64 + 40 - 13 = 91 J available, 20 J required, YES` instead of `11 J, NO`, recovering relapses at `356`, `371`, `387`).",
        "extra_link": "runs/vector-chance-final-experiment-installed-348-recovery/final-state.json",
    },
    {
        "name": "vector-chance-descent-sign-swap",
        "step": "4 (transplant)",
        "hypothesis": "Isolating either the learned attention-output family or the learned FFN down-projection family from the repaired lineage into the early failing checkpoint tests whether one parameter family monopolizes the sign repair.",
        "outcome": "**Dual-pathway sufficiency:** Early control gives `11 J / NO`; transplanting **either** late `attn_output` weights alone **or** late `ffn_down` weights alone independently produces correct `91 J / YES` (`attn_output` reproduces late wording; `ffn_down` solves it via distinct phrasing).",
        "extra_link": "runs/vector-chance-descent-sign-swap/result.json",
    },
    {
        "name": "frozen-energy-generalization",
        "step": "3",
        "hypothesis": "Lower fitted computational trace loss along the one-pair lineage (`original -> 175 -> 189 -> 390`) improves complete multi-step energy chains on unseen interpolated velocities and reordered problem templates.",
        "outcome": "Across 44 blinded native generations on 8 unseen formulations, complete valid chains rose monotonically **`1/8` (original) -> `2/8` (175) -> `2/8` (189) -> `3/8` (390)** (repairing 2 explicit errors and 1 capped run, while regressing 1 original success; 0/4 on reordered prose).",
        "extra_link": "runs/frozen-energy-generalization/result.json",
    },
    {
        "name": "same-input-operation-comparison",
        "step": "4",
        "hypothesis": "Evaluating two distinct successful weight repairs and one failed projected update on identical incoming states separates direct weight-operator changes from shifts inherited via altered token history.",
        "outcome": "Before histories forked (73 shared predictions), both working repairs moved full traces coherently (`0.7364` whole-prefix cosine vs `0.2991` / `0.2006` against the failed edit), even though their local direct signs at the descent merge opposed each other.",
        "extra_link": "runs/same-input-operation-comparison/README.md",
    },
    {
        "name": "descent-landmark-route-comparison",
        "step": "3",
        "hypothesis": "Conditioning downstream layer transformations on a shared `FFN20` computational landmark separates failing downhill subtractions from valid descents and valid friction subtractions.",
        "outcome": "Multi-turn consensus corrected two 32-step-late donor registrations; at the true pre-descent event, triggered on **13/13 wrong** and **0/15 valid** original descents (plus reordered failures), showing precursor divergence exists prior to emitting `-` vs `+`.",
        "extra_link": "runs/descent-landmark-route-comparison/registration-results.json",
    },
    {
        "name": "descent-operand-identity",
        "step": "3",
        "hypothesis": "Aligned residual-transformation prefixes distinguish *which physical operation* is being applied (downhill work vs. friction dissipation) independently of whether its sign is right or wrong.",
        "outcome": "On 30 public originals + 5 controls, local pre-choice readout separated descent vs. friction in **28/30** two-way comparisons each and flagged **15/15 wrong descents**, **3/15 valid descents**, and **0/30 friction subtractions**.",
        "extra_link": "runs/descent-operand-identity/result.json",
    },
    {
        "name": "joint-climb-conditional-overlap",
        "step": "4",
        "hypothesis": "Holding downhill behavior matched (both subtracting downhill work) exposes an orthogonal recurring trace discrepancy for double-counting climb height (`115 J` problem).",
        "outcome": "Confirmed: a climb-only repair reduced conditional climb MSE by **86.9%** (`0.06863 -> 0.00899` across 670 cells, 661 improving) and separated wrong vs. valid climbs across 640–673 / 673 cells on withheld families.",
        "extra_link": "runs/joint-climb-conditional-overlap/README.md",
    },
    {
        "name": "early-27-ffn20-native-effect",
        "step": "4",
        "hypothesis": "A pure phase-registered pullback from the `FFN20` correct-witness target moves native BF16 computation toward addition on fixed history, with finite anchored forward outperforming rounded JVP.",
        "outcome": "Moved native `FFN20` projection `-0.05414` toward the `-0.13706` target (finite forecast error **0.154%**, whereas rounded JVP overpredicted by **4.79x**) and narrowed the native `+` vs `-` logit margin from **-2.524 to -0.047** on fixed history.",
        "extra_link": "runs/early-27-ffn20-native-effect/README.md",
    },
    {
        "name": "joint-stock-prechoice-training",
        "step": "5",
        "hypothesis": "Combining pre-choice `FFN20` downhill discrepancy, conditional climb discrepancy, and friction preservation into a single stock-origin BF16 gradient simultaneously repairs `91 J` descent and `95 J` climb.",
        "outcome": "Update 1 cut first-event squared discrepancy **28.03%** (`-28.6%` total carried loss) and permanently repaired the `95 J` climb, but subsequent updates lost matcher registration on 2,000+ cells—trading preserved operations (`51x` carried-loss rise) for logged-loss drops.",
        "extra_link": "runs/joint-stock-prechoice-training/result.json",
    },
]


# Nov 17 runs documented in ROUND-2.md & input-archives whose raw run dirs were untracked under .gitignore
NOV17_DOCUMENTED_RUNS: list[dict[str, str]] = [
    {
        "name": "vector-chance-descent-sign-swap",
        "step": "4",
        "status": "ROUND-2.md record",
        "link": "ROUND-2.md",
        "summary": "Family-isolated weight transplant on `discovery-05`: transplanting either late `attn_output` alone or late `ffn_down` alone into the early `11 J / NO` control independently yields `91 J / YES`.",
    },
    {
        "name": "vector-chance-family-swap-175-189",
        "step": "4",
        "status": "ROUND-2.md record",
        "link": "ROUND-2.md",
        "summary": "Complementary native family swaps between checkpoints `175` and `189`: generated history follows the `ffn_down` donor on both `attn_output` backgrounds while preserving valid energy chains.",
    },
    {
        "name": "vector-chance-crossed-history-175-189",
        "step": "4",
        "status": "ROUND-2.md + dev record",
        "link": "../../records/development/2026-11-17-crossed-history/README.md",
        "summary": "2x2 weights-by-history factorial across the 14-update `175 -> 189` drop (`-56.26%`): supplying the `189` token history at fixed weights accounts for `99.66–99.80%` of the diagonal trace-loss decrease.",
    },
    {
        "name": "vector-chance-development-profile-175-189",
        "step": "3",
        "status": "ROUND-2.md + archive",
        "link": "ROUND-2.md",
        "summary": "Four complete native generations across `175` and `189` confirming preserved `44 J` reference arithmetic and identical partial climb repair on the unfitted `115 J` case.",
    },
    {
        "name": "vector-chance-reference-preservation-348",
        "step": "3",
        "status": "ROUND-2.md + dev record",
        "link": "../../records/development/2026-11-17-reference-preservation/reference-preservation-result.md",
        "summary": "Same-device preservation audit at step `348`: reference prompt trace loss rose (`1.30e-7 -> 9.35e-5` across all 105 merges) while generated solution preserved exact `44 J / 20 J / YES` arithmetic.",
    },
    {
        "name": "vector-chance-development-transfer-348",
        "step": "3",
        "status": "ROUND-2.md + dev record",
        "link": "../../records/development/2026-11-17-reference-preservation/development-transfer-result.md",
        "summary": "Unfitted `115 J` transfer diagnostic at step `348`: repaired required climb energy (`60 -> 30 J`) and verdict (`NO -> YES`), while downhill subtraction (`55 J` vs `115 J`) persisted.",
    },
    {
        "name": "vector-chance-descent-context-175",
        "step": "4",
        "status": "ROUND-2.md + dev record",
        "link": "../../records/development/2026-11-17-descent-context/README.md",
        "summary": "Pre-sign generated-framing swap at frozen checkpoint `175`: swapping prose framing before the sign token flipped `91 J -> 11 J` on the fitted problem and rescued `55 J -> 115 J` on the unfitted problem.",
    },
    {
        "name": "vector-chance-descent-context-heading-175",
        "step": "4",
        "status": "ROUND-2.md + dev record",
        "link": "../../records/development/2026-11-17-descent-context/heading.md",
        "summary": "Single-token heading intervention at token `72` on frozen `175`: altering only the section-heading token induced the `11 J` sign error on `discovery-05`.",
    },
    {
        "name": "vector-chance-isolated-pre-error-device-continuation",
        "step": "5",
        "status": "ROUND-2.md + dev record",
        "link": "../../records/development/2026-11-17-isolated-pre-error-device-continuation/RESULTS.md",
        "summary": "Completed single-update directional vs. standard `FFN20` comparison (continuing `vector-chance-isolated-pre-error-update`): directional arm lowered fitted loss `0.290%` while standard arm raised loss `5.55x`.",
    },
]


FAMILIES: list[tuple[str, str]] = [
    (
        "1. Cohort Overlaps, Acquisition Qualification & Trajectory Registration",
        "Step 1–3 multi-trace capture qualification on GPU0, whole-cohort correct/wrong overlaps (`30` vs `30`), DTW turn alignment, and fixed-prefix contribution diagnostics.",
    ),
    (
        "2. Analytical Path Edit Ladder (`a2` .. `d2-path`)",
        "Early Step 5–6 multi-matrix analytical edits built directly from cohort turn/state overlaps before iterative on-policy regeneration.",
    ),
    (
        "3. Controlled BF16 Variance-Tube & Subspace Sweeps",
        "Native BF16 constrained sweeps testing split-half/third feasibility tubes, mean-vs-row displacement budgets, and late-layer (`17`, `19`, `29–34`) sign chains.",
    ),
    (
        "4. Whole-Gap & Relational Trajectory Loops",
        "Transition from post-merge state targets to input-output merge relations, reference/envelope compilations, and anchored Torch trajectory loops.",
    ),
    (
        "5. Direct Computational Backprop (`vector-chance` Lineage & Precision/Curvature Diagnostics)",
        "Closed-loop one-pair computational differentiation (`expanded-004-h1` reference vs `discovery-05` failure) through `70`–`538` native BF16 tensors up to checkpoint `390`, plus operand-precision, JVP/finite, landscape, and generalization panels.",
    ),
    (
        "6. Mechanistic Interventions, Family Transplants, Same-Input Counterfactuals & KV-Cache Anchoring",
        "Causal replay and counterfactual attribution isolating native K-cache anchoring, block-0 Q/K/V contributions, weight-vs-history effects, and `attn_output` vs `ffn_down` transplants.",
    ),
    (
        "7. Operation Localization, Prefix Direction & Joint Climb+Descent Loops",
        "Pre-choice landmark/operand-identity detectors (`FFN20`), prefix-direction training (`390 -> 401`), unbanded group contrasts, and multi-operation stock training with streamed host memory.",
    ),
]


def classify_family(name: str) -> int:
    if name in {
        "a2",
        "a2-diverge",
        "a2-path",
        "a2-path-steps",
        "a2-pattern",
        "a2-sent",
        "b2-path",
        "c2-path",
        "c2-protected",
        "d2-path",
        "sign-route-a2-path",
    }:
        return 1
    if (
        name.startswith(("controlled-", "correction-", "sign-chain-", "sign-33-", "sign-request-"))
        or name in {
            "separable-at-17-original",
            "mixture-at-17-original",
            "line-chain-18-20-21-original",
            "overlap-chain-29-34-original",
            "terminal-rehearsal-34-original",
            "on-policy-33-original",
        }
    ):
        return 2
    if (
        name.startswith(
            (
                "whole-context-",
                "whole-gap-",
                "envelope-loop-",
                "relation-",
                "same-history-control-",
                "success-set",
                "functional-",
                "turn-",
            )
        )
        or name
        in {
            "recipient-prefix-target-original",
            "sign-route-whole-gap-mates",
            "plain-first-step-state",
            "shared-relation-loop-torch",
            "single-reference-original",
            "signed-residual-contrast-original",
        }
    ):
        return 3
    if (
        name.startswith(("directional-intervention-", "same-input-"))
        or name
        in {
            "finite-installed-diagnostic-replay",
            "descent-donor-on-descendants",
            "frozen-repair-pattern",
        }
    ):
        return 5
    if (
        name.startswith(
            (
                "descent-",
                "discovered-turn-",
                "prefix-",
                "early-27-",
                "unbanded-group-",
                "joint-",
            )
        )
        or name
        in {
            "conditional-whole-trace-stage",
            "whole-prefix-frozen-stages",
            "frozen-prefix-discrepancies",
            "calibrated-descent-development-outcome",
        }
    ):
        return 6
    if (
        name.startswith(("vector-chance-", "computational-loss-landscape-", "frozen-energy-"))
        or name in {"projection-gradient-cuda-fixture", "one-pair-landscape-and-traces"}
    ):
        return 4
    return 0


def format_step(val: Any) -> str:
    if isinstance(val, list):
        return ",".join(str(x) for x in val)
    return str(val) if val is not None else "—"


def clean_one_line(val: Any, limit: int = 155) -> str:
    if val is None:
        return ""
    if isinstance(val, str):
        text = " ".join(val.split())
    elif isinstance(val, dict):
        # Extract the most informative concise string value if available
        parts: list[str] = []
        for k, v in val.items():
            if isinstance(v, str):
                parts.append(f"{k}: {' '.join(v.split())}")
            elif isinstance(v, (int, float, bool)):
                parts.append(f"{k}={v}")
            elif isinstance(v, list) and v and isinstance(v[0], str):
                parts.append(f"{k}: {' '.join(v[0].split())}")
        text = "; ".join(parts) if parts else json.dumps(val, ensure_ascii=False)
    elif isinstance(val, list):
        text = " ".join(str(x) for x in val)
    else:
        text = str(val)
    text = text.replace("|", "\\|")
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return text


def summarize_run(folder: Path, decl: dict[str, Any], res: dict[str, Any] | None) -> tuple[str, str]:
    has_readme = (folder / "README.md").exists()
    if res is not None:
        status = "Completed (`result.json`)"
        pvr = res.get("prediction_vs_result")
        wtc = res.get("what_this_changes")
        summary = clean_one_line(pvr, 150)
        if not summary or summary.startswith("{"):
            alt = clean_one_line(wtc, 150)
            if alt and not alt.startswith("{"):
                summary = alt
        if not summary:
            summary = clean_one_line(decl.get("hypothesis"), 150)
    else:
        # Check execution/stop receipts for accurate status label
        if (folder / "final-state.json").exists() or (folder / "final-checkpoint").exists():
            status = "Checkpoint / handoff"
        elif (folder / "authorized-stop.json").exists() or (folder / "launcher-exit.json").exists():
            status = "Stopped / superseded"
        elif has_readme:
            status = "Recorded (`README.md`)"
        else:
            status = "Declared / diagnostic"
        summary = clean_one_line(decl.get("hypothesis"), 150)
    return status, summary


def build_catalog() -> str:
    fallback_runs = Path("~/Documents/dev/research/weight-patterns/studies/04-round-2-overlap/runs")
    folder_map: dict[str, Path] = {}
    if fallback_runs.is_dir():
        for p in sorted(fallback_runs.iterdir()):
            if p.is_dir() and ((p / "declaration.json").is_file() or (p / "result.json").is_file() or (p / "README.md").is_file()):
                folder_map[p.name] = p
    for p in sorted(RUNS_DIR.iterdir()):
        if p.is_dir():
            folder_map[p.name] = p

    run_folders = [folder_map[k] for k in sorted(folder_map)]
    present_names = set(folder_map)
    doc_only = [r for r in NOV17_DOCUMENTED_RUNS if r["name"] not in present_names]

    grouped: dict[int, list[tuple[str, str, str, str, str]]] = {i: [] for i in range(len(FAMILIES))}
    completed_count = 0

    for folder in run_folders:
        name = folder.name
        alt_folder = fallback_runs / name if fallback_runs.is_dir() else folder
        decl_path = folder / "declaration.json" if (folder / "declaration.json").is_file() else (alt_folder / "declaration.json")
        res_path = folder / "result.json" if (folder / "result.json").is_file() else (alt_folder / "result.json")
        readme_path = folder / "README.md" if (folder / "README.md").is_file() else (alt_folder / "README.md")

        decl = json.loads(decl_path.read_text()) if decl_path.is_file() else {}
        res = json.loads(res_path.read_text()) if res_path.is_file() else None
        if res is not None:
            completed_count += 1

        step_str = format_step(decl.get("method_step"))
        has_final = (folder / "final-state.json").is_file() or (alt_folder / "final-state.json").is_file()
        status, summary = summarize_run(folder, decl, res)
        if res is None and has_final:
            status = "Checkpoint / handoff"

        links = []
        if decl_path.is_file():
            links.append(f"[decl](runs/{name}/declaration.json)")
        if res_path.is_file():
            links.append(f"[result](runs/{name}/result.json)")
        if readme_path.is_file():
            links.append(f"[README](runs/{name}/README.md)")
        link_cell = f"`{name}` (" + " · ".join(links) + ")"

        fam_idx = classify_family(name)
        grouped[fam_idx].append((name, step_str, status, link_cell, summary))

    total_indexed = len(run_folders) + len(doc_only)

    lines: list[str] = []
    lines.append("# Round 2 Run Catalog (`04-round-2-overlap`)")
    lines.append("")
    lines.append(
        "Structured guide to all Round 2 multi-trace overlap, native BF16 weight-editing, "
        "direct computational-backpropagation, and mechanistic attribution runs on **Gemma 4 E2B BF16**."
    )
    lines.append("")
    lines.append(
        f"- **Indexed runs:** **{total_indexed} total** (**{len(run_folders)}** tracked `runs/<name>/` directories "
        f"with prospective `declaration.json` — **{completed_count}** carrying formal `result.json` verdicts and "
        f"**{len(run_folders) - completed_count}** checkpointed continuations, diagnostics, or superseded starts — "
        f"plus **{len(doc_only)}** November 17 transplant/context diagnostics recorded directly in "
        "[ROUND-2.md](ROUND-2.md) and `records/development/2026-11-17-*/`)."
    )
    lines.append(
        "- **Method steps (`round-2-method.json`):** `1` Correct-cohort overlap · `2` Wrong-cohort overlap · "
        "`3` Recurring discrepancy & reference compilation · `4` Direct counterfactual / target qualification · "
        "`5` Native BF16 trace-driven weight training · `6` Continuation from reached checkpoint."
    )
    lines.append(
        "- **Companion documents:** [ROUND-2.md](ROUND-2.md) (chronological lab log) · "
        "[round-2-method.json](round-2-method.json) (machine-checked invariants) · "
        "[../../JOURNEY.md](../../JOURNEY.md) (end-to-end research narrative) · "
        "[../../reports/descent-trace-repair/README.md](../../reports/descent-trace-repair/README.md) (case-study paper)."
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. Landmark Runs (Start Here)")
    lines.append("")
    lines.append(
        "These **16 pivotal runs** capture every major discovery, repair milestone, negative result, "
        "and mechanistic control in Round 2:"
    )
    lines.append("")
    lines.append("| # | Run & Evidence Links | Step | Prospective Hypothesis | Quantitative Outcome |")
    lines.append("| --- | --- | --- | --- | --- |")

    for idx, lm in enumerate(LANDMARKS, 1):
        name = lm["name"]
        if name in present_names:
            links = [f"[decl](runs/{name}/declaration.json)"]
            extra = lm.get("extra_link", "")
            alt_extra = Path("~/Documents/dev/research/weight-patterns/studies/04-round-2-overlap") / extra if extra else None
            if extra and ((HERE / extra).is_file() or (alt_extra is not None and alt_extra.is_file())):
                label = Path(extra).name
                links.append(f"[{label}]({extra})")
            run_cell = f"**`{name}`**<br>(" + " · ".join(links) + ")"
        else:
            extra = lm.get("extra_link", "ROUND-2.md")
            run_cell = f"**`{name}`**<br>([ROUND-2.md]({extra}))"
        lines.append(
            f"| {idx} | {run_cell} | `{lm['step']}` | {lm['hypothesis']} | {lm['outcome']} |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"## 2. Complete {total_indexed}-Run Thematic Index (7 Scientific Families)")
    lines.append("")
    lines.append("### Family Summary")
    lines.append("")
    lines.append("| Family | Tracked Run Dirs | Additional Nov 17 Logged Runs | Total |")
    lines.append("| --- | ---: | ---: | ---: |")
    for idx, (title, _) in enumerate(FAMILIES):
        extra_n = len(doc_only) if idx == 5 else 0
        anchor = (
            title.lower()
            .replace(".", "")
            .replace("`", "")
            .replace("(", "")
            .replace(")", "")
            .replace("+", "")
            .replace("&", "")
            .replace(",", "")
            .replace("/", "")
        )
        anchor = "-".join(anchor.split())
        lines.append(
            f"| [{title}](#{anchor}) | {len(grouped[idx])} | {extra_n} | {len(grouped[idx]) + extra_n} |"
        )
    lines.append(f"| **Total** | **{len(run_folders)}** | **{len(doc_only)}** | **{total_indexed}** |")
    lines.append("")

    for idx, (title, desc) in enumerate(FAMILIES):
        lines.append(f"### {title}")
        lines.append("")
        lines.append(desc)
        lines.append("")
        lines.append("| Run & Links | Step | Status | Concise Summary (Hypothesis / Result) |")
        lines.append("| --- | --- | --- | --- |")
        for _, step_str, status, link_cell, summary in grouped[idx]:
            lines.append(f"| {link_cell} | `{step_str}` | {status} | {summary} |")
        if idx == 5 and doc_only:
            for item in doc_only:
                link_cell = f"`{item['name']}` ([record]({item['link']}))"
                lines.append(
                    f"| {link_cell} | `{item['step']}` | {item['status']} | {item['summary']} |"
                )
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*Generated deterministically by [`make_catalog.py`](make_catalog.py).*")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    OUT_PATH.write_text(build_catalog(), encoding="utf-8")
    print(f"Wrote {OUT_PATH} ({OUT_PATH.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
