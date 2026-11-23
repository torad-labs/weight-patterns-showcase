# Manuscripts, Specifications & Interactive Explorers

**Marcos Damasceno · Torad Labs (November 2026)**

This directory collects the working papers, method guides, formal specifications, empirical tables, and interactive visualizers for the **computational trace repair** research program on **Gemma 4 E2B** (native BF16 / FP32 in `llama.cpp`).

Across four progressive studies, the program tests one hypothesis: *when a model exercises a capability, recurring differences between the internal computational traces of successful and failing executions expose where the calculation departs, and updating the weights responsible for those departures repairs the underlying operation without token-loss fine-tuning or answer-key supervision.*

---

## Reading Order: Four Working Papers & Bridge Proposal

All four studies investigate multi-step conservation-of-energy reasoning in Gemma 4 E2B ($K_{\text{available}} = K_0 + mgd - Q \ge mgH$), where the base model shows two recurring failures:
1. **Climb height conflation (`final_climb_includes_descent`):** adding an earlier downhill drop $d$ to the final climb height $H$ when computing required potential energy $mgH$.
2. **Downhill sign inversion (`descent_subtracted`):** subtracting positive gravitational work $mgd$ instead of adding it to initial kinetic energy $K_0$.

Every recording is captured bit-identically from `llama.cpp` graph callbacks (verified by exact token identity and full 262,144-token vocabulary-score SHA-256 hashes at every decoded step), and every weight edit is installed into native model storage and evaluated under unconstrained greedy generation.

| Stage | Document | Formats & Source | Core Question, Method & Measured Findings |
| :--- | :--- | :--- | :--- |
| **1. Single-Pair Weight Edits & Selectivity** | *Fixing a Reasoning Mistake by Editing the Weights: Four Edits to a Small Language Model and What They Taught Us* | [PDF](fixing-a-reasoning-mistake-by-editing-the-weights.pdf) · [HTML](../studies/03-round-1-single-pair-edits/paper/paper.html) | **Can a rank-one native BF16 edit derived from a single right/wrong trace pair fix a reasoning mistake without training?** Four edits (`A`–`D`) to `blk.16.attn_output.weight` repair the `120 -> 90 J` climb error and a sled variant (`0 -> 80 J`). Conditioning the read key against background second moments (`D` / `D1`) cuts RMS projection action on held-out unrelated text by **85.5%** while preserving all 7 prior correct chains and restoring a `4 kg` case broken by `A`, while leaving four physics prompts outside `D`'s fit at `1/4 -> 1/4`. |
| **2. Multi-Layer Contextual Trace Targets** | *How the Trace-Derived Edit Works: An Internal Guide to the Method, Step by Step* *(Nov 8 internal guide, shipped byte-identical)* | [PDF](how-the-trace-derived-edit-works.pdf) · [HTML](../studies/03-round-1-single-pair-edits/method-guide/guide.html) | **How do we represent whole-model signal routing and transfer a repair to unseen problem parameters?** Introduces the normalized residual-stream *turn* observable $T = h_{\text{out}}/\|h_{\text{out}}\| - h_{\text{in}}/\|h_{\text{in}}\|$ across all $35 \times 3 = 105$ residual merges. Solving contextual trace targets across three matrices (`FK`: `blk.11.ffn_down` and `blk.14` K/V) repairs **15 of 20** wrong or missing climb fields on 40 excluded prompts with unseen speeds and heights (`2/31` full chains, `8/9` prior successes damaged), and a 46-matrix pass (`old A2`) repairs **25/36** fitted failures and **9/31** complete unseen chains (`6/9` damaged). |
| **3. Pure Multi-Trace Computational Overlap** | *Editing the Weights from the Overlap of Many Answers: How the Path Edit Is Built and What Its First Run Did* | [PDF](how-the-path-edit-works.pdf) · [HTML](../studies/04-round-2-overlap/method-guide/guide.html) | **Can variable-length traces be aligned purely by computation (with zero token or word labels) to repair a capability from group overlap?** Aligns 60 natural generations (30 wrong vs. 30 correct descents) via 105-merge Dynamic Time Warping (`1 - cos`), isolating a connected stretch `5.8×` chance. Solving 18 matrices across layers 0–8 sequentially in path order (`a2-path`) flips the downhill sign error in **29 of 30** fitted cases (**17/30** complete, `12/30` overlap controls broken) and transfers to **7 of 21** excluded Round-1 failures (**11/21** adding descent, `4/7` controls broken), while traces drift sideways by **2.6–2.7** local gaps outside the targeted interval. |
| **Bridge Proposal & Assessment (Nov 15)** | *Iterative Learning from Computational Traces* *(external assessment & proposal prepared for the author, AI-assisted)* | [DOCX](Iterative_Learning_from_Computational_Traces.docx) · [Spec](../FINAL-EXPERIMENT.md) | **Why replace one-shot matrix solves with iterative trace backpropagation?** Formulates closed-loop computational learning: differentiate local discrepancies exceeding tolerance $\varepsilon_j$ through the connected forward graph into shared weights, install in native precision, and regenerate freely after every step (subsequently formalized with witness-calibrated $\tau_{im}$ and carried FP32 remainders in [`FINAL-EXPERIMENT.md`](../FINAL-EXPERIMENT.md)). |
| **4. Iterative Trace Backpropagation & Operation Localization** | *Training on Local Divergence Between Computational Traces* (18 Nov 2026) | [PDF](descent-trace-repair/paper.pdf) · [HTML](descent-trace-repair/paper.html) · [Typst](descent-trace-repair/paper.typ) · [Source Map](descent-trace-repair/README.md) · [Milestone](descent-trace-repair/milestone.md) | **Does iterative backpropagation of trace divergence produce reproducible, operation-specific repair?** Reduces fresh fitted trace loss by **78.09%** through update 390 (`70` matrices over `0–317` producing `11 -> 91 J` and `-77.95%` drop; all `538` eligible tensors over `317–390`), preserving the `44 J` reference and rough-section friction (`35/35` audited captures). Shows via native transplants that either all 35 `attn_output` or all 35 `ffn_down` matrices suffice to restore `91 J`; evaluates checkpoint-blind unseen transfer (`3/8` full chains at `390` vs. `1/8` baseline, all in channel wording; `3/8` vs. `4/8` downhill signs); isolates F16 KV-cache anchoring (`-90.7%` forecast error on the primary displacement); and reports negative results on a 1-update directional cap and an 11-update continuation (`390 -> 401`). |

> **Provenance & Timeline Notes:**
> - **Paper 2 header:** `how-the-trace-derived-edit-works.pdf` retains its original November 8 `"INTERNAL GUIDE · NOT FOR PUBLICATION"` banner untouched so its binary SHA-256 (`10de7506371d4bc4...`) matches the cryptographic table in [`../FINAL-EXPERIMENT.md`](../FINAL-EXPERIMENT.md).
> - **Original 40-question set vs. `reserved-40b`:** Papers 2 and 3 note that the original 40-question set was unread when they were drafted; that initial set was subsequently evaluated and retired in the [November 9 retrospective](../records/retrospectives/2026-11-09-round-two-repairs/STUDY.md) and replaced by the separate `reserved-40b` evaluation set.

---

## Conceptual Foundation, Specifications & Reviews

| Resource | Path | Role in the Repository |
| :--- | :--- | :--- |
| **Hypothesis Overview** | [`../hypothesis/RESIDUAL-TRACE-HANDOFF.md`](../hypothesis/RESIDUAL-TRACE-HANDOFF.md) | Concise entry point to the *residual trace* hypothesis: observing connected transformations across varied capability executions ("completing the dots") without requiring full circuit recovery. |
| **Full Theoretical Framework** | [`../hypothesis/RESIDUAL-TRACE-THEORY.md`](../hypothesis/RESIDUAL-TRACE-THEORY.md) | November 6 theoretical synthesis connecting continued pretraining, RL, small-network weight recognition, control repair, and trace-guided editing, keeping analogies separate from measured claims. |
| **Two Research Fronts** | [`../hypothesis/TWO-RESEARCH-FRONTS.md`](../hypothesis/TWO-RESEARCH-FRONTS.md) | Methodological contract pairing single-pair causal investigation with multi-trace overlap discovery under shared native evaluation standards. |
| **The Final Experiment Specification** | [`../FINAL-EXPERIMENT.md`](../FINAL-EXPERIMENT.md) | Mathematical and architectural specification of the 35-block / 105-merge Gemma 4 E2B trace observable, DTW correspondence, calibrated radial tolerance, 538-tensor reverse-mode derivative, response-scaled step $\eta\alpha$, and mixed BF16/FP32 remainder installer. |
| **Final Experiment Research Record** | [`../final-experiment/README.md`](../final-experiment/README.md) | Chronological dossier of the November 17–18 experiments: [training & stopping](../final-experiment/training-and-stopping.md), [descent-sign mechanism](../final-experiment/descent-sign-mechanism.md), [operand identity (descent vs. friction)](../final-experiment/operand-identity.md), [one-update experiment](../final-experiment/one-update-experiment.md), [multiple-trace signals](../final-experiment/multiple-trace-signals.md), [accumulating climb + descent repairs](../final-experiment/accumulating-repairs.md), and [joint fit & transfer](../final-experiment/joint-fit-and-transfer.md). |
| **Independent Scientific & Build Reviews** | [`descent-trace-repair/review.md`](descent-trace-repair/review.md) · [`descent-trace-repair/unified-review.md`](descent-trace-repair/unified-review.md) | Source-bound audits verifying SHA-256 digests, checkpoint-blind arithmetic gradings, alignment-conditional lower bounds, Q/K/V native replay factorials, and negative precursor findings. |
| **Prospective Forecast Protocol & Scorer** | [`descent-trace-repair/transfer-protocol.md`](descent-trace-repair/transfer-protocol.md) · [`descent-trace-repair/score_forecasts.py`](descent-trace-repair/score_forecasts.py) | Preregistered protocol and standalone Python scorer for testing before-edit operation-level forecasts (`repair`, `preserve`, `persist`, `regress`, `abstain`) on unseen cohorts. |
| **Consolidated Round-1 & Early Findings** | [`../publications/findings/FINDINGS.md`](../publications/findings/FINDINGS.md) · [`EXPERIMENTS.md`](../publications/findings/EXPERIMENTS.md) · [`TABLES.md`](../publications/findings/TABLES.md) | Cross-study summaries, 22-study chronological index, and unpooled per-population score tables. |

---

## Interactive HTML Explorers & Visualizers

Both explorers run directly in a browser using committed display artifacts, with no GPU, model checkpoint, or inference server required.

1. **[Paired Weight-Slice & 105-Merge Trace Explorer](../studies/04-round-2-overlap/runs/one-pair-landscape-and-traces/explorer.html)**
   *Self-contained single-file HTML interactive figure accompanying Paper 4 (§6).*
   Inspects a measured two-dimensional slice through native weight space alongside the aligned 105-merge computational turn field across every generated event, showing visually why the largest post-decision trace peak (aligned pair 270) sits far downstream of the actual physical sign decision (aligned pair 143).

2. **[Decode Machine: Inside Gemma 4 E2B (`../apps/decode-machine/`)](../apps/decode-machine/README.md)**
   *Interactive 3D/2D execution observatory and profiling inspector ([`index.html`](../apps/decode-machine/index.html) · [`teaching.html`](../apps/decode-machine/teaching.html)).*
   Visualizes token-by-token signal flow across Gemma 4 E2B's 35 layers, 8 query heads, owned vs. shared KV caches (layers 13–14 feeding 15–34), GeGLU feed-forward branches, BF16/quantized rounding gauges, and worker/warp execution timelines, linking directly to the early physics-trace and persistent weight-edit studies.

---

## Rebuilding Paper 4 (`descent-trace-repair/`)

All six SVG figures and both typeset editions (`paper.pdf` and self-contained `paper.html`) compile deterministically from [`descent-trace-repair/`](descent-trace-repair/):

```bash
cd reports/descent-trace-repair
python3 make_figures.py
uv run --no-project --with typst==0.15.2 python render.py
```
