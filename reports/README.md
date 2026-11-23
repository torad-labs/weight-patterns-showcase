# Manuscripts, Specifications & Interactive Explorers

**Marcos Damasceno · Torad Labs (November 2026)**

This directory collects the manuscripts, method guides, formal specifications, empirical tables, and interactive visualizers for the **computational trace repair** research program on **Gemma 4 E2B** (native BF16 / FP32 in `llama.cpp`).

Across four progressive studies, the program tests a single core hypothesis: *when a model exercises a capability, recurring differences between the internal computational traces of successful and failing executions expose where the calculation departs—and updating the weights responsible for those departures repairs the underlying operation without conventional token-loss fine-tuning or answer-key supervision.*

---

## Reading Order: The Four-Paper Progression

All four studies investigate multi-step conservation-of-energy reasoning in Gemma 4 E2B ($K_{\text{available}} = K_0 + mgd - Q \ge mgH$), where the base model exhibits two distinct, recurring failures:
1. **Climb height conflation (`final_climb_includes_descent`):** adding an earlier downhill drop $d$ to the final climb height $H$ when computing required potential energy $mgH$.
2. **Downhill sign inversion (`descent_subtracted`):** subtracting positive gravitational work $mgd$ instead of adding it to initial kinetic energy $K_0$.

Every recording is captured bit-identically from `llama.cpp` graph callbacks (verified by exact token identity and full 262,144-token vocabulary-score SHA-256 hashes at every decoded step), and every weight edit is installed into native model storage and evaluated under unconstrained greedy generation.

| Stage | Manuscript | Formats & Source | Core Question, Method & Measured Findings |
| :--- | :--- | :--- | :--- |
| **1. Single-Pair Weight Edits & Selectivity** | *Fixing a Reasoning Mistake by Editing the Weights: Four Edits to a Small Language Model and What They Taught Us* | [PDF](fixing-a-reasoning-mistake-by-editing-the-weights.pdf) · [HTML](paper-a-d/paper.html) | **Can a rank-one native BF16 edit derived from a single right/wrong trace pair fix a reasoning mistake without training?** Four edits (`A`–`D`) to `blk.16.attn_output.weight` repair the `120 → 90 J` climb error and a sled variant (`0 → 80 J`). Conditioning the read key against background second moments (`D`) cuts RMS projection action on held-out unrelated text by **85.5%** while preserving all 7 prior correct chains and a protected 4 kg case—but single-position edits do not generalize to unseen physics prompts (`1/4 → 1/4`), and two edits matching at the target anchor diverge 34 tokens later. |
| **2. Multi-Layer Contextual Trace Targets** | *How the Trace-Derived Edit Works: An Internal Guide to the Method, Step by Step* | [PDF](how-the-trace-derived-edit-works.pdf) · [HTML](../studies/03-round-1-single-pair-edits/method-guide/guide.html) | **How do we represent whole-model signal routing and transfer a repair to unseen problem parameters?** Introduces the normalized residual-stream *turn* observable $T = h_{\text{out}}/\|h_{\text{out}}\| - h_{\text{in}}/\|h_{\text{in}}\|$ across all $35 \times 3 = 105$ residual merges (accounting for KV cache sharing across layers 14, 19, 24, 29, and 34). Solving contextual trace targets across three matrices (`FK`: `blk.11.ffn_down` and `blk.14` K/V) repairs **15 of 20** wrong or missing climb calculations on 40 excluded prompts with unseen speeds and heights, and a 46-matrix pass (`A2`) repairs **25/36** fitted failures and **9/31** complete unseen chains—while exposing collateral regressions (`8/9` and `6/9`) from uncoupled layer solves. |
| **3. Pure Multi-Trace Computational Overlap** | *Editing the Weights from the Overlap of Many Answers: How the Path Edit Is Built and What Its First Run Did* | [PDF](how-the-path-edit-works.pdf) · [HTML](../studies/04-round-2-overlap/method-guide/guide.html) | **Can variable-length traces be aligned purely by computation—with zero token or word labels—to repair a capability from group overlap?** Aligns 60 natural generations (30 wrong vs. 30 correct descents) via 105-merge Dynamic Time Warping (`1 − cos`), isolating a connected stretch ~6× chance. Solving 18 matrices across layers 0–8 sequentially in path order (`a2-path`) flips the downhill sign error in **29 of 30** fitted cases (**17/30** complete) and transfers to **7 of 21** excluded Round-1 failures (**11/21** adding descent), while **12 of 30** original overlap successes regress without protection and traces drift sideways by **2.6–2.7** local gaps outside the targeted interval. |
| **Bridge Proposal** | *Iterative Learning from Computational Traces* | [DOCX](Iterative_Learning_from_Computational_Traces.docx) · [Spec](../FINAL-EXPERIMENT.md) | **Why replace one-shot matrix solves with iterative trace backpropagation?** Formulates closed-loop computational learning: differentiate only local discrepancies exceeding calibrated correct-to-correct variation $\tau_{im}$, propagate gradients jointly through the connected forward graph into shared weights, install with native rounding remainders, and regenerate freely after every step so coupled layers co-adapt on the model's actual trajectory. |
| **4. Iterative Trace Backpropagation & Operation Localization** | *Training on Local Divergence Between Computational Traces* (18 Nov 2026) | [PDF](descent-trace-repair/paper.pdf) · [HTML](descent-trace-repair/paper.html) · [Typst](descent-trace-repair/paper.typ) · [Source Map](descent-trace-repair/README.md) · [Milestone](descent-trace-repair/milestone.md) | **Does end-to-end backpropagation of trace divergence across 538 parameter tensors produce reproducible, operation-specific repair?** Reduces fresh fitted trace loss by **78.09%** through update 390, repairing downhill energy (`11 → 91 J`) while preserving the `44 J` reference and local rough-section friction across all 35 audited captures. Shows via native transplants that either all 35 attention-output or all 35 FFN-down matrices suffice to restore the sign; evaluates blind unseen transfer (`3/8` full chains at 390 vs. `1/8` at baseline, all in channel wording); isolates native F16 KV-cache rounding boundaries (cutting local finite-step forecast error by **90.7%**); and reports rigorous negative results on a 1-update directional cap and an 11-update shared-prefix continuation (`390 → 401`). |

---

## Conceptual Foundation, Specifications & Reviews

| Resource | Path | Role in the Repository |
| :--- | :--- | :--- |
| **Hypothesis Overview** | [`../hypothesis/RESIDUAL-TRACE-HANDOFF.md`](../hypothesis/RESIDUAL-TRACE-HANDOFF.md) | Concise entry point to the *residual trace* hypothesis: observing connected transformations across varied capability executions ("completing the dots") without requiring full circuit recovery. |
| **Full Theoretical Framework** | [`../hypothesis/RESIDUAL-TRACE-THEORY.md`](../hypothesis/RESIDUAL-TRACE-THEORY.md) | Complete November 6 theoretical synthesis connecting continued pretraining, RL, small-network weight recognition, control repair, and trace-guided editing, keeping analogies strictly separated from measured claims. |
| **Two Research Fronts** | [`../hypothesis/TWO-RESEARCH-FRONTS.md`](../hypothesis/TWO-RESEARCH-FRONTS.md) | Methodological contract pairing single-pair causal investigation with multi-trace overlap discovery under shared native evaluation standards. |
| **The Final Experiment Specification** | [`../FINAL-EXPERIMENT.md`](../FINAL-EXPERIMENT.md) | Complete mathematical and architectural specification of the 35-block / 105-merge Gemma 4 E2B trace observable, DTW correspondence, calibrated radial tolerance, 538-tensor reverse-mode derivative, response-scaled step $\eta\alpha$, and mixed BF16/FP32 remainder installer. |
| **Final Experiment Research Record** | [`../final-experiment/README.md`](../final-experiment/README.md) | Chronological dossier of the November 17–18 experiments: [training & stopping](../final-experiment/training-and-stopping.md), [descent-sign mechanism](../final-experiment/descent-sign-mechanism.md), [operand identity (descent vs. friction)](../final-experiment/operand-identity.md), [one-update experiment](../final-experiment/one-update-experiment.md), [multiple-trace signals](../final-experiment/multiple-trace-signals.md), [accumulating climb + descent repairs](../final-experiment/accumulating-repairs.md), and [joint fit & transfer](../final-experiment/joint-fit-and-transfer.md). |
| **Independent Scientific & Build Reviews** | [`descent-trace-repair/review.md`](descent-trace-repair/review.md) · [`descent-trace-repair/unified-review.md`](descent-trace-repair/unified-review.md) | Source-bound audits verifying SHA-256 digests, blinded human arithmetic gradings, alignment-conditional lower bounds, Q/K/V native replay factorials, and negative precursor findings. |
| **Prospective Forecast Protocol & Scorer** | [`descent-trace-repair/transfer-protocol.md`](descent-trace-repair/transfer-protocol.md) · [`descent-trace-repair/score_forecasts.py`](descent-trace-repair/score_forecasts.py) | Preregistered protocol and standalone Python scorer for testing before-edit operation-level forecasts (`repair`, `preserve`, `persist`, `regress`, `abstain`) on unseen cohorts. |
| **Consolidated Round-1 & Early Findings** | [`FINDINGS.md`](FINDINGS.md) · [`EXPERIMENTS.md`](EXPERIMENTS.md) · [`TABLES.md`](TABLES.md) | Cross-study summaries, 22-study chronological index, and unpooled per-population score tables (`../publications/findings/`). |
| **Standalone Release Exporter** | [`../publications/descent-paper-release/`](../publications/descent-paper-release/) | Deterministic, SHA-256-verified export script (`export.py` + `FILES.txt`) packaging Paper 4, its raw JSON data, figure generators, and the CPU-testable 538-tensor learner. |

---

## Interactive HTML Explorers & Visualizers

Both explorers run directly in a browser using committed display artifacts—no GPU, model checkpoint, or inference server required.

1. **[Paired Weight-Slice Landscape & 105-Merge Trace Explorer](../studies/04-round-2-overlap/runs/one-pair-landscape-and-traces/explorer.html)**
   *Self-contained single-file HTML interactive figure accompanying Paper 4 (§6).*
   Inspects a measured two-dimensional slice through native weight space alongside the aligned 105-merge computational turn field across every generated event, showing visually why the largest post-decision trace peak (aligned pair 270) sits far downstream of the actual physical sign decision (aligned pair 143).

2. **[Decode Machine — Inside Gemma 4 E2B (`../apps/decode-machine/`)](../apps/decode-machine/README.md)**
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
