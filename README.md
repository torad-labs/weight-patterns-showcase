# Weight Patterns: Repairing Capabilities by Differentiating Computational Traces

**Marcos Damasceno · Torad Labs · November 2026**  
*Mechanistic interpretability, native BF16 weight editing, and computational-trace training on Gemma 4 E2B*

> **Public Showcase (`torad-labs/weight-patterns-showcase`)**
> - This repository bundles all four papers, formal specifications, the C++ `llama.cpp` trace recorder, the 538-tensor PyTorch training loop, runnable CPU verification tests, primary empirical records across all four study phases, all 39 lab journal entries, and interactive HTML visualizers. Every file is listed in `MANIFEST.sha256` with zero symlinks.
> - The full 226-run historical lab bench (`torad-labs/weight-patterns`) and multi-GB R2 `.f32`/`.gguf` captures stay private to keep the sealed `reserved-40b` evaluation holdout out of public git. See [`studies/04-round-2-overlap/RUN-CATALOG.md`](studies/04-round-2-overlap/RUN-CATALOG.md) for the full run index; specific intermediate records are available on request.

---

## 30-Second CPU Verification

Every check below runs deterministically on **CPU** (`CUDA_VISIBLE_DEVICES=""`) in seconds, with no GPU, model weights, or external downloads required.

```bash
# 1. Verify SHA-256 checksums for every file in this repository
sha256sum -c MANIFEST.sha256
```

```bash
# 2. Rebuild all 6 Paper 4 SVG figures from primary JSON measurements (Python 3 stdlib only)
make figures
```

```bash
# 3. Verify the preregistered operation-forecast scoring protocol (Python 3 stdlib only)
make score-protocol
```

```bash
# 4. Run preregistration, parameter-provenance, and archive guard unit tests (<1s, stdlib only)
make test-guard
```

```bash
# 5. Run the synthetic 35-block, 538-tensor CPU training & overlap suite (PyTorch + NumPy + Matplotlib)
make test-cpu
```

<details>
<summary><strong>Run without <code>make</code> (direct <code>python3</code> / <code>uv</code> commands)</strong></summary>

```bash
# Stdlib-only checks:
(cd reports/descent-trace-repair && python3 make_figures.py)
PYTHONPATH=code/signal-trace CUDA_VISIBLE_DEVICES="" python3 -m unittest \
  tests.test_recorded_losses tests.test_parameter_sources tests.test_forbidden_bytes

# Full 35-block synthetic Gemma 4 E2B learner, DTW overlap, and group-contrast tests via uv:
CUDA_VISIBLE_DEVICES="" uv run --project code/signal-trace python -m unittest \
  tests.test_final_experiment tests.test_full_graph tests.test_complete_overlap tests.test_group_contrast
```
</details>

---

## Central Question

**Hypothesis:** When a language model exercises a reasoning capability, recurring geometric transformations appear in its connected residual stream across layers and token positions. Differentiating local divergence in those internal traces directly into stored weights can repair the underlying operation without token-level cross-entropy or answer-key supervision.

All experiments run on **Gemma 4 E2B** (`35` transformer blocks, residual width `1536`, `3` residual merges per block = **`105` merges per token**, KV cache sharing from producer blocks `13/14` into consumer blocks `15–34`, mixed `BF16`/`FP32` native storage in `llama.cpp`, SHA-256 `6a737131...`). Instead of matching output tokens or raw post-merge states $h^{\text{out}}$ (which carry prompt-specific numbers and wording), we align variable-length free generations by Dynamic Time Warping (`1 - cos`) over normalized **branch turns**:

$$u(v) = \frac{v}{\|v\|_2}, \qquad T_{t,l,b} = u\!\left(h^{\text{out}}_{t,l,b}\right) - u\!\left(h^{\text{in}}_{t,l,b}\right)$$

On multi-step conservation-of-energy problems ($K_{\text{avail}} = K_0 + mgd - Q_{\text{friction}} \ge mgH_{\text{climb}}$), stock Gemma 4 E2B shows two systematic reasoning bugs:
1. **Climb height conflation:** adding an earlier downhill drop $d$ to the final climb height $H$ when computing required potential energy $mgH$.
2. **Downhill sign inversion:** subtracting gravitational work gained on a descent ($K_0 - mgd$) while correctly subtracting rough-patch friction ($- Q_{\text{friction}}$) a few tokens later.

---

## Empirical Findings (Repairs, Transfer, and Negative Results)

All counts below keep their population denominators, regressions, and falsified ideas visible. Read **[`JOURNEY.md`](JOURNEY.md)** for the full 19-day chronological log and a short retrospective on running this research with coding agents.

### 1. Two Computational Trace Signals Isolated Without Token Supervision
- **Pre-choice downhill sign vs. valid friction** ([`operand-identity.md`](final-experiment/operand-identity.md), [`descent-operand-identity`](studies/04-round-2-overlap/runs/descent-operand-identity/result.json)): Across 30 natural stock responses containing 60 signed energy steps, phase-matched turn divergence immediately before operator emission flags **15/15** wrong descents (`13/13` with **0/15** false positives under landmark-consensus registration in [`landmark-route-comparison`](studies/04-round-2-overlap/runs/descent-landmark-route-comparison/result.json)) and **0/30** valid friction subtractions. The internal trace separates the physical role of the operand from the `-` token itself.
- **105-merge conditional climb signal** ([`joint-climb-conditional-overlap`](studies/04-round-2-overlap/runs/joint-climb-conditional-overlap/README.md)): Holding the downhill bug constant across 16 traces isolates a 712-cell 105-merge field that classifies a withheld problem's extra-height error on **643/673** cells and drops **86.9%** (`0.06863 -> 0.00899`) when checkpoint `175` repairs that climb (`60 J -> 30 J`).

### 2. Native BF16 Weight Repairs and Convergent Causal Pathways
- **Single-pair edits (`A -> B -> C -> D1`)** ([Paper 1](reports/fixing-a-reasoning-mistake-by-editing-the-weights.pdf), [`FINDINGS.md`](publications/findings/FINDINGS.md)): A rank-1 native BF16 edit to `blk.16.attn_output.weight` (`0.0297%` norm change, `0` bytes altered outside the matrix) repaired `120 J -> 90 J` (computing the recipient's own $3\text{ m} \times 30\text{ N}$ instead of copying the donor's $30\text{ J}$) and a sled prompt (`0 -> 80 J`). Conditioning on inverse background second moments (`D1`, [`D-background-sensitivity`](studies/03-round-1-single-pair-edits/D-background-sensitivity/run/REPORT.md)) preserved all repairs and controls while cutting generic background RMS action by **85.50%**.
- **60-trace DTW overlap (`a2-path`)** ([Paper 3](reports/how-the-path-edit-works.pdf), [`a2-path/result.json`](studies/04-round-2-overlap/runs/a2-path/result.json)): Aligning 30 correct and 30 wrong traces by 105-merge cosine (`5.8×` sampling variation, `0.825` split-half cosine) and editing layers `0–8` flipped **29/30** fitted descents (`17/30` complete chains) and **7/21** unseen Round-1 failures (`11/21` adding descent), with collateral regressions on prior successes. Earlier contextual multi-matrix solves (`FK`, `old A2` in [Paper 2](reports/how-the-trace-derived-edit-works.pdf) and [`2026-11-09-round-two-repairs`](records/retrospectives/2026-11-09-round-two-repairs/STUDY.md)) transferred explicit $mgH$ arithmetic to **15/20** unseen speed/height prompts.
- **Closed-loop 538-tensor trace backpropagation** ([Paper 4](reports/descent-trace-repair/paper.pdf), [`FINAL-EXPERIMENT.md`](FINAL-EXPERIMENT.md)): Differentiating calibrated local turn excess through all **538 eligible native tensors** reduced fresh computational loss **78.09%** (`0.00029234 -> 0.00006406` at checkpoint `390`), repairing `64 - 40 - 13 = 11 J (NO)` to **`64 + 40 - 13 = 91 J (YES)`** while preserving rough-patch friction (`35/35` captures) and the `44 J` reference chain.
- **Redundant causal sufficiency and shared prefix geometry** ([`descent-sign-swap`](records/development/2026-11-17-descent-sign-swap/README.md), [`same-input-operation-comparison`](studies/04-round-2-overlap/runs/same-input-operation-comparison/README.md)): Transplanting *either* the 35 `attn_output` matrices alone *or* the 35 `ffn_down` matrices alone into the failing `early-27` checkpoint independently restores `64 + 40 = 104 J` and `91 J / YES` (`99.84%` and `99.21%` of trace-loss reduction). Across the `73` identical pre-fork steps before any token differs, both repairs move the 105-merge trace along a common direction (**cosine `0.7364`**, **`0.9991`** at FFN20) that a failed edit misses (**`0.2991` / `0.2006`**).
- **Blinded out-of-sample panel** ([`frozen-energy-generalization`](studies/04-round-2-overlap/runs/frozen-energy-generalization/result.json)): On a prespecified, blind-graded 8-formulation unseen panel across 4 frozen checkpoints (`stock`, `175`, `189`, `390`), complete valid chains rose **`1/8 -> 2/8 -> 2/8 -> 3/8`** with friction preserved **`8/8`** throughout (2 errors fixed, 1 length cap resolved, 1 baseline success regressed; raw blinded transcripts in [`blinded-readings.json`](studies/04-round-2-overlap/runs/frozen-energy-generalization/blinded-readings.json)).

### 3. Negative Results and Falsified Hypotheses
| Negative / Falsification | What Happened and Why It Mattered | Primary Record |
| :--- | :--- | :--- |
| **Single-step activation patching & single-anchor edit `B`** | 1-step activation substitution changed `0` tokens (`1,860` decisions); edit `B` matched `A` at step 235 (`0.95%` vs `0.92%`) but decayed `10×` by step 269 and lost the repair. Multi-step capability fixes require sustained connected trajectory support. | [`physics-intervention`](studies/02-physics-trace-observation/physics-intervention/STUDY.md), [`B-selective-footprint`](studies/03-round-1-single-pair-edits/B-selective-footprint/run/REPORT.md) |
| **Per-layer variance tubes & state-hull targets** | Hard per-layer variance caps (`tube.py`) blocked downstream compensation; matching post-merge states $h^{\text{out}}$ leaked donor numbers into recipients. Led directly to turn vectors $T$ plus non-self witness radii. | [`2026-11-12-correction-not-protection.md`](journal/2026-11-12-correction-not-protection.md), [`2026-11-15-computation-not-state-values.md`](journal/2026-11-15-computation-not-state-values.md) |
| **Single-site FFN20 pullback (`early-27`)** | Reduced fixed-history sign margin from `-2.524 -> -0.047` (finite model accurate to `0.154%`), but under free generation the model changed wording at token 4 and routed back to subtraction (`-8.851`). | [`pure-descent-native-read`](records/retrospectives/2026-11-18-pure-descent-native-read/README.md), [`early-27-ffn20-native-effect`](studies/04-round-2-overlap/runs/early-27-ffn20-native-effect/README.md) |
| **Shared-prefix continuation (`390 -> 401`)** | 11 native updates adding a 7,299-cell early-repair hinge kept `91 J` and cut ordinary loss `0.693%`, but yielded **`0/8`** newly repaired descents on 4 unseen physical setups in 2 orderings. | [Paper 4 §12](reports/descent-trace-repair/paper.pdf), [`prefix-direction-generalization`](studies/04-round-2-overlap/runs/prefix-direction-generalization/README.md) |
| **Matcher-dropout reward hacking in sparse op-only training** | Optimizing sparse labelled-operation cells alone lowered *logged* loss **25.7%** (`0.0272 -> 0.0202`) while breaking all `3/3` descents because unaligned cells contributed `0` loss. Charging lost cells their last measured error showed true loss **rose `+22.6%`** (`0.0333`), motivating the combined whole-trace plus carried-cell objective. | [`2026-11-22-joint-loss-recheck`](records/development/2026-11-22-joint-loss-recheck/README.md), [`recheck.py`](records/development/2026-11-22-joint-loss-recheck/recheck.py) |

---

## Systems and Instrumentation

1. **Native C++ stream recorder** ([`observe_stream.cpp`](code/signal-trace/recorder/observe_stream.cpp), [`signal_trace.hpp`](code/signal-trace/recorder/signal_trace.hpp), [`STREAM-RECORDER.md`](code/signal-trace/recorder/STREAM-RECORDER.md)): Hooks `llama.cpp`'s GGML graph scheduler to stream all `105` residual merges and projection operands in native BF16/FP32 without per-op `cudaDeviceSynchronize`, with `stop_after_last_label` early graph termination (`35×` layer-0 speedup) and `late_start` subgraph seeding (`207/207` bitwise-exact passes). Qualified bit-for-bit across `12,023` BF16 and `13,462` Q4 decisions.
2. **Differentiable 538-tensor native-anchored graph** ([`full_graph.py`](code/signal-trace/pipeline/weight_edit/full_graph.py), [`native_key_anchor.py`](code/signal-trace/pipeline/weight_edit/native_key_anchor.py), [`ARCHITECTURE.md`](code/signal-trace/ARCHITECTURE.md)): Reconstructs Gemma 4 E2B's GGML forward conventions (windowed vs. global RoPE, blocks `0–14` KV production reused by blocks `15–34`, per-layer-input gates, layer output scales) anchored to native captures for VJP ($\nabla_W L$) and JVP ($J_r S$) response-scaled step sizing ($\Delta W = \eta \alpha S$, $\alpha = -\langle r,q\rangle/\langle q,q\rangle$). Anchoring pre-cache FP32->FP16 keys across all 15 KV producers fixed sign-flip forecast errors and cut finite-step error by **90.66%**.
3. **Sub-ULP BF16/FP32 weight installer and streamed memory engine** ([`full_state.py`](code/signal-trace/pipeline/weight_edit/full_state.py), [`group_disk_state.py`](code/signal-trace/pipeline/weight_edit/group_disk_state.py)): Accumulates updates in `float64` (`W + E + ΔW`), writes native `BF16` (`278` tensors) and `FP32` (`260` tensors) in-place with carried `FP32` rounding remainders `E` so small multi-step gradients survive BF16 quantization, and streams leaf gradients within `~15 GB` host RAM.
4. **Automated research hygiene checks** ([`verify_round2.py`](code/signal-trace/pipeline/research_guard/verify_round2.py), [`round-2-method.json`](studies/04-round-2-overlap/round-2-method.json)): Enforces prospective `declaration.json` fields (`why_this_direction`, `repeating_pattern_check`, `hypothesis`, `prediction`, `would_weaken_it`), requires parameter provenance (`measured_from`, `derivation`, `scales_with`), blocks token-based training metrics (`forbidden_bytes.py`), and guards sealed holdouts.

---

## Repository Map

| Directory / File | What to Read & Inspect |
| :--- | :--- |
| **[`JOURNEY.md`](JOURNEY.md)** | Start here for the 6-chapter chronological narrative (Nov 4–22, 2026) and a short retrospective on conducting the work with coding agents. |
| **[`reports/`](reports/README.md)** | All 4 papers and index: [Paper 1 (Single-Pair BF16 Edits)](reports/fixing-a-reasoning-mistake-by-editing-the-weights.pdf), [Paper 2 (Contextual Multi-Trace Method Guide)](reports/how-the-trace-derived-edit-works.pdf), [Paper 3 (60-Trace Path Overlap)](reports/how-the-path-edit-works.pdf), [Bridge Proposal (DOCX)](reports/Iterative_Learning_from_Computational_Traces.docx), and **[Paper 4 (`descent-trace-repair/paper.pdf`)](reports/descent-trace-repair/paper.pdf)** plus [`milestone.md`](reports/descent-trace-repair/milestone.md) and [`transfer-protocol.md`](reports/descent-trace-repair/transfer-protocol.md). |
| **[`FINAL-EXPERIMENT.md`](FINAL-EXPERIMENT.md)** & **[`final-experiment/`](final-experiment/README.md)** | Equations for the 35-block / 105-merge learner plus mechanism dossiers ([`operand-identity.md`](final-experiment/operand-identity.md), [`descent-sign-mechanism.md`](final-experiment/descent-sign-mechanism.md), [`accumulating-repairs.md`](final-experiment/accumulating-repairs.md), [`joint-fit-and-transfer.md`](final-experiment/joint-fit-and-transfer.md)). |
| **[`code/signal-trace/`](code/signal-trace/ARCHITECTURE.md)** | C++ recorder (`recorder/`), Python/PyTorch trace alignment & weight-update engine (`pipeline/`), [`TRAINING.md`](code/signal-trace/TRAINING.md), [`pyproject.toml`](code/signal-trace/pyproject.toml), and runnable CPU `tests/`. |
| **[`studies/`](studies/README.md)** | Curated study reports and primary JSON records across [`01-early-exploration`](studies/01-early-exploration/README.md), [`02-physics-trace-observation`](studies/02-physics-trace-observation/README.md), [`03-round-1-single-pair-edits`](studies/03-round-1-single-pair-edits/README.md), and [`04-round-2-overlap`](studies/04-round-2-overlap/RUN-CATALOG.md) (including [interactive 105-merge trace explorer](studies/04-round-2-overlap/runs/one-pair-landscape-and-traces/explorer.html)). |
| **[`hypothesis/`](hypothesis/RESIDUAL-TRACE-THEORY.md)** & **[`journal/`](journal/README.md)** | Theoretical framework ([`RESIDUAL-TRACE-THEORY.md`](hypothesis/RESIDUAL-TRACE-THEORY.md), [`TWO-RESEARCH-FRONTS.md`](hypothesis/TWO-RESEARCH-FRONTS.md)) and all 39 dated lab memos. |
| **[`records/`](records/README.md)** & **[`publications/findings/`](publications/findings/FINDINGS.md)** | Mechanistic retrospectives, causal transplant readbacks, [`2026-11-22-joint-loss-recheck`](records/development/2026-11-22-joint-loss-recheck/README.md), and consolidated unpooled score tables. |
| **[`apps/decode-machine/`](apps/decode-machine/README.md)** | Interactive 3D/2D Gemma 4 E2B execution & trace visualizer (TypeScript / Vite / WebGL). |
| **Full Historical Lab Archive** | Private (`torad-labs/weight-patterns`, protecting sealed `reserved-40b` holdout and multi-GB R2 captures); see [`RUN-CATALOG.md`](studies/04-round-2-overlap/RUN-CATALOG.md) for the complete 226-run index. |
