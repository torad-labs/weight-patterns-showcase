# Weight Patterns: Repairing Capabilities by Differentiating Computational Traces

**Marcos Damasceno · Torad Labs · November 2026 · [MIT License](LICENSE) · [`CITATION.cff`](CITATION.cff)**  
*Mechanistic interpretability, native BF16 weight editing, and computational-trace training on Gemma 4 E2B*

> **Public Research Showcase (`torad-labs/weight-patterns-showcase`)**
> - This standalone repository packages all four working papers (plus the Nov 15 bridge proposal), formal specifications, the C++ `llama.cpp` stream recorder, the complete `code/signal-trace/` Python/PyTorch engine, runnable CPU verification tests, primary JSON declarations and results across all **226 Round-2 runs** ([`RUN-CATALOG.md`](studies/04-round-2-overlap/RUN-CATALOG.md)), all **40 lab journal memos**, and offline interactive HTML visualizers—with zero symlinks, zero broken relative links, and every shipped file verified via `sha256sum -c MANIFEST.sha256` (which lists every file except itself).
> - Multi-GB raw `.f32` activation streams, `.gguf` checkpoints in Cloudflare R2, and the unedited working bench (`torad-labs/weight-patterns`, holding the `reserved-40b` evaluation folder; see [Nov 21 audit](journal/2026-11-21-consolidation-takeover.md) for an unresolved automated directory-walk check on that folder) remain outside public git.

---

## 30-Second CPU Verification

Every check below runs deterministically on **CPU** (`CUDA_VISIBLE_DEVICES=""`) in seconds, with no GPU, model weights, or external downloads required.

```bash
# 1. Verify SHA-256 checksums for every file in MANIFEST.sha256
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

## Central Question & Where This Research Stands

**Hypothesis:** When a language model exercises a reasoning capability, recurring geometric transformations appear in its connected residual stream across layers and token positions. Differentiating local divergence in those internal traces directly into stored weights can repair the underlying operation without token-level cross-entropy or answer-key supervision.

> **Where This Stands (November 22, 2026):**
> 1. **Stored-weight repair under free generation is proven across five constructions:** single-pair analytical edits (`D1`), contextual multi-matrix solves (`FK`, `old A2`), pure 60-trace DTW overlap (`a2-path`), iterative whole-trace backpropagation (`checkpoint 390`), and independent 35-matrix `attn_output` / `ffn_down` causal transplants.
> 2. **Multiple computational signals cleanly localize the failing operations:** pre-choice turn divergence separates wrong downhill subtractions from valid rough-patch friction subtractions before `-` is emitted, and a 105-merge conditional overlap isolates the climb-height conflation error.
> 3. **What is not yet solved (active frontier):** we do not yet have a single unified training loop that composes these signals to repair failing answers, preserve valid ones, and generalize reliably without matcher-dropout drift. Returning the differentiated whole-trace anchor to the gradient alongside carried-cell operation penalties—paired against a matched next-token cross-entropy twin baseline ([`2026-11-22-token-loss-twin.md`](journal/2026-11-22-token-loss-twin.md), built and awaiting engine re-qualification)—is the immediate next run. All measurements to date cover one model (`Gemma 4 E2B`) and one problem family, with no evaluation run on `reserved-40b`.

All experiments run on **Gemma 4 E2B** (`35` transformer blocks, residual width `1536`, `3` residual merges per block = **`105` merges per token**, KV cache sharing from producer blocks `13/14` into consumer blocks `15–34`, mixed `BF16`/`FP32` native storage in `llama.cpp`, SHA-256 `6a737131...`). Instead of matching output tokens or raw post-merge states $h^{\text{out}}$ (which carry prompt-specific numbers and wording), we align variable-length free generations by Dynamic Time Warping (`1 - cos`) over normalized **branch turns**:

$$u(v) = \frac{v}{\|v\|_2}, \qquad T_{t,l,b} = u\!\left(h^{\text{out}}_{t,l,b}\right) - u\!\left(h^{\text{in}}_{t,l,b}\right)$$

On multi-step conservation-of-energy problems ($K_{\text{avail}} = K_0 + mgd - Q_{\text{friction}} \ge mgH_{\text{climb}}$), stock Gemma 4 E2B shows two systematic reasoning bugs:
1. **Climb height conflation:** adding an earlier downhill drop $d$ to the final climb height $H$ when computing required potential energy $mgH$.
2. **Downhill sign inversion:** subtracting gravitational work gained on a descent ($K_0 - mgd$) while correctly subtracting rough-patch friction ($- Q_{\text{friction}}$) a few tokens later.

---

## Empirical Findings (Repairs, Transfer, and Negative Results)

All counts below keep their population denominators, collateral regressions, and falsified ideas visible. Read **[`JOURNEY.md`](JOURNEY.md)** for the full 18-day chronological log (Nov 5–22, 2026) and a closing retrospective on running this research with coding agents.

### 1. Two Computational Trace Signals Isolated Without Token Supervision
- **Pre-choice downhill sign vs. valid friction** ([`operand-identity.md`](final-experiment/operand-identity.md), [`descent-operand-identity`](studies/04-round-2-overlap/runs/descent-operand-identity/result.json), [`landmark-route-comparison.md`](final-experiment/landmark-route-comparison.md)): On 30 natural stock responses (60 signed energy steps), a local two-turn block-20 matcher flags **15/15** wrong descents and **0/30** valid friction subtractions immediately before operator selection (with **3/15** valid descents flagged). A refined landmark-consensus donor registration ([`registration-results.json`](studies/04-round-2-overlap/runs/descent-landmark-route-comparison/registration-results.json)) separates **13/13** wrong from **0/15** valid descents at the pre-choice event on the 28-response bank, though scanning whole responses still alerts elsewhere on valid answers (and sequence/equation order remains a competing explanation per Paper 4 §8).
- **105-merge conditional climb signal** ([`joint-climb-conditional-overlap`](studies/04-round-2-overlap/runs/joint-climb-conditional-overlap/README.md)): Holding the downhill bug constant across 16 traces isolates a retrospective 712-cell 105-merge field that separates a withheld problem's extra-height error on **643/673** cells and drops **86.9%** (`0.06863 -> 0.00899` across `670` shared cells) when checkpoint `175` repairs that climb (`60 J -> 30 J`).

### 2. Native BF16 Weight Repairs and Convergent Causal Pathways
- **Single-pair edits (`A -> B -> C -> D1`)** ([Paper 1](reports/fixing-a-reasoning-mistake-by-editing-the-weights.pdf), [`FINDINGS.md`](publications/findings/FINDINGS.md)): Rank-1 native BF16 edit `A` on `blk.16.attn_output.weight` (`0.0297%` norm change, `0` bytes altered outside the matrix) repaired `120 J -> 90 J` (computing the recipient's own $3\text{ m} \times 30\text{ N}$ rather than copying the donor's $30\text{ J}$) and a sled prompt (`0 -> 80 J`), while regressing a `4 kg` block (`60/80/NO -> 60/40/YES`). Connected preservation (`C`) restored `4 kg`, and inverse-second-moment background conditioning (`D` / `D1`, [`D-background-sensitivity`](studies/03-round-1-single-pair-edits/D-background-sensitivity/run/REPORT.md)) kept all repairs and `7/7` prior chains while cutting generic background RMS action by **85.50%** (`1/4 -> 1/4` on four physics prompts outside `D1`'s fit).
- **60-trace DTW overlap (`a2-path`) & contextual multi-matrix edits (`FK`, `old A2`)** ([Paper 3](reports/how-the-path-edit-works.pdf), [`a2-path/result.json`](studies/04-round-2-overlap/runs/a2-path/result.json), [Paper 2](reports/how-the-trace-derived-edit-works.pdf), [`2026-11-09-round-two-repairs`](records/retrospectives/2026-11-09-round-two-repairs/STUDY.md)): Aligning 30 correct and 30 wrong traces by 105-merge cosine (`5.8×` sampling variation, `0.825` split-half cosine) and editing layers `0–8` flipped **29/30** fitted descents (`17/30` complete chains, while breaking `12/30` correct overlap controls) and **7/21** Round-1 failures outside the fit (`11/21` adding descent, breaking `4/7` Round-1 controls). Earlier contextual solves corrected **15/20** wrong or missing climb-energy *fields* on 40 excluded prompts (`FK`: `2/31` full chains repaired, `8/9` prior successes damaged; `old A2`: `9/31` full chains repaired, `6/9` damaged).
- **Closed-loop whole-trace backpropagation (`0 -> 390`)** ([Paper 4](reports/descent-trace-repair/paper.pdf), [`FINAL-EXPERIMENT.md`](FINAL-EXPERIMENT.md)): Whole-trace backpropagation against fixed correct reference `expanded-004-h1` reduced fresh computational loss **78.09%** (`0.00029234 -> 0.00006406` at checkpoint `390`) and repaired `64 - 40 - 13 = 11 J (NO)` to **`64 + 40 - 13 = 91 J (YES)`** while preserving the `44 J` reference chain (and rough-patch friction across `35/35` audited recordings—30 stock, 1 reordered, 4 checkpoint captures—plus `11/11` panel generations at `390`). Updates `0–317` trained the **70** `attn_output` and `ffn_down` matrices, producing the `91 J` repair (by checkpoint `174/175`) and `77.95%` of the reduction; updates `317–390` differentiated all **538** eligible native tensors (`-0.64%` further reduction, recovering one-step relapses at `356`, `371`, `387`).
- **Redundant causal sufficiency and shared prefix geometry** ([`descent-sign-swap`](records/development/2026-11-17-descent-sign-swap/README.md), [`same-input-operation-comparison`](studies/04-round-2-overlap/runs/same-input-operation-comparison/README.md)): Transplanting *either* the 35 `attn_output` matrices alone *or* the 35 `ffn_down` matrices alone from the repaired lineage into `early-27` restores `64 + 40 = 104 J` and `91 J / YES` on its own (`99.84%` and `99.21%` of trace-loss reduction). Across the `73` pre-fork predictions before any generated token differs, both repairs move the 105-merge turn field along a shared direction (**cosine `0.7364`**) that a failed edit misses (**`0.2991` / `0.2006`**); at a later block-20 site after histories diverge, the two repairs agree at **`0.9991`**.
- **Blinded out-of-sample panel** ([`frozen-energy-generalization`](studies/04-round-2-overlap/runs/frozen-energy-generalization/result.json), [`paired-outcomes.json`](studies/04-round-2-overlap/runs/frozen-energy-generalization/paired-outcomes.json)): On a prespecified, checkpoint-blind 8-formulation panel (4 parameter-novel setups $\times$ 2 templates) across `stock`, `175`, `189`, and `390`, complete valid chains rose **`1/8 -> 2/8 -> 2/8 -> 3/8`** (all 3 in the channel template: 2 wrong chains and 1 capped chain became correct, 1 baseline success regressed), while correct downhill signs went **`4/8 -> 3/8`** (1 repaired, 2 regressed) and friction stayed **`8/8`** throughout (blinded judgments in [`blinded-readings.json`](studies/04-round-2-overlap/runs/frozen-energy-generalization/blinded-readings.json), full outputs in [`blinded-responses.json`](studies/04-round-2-overlap/runs/frozen-energy-generalization/blinded-responses.json)).

### 3. Negative Results and Falsified Hypotheses
| Negative / Falsification | What Happened and Why It Mattered | Primary Record |
| :--- | :--- | :--- |
| **Single-step activation patching & single-anchor edit `B`** | 1-step activation substitution changed `0` tokens (`1,860` decisions); edit `B` matched `A` at step 235 (`0.95%` vs `0.92%`) but shrank to a tenth of `A` by step 269 (`0.07%` vs `0.72%`) and lost the repair, suggesting multi-step fixes need sustained connected support. | [`physics-intervention`](studies/02-physics-trace-observation/physics-intervention/STUDY.md), [`B-selective-footprint`](studies/03-round-1-single-pair-edits/B-selective-footprint/run/REPORT.md) |
| **Per-layer variance tubes & state-hull targets** | Hard per-layer variance caps (`tube.py`) blocked downstream compensation; matching post-merge states $h^{\text{out}}$ risked pulling donor content into recipients (`success-sets-loop-from-mates` halted at write 42/70). Led to turn vectors $T$ plus non-self witness radii. | [`2026-11-12-correction-not-protection.md`](journal/2026-11-12-correction-not-protection.md), [`2026-11-15-computation-not-state-values.md`](journal/2026-11-15-computation-not-state-values.md) |
| **Single-site FFN20 pullback (`early-27`)** | Reduced fixed-history sign margin from `-2.524 -> -0.047` (finite FFN20 projection forecast accurate to `0.154%`), but `-` stayed greedy and under free generation the model changed wording at token 4 and routed back to subtraction (`-8.851`). | [`pure-descent-native-read`](records/retrospectives/2026-11-18-pure-descent-native-read/README.md), [`early-27-ffn20-native-effect`](studies/04-round-2-overlap/runs/early-27-ffn20-native-effect/README.md) |
| **Shared-prefix continuation (`390 -> 401`)** | 11 native updates adding a 7,299-cell early-repair hinge kept `91 J` and cut ordinary loss `0.693%` while auxiliary prefix loss rose `2.461%`, yielding **`0/8`** complete chains or newly repaired descents on a `±1 J` boundary panel (`390` also scored `0/8` there). | [Paper 4 §12](reports/descent-trace-repair/paper.pdf), [`prefix-direction-generalization`](studies/04-round-2-overlap/runs/prefix-direction-generalization/README.md) |
| **Matcher-dropout reward hacking in sparse op-only training** | Optimizing sparse labelled-operation cells alone lowered *logged* loss **25.7%** (`0.0272 -> 0.0202`) while wrong descents spread from `1/3 -> 3/3` (`2` newly broken) because unaligned cells added `0` loss. Under a last-measured-value charge, loss **rose `+22.6%`** (`+42.0%` on `3,845` common cells). | [`2026-11-22-joint-loss-recheck`](records/development/2026-11-22-joint-loss-recheck/README.md), [`recheck.py`](records/development/2026-11-22-joint-loss-recheck/recheck.py) |

---

## Systems and Instrumentation

1. **Native C++ stream recorders** ([`observe_stream.cpp`](code/signal-trace/recorder/observe_stream.cpp), [`signal_trace.hpp`](code/signal-trace/recorder/signal_trace.hpp), [`STREAM-RECORDER.md`](code/signal-trace/recorder/STREAM-RECORDER.md)): Hooks `llama.cpp`'s GGML graph scheduler (`f089c9cb`) to stream all `105` residual merges and projection operands in native BF16/FP32 without per-op `cudaDeviceSynchronize`, with `stop_after_last_label` early graph termination (`35×` layer-0 speedup) and `late_start` subgraph seeding (`207/207` bitwise-exact passes). Bitwise inference equivalence was verified on `12,023` BF16 and `13,462` Q4 decisions for `observe`, and `480` residual + `480` projection + `265` trace-excerpt decisions for `observe_stream` on GPU0.
2. **Differentiable 538-tensor native-anchored graph** ([`torch_forward.py`](records/development/2026-11-12-path-forward/torch_forward.py), [`full_parameters.py`](code/signal-trace/pipeline/weight_edit/full_parameters.py), [`full_graph.py`](code/signal-trace/pipeline/weight_edit/full_graph.py), [`native_key_anchor.py`](code/signal-trace/pipeline/weight_edit/native_key_anchor.py), [`ARCHITECTURE.md`](code/signal-trace/ARCHITECTURE.md)): Reconstructs Gemma 4 E2B's GGML forward path (windowed vs. global RoPE, producer blocks `13/14` KV caches reused by blocks `15–34`, GeGLU, per-layer-input gates) over all `538` eligible parameters (`601` GGUF tensors minus `63` fixed/unused late-KV buffers) for VJP ($\nabla_W L$) and JVP ($J_r S$) step sizing ($\Delta W = \eta \alpha S$, $\alpha = -\langle r,q\rangle/\langle q,q\rangle$). Anchoring pre-cache FP32->FP16 keys across all 15 KV producers corrected the finite forecast's sign and cut error by **90.66%** on the primary displacement (`-9.94%` on a second archived displacement).
3. **BF16/FP32 remainder installer and streamed memory engine** ([`full_state.py`](code/signal-trace/pipeline/weight_edit/full_state.py), [`group_disk_state.py`](code/signal-trace/pipeline/weight_edit/group_disk_state.py), [`2026-11-19-streamed-working-state`](records/development/2026-11-19-streamed-working-state/README.md)): Accumulates updates in `float64` (`W + E + ΔW`), writes native `BF16` (`278` tensors) and `FP32` (`260` tensors) via 4 KiB changed-page `pwrite` with `FP32` rounding remainders `E` carried forward rather than discarded, and streams leaf gradients within `~15.6 GB` peak host RSS (down from `63.4 GB`).
4. **Automated research hygiene checks** ([`verify_round2.py`](code/signal-trace/pipeline/research_guard/verify_round2.py), [`round-2-method.json`](studies/04-round-2-overlap/round-2-method.json)): Enforces prospective `declaration.json` fields (`why_this_direction`, `repeating_pattern_check`, `hypothesis`, `prediction`, `would_weaken_it`), parameter provenance (`measured_from`, `derivation`, `scales_with`), and token-supervision bans (`forbidden_bytes.py`) across the full bench (standalone unit tests run via `make test-guard`).

---

## Research Process & Multi-Agent Engineering

I led this 18-day research program solo—owning the hypotheses, mathematical formulations, experimental designs, and every scientific pivot—while heavily using **frontier models from Anthropic and OpenAI as a coordinated engineering and verification team** (parallel implementation peers for C++/PyTorch infrastructure, an independent source-bound reviewer auditing code and empirical claims directly against Papers 1–4, and automated repository guards). Working at high velocity with frontier coding agents surfaced recurring failure modes—**proxy regression** back toward surface tokens or unpenalized matcher-dropout metrics, **precision shortcuts** like FP32 upcasting, and **sycophancy / defensive over-engineering**—which directly motivated `forbidden_bytes.py`, `verify_round2.py`, and carried-loss accounting. See **[`JOURNEY.md` § Retrospective](JOURNEY.md#retrospective-conducting-empirical-research-with-coding-agents)** for lessons learned and **[`journal/README.md`](journal/README.md#3-lab-roles-verification-workflow--multi-agent-glossary)** for the lab role glossary.

---

## Repository Map

| Directory / File | What to Read & Inspect |
| :--- | :--- |
| **[`JOURNEY.md`](JOURNEY.md)** | Start here for the 6-chapter chronological narrative (Nov 5–22, 2026) and the closing retrospective on conducting empirical research with frontier coding agents. |
| **[`reports/`](reports/README.md)** | Four working papers + Nov 15 bridge proposal: [Paper 1 (Single-Pair BF16 Edits)](reports/fixing-a-reasoning-mistake-by-editing-the-weights.pdf), [Paper 2 (Contextual Multi-Trace Guide)](reports/how-the-trace-derived-edit-works.pdf), [Paper 3 (60-Trace Path Overlap)](reports/how-the-path-edit-works.pdf), [Bridge Proposal & Assessment (DOCX)](reports/Iterative_Learning_from_Computational_Traces.docx), and **[Paper 4 (`descent-trace-repair/paper.pdf`)](reports/descent-trace-repair/paper.pdf)** plus [`milestone.md`](reports/descent-trace-repair/milestone.md) and [`transfer-protocol.md`](reports/descent-trace-repair/transfer-protocol.md). |
| **[`FINAL-EXPERIMENT.md`](FINAL-EXPERIMENT.md)** & **[`final-experiment/`](final-experiment/README.md)** | Equations for the 35-block / 105-merge learner plus mechanism dossiers ([`operand-identity.md`](final-experiment/operand-identity.md), [`descent-sign-mechanism.md`](final-experiment/descent-sign-mechanism.md), [`accumulating-repairs.md`](final-experiment/accumulating-repairs.md), [`joint-fit-and-transfer.md`](final-experiment/joint-fit-and-transfer.md)). |
| **[`code/signal-trace/`](code/signal-trace/ARCHITECTURE.md)** | C++ recorder (`recorder/`), complete Python/PyTorch trace alignment & weight-update engine (`pipeline/`), [`TRAINING.md`](code/signal-trace/TRAINING.md), [`pyproject.toml`](code/signal-trace/pyproject.toml), and runnable CPU `tests/`. |
| **[`studies/`](studies/README.md)** | Study reports and primary JSON declarations/results across [`01-early-exploration`](studies/01-early-exploration/README.md), [`02-physics-trace-observation`](studies/02-physics-trace-observation/README.md), [`03-round-1-single-pair-edits`](studies/03-round-1-single-pair-edits/README.md), and all **226 runs** in [`04-round-2-overlap`](studies/04-round-2-overlap/RUN-CATALOG.md) (including [interactive 105-merge trace explorer](studies/04-round-2-overlap/runs/one-pair-landscape-and-traces/explorer.html)). |
| **[`hypothesis/`](hypothesis/RESIDUAL-TRACE-THEORY.md)** & **[`journal/`](journal/README.md)** | Theoretical framework ([`RESIDUAL-TRACE-THEORY.md`](hypothesis/RESIDUAL-TRACE-THEORY.md), [`TWO-RESEARCH-FRONTS.md`](hypothesis/TWO-RESEARCH-FRONTS.md)) and all 40 dated lab memos + role glossary. |
| **[`records/`](records/README.md)** & **[`publications/findings/`](publications/findings/FINDINGS.md)** | Master empirical ledger ([`KNOWN.md`](records/KNOWN.md)), mechanistic retrospectives, causal transplant readbacks, [`2026-11-22-joint-loss-recheck`](records/development/2026-11-22-joint-loss-recheck/README.md), and unpooled score tables. |
| **[`apps/decode-machine/`](apps/decode-machine/README.md)** | Interactive 3D/2D Gemma 4 E2B execution & trace visualizer (TypeScript / Vite / WebGL, `48/48` tests passing). |
| **[`LICENSE`](LICENSE)** & **[`CITATION.cff`](CITATION.cff)** | MIT License, third-party open-source notices, and machine-readable citation metadata. |
