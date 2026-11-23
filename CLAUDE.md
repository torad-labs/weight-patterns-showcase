# CLAUDE.md — Guide for Reviewers & AI Coding Agents

Welcome to **`weight-patterns-showcase`**, the curated, self-contained **Clean Showcase** of Marcos Damasceno's (Torad Labs) mechanistic interpretability and computational-trace weight-repair research on **Gemma 4 E2B**.

## 1. Repository Layout (Public Showcase & Private Full Archive)

1. **This Repository (`torad-labs/weight-patterns-showcase`):**
   - Deterministically exported with every file cryptographically bound in [`MANIFEST.sha256`](MANIFEST.sha256) and zero symlinks.
   - Contains all **4 papers**, mathematical specifications, the C++ `llama.cpp` stream recorder (`observe_stream.cpp`), the **538-tensor PyTorch differentiable trace learner** (`code/signal-trace/`), **runnable CPU unit tests**, curated landmark results across Studies 01–04, all 39 methodological journal memos, and interactive HTML visualizers.
2. **Full Historical Lab Archive (`torad-labs/weight-patterns`, private):**
   - Preserves the complete unedited laboratory bench (all **226 Round-2 run directories**, raw intermediate diagnostics, and R2 bulk-tensor archive receipts) while keeping the sealed `reserved-40b` evaluation holdout and multi-GB `.f32`/`.gguf` artifacts out of public git.
   - Whenever a historical note or provenance field cites an intermediate run not bundled in this showcase, see [`studies/04-round-2-overlap/RUN-CATALOG.md`](studies/04-round-2-overlap/RUN-CATALOG.md) for the full index.

---

## 2. Fast Orientation for Summarizing or Auditing This Repo

If a reviewer asks you to summarize the research, verify claims against primary data, or inspect the math and code, read in this order:

1. **[`JOURNEY.md`](JOURNEY.md)** — Complete 6-chapter chronological arc (Nov 4–22, 2026), connecting every pivot, positive repair, generalization panel, and falsified hypothesis.
2. **[`reports/README.md`](reports/README.md) & [`reports/descent-trace-repair/milestone.md`](reports/descent-trace-repair/milestone.md)** — Overview of all 4 manuscripts and the claim-by-claim evidence map for Paper 4 ([`reports/descent-trace-repair/paper.typ`](reports/descent-trace-repair/paper.typ) / [`paper.pdf`](reports/descent-trace-repair/paper.pdf)).
3. **[`FINAL-EXPERIMENT.md`](FINAL-EXPERIMENT.md)** — Exact equations for the 35-block / 105-merge normalized turn observable $T_{t,l,b} = u(h^{\text{out}}) - u(h^{\text{in}})$, DTW computational alignment, non-self witness radial calibration $\tau_{im}$, reverse-mode weight pullback across 538 native tensors, JVP response-scaled step length $\Delta W = \eta\alpha S$, and BF16/FP32 rounding-remainder installation.
4. **[`code/signal-trace/ARCHITECTURE.md`](code/signal-trace/ARCHITECTURE.md)** — Map of the C++ recorder (`recorder/`) and Python/PyTorch modules (`pipeline/`).

### Key Code Entry Points (`code/signal-trace/`)
- **C++ Native Stream Recorder:** [`code/signal-trace/recorder/observe_stream.cpp`](code/signal-trace/recorder/observe_stream.cpp) & [`signal_trace.hpp`](code/signal-trace/recorder/signal_trace.hpp) — GGML scheduler callbacks capturing all 105 merges/token bit-identically to ordinary `llama.cpp` inference (`0` per-op `cudaDeviceSynchronize`), plus `stop_after_last_label` and `late_start` (`observe_stream.late.cpp`).
- **DTW 105-Merge Computational Alignment:** [`code/signal-trace/pipeline/overlap/complete_overlap.py`](code/signal-trace/pipeline/overlap/complete_overlap.py) — Monotone full-coverage DTW alignment using flattened cosine similarity over 105-merge turn fields (zero token/word/position heuristics).
- **35-Block / 538-Tensor Differentiable Native Graph:** [`code/signal-trace/pipeline/weight_edit/full_graph.py`](code/signal-trace/pipeline/weight_edit/full_graph.py) & [`native_key_anchor.py`](code/signal-trace/pipeline/weight_edit/native_key_anchor.py) — Exact Gemma 4 E2B forward graph (shared KV caches from layers `13/14` into `15–34`, local/global RoPE, GeGLU, per-layer-input gates) anchored to native captures with FP16 pre-cache key anchoring (`-90.66%` finite-step error).
- **Mixed BF16/FP32 State & Rounding-Remainder Installer:** [`code/signal-trace/pipeline/weight_edit/full_state.py`](code/signal-trace/pipeline/weight_edit/full_state.py) — `float64` accumulation (`W + E + ΔW`) with native BF16 (`278` tensors) and FP32 (`260` tensors) writes plus carried `FP32` remainders `E`.
- **One-Pair & Group-Contrast Training Drivers:** [`code/signal-trace/pipeline/weight_edit/final_experiment.py`](code/signal-trace/pipeline/weight_edit/final_experiment.py), [`group_contrast_objective.py`](code/signal-trace/pipeline/weight_edit/group_contrast_objective.py), [`group_contrast_driver.py`](code/signal-trace/pipeline/weight_edit/group_contrast_driver.py), and [`common_prefix_driver.py`](code/signal-trace/pipeline/weight_edit/common_prefix_driver.py).
- **Preregistration & Provenance Guard:** [`code/signal-trace/pipeline/research_guard/verify_round2.py`](code/signal-trace/pipeline/research_guard/verify_round2.py) — Enforces parameter derivation lineage (`measured_from`, `derivation`, `scales_with`) and blocks token-supervision leakage (`forbidden_bytes.py`).

---

## 3. Runnable CPU Verification Commands

Always keep `CUDA_VISIBLE_DEVICES=""` set when running checks so tests remain strictly on CPU.

| Check | Command | Runtime & Dependencies |
| :--- | :--- | :--- |
| **Verify showcase SHA-256 manifest** | `sha256sum -c MANIFEST.sha256` | `<1s`, coreutils only |
| **Run guard & provenance unit tests** | `make test-guard` | `<1s`, Python 3 stdlib only |
| **Rebuild all 6 Paper 4 SVG figures** | `make figures` | `<1s`, Python 3 stdlib only (reads included JSON evidence) |
| **Verify prospective forecast scorer** | `make score-protocol` | `<1s`, Python 3 stdlib only ([`score_forecasts.py`](reports/descent-trace-repair/score_forecasts.py)) |
| **Run full synthetic 35-block CPU suite** | `make test-cpu` | `~10s`, Python 3.12 + PyTorch + NumPy + Matplotlib (auto-detects `python3` or `uv`) |
| **Run every standalone check at once** | `make verify` | `~12s`, runs checksums + guard + protocol + figures + CPU suite |
| **Typeset Paper 4 PDF & HTML** | `make paper` | `~3s`, uses `uv run --no-project --with typst==0.15.2` |

---

## 4. Scientific & Epistemic Conventions When Summarizing Results

When reporting or answering questions about results in this repository, preserve the lab's strict epistemic standards:

1. **Keep denominators and populations unpooled:** Never merge discovery/fitted counts with held-out transfer panels, or Round-1 28-question baselines with Round-2 30+30 overlap cohorts (see [`publications/findings/FINDINGS.md`](publications/findings/FINDINGS.md)).
2. **Distinguish fixed-history forecasts from free native generation:** A JVP linear forecast or fixed-history replay isolates local weight sensitivity; actual capability repair is established *only* after installing native BF16/FP32 weights into the GGUF model and decoding freely from the prompt (`llama.cpp` greedy decode).
3. **Separate the two trace mechanisms clearly:**
   - Checkpoint `390`'s `11 J → 91 J` repair (`-78.09%` fresh trace loss, `1/8 → 3/8` blind unseen panel in [`frozen-energy-generalization/result.json`](studies/04-round-2-overlap/runs/frozen-energy-generalization/result.json)) was produced by **whole-trace backpropagation** against fixed reference `expanded-004-h1` (`FINAL-EXPERIMENT.md`).
   - The **pre-choice descent signal** (`15/15` local, `13/13` landmark wrong descents vs. `0/30` friction false alarms in [`descent-operand-identity/result.json`](studies/04-round-2-overlap/runs/descent-operand-identity/result.json)) localizes operand identity before the sign token; when trained in isolation without the whole-trace anchor (`joint-stock-prechoice-training`), matcher dropout caused logged loss to drop `25.7%` while carried loss rose `+22.6%` ([`records/development/2026-11-22-joint-loss-recheck/README.md`](records/development/2026-11-22-joint-loss-recheck/README.md)). Combining the whole-trace anchor with carried-cell operation residuals is the active next experiment.
4. **Preserve honest negative results:** Always report regressions alongside repairs (e.g., `390` repairing 2 errors + 1 cap while regressing 1 baseline success on the 8-question panel; `390 → 401` prefix continuation yielding `0/8` new unseen repairs; single-anchor `B` and single-site `early-27` failing under free generation).
