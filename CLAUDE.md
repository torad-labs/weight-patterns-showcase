# CLAUDE.md: Guide for Reviewers & AI Coding Agents

This is **`torad-labs/weight-patterns-showcase`**, the self-contained public repository for Marcos Damasceno's (Torad Labs) mechanistic interpretability and computational-trace weight-repair research on **Gemma 4 E2B** ([MIT License](LICENSE)).

## 1. Repository Layout & Scope

1. **What Ships in This Repository (`torad-labs/weight-patterns-showcase`):**
   - Deterministically exported with every file listed in [`MANIFEST.sha256`](MANIFEST.sha256) (except `MANIFEST.sha256` itself), zero symlinks, and zero broken relative links.
   - Contains all **4 working papers** plus the Nov 15 bridge proposal/assessment, mathematical specifications, the C++ `llama.cpp` stream recorder (`observe_stream.cpp` + GGML scheduler patches), the complete `code/signal-trace/` **PyTorch differentiable trace engine**, **runnable CPU unit tests**, primary declarations and results for **all 226 Round-2 runs** ([`RUN-CATALOG.md`](studies/04-round-2-overlap/RUN-CATALOG.md)), all **40 journal memos**, [`records/KNOWN.md`](records/KNOWN.md), and offline HTML visualizers.
2. **What Remains Outside Public Git (`torad-labs/weight-patterns`, private):**
   - Multi-GB `.f32` activation captures and `.gguf` checkpoints stored in Cloudflare R2 (`studies/04-round-2-overlap/input-archives/README.md`), plus the `reserved-40b` evaluation holdout folder (see [`2026-11-21-consolidation-takeover.md`](journal/2026-11-21-consolidation-takeover.md) regarding an unresolved Nov 21 automated directory-walk check on that folder; no evaluation has been run on `reserved-40b`).

---

## 2. Fast Orientation for Auditing or Summarizing This Repo

If a reviewer asks you to summarize the research, verify claims against primary data, or inspect the math and code, read in this order:

1. **[`JOURNEY.md`](JOURNEY.md)**: 6-chapter chronological log (Nov 5–22, 2026) connecting every pivot, repair, generalization panel, negative result, and the closing retrospective on directing frontier Anthropic and OpenAI coding agents.
2. **[`reports/README.md`](reports/README.md) & [`reports/descent-trace-repair/milestone.md`](reports/descent-trace-repair/milestone.md)**: Overview of Papers 1–4 + the Nov 15 bridge proposal, and the claim-by-claim evidence map for Paper 4 ([`reports/descent-trace-repair/paper.typ`](reports/descent-trace-repair/paper.typ) / [`paper.pdf`](reports/descent-trace-repair/paper.pdf)).
3. **[`FINAL-EXPERIMENT.md`](FINAL-EXPERIMENT.md)**: Equations for the 35-block / 105-merge normalized turn observable $T_{t,l,b} = u(h^{\text{out}}) - u(h^{\text{in}})$, DTW computational alignment, non-self witness radial calibration $\tau_{im}$, reverse-mode weight pullback across 538 native tensors, JVP response-scaled step length $\Delta W = \eta\alpha S$, and BF16/FP32 rounding-remainder installation.
4. **[`code/signal-trace/ARCHITECTURE.md`](code/signal-trace/ARCHITECTURE.md)**: Map of the C++ recorder (`recorder/`) and Python/PyTorch modules (`pipeline/`).

### Key Code Entry Points (`code/signal-trace/` & `records/`)
- [`code/signal-trace/recorder/observe_stream.cpp`](code/signal-trace/recorder/observe_stream.cpp) & [`signal_trace.hpp`](code/signal-trace/recorder/signal_trace.hpp): C++ native stream recorder hooking GGML scheduler callbacks to capture all 105 merges/token bit-identically to ordinary `llama.cpp` inference (`0` per-op `cudaDeviceSynchronize`), plus `stop_after_last_label` and `late_start` (`observe_stream.late.cpp`).
- [`code/signal-trace/pipeline/overlap/complete_overlap.py`](code/signal-trace/pipeline/overlap/complete_overlap.py): Monotone full-coverage DTW alignment using flattened cosine similarity over 105-merge turn fields (zero token, word, or position heuristics).
- [`records/development/2026-11-12-path-forward/torch_forward.py`](records/development/2026-11-12-path-forward/torch_forward.py), [`code/signal-trace/pipeline/weight_edit/full_parameters.py`](code/signal-trace/pipeline/weight_edit/full_parameters.py), [`full_graph.py`](code/signal-trace/pipeline/weight_edit/full_graph.py) & [`native_key_anchor.py`](code/signal-trace/pipeline/weight_edit/native_key_anchor.py): Gemma 4 E2B forward arithmetic (shared KV caches from layers `13/14` into `15–34`, local/global RoPE, GeGLU, per-layer-input gates), `601 -> 538` eligible tensor inventory, VJP/JVP wrapper, and FP16 pre-cache key anchoring (`-90.66%` finite-step error on the primary displacement, `-9.94%` on a second).
- [`code/signal-trace/pipeline/weight_edit/full_state.py`](code/signal-trace/pipeline/weight_edit/full_state.py): Mixed BF16/FP32 weight installer (`float64` accumulation `W + E + ΔW`, native writes across `278` BF16 and `260` FP32 tensors with carried `FP32` remainders `E`).
- [`code/signal-trace/pipeline/weight_edit/final_experiment.py`](code/signal-trace/pipeline/weight_edit/final_experiment.py), [`group_contrast_objective.py`](code/signal-trace/pipeline/weight_edit/group_contrast_objective.py), [`group_contrast_driver.py`](code/signal-trace/pipeline/weight_edit/group_contrast_driver.py), and [`common_prefix_driver.py`](code/signal-trace/pipeline/weight_edit/common_prefix_driver.py): One-pair and group-contrast training drivers.
- [`code/signal-trace/pipeline/research_guard/verify_round2.py`](code/signal-trace/pipeline/research_guard/verify_round2.py): Preregistration and provenance guard enforcing parameter lineage (`measured_from`, `derivation`, `scales_with`) and blocking token-supervision leakage (`forbidden_bytes.py`).

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

## 4. Reporting Conventions When Summarizing Results

When reporting or answering questions about results in this repository, preserve these conventions:

1. **Respect chronological timeline progression:** Papers 1–4, `FINAL-EXPERIMENT.md`, and the 40 memos in `journal/` were written on different dates as the method evolved (e.g., post-merge state targets on Nov 10 $\to$ branch turns with witness radii on Nov 15 $\to$ unbanded group contrasts on Nov 18). Keep populations and dates unpooled (see [`publications/findings/FINDINGS.md`](publications/findings/FINDINGS.md)).
2. **Distinguish fixed-history forecasts from free native generation:** A JVP linear forecast or fixed-history replay isolates local weight sensitivity; actual capability repair is established only after installing native BF16/FP32 weights into the GGUF model and decoding freely from the prompt (`llama.cpp` greedy decode).
3. **Separate the trace mechanisms and phases accurately:**
   - Checkpoint `390`'s `11 J -> 91 J` repair (`-78.09%` cumulative fresh trace loss: `70` matrices over updates `0–317` producing `91 J` by `174/175` and `-77.95%` drop, followed by all `538` eligible tensors over `317–390`; `1/8 -> 3/8` complete chains and `4/8 -> 3/8` downhill signs on the blind unseen panel in [`frozen-energy-generalization/paired-outcomes.json`](studies/04-round-2-overlap/runs/frozen-energy-generalization/paired-outcomes.json)) came from **whole-trace backpropagation** against fixed reference `expanded-004-h1` (`FINAL-EXPERIMENT.md`).
   - The **pre-choice descent detectors** (`15/15` wrong, `3/15` valid descents, `0/30` friction in local two-turn matching [`descent-operand-identity/result.json`](studies/04-round-2-overlap/runs/descent-operand-identity/result.json); `13/13` wrong, `0/15` valid at pre-choice in landmark consensus [`descent-landmark-route-comparison/registration-results.json`](studies/04-round-2-overlap/runs/descent-landmark-route-comparison/registration-results.json)) localize the error before `-` is emitted; when trained without the whole-trace anchor (`joint-stock-prechoice-training`), matcher dropout caused logged loss to drop `25.7%` while carried loss rose `+22.6%` ([`records/development/2026-11-22-joint-loss-recheck/README.md`](records/development/2026-11-22-joint-loss-recheck/README.md)). Combining the whole-trace anchor with carried-cell operation residuals (plus a matched token-loss baseline in [`journal/2026-11-22-token-loss-twin.md`](journal/2026-11-22-token-loss-twin.md)) is the active next step.
4. **Keep negative results alongside repairs:** Always report regressions and open limits alongside gains (e.g., `390`'s `1` regressed chain and `4/8 -> 3/8` downhill count; `390 -> 401` yielding `0/8` on the `±1 J` panel; single-anchor `B` and single-site `early-27` failing under free generation; no unified multi-signal training loop yet).
