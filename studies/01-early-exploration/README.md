# Study Phase 01: Early Weight & Activation Exploration

Executed November 5–6, 2026 on **Gemma 4 E2B** (`35` layers, `1,536` hidden width, shared K/V in layers `15–34`), Phase 01 comprises **14 foundational studies** plus two cross-study orientation documents. Together they establish how quantized parameters are stored and rounded, transition the experimental harness from the production FP4 (`TQ4 E2M1`) megakernel to an instrumented `llama.cpp` fork, demonstrate bidirectional behavioral steering via a real quantized query-projection edit, and clarify the conceptual boundary between perturbation footprints and distributed capability encodings.

## Synthesis & Orientation Documents

- [WEIGHT-SPACE-ORIENTATION.md](WEIGHT-SPACE-ORIENTATION.md) — Architectural and numerical map of Gemma 4 E2B (`317` text projection matrices, layers `0–14` K/V producers vs. layers `15–34` shared-KV readers, `6,144` vs. `12,288` GeGLU widths, `E2M1` + `UE8M0/UE4M3` scale bake) and negative result ruling out naive largest-product channel selection.
- [WEIGHT-INVESTIGATIONS.md](WEIGHT-INVESTIGATIONS.md) — Relational schema (`qgre.weight-investigation.v1` / `qgre.weight-case.v1`) linking exact token spans, diagnostic captures, architectural routes, and raw checkpoint slices without conflating static correlation with causation.

---

## Categorized Index of All 14 Early Studies

### 1. Decode Telemetry & Observational Tooling (QGRE `TQ4 E2M1`)

| Study | Primary Document | One-Line Empirical Finding |
| --- | --- | --- |
| `decode-observatory` | [run/ANALYSIS.md](decode-observatory/run/ANALYSIS.md) | Profiled `30,000` generated tokens across `187` turns (up to `39.5k` context), isolating linear decode latency growth (`2.725 ms / 1k tokens`, $R^2 = 0.9995$) to full-history attention tails (`390 ns/token/layer`) and uncovering a `CU_DEVICE_ATTRIBUTE_CLOCK_RATE` boost-ceiling bug in phase-clock validation. |
| `long-context-capture` | [run/long-instrumented/manifest.json](long-context-capture/run/long-instrumented/manifest.json) · [run/wrap-instrumented/manifest.json](long-context-capture/run/wrap-instrumented/manifest.json) | Captured multi-layer K/V cache views (`kvl0`, `kvl4`, `kvl9`, `kvl14`), launch tables, and phase-clock traces past the `1,024`-token sliding-window wrap boundary (`pos_01015` and `pos_01067`) to verify long-context cache indexing. |
| `ghost-inspection` | [STUDY.md](ghost-inspection/STUDY.md) · [run/analysis.md](ghost-inspection/run/analysis.md) | Built exact dyadic `E2M1`/`E4M3` activity inventory and `baseline / A / B / AB` nonadditivity comparator (`ghost-inspect`), exposing a `6,144`-vs-`12,288` GeGLU stride gap in layers `15–34` of early diagnostic taps. |

### 2. Static Alignment & Bounded FP4 Weight Interventions (QGRE `TQ4 E2M1`)

| Study | Primary Document | One-Line Empirical Finding |
| --- | --- | --- |
| `ghost-retrieval` | [STUDY.md](ghost-retrieval/STUDY.md) · [run/PLAN.md](ghost-retrieval/run/PLAN.md) | Nominated layer-9 attention head `H7` by static `Q/K` and `O/V^T` alignment; zeroing `H7` (`711,561` weights) caused `21/24` confirmation losses vs. `4–6` on norm-matched control heads (`H6, H3, H2`), proving strong selective sensitivity while showing downstream norm ratios lose proportionality (`0.506` at L9 → `0.970` at L34). |
| `mechanism-study` | [run/COLOR-PROBE.md](mechanism-study/run/COLOR-PROBE.md) · [WEIGHT-SPACE-ORIENTATION.md](WEIGHT-SPACE-ORIENTATION.md) | Showed failing `Lom = blue → 2` lookups already rank `blue` first in `5/8` layer-29 query heads; changing prompt wording (`value` → `color`) yielded `2` rescues and `1` regression (`3/6 → 4/6`), failing selective repair and ruling out simple retrieval-miss or single-channel (`4302` / `10884`) explanations. |
| `targeted-lookup` | [STUDY.md](targeted-lookup/STUDY.md) · [run/SELECTION.md](targeted-lookup/run/SELECTION.md) | Tested three `32`-column-aligned block-scale edits inside `H6`; halving the last `128` columns (`[3456, 3584)`, `4.28%` of input energy) repaired `Lom = blue` (`12/12` discovery, `0` control losses) but failed fresh confirmation (`30/36 → 27/36`, `4` regressions). |
| `individual-weight-edit` | [STUDY.md](individual-weight-edit/STUDY.md) · [run/PLAN.md](individual-weight-edit/run/PLAN.md) | Kept all `E8M0` block scales fixed and rounded `177,596` individual `E2M1` nibbles on GPU via `e4m3s_row_reduce_e2m1_encode_rne` (`3/4`, `5/8`, `1/2` fractions); the `3/4` edit fixed `Lom = blue`, but all three candidates regressed `7/56` baseline-correct development cases and were rejected. |

### 3. Failure-Family Diagnosis & Harness Transition (`QGRE` → `llama.cpp`)

| Study | Primary Document | One-Line Empirical Finding |
| --- | --- | --- |
| `behavior-pattern` | [run/STOPPED.md](behavior-pattern/run/STOPPED.md) · [run/PLAN.md](behavior-pattern/run/PLAN.md) | Exploratory shortened-wording batch (`What is <key>?`) halted early when prompt changes confounded the target failure mode; recorded the methodological correction that behavioral computation—not surface wording—defines cohort membership. |
| `lookup-failure-family` | [run/RESULTS.md](lookup-failure-family/run/RESULTS.md) · [run/FINISH-OBSERVATIONS.md](lookup-failure-family/run/FINISH-OBSERVATIONS.md) | Mapped `8/20` reproducible number-for-word lookup substitutions on original `TQ4` weights (`6/8` numbers did not match row indices, refuting a pure row-index bug) and isolated a matched `Nek = Silver` presentation pair (`24` in row 2 vs. `Silver` in row 3). |
| `llamacpp-lookup-comparison` | [run/RESULTS.md](llamacpp-lookup-comparison/run/RESULTS.md) · [run/PLAN.md](llamacpp-lookup-comparison/run/PLAN.md) | Evaluated the exact same `20` tokenized association prompts on `google_gemma-4-E2B-it-Q4_K_M.gguf` via `llama.cpp`, returning **20/20 concise correct answers** and proving the earlier numeric-substitution failures were specific to the `TQ4`/QGRE conversion path rather than intrinsic to Gemma 4 E2B. |

### 4. Mechanism-Oriented `llama.cpp` Editing & Conceptual Synthesis

| Study | Primary Document | One-Line Empirical Finding |
| --- | --- | --- |
| `llamacpp-preference-edit` | [run/REPORT.md](llamacpp-preference-edit/run/REPORT.md) · [run/PLAN.md](llamacpp-preference-edit/run/PLAN.md) | Applied a control-projected `Q4_K` query edit (`42,377` weights in `blk.29.attn_q.weight` head 5, cosine `0.7948` after rounding) for repeated-key tables: forward `+2` shifted first-occurrence preference `5/16 → 10/16` on discovery and `7/16 → 10/16` on withheld keys, reverse `-2` shifted last-occurrence `6/16 → 12/16`, and all `84` explicit/unique-key controls stayed unchanged. |
| `path-inspection` | [run/REPORT.md](path-inspection/run/REPORT.md) | Decomposed the layer-29 query edit's downstream propagation through layers `29–34` (where layer-34 FFN carried `80.27%` of the output difference projection) and formally withdrew an overclaim equating perturbation propagation with the underlying capability's encoded path. |
| `lookup-pattern-observation` | [run/REPORT.md](lookup-pattern-observation/run/REPORT.md) · [run/PREDICTIONS.md](lookup-pattern-observation/run/PREDICTIONS.md) | Confirmed `60/60` exact three-way output agreement on unedited `Q4_K_M` across `6` vocabulary families (`86.9%` vs. `0.8%` L29H5 diagnostic affinity on value vs. position queries), while showing that incoming pre-projection activations already carry the task contrast and literal copying overlaps value retrieval (`0.824` cosine). |
| `distributed-pathways` | [run/REFERENCE.md](distributed-pathways/run/REFERENCE.md) · [run/MODEL-HANDOFF.md](distributed-pathways/run/MODEL-HANDOFF.md) | Consolidated the theoretical distinction between persistent weight substrate, distributed capability organization, execution expression, and recorded trace observations—motivating the shift in [Phase 02](../02-physics-trace-observation/README.md) to multi-step physics reasoning where genuine reasoning failures occur in BF16. |
