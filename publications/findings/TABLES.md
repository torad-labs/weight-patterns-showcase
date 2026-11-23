# Original tables, collected by source

Extracted verbatim from the application research documents and the top-level Markdown documents in the original 18 studies, with later study tables appended through the current 22-study collection. These include plans, metadata, earlier findings, and tables repeated in later summaries. They are not one merged dataset. Follow the source for definitions, units, and scope; use [Findings](FINDINGS.md) for the current synthesis. Navigation-only edits to three application documents are resolved to their saved original versions so line references remain exact.

## 1. The visual story

Source: [apps/decode-machine/EVIDENCE-ONTOLOGY.md](../../apps/decode-machine/EVIDENCE-ONTOLOGY.md), line 52.

| Question | Representation | Interaction |
| --- | --- | --- |
| Where am I? | Recording → layer → operation → selected worker; an enclosing-operation branch for nested stamps | Keep selection synchronized with the machinery |
| What occupied this worker? | Chronological strip of resolved work, wait, mixed and unassigned spans; the selected observation outlined | Select a span to seek; use the existing clock |
| What comes before and after? | Connected operation nodes; solid lines for recorded order, enclosing bracket for nesting, dashed missing-producer branch | Follow a node without moving time |
| Do workers arrive together? | One selectable dot per valid observation, positioned by exit time; first/median/last landmarks | Select a worker; explicitly seek to the last exit |
| Does this appear in both recordings? | Aligned median-to-last ranges for the same source record in both captures | Inspect either capture; do not label this A/B |
| What is measured and what is missing? | Connected architecture areas with measured, derived and unavailable evidence states | Select an area to reveal the distinguishing measurement |
| What should we do next? | Observation → alternative explanations → required evidence → conditional experiment → validation | Choose an architecture area; export the same investigation |

## 2. Entities and identity

Source: [apps/decode-machine/EVIDENCE-ONTOLOGY.md](../../apps/decode-machine/EVIDENCE-ONTOLOGY.md), line 68.

| Entity table | Identity and purpose |
| --- | --- |
| runs | One capture; token position, clock domain, tape/table hashes, bounds and original stamp semantics |
| layers | A captured layer in one run; kind, dimensions and K/V owner index |
| operations | A source record in one run; role, taps, kind, description and anomalies |
| workers | A logical block in one run, never implicitly a physical SM |
| observations | One raw tuple linked to its operation and worker, including invalid/zero tuples with validity marked |
| segments | Resolved, clipped intervals used for the selected worker's totals, linked back to source observations |
| areas | Architectural domains: host, coordination, memory, transfer, execution, resources, numerical correctness |
| sources | Local NVIDIA reference chapters and official documentation with scope and provenance |
| checkpoints | Conceptual stages to investigate in an architecture area; their event timings are not captured |
| measurements | Derived values with units, method, input references, scope and validity |
| hypotheses | Competing explanations; unproven unless additional evidence discriminates them |
| requirements | The observation needed to distinguish a hypothesis, tool, scope and availability |
| experiments | Conditional changes, predicted direction, companion metrics and refutation criteria |

## 3. Architectural domains and discriminating evidence

Source: [apps/decode-machine/EVIDENCE-ONTOLOGY.md](../../apps/decode-machine/EVIDENCE-ONTOLOGY.md), line 125.

| Area | Visible clue, not diagnosis | Evidence to acquire | Conditional action and refutation |
| --- | --- | --- | --- |
| Host and submission | Gaps in whole-request progress; current layer-only capture cannot see them | Nsight Systems CPU/CUDA/NVTX timeline, launch/queue/execution split and request boundaries | If host round trips dominate, change host scheduling or graph preparation. Refuted if wall time does not improve or the gap is GPU queueing |
| Coordination and assignment | Long software waits or separated worker exits | Producer/consumer IDs, readiness generation, equal-work grouping, all barrier arrivals, aligned phase clocks | If unequal assigned work delays a required producer, change assignment in policy. Refuted if critical completion is unchanged or another group slows |
| Memory and reuse | Long weight-consuming region | Phase-local DRAM/L2 bytes and throughput, demand-only hit rate, sectors, working set, traffic class and clock | If redundant demand traffic is proven, change reuse/layout. If bandwidth is saturated, remove bytes. Refuted if bytes fall but wall time does not or read amplification rises |
| TMA and staging | Waits near a transfer-consuming region | Issued transfer ID, descriptor/box, bytes, completion generation, consumer acquire, release and slot reuse | If exposed delivery latency is proven, change overlap/depth in policy. Refuted by more occupancy pressure, later completion, incorrect slot reuse or increased wall time |
| Warp issue and execution | Long arithmetic or mixed region | SchedulerStats, SourceCounters, WarpStateStats, per-pipe throughput, PC/SASS correlation and assigned work | If missed issue slots dominate, investigate the producer of the stalled operand. If a pipe is saturated, remove work on that pipe. Refuted if only samples move while duration does not |
| Registers and shared memory | Uneven progress or inability to keep independent work ready | Launch limits, compiled register/SMEM use, actual spills and bank-conflict wavefronts, eligible/issued warps | If resource pressure is limiting, shorten live ranges or change staging/geometry. Refuted by extra spills, lower issue, or slower completion |
| Numerical correctness | A faster candidate changes the emitted answer or state | Existing exact cases, equation invariants, deterministic replay and the production coherence gate | Keep the approved low-precision carrier and megakernel path. A correctness regression rejects the optimization; wider recompute is not the oracle |

## 4. Relational artifact: `qgre.ghost-observations.v1`

Source: [apps/decode-machine/GHOST-INSTRUMENTATION.md](../../apps/decode-machine/GHOST-INSTRUMENTATION.md), line 37.

| Table | Identity and joins | What it establishes |
| --- | --- | --- |
| `captures` | Original `id`, reply and token correspondence | Which saved execution was sampled; the original correspondence caveats remain |
| `observations` | `id`, `captureId`, `groupId`, `activationId` | A measured state of one group, and exact byte ranges for inspecting it |
| `groups` | `id`, `matrixId`, half-open `coordinateRange` | A fixed identity across captures, with activity/state counts and coverage |
| `shortlists` | `layer`, `matrixId`, ordered `groupIds` | Which groups to inspect first under one explicit observational ordering |
| `activations` | Original `id`, code/scale paths, SHA-256, offsets | Recoverable raw recorded values, interpreted relative to the input study directory |
| `matrices` | Original `id`, `tensorIds`, input/output | The expected architectural relationship, not a measured physical transfer |
| `tensors`, `sources` | Original identities | Named parameter storage and the inherited evidence inventory |
| `relations` | `from`, `to`, `kind`, `basis` | Explicit architectural group-to-matrix edges |

## 5. The candidate and its counterfactuals

Source: [apps/decode-machine/GHOST-RETRIEVAL-EXPERIMENT.md](../../apps/decode-machine/GHOST-RETRIEVAL-EXPERIMENT.md), line 23.

| Head | Halved by half edit | Removed at the scale floor | Already zero | Removed by zero edit |
|---|---:|---:|---:|---:|
| H7 candidate | 711,471 | 90 | 74,871 | 711,561 |
| H6 control | 709,530 | 216 | 76,686 | 709,746 |
| H3 control | 705,399 | 120 | 80,913 | 705,519 |
| H2 control | 710,654 | 182 | 75,596 | 710,836 |

## 6. Behavioral results

Source: [apps/decode-machine/GHOST-RETRIEVAL-EXPERIMENT.md](../../apps/decode-machine/GHOST-RETRIEVAL-EXPERIMENT.md), line 46.

| Checkpoint / intervention | Discovery / 12 | Confirmation / 24 | Unrelated controls / 12 |
|---|---:|---:|---:|
| Original | 9 | 24 | 12 |
| Unchanged republished checkpoint | 9 | 24 | 12 |
| H7 candidate, quantized half-scale | 6 | 17 | 12 |
| H7 candidate, zero | 2 | 3 | 10 |
| H6 control, quantized half-scale | 11 | 24 | 12 |
| H6 control, zero | 4 | 18 | 12 |
| H3 control, quantized half-scale | 6 | 22 | 12 |
| H3 control, zero | 2 | 20 | 12 |
| H2 control, quantized half-scale | 4 | 22 | 12 |
| H2 control, zero | 5 | 19 | 11 |

## 7. How the edit propagates

Source: [apps/decode-machine/GHOST-RETRIEVAL-EXPERIMENT.md](../../apps/decode-machine/GHOST-RETRIEVAL-EXPERIMENT.md), line 71.

| Recorded stage (zero-based layer) | Half/full change length | Direction agreement |
|---|---:|---:|
| Layer 9 output projection | 0.5055 | 0.9811 |
| Layer 9 post-attention | 0.5290 | 0.9377 |
| Layer 9 output | 0.6025 | 0.8179 |
| Layer 12 output | 0.5138 | 0.3143 |
| Layer 20 output | 0.8250 | 0.4371 |
| Layer 32 output | 0.9218 | 0.7071 |
| Layer 34 output | 0.9698 | 0.9365 |
| Final normalized state | 0.8915 | 0.8072 |

## 8. Individual quantized weight editing

Source: [apps/decode-machine/INDIVIDUAL-WEIGHT-EXPERIMENT.md](../../apps/decode-machine/INDIVIDUAL-WEIGHT-EXPERIMENT.md), line 27.

| Requested fraction of original | Changed codes | Unchanged codes | Nonzero values rounded to zero | Changed scales |
| --- | ---: | ---: | ---: | ---: |
| 3/4 | 107,702 | 69,894 | 0 | 0 |
| 5/8 | 140,952 | 36,644 | 0 | 0 |
| 1/2 | 177,596 | 0 | 36,644 | 0 |

## 9. Individual quantized weight editing

Source: [apps/decode-machine/INDIVIDUAL-WEIGHT-EXPERIMENT.md](../../apps/decode-machine/INDIVIDUAL-WEIGHT-EXPERIMENT.md), line 39.

| Model / requested fraction | Target answer (wanted `blue`) | Correct / 66 | Original-correct preserved / 56 | Newly correct | Regressed | Decision |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Original | `2` | 56 | 56 | — | — | Retained |
| 3/4 | `blue` | 54 | 49 | 5 | 7 | Rejected |
| 5/8 | `green` | 51 | 49 | 2 | 7 | Rejected |
| 1/2 | `2` | 54 | 49 | 5 | 7 | Rejected |

## 10. Coverage

Source: [apps/decode-machine/INTERNAL-EQUIPMENT.md](../../apps/decode-machine/INTERNAL-EQUIPMENT.md), line 29.

| Operation | Physical equipment and behavior |
| --- | --- |
| Gather and layout | Forks and rollers gather a carrier into the next machine's lanes. |
| Weight read | Amber balls remain in their bins; read copies feed the press. |
| Matrix multiplication | Crossing operand beds and a descending platen combine input and weights; the receiving fork gathers output. |
| Normalization | Balance pans and a gain mechanism represent measuring magnitude and rescaling. |
| Position rotation | Paired turntables rotate pairs together using position settings. |
| Attention score | A comb compares the query with retained key material. |
| Probability and value mix | A maximum selector, exponential cam, parallel value press and sum beam feed a final gathering mechanism. |
| GEGLU | Cubic, coefficient, tanh and half-gain mechanisms shape the gate. The up branch bypasses GELU and joins at multiplication. |
| Residual addition | Two arriving carriers meet at a combining beam. |
| Cache | Retained racks store owner-layer context; shared layers use owner couplings and read forks. |
| Dependency | A counter mechanism holds a latch until its recorded dependency completes. |

## 11. What the model did

Source: [apps/decode-machine/PHYSICS-TRACE-BF16.md](../../apps/decode-machine/PHYSICS-TRACE-BF16.md), line 22.

| Observation across the same questions | Q4_K_M | BF16 |
| --- | ---: | ---: |
| Correct explicit final verdict, all cases | 8/24 | 13/24 |
| Complete correct arithmetic calculation and verdict | 1/6 | 4/6 |
| Complete correct physical numerical chain and verdict, including paraphrases | 0/18 | 2/18 |
| Physically consistent completed explanation | 0/18 | 0/18 |
| Responses reaching the 768-token cap | 10/24 | 8/24 |

## 12. A correct partial calculation with a controlled comparison

Source: [apps/decode-machine/PHYSICS-TRACE-BF16.md](../../apps/decode-machine/PHYSICS-TRACE-BF16.md), line 64.

| Observable calculation moment | Decision, zero-based | FF14 relative ratio | FF15 relative ratio |
| --- | ---: | ---: | ---: |
| Adds downhill gain: 54 + 30 | 166 | 0.493 | 1.471 |
| Produces 84 J | 177 | 0.546 | 1.425 |
| Subtracts friction: 84 − 24 | 222 | 0.527 | 1.425 |
| Produces 60 J | 233 | 0.583 | 1.281 |

## 13. How broadly the earlier progression persists

Source: [apps/decode-machine/PHYSICS-TRACE-BF16.md](../../apps/decode-machine/PHYSICS-TRACE-BF16.md), line 95.

| Pair | Q4 progression / eligible | BF16 progression / eligible | BF16 median FF14 ratio | BF16 median FF15 ratio |
| --- | ---: | ---: | ---: | ---: |
| cart | 94/94 | 297/297 | 0.510 | 1.537 |
| sled | 104/104 | 306/306 | 0.502 | 1.519 |
| bead | 4/4 | 20/20 | 0.482 | 1.530 |
| coaster | 223/223 | 270/270 | 0.502 | 1.547 |
| block | 4/4 | 253/254 | 0.526 | 1.492 |
| trolley | 251/251 | 307/307 | 0.503 | 1.489 |
| cart-paraphrase | 17/17 | 35/35 | 0.510 | 1.613 |
| coaster-paraphrase | 16/16 | 50/50 | 0.497 | 1.606 |
| block-paraphrase | 40/40 | 47/47 | 0.515 | 1.565 |
| cart-arithmetic | 25/25 | 72/72 | 0.541 | 1.682 |
| coaster-arithmetic | 34/34 | 25/25 | 0.563 | 1.714 |
| block-arithmetic | 34/34 | 25/25 | 0.585 | 1.657 |

## 14. The connected progression

Source: [apps/decode-machine/PHYSICS-TRACE-FOLLOWUP.md](../../apps/decode-machine/PHYSICS-TRACE-FOLLOWUP.md), line 44.

| Pair | Kind | Predicted sequence / eligible | Median FF14 ratio | Median FF15 ratio |
| --- | --- | ---: | ---: | ---: |
| cart | physics | 94/94 | 0.559 | 1.506 |
| sled | physics | 104/104 | 0.567 | 1.515 |
| bead | physics | 4/4 | 0.506 | 1.510 |
| coaster | physics | 223/223 | 0.556 | 1.497 |
| block | physics | 4/4 | 0.493 | 1.405 |
| trolley | physics | 251/251 | 0.574 | 1.513 |
| cart-paraphrase | paraphrase | 17/17 | 0.535 | 1.504 |
| coaster-paraphrase | paraphrase | 16/16 | 0.552 | 1.530 |
| block-paraphrase | paraphrase | 40/40 | 0.545 | 1.531 |
| cart-arithmetic | arithmetic | 25/25 | 0.563 | 1.625 |
| coaster-arithmetic | arithmetic | 34/34 | 0.556 | 1.690 |
| block-arithmetic | arithmetic | 34/34 | 0.568 | 1.682 |

## 15. The connected progression

Source: [apps/decode-machine/PHYSICS-TRACE-FOLLOWUP.md](../../apps/decode-machine/PHYSICS-TRACE-FOLLOWUP.md), line 76.

| Layer | Absolute separation ratio | State-length growth | Update alignment cosine | Before/after difference cosine |
| --- | ---: | ---: | ---: | ---: |
| 14 | 1.721–1.942 | 3.284–3.387 | -0.067–0.029 | 0.514–0.552 |
| 15 | 3.508–4.041 | 2.382–2.547 | -0.041–0.090 | 0.232–0.360 |

## 16. What the model actually did

Source: [apps/decode-machine/PHYSICS-TRACE-FOLLOWUP.md](../../apps/decode-machine/PHYSICS-TRACE-FOLLOWUP.md), line 93.

| Kind | Cases | Correct explicit verdicts | Complete correct solutions | Capped |
| --- | ---: | ---: | ---: | ---: |
| physics | 12 | 3 | 0 | 7 |
| paraphrase | 6 | 2 | 0 | 3 |
| arithmetic | 6 | 3 | 1 | 0 |

## 17. Connecting concrete calculation moments

Source: [apps/decode-machine/PHYSICS-TRACE-FOLLOWUP.md](../../apps/decode-machine/PHYSICS-TRACE-FOLLOWUP.md), line 125.

| Execution | Observable progression | Recorded decision indices (zero-based) |
| --- | --- | --- |
| sled-base | Adds downhill gain; obtains 90 J; rejects 50−40; later uses the rejected 10 J | 272, 283, 331, 538 |
| cart-paraphrase-changed | States addition of downhill gain; later substitutes subtraction | 387, 466 |
| cart-arithmetic-base | Calculates 60 and 40 correctly; answers NO incorrectly | 120 |
| cart-arithmetic-changed | Calculates 60 and 80 correctly; answers NO correctly | 79 |

## 18. The observable target

Source: [apps/decode-machine/PHYSICS-TRACE-INTERVENTION.md](../../apps/decode-machine/PHYSICS-TRACE-INTERVENTION.md), line 30.

| Recipient | Correct available / required / verdict | Recorded baseline |
| --- | --- | --- |
| 1 m climb, `block-base` | 60 J / 30 J / YES | Correct numerical chain and final triple; top/bottom prose labels are reversed |
| 3 m climb, `block-changed` | 60 J / 90 J / NO | 60 J / **120 J** / NO; it incorrectly adds the earlier descent to the final climb |

## 19. The additional numerical variants

Source: [apps/decode-machine/PHYSICS-TRACE-INTERVENTION.md](../../apps/decode-machine/PHYSICS-TRACE-INTERVENTION.md), line 61.

| Pair | Correct available energy | Required energy, base / changed | Shared generated prefix | Observation |
| --- | ---: | ---: | ---: | --- |
| 2 kg, speed 6 m/s, loss 16 J | 40 J | 20 / 60 J | 18 tokens | Base gets initial kinetic energy wrong, later has 2 J, and hits the token cap; changed has the correct final triple but inconsistent descent accounting |
| 4 kg, speed 4 m/s, loss 12 J | 60 J | 40 / 80 J | 88 tokens | Both final triples are correct; changed nevertheless makes an extra 40 J subtraction and derives an incorrect speed |

## 20. What the swaps changed

Source: [apps/decode-machine/PHYSICS-TRACE-INTERVENTION.md](../../apps/decode-machine/PHYSICS-TRACE-INTERVENTION.md), line 75.

| Recipient | Swap time | Difference in substituted branch, relative to recipient branch | Changed vocabulary-score decisions | Final answer |
| --- | --- | ---: | ---: | --- |
| Correct 1 m case | After 60 J, prediction 235 | 1.777% | Exactly 235 | Unchanged: 60 / 30 / YES |
| Correct 1 m case | First prediction, 0 | 2.939% | Exactly 0 | Unchanged: 60 / 30 / YES |
| Failing 3 m case | After 60 J, prediction 235 | 1.776% | Exactly 235 | Unchanged: 60 / 120 / NO |
| Failing 3 m case | First prediction, 0 | 2.940% | Exactly 0 | Unchanged: 60 / 120 / NO |

## 21. What the swaps changed

Source: [apps/decode-machine/PHYSICS-TRACE-INTERVENTION.md](../../apps/decode-machine/PHYSICS-TRACE-INTERVENTION.md), line 91.

| Recorded boundary | Difference from the unmodified recipient, relative L2 |
| --- | ---: |
| Layer 15 exit, preceding residual | Exactly zero |
| Substituted normalized attention branch 16 | 1.7761% |
| Native attention addition 16 | 0.5605% |
| Layer 16 exit | 0.6164% |
| Layer 17 exit | 0.5449% |
| Layer 34 exit | 0.0471% |
| Final normalized state | 0.1038% |
| Each retained boundary at prediction 236 | Byte-identical to baseline |

## 22. What we asked

Source: [apps/decode-machine/PHYSICS-TRACE-PILOT.md](../../apps/decode-machine/PHYSICS-TRACE-PILOT.md), line 26.

| Server pilot format | Correct YES/NO verdict | Reached output limit | Interpretation |
| --- | ---: | ---: | --- |
| Verdict first; 256-token cap | 9/18 | 18/18 | Every answer begins NO; no completed solutions |
| Calculation before verdict; 768-token cap | 11/18 | 1/18 | Most responses omit the requested numerical calculation |
| Explicit numerical calculation; 768-token cap | 6/18 | 10/18 | One complete correct calculation; one correct quantitative progression cut off before verdict |

## 23. Native representation and checkpoint integrity

Source: [apps/decode-machine/PHYSICS-TRACE-WEIGHT-EDIT.md](../../apps/decode-machine/PHYSICS-TRACE-WEIGHT-EDIT.md), line 70.

| Measurement | Result |
| --- | ---: |
| Values in the selected matrix | 3,145,728 |
| Values whose stored representation changed | 200,965 (6.3885% of this matrix) |
| Changed bytes | 204,565 |
| Intended change / original matrix norm | 0.036654% |
| Rounded change / original matrix norm | 0.029656% |
| Intended versus realized matrix-change cosine | 0.702831 |
| Remaining distance to donor projection at the fixed source input, CPU equation diagnostic | 44.4103% of the original distance |
| Zero-update BF16 decode/re-encode | Every value byte-identical |
| Bytes compared across the complete original and edited GGUF | 9,311,304,896 |
| Bytes changed outside the selected matrix | **0** |

## 24. All 28 behavioral comparisons

Source: [apps/decode-machine/PHYSICS-TRACE-WEIGHT-EDIT.md](../../apps/decode-machine/PHYSICS-TRACE-WEIGHT-EDIT.md), line 102.

| Measure | Original | Edited |
| --- | ---: | ---: |
| Complete correct numerical calculation | 7 / 28 | **9 / 28** |
| Correct explicit YES/NO verdict | 16 / 28 | **16 / 28** |
| Reached the 768-token cap | 9 / 28 | 9 / 28 |
| Token-for-token identical to original | — | 18 / 28 |

## 25. All 28 behavioral comparisons

Source: [apps/decode-machine/PHYSICS-TRACE-WEIGHT-EDIT.md](../../apps/decode-machine/PHYSICS-TRACE-WEIGHT-EDIT.md), line 115.

| Case | Original available / required / verdict | Edited | Interpretation |
| --- | --- | --- | --- |
| Original 3 kg block, 3 m climb | 60 / **120** / NO | 60 / **90** / NO | Targeted numerical error repaired; the earlier 1 m descent is no longer added to the 3 m climb |
| Paired 3 kg block, 1 m climb | 60 / 30 / YES | Identical response | Existing numerical success preserved |
| Sled, 1 m climb | **0** / 40 / **NO** | **80** / 40 / **YES** | Correctly adds the 40 J downhill gain, then subtracts the 10 J loss |
| 4 kg block, 2 m climb | 60 / 80 / NO | 60 / **40** / **YES** | New failure: subtracts the earlier 1 m descent from the destination height |

## 26. Following the effect

Source: [apps/decode-machine/PHYSICS-TRACE-WEIGHT-EDIT.md](../../apps/decode-machine/PHYSICS-TRACE-WEIGHT-EDIT.md), line 151.

| Boundary | Difference from original, relative L2 |
| --- | ---: |
| Input to the edited projection | Exactly zero |
| Raw projection output | 0.9191% |
| Normalized attention contribution | 1.0273% |
| Layer 16 exit | 0.3570% |
| Layer 34 exit | 0.02357% |
| Final normalized state | 0.05797% |

## 27. Following the effect

Source: [apps/decode-machine/PHYSICS-TRACE-WEIGHT-EDIT.md](../../apps/decode-machine/PHYSICS-TRACE-WEIGHT-EDIT.md), line 167.

| Candidate token | Original logit | Edited logit |
| --- | ---: | ---: |
| ` reached` | **16.181353** | 16.177919 |
| ` gained` | 16.174959 | **16.186314** |

## 28. What each producer must establish

Source: [apps/decode-machine/PROFILING-LENSES.md](../../apps/decode-machine/PROFILING-LENSES.md), line 68.

| Lens | Required relationship | Do not infer |
| --- | --- | --- |
| Dependencies | Producer task → generation-specific readiness event → consumer task; explicit common clock | A global critical path or recoverable latency from the last observed input |
| Instruction supply | Compiled location and instruction/fetch evidence → task → instrument/pass | Instruction-cache misses from a long task interval alone |
| Useful work | Valid work and issued work in the same counting unit; requested and transferred bytes at a named boundary | Padding cost from matrix shape alone, or DRAM traffic from logical weight bytes |
| Buffer lifetime | One allocation generation, capacity, request, ready, first use, last use and release/observation end | An actual release when an object is retained, or saved time from byte-nanoseconds |
| Supply/activity | Metric, units and observation interval → collection pass | Simultaneity between separate replays, or zero activity in unsampled gaps |
| Values | Named boundary, shared bin edges, bin counts, zero and saturation counts | A speedup from sparsity without an execution path that exploits it |
| Tokens | Position, sequence, interval, context and batch → instrument; clocks/power/limits when captured | Continuity between independently recorded positions or causality from correlation |
| Instruments | Every row → method, source, pass, cache treatment, clock treatment, sampling width and binary identity where known | Production timing from a diagnostic replay |

## 29. The other analogies

Source: [references/history/pre-organization-RESIDUAL-TRACE-THEORY.md](../../references/history/pre-organization-RESIDUAL-TRACE-THEORY.md), line 162.

| Analogy | Concept it conveys |
|---|---|
| Physics as the trace | A learned capability is recognizable through coherent transformations across changing situations; the correct numerical answer alone is only an endpoint. |
| Dotted line or jigsaw | Enough surrounding functional context can make a missing or defective continuation understandable. |
| Sine-wave approximation and the golden ratio | The organization of changes can reveal a pattern that individual numbers do not make evident. These are explanatory examples, not claims about a Transformer's training algorithm. |
| Neurons, electricity, conductors, and bits | Distinguish the physical/numerical substrate from the information-bearing operation distributed through it. No literal binary-bit traversal or biological storage mechanism is asserted by the analogy. |
| Gravity, cylinders, roads, and geometry | Express influence, coordination, and coupled effects. No sphere, rigidity law, metric, dimensionality, or guaranteed safe propagation rule has been established. |
| Observation or quantum collapse | A request conditions which effects become expressed and observable. It does not invoke quantum hardware or require one predetermined continuation. |

## 30. 7. Misunderstandings to avoid

Source: [references/history/pre-organization-RESIDUAL-TRACE-THEORY.md](../../references/history/pre-organization-RESIDUAL-TRACE-THEORY.md), line 350.

| Misreading | Meaning to retain |
|---|---|
| Find values that contain a capability, then connect similar values. | Observe connected transformations during relevant executions; seek recurring portions of their influence. |
| Recover a full signature uniquely belonging to one capability first. | Partial, shared, context-dependent residue can already be useful. |
| The path is defined to exclude the inference trace. | The present target is recognizable residual pattern in how information unfolds; do not restore the earlier frozen-pattern-only requirement. |
| All examples must use identical weights, states, or directions. | What recurs may be how changes develop together through varying circumstances. |
| Any common activation or architectural route is the pattern. | Common machinery is a starting observation; it does not explain how the capability contributes. |
| More arbitrary prompts will reveal it automatically. | Shape requests so the relevant capability contributes strongly and repeatedly. |
| An edit must reconstruct or improve the whole capability. | It may be useful to alter a tendency toward an unproductive branch while much else continues to work. |
| The analogies specify literal physics or a training algorithm. | They communicate distributed influence, continuity, and partial observability. |
| The same reward means equal numerical gradients. | Marcos explicitly rejects that as a description of his argument. |
| A polished visualization or a changed answer completes the research. | The intended result is meaningful recognition of the unfolding influence; representation and intervention serve that purpose. |

## 31. Targeted key/value lookup edit

Source: [apps/decode-machine/TARGETED-LOOKUP-EXPERIMENT.md](../../apps/decode-machine/TARGETED-LOOKUP-EXPERIMENT.md), line 5.

| Run | Target answer | Discovery correct / 12 | Controls correct / 18 | Losses among 26 original-correct discovery/controls |
| --- | --- | ---: | ---: | ---: |
| Original | `2` | 9 | 17 | — |
| Earlier whole-head half-scale control | `blue` | 11 | 17 | 1 |
| Candidate 1: first 32 columns | `Blue` | 7 | 14 | 6 |
| Candidate 2: first 384 columns | `2` | 10 | 17 | 1 |
| Candidate 3: last 128 columns | `blue` | 12 | 17 | 0 |

## 32. Targeted key/value lookup edit

Source: [apps/decode-machine/TARGETED-LOOKUP-EXPERIMENT.md](../../apps/decode-machine/TARGETED-LOOKUP-EXPERIMENT.md), line 62.

| Recorded location | Target: changed / total | Protected red case: changed / total |
| --- | ---: | ---: |
| Layer 9 output projection | 792 / 1536 | 796 / 1536 |
| Layer 9 post-attention | 620 / 1536 | 623 / 1536 |
| Layer 9 layer output | 1234 / 1536 | 1185 / 1536 |
| Final normalization | 1461 / 1536 | 1466 / 1536 |
| Answer | `2`→`blue` | `red`→`red` |

## 33. Observations and the first test

Source: [apps/decode-machine/WEIGHT-SPACE-ORIENTATION.md](../../apps/decode-machine/WEIGHT-SPACE-ORIENTATION.md), line 62.

| Pair | Expected | Original wording | Color wording |
| --- | --- | --- | --- |
| 01 | blue | 2 | green |
| 03 | blue | 3 | blue |
| 10 | blue | explanatory text, cap reached | blue |
| 00 | red | red | red |
| 02 | green | green | green |
| 08 | blue | blue | green |

## 34. Workload and what actually ran

Source: [runs/legs/gemma4-e2b/decode-observatory-30k-2026-11-05/ANALYSIS.md](../../studies/01-early-exploration/decode-observatory/run/ANALYSIS.md), line 13.

| Measured scope | Result | Denominator / qualification |
|---|---:|---|
| New tokens | 30,000 | Includes first output tokens produced by prefill and emitted end-of-turn tokens |
| Requests | 187 | One sequential, retained conversation |
| Sum of request wall times | 2,442.492 s | Excludes gaps between requests and initial server startup |
| Persistent decode GPU time | 1,673.372 s | 29,813 decode-loop tokens; one output token per request belongs to prefill |
| Prefill GPU time | 724.745 s | 9,758 rows processed across initial and continuation prompts |
| Other time within requests | 44.375 s | Wall time minus reported prefill and decode GPU time; not a per-cause attribution |
| Mean decode-loop time | 56.129 ms/token | Weighted by the measured decode-loop token population |
| First reply mean decode time | 10.516 ms/token | 2,098 decode-loop tokens |
| Last reply mean decode time | 114.385 ms/token | 77 decode-loop tokens |

## 35. What the early Nsight Compute capture adds

Source: [runs/legs/gemma4-e2b/decode-observatory-30k-2026-11-05/ANALYSIS.md](../../studies/01-early-exploration/decode-observatory/run/ANALYSIS.md), line 58.

| Counter observation | Recorded result | Implication to test |
|---|---:|---|
| Cycles with no eligible warp | 89.35% | Separate dependency readiness, load latency and intentional sleeping before changing scheduling |
| Active / eligible warps per scheduler | 2.00 / 0.12 | Resident work often cannot issue its next instruction |
| Sleeping stall contribution | 8.55 of 18.78 cycles per issued instruction, about 45.5% | Find which producer each sleeping consumer awaits; removing sleeps could merely replace waiting with polling |
| Long-scoreboard stall contribution | 3.12 cycles per issued instruction | Memory dependencies also matter; this is not the same as saturated bandwidth |
| Register count / rounded allocation | 242 / 248 per thread | Register allocation limits residency to one block per SM |
| Dynamic / static / driver shared memory | 51,200 / 44,032 / 1,024 bytes per block | Total allocation is 96,256 bytes; shared memory independently limits residency to one block |
| Theoretical / achieved occupancy | 16.67% / 16.71% | The kernel reaches its low resource-limited resident-warp budget |
| DRAM read / write bytes | 1,304,050,688 / 949,760 | Whole-launch traffic, not a byte map to individual weights or buffers |
| DRAM throughput | 23.70% of peak sustained elapsed | The early fixture is not saturating aggregate DRAM bandwidth |
| L2 sector hit rate | 14.04% | Aggregate cache behavior; it does not identify which object missed |
| Tensor / TMA instruction activity | 1.87% / 0.38% of their reported active-cycle peaks | Distinct metric denominators; neither measures useful-operation coverage |

## 36. Action order from the inspection ontology

Source: [runs/legs/gemma4-e2b/decode-observatory-30k-2026-11-05/ANALYSIS.md](../../studies/01-early-exploration/decode-observatory/run/ANALYSIS.md), line 82.

| Priority and architecture area | Evidence and hypothesis | Smallest next experiment | What would refute it |
|---|---|---|---|
| 0. Capture integrity | The later tape uses 2.73 GHz as a cycle upper bound, although the saved device report permits 3.135 GHz | Make the capture writer record a proven upper bound with its device provenance, or use an unambiguous clock representation; preserve the ordering checks and recapture the affected slice | A nominal or sampled frequency is still treated as a guaranteed maximum, or the old receipt is inflated merely to pass admission |
| 1. Coordination / work distribution | Production growth and full-history attention tail growth agree; one position slice leaves the long operation concentrated in eight observed workers | Build a known-revision baseline, then vary position slicing as a policy bundle with distinct owned work. Measure the context slope, early cost, and dependent completion; run the existing coherence gate | The slope does not fall, dependent completion does not advance, or overhead merely moves to another group |
| 2. Warp execution / local resources | Low eligibility, sleep and scoreboard samples, and two independent one-block residency limits | Bind the captured hot PCs to logical operations and readiness events. Test useful independent work and resource lifetime changes together where the data supports them | Occupancy or stall counters improve but the same production workload does not finish earlier, or spills / other regions regress |
| 3. Continuation prefill | 724.745 s of GPU prefill across the conversation, including 5.750 s for the final 52 fresh rows | Profile the real cached-prefix M-row continuation path separately, retaining its large history and row count | A short standalone prefill improves while actual continuation request time is unchanged |
| 4. Memory capacity and reuse | 2 GiB pinned lot survives the first decode; allocated size alone says nothing about occupied capacity | Add occupied-capacity and first/last-use events for this existing buffer, then assess allocation timing or reuse | Most capacity is needed, or smaller/later allocation adds transfer or request latency |
| 5. Output utility | Late replies reuse most eight-token phrases from earlier replies | Add a task-specific progress criterion alongside throughput and retain the actual text / token IDs | A faster configuration reduces useful task progress or required correctness, even if raw token rate rises |

## 37. Desktop, 3840x2050: mean 6.7

Source: [runs/legs/gemma4-e2b/decode-observatory-30k-2026-11-05/design-review.md](../../studies/01-early-exploration/decode-observatory/run/design-review.md), line 7.

| # | Criterion | Score | Reason |
|---|---|---|---|
| 1 | Room and width | 6 | Content runs edge to edge (about x 40 to 3800) and nothing leaves its box. The "Busy versus productive" column ends about 40% down its pane, so the lower ~58% of that pane is empty. |
| 2 | Show, don't tell | 8 | The bars, the dotted line chart, the activity meters, the reuse bar and the slice chips with mini bars are all drawn. "Connected evidence" is a key/value text table. |
| 3 | Rich components | 7 | Charts, the reply card and the chips have their own shapes. Layer, Operation and Worker are plain native selects, and "Connected evidence" is plain rows. |
| 4 | Navigating the data | 8 | The page scrolls once. Detail opens in place behind disclosure triangles, the chip list is bounded, and the side list marks the active item. Nothing sorts by what hurts. |
| 5 | States as shapes | 6 | "Zeros / saturated: 0 / Unknown" is a word where a shape belongs. The big "What happens to the values?" panel shows only a sentence ("Same bins. Same count scale.") and a clipped bar. |
| 6 | Hierarchy and type | 4 | Chart labels are about 23 px, larger than the pane titles (about 19 px) and about twice the neighbouring HTML text. Many labels are about 9 to 11 px, which is tiny at 3840. |
| 7 | Spacing and rhythm | 7 | The three columns sit on one grid, with shared left edges at x 31, 761 and 1491 (display px). Meter values end on one right edge. Chart insets differ from the pane edges, and there are large empty blocks. |
| 8 | Prose economy | 8 | Lines are short. "Population" runs 16 words and "Next useful question" about 11. |
| 9 | Its people's words | 7 | The jargon fits GPU analysts. A raw tensor name, "Stored attn_out, layer 5", sits in plain view. |
| 10 | Colour roles | 6 | The sage green means "decode time" in the bars and "sliding window" in the line, while the legend calls the line teal. Amber means "full history" in the line chart and also fills the value-distribution bar. |
| 11 | The project's own look | 7 | Pale paper, dark green ink and restrained brass read well. The native selects fall out of the look, and no mechanical scene is visible. |

## 38. Laptop, 1440x900: mean 7.5

Source: [runs/legs/gemma4-e2b/decode-observatory-30k-2026-11-05/design-review.md](../../studies/01-early-exploration/decode-observatory/run/design-review.md), line 23.

| # | Criterion | Score | Reason |
|---|---|---|---|
| 1 | Room and width | 9 | Content fills about 96% of the width (x 28 to 1411), nothing runs out of a box, and both chart panes are filled. |
| 2 | Show, don't tell | 8 | Bars, the dotted line chart, the progress bar and the reuse bar are drawn. The Reply 45 figures are big numbers with a bar only for reuse. |
| 3 | Rich components | 7 | The reply card with its accent edge and the charts have their own shapes. The selects are native. |
| 4 | Navigating the data | 8 | Detail opens in place ("Timing scope", "What this interval means") and the page scrolls once. |
| 5 | States as shapes | 7 | The unselected state is said as a sentence under the chart: "Select a dot to open its recorded attention machinery." |
| 6 | Hierarchy and type | 5 | Each part has a clear focal. Labels sit at about 11 px, and the bar numbers 1 to 16 are about 8 px. |
| 7 | Spacing and rhythm | 8 | Content shares one left edge, and label baselines match across the two charts (y 473 and y 672). Select gutters are about 20 px against 26 px between the columns. |
| 8 | Prose economy | 9 | Every line is 12 words or fewer. |
| 9 | Its people's words | 8 | "Layer 5 · Full context", "Worker 16" and "Reply 45" are plain. "864 µs tail" and "seconds, request" are a bit cryptic. |
| 10 | Colour roles | 6 | The sage green is shared by the decode bars and the sliding-window line, and the legend calls the line teal. Amber is used only for full history. |
| 11 | The project's own look | 7 | The paper, ink and teal look reads. The native selects fall out of it, and no mechanical scene is in the crop. |

## 39. Phone, 390x844: mean 7.5 over 10 criteria

Source: [runs/legs/gemma4-e2b/decode-observatory-30k-2026-11-05/design-review.md](../../studies/01-early-exploration/decode-observatory/run/design-review.md), line 41.

| # | Criterion | Score | Reason |
|---|---|---|---|
| 1 | Room and width | 9 | It is one column with nothing past the right edge (card x 14 to 376). Padding is doubled, so the content is about 86% of the width. |
| 2 | Show, don't tell | 8 | The decode-time bars, the progress bar and the reuse bar are drawn. |
| 3 | Rich components | 7 | The reply card and chart have shapes. The selects are native. |
| 4 | Navigating the data | 8 | The disclosure ("Timing scope") opens in place and the page scrolls once. |
| 6 | Hierarchy and type | 5 | There is a clear focal ("E2B · a conversation growing to 30k", and the 17 px Reply figures). The unit labels and "RECORDED CONVERSATION" are about 11 px. |
| 7 | Spacing and rhythm | 7 | Three left edges coexist: body text at x 27, reply card text at 43, chart at 49. |
| 8 | Prose economy | 9 | Lines are short. The subtitle wraps to two lines at 12 words. |
| 9 | Its people's words | 8 | The names are plain. "seconds, request" is a little odd. |
| 10 | Colour roles | 7 | Sage for data, dark green for ink and progress, teal for reuse. The legend conflict is not visible at this size. |
| 11 | The project's own look | 7 | The paper and green ink read. The native selects fall out of the look. |

## 40. Actual outputs

Source: [runs/legs/gemma4-e2b/lookup-failure-family-2026-11-06/RESULTS.md](../../studies/01-early-exploration/lookup-failure-family/run/RESULTS.md), line 9.

| Case | Requested value | Requested row | Actual text | Short-run classification |
|---|---|---:|---|---|
| known-01 | blue | 2 | 2 | numeric-substitution |
| known-03 | blue | 2 | 3 | numeric-substitution |
| probe-03 | mango | 1 | Mango | correct-value |
| probe-04 | mango | 2 | Mango | correct-value |
| probe-05 | mango | 3 | Mango | correct-value |
| probe-06 | falcon | 1 | Panda | wrong-listed-value |
| probe-07 | falcon | 2 | 3 | numeric-substitution |
| probe-08 | falcon | 3 | The question asks for the value of " | incomplete-at-cap |
| probe-09 | violin | 1 | Please provide the table you are referring to | incomplete-at-cap |
| probe-10 | violin | 2 | 3 | numeric-substitution |
| probe-11 | violin | 3 | 3 | numeric-substitution |
| probe-12 | silver | 1 | Copper | wrong-listed-value |
| probe-13 | silver | 2 | 24 | numeric-substitution |
| probe-14 | silver | 3 | Silver | correct-value |
| probe-15 | oslo | 1 | Tokyo | wrong-listed-value |
| probe-16 | oslo | 2 | The question asks for the value of " | incomplete-at-cap |
| probe-17 | oslo | 3 | Lima | wrong-listed-value |
| probe-18 | train | 1 | Train | correct-value |
| probe-19 | train | 2 | 1 | numeric-substitution |
| probe-20 | train | 3 | 1 | numeric-substitution |

## 41. Follow-up observations completed

Source: [runs/legs/gemma4-e2b/lookup-failure-family-2026-11-06/RESULTS.md](../../studies/01-early-exploration/lookup-failure-family/run/RESULTS.md), line 75.

| Case | Longer observation | Classification |
|---|---|---|
| probe-08 | Explicitly states that Sop's value is Falcon, continues explaining, reaches 128-token cap | Correct association reached; verbose and unfinished |
| probe-09 | "Please provide the table you are referring to. I need the table to answer the question." Stops naturally after 19 tokens | Fails to use the supplied table; distinct from numeric substitution |
| probe-16 | Explicitly states Zef is associated with Oslo, continues reconsidering the question, reaches 128-token cap | Correct association reached; verbose and unfinished |

## 42. The local llama.cpp fork answers all twenty association questions correctly

Source: [runs/legs/gemma4-e2b/llamacpp-lookup-comparison-2026-11-06/RESULTS.md](../../studies/01-early-exploration/llamacpp-lookup-comparison/run/RESULTS.md), line 5.

| Case | Requested value | QGRE short answer | llama.cpp answer |
|---|---|---|---|
| known-01 | blue | 2 | blue |
| known-03 | blue | 3 | blue |
| probe-03 | mango | Mango | Mango |
| probe-04 | mango | Mango | Mango |
| probe-05 | mango | Mango | Mango |
| probe-06 | falcon | Panda | Falcon |
| probe-07 | falcon | 3 | Falcon |
| probe-08 | falcon | The question asks for the value of " | Falcon |
| probe-09 | violin | Please provide the table you are referring to | Violin |
| probe-10 | violin | 3 | Violin |
| probe-11 | violin | 3 | Violin |
| probe-12 | silver | Copper | Silver |
| probe-13 | silver | 24 | Silver |
| probe-14 | silver | Silver | Silver |
| probe-15 | oslo | Tokyo | Oslo |
| probe-16 | oslo | The question asks for the value of " | Oslo |
| probe-17 | oslo | Lima | Oslo |
| probe-18 | train | Train | Train |
| probe-19 | train | 1 | Train |
| probe-20 | train | 1 | Train |

## 43. Ordinary-server evaluation

Source: [runs/legs/gemma4-e2b/llamacpp-preference-edit-2026-11-06/REPORT.md](../../studies/01-early-exploration/llamacpp-preference-edit/run/REPORT.md), line 31.

| Measurement | Original | Alpha 1 | Selected alpha 2 |
|---|---:|---:|---:|
| Discovery: first value only | 5/16 | 7/16 | 10/16 |
| Discovery: last value only | 6/16 | 4/16 | 1/16 |
| Discovery: both values | 5/16 | 5/16 | 5/16 |
| Discovery: explicit first correct | 14/16 | 14/16 | 14/16 |
| Discovery: explicit last correct | 16/16 | 16/16 | 16/16 |
| Discovery: unique-key correct | 20/20 | 20/20 | 20/20 |
| Withheld: first value only | 7/16 | Not evaluated | 10/16 |
| Withheld: last value only | 5/16 | Not evaluated | 2/16 |
| Withheld: both values | 4/16 | Not evaluated | 4/16 |
| Withheld: explicit first correct | 15/16 | Not evaluated | 15/16 |
| Withheld: explicit last correct | 15/16 | Not evaluated | 15/16 |

## 44. Ordinary-server evaluation

Source: [runs/legs/gemma4-e2b/llamacpp-preference-edit-2026-11-06/REPORT.md](../../studies/01-early-exploration/llamacpp-preference-edit/run/REPORT.md), line 49.

| Case | First value | Last value | Original answer | Edited answer |
|---|---|---|---|---|
| confirmation-1-1-default | Ruby | Opal | Opal | Ruby |
| confirmation-4-1-default | Tennis | Golf | Golf | Tennis |
| confirmation-6-1-default | Circle | Square | Square | Circle |

## 45. Reversed-direction control

Source: [runs/legs/gemma4-e2b/llamacpp-preference-edit-2026-11-06/REPORT.md](../../studies/01-early-exploration/llamacpp-preference-edit/run/REPORT.md), line 61.

| Discovery default answer | Reverse, −2 | Original | Toward first, +1 | Toward first, +2 |
|---|---:|---:|---:|---:|
| First value only | 2/16 | 5/16 | 7/16 | 10/16 |
| Last value only | 12/16 | 6/16 | 4/16 | 1/16 |
| Both values | 2/16 | 5/16 | 5/16 | 5/16 |

## 46. Internal propagation

Source: [runs/legs/gemma4-e2b/llamacpp-preference-edit-2026-11-06/REPORT.md](../../studies/01-early-exploration/llamacpp-preference-edit/run/REPORT.md), line 77.

| Diagnostic | Original | Edited |
|---|---:|---:|
| First value's share of the two candidates' affinity | 59.09% | 88.66% |
| Absolute affinity mass on first value | 50.39% | 62.86% |
| Absolute affinity mass on last value | 32.45% | 7.48% |
| Total affinity mass on either value | 82.84% | 70.34% |

## 47. What ran

Source: [runs/legs/gemma4-e2b/lookup-pattern-observation-2026-11-06/REPORT.md](../../studies/01-early-exploration/lookup-pattern-observation/run/REPORT.md), line 15.

| Condition | Discovery correct / total | Confirmation correct / total | Three-way exact answer-token agreement, total |
|---|---:|---:|---:|
| Assigned value | 16 / 16 | 8 / 8 | 24 / 24 |
| Position | 16 / 16 | 8 / 8 | 24 / 24 |
| Different queried key | 4 / 4 | 2 / 2 | 6 / 6 |
| Literal copy | 4 / 4 | 2 / 2 | 6 / 6 |
| **Total** | **40 / 40** | **20 / 20** | **60 / 60** |

## 48. Observable contrast and held-aside confirmation

Source: [runs/legs/gemma4-e2b/lookup-pattern-observation-2026-11-06/REPORT.md](../../studies/01-early-exploration/lookup-pattern-observation/run/REPORT.md), line 33.

| Inspection location, zero-based | Ask for value, n=8 | Ask for position, n=8 | Ask about alternate key, n=2 | Literal copy, n=2 |
|---|---:|---:|---:|---:|
| Layer 29, head 5 | 86.94% | 0.80% | 93.34% | 43.61% |
| Layer 27, head 7 | 64.99% | 0.33% | 64.93% | 19.49% |
| Layer 24, head 3 | 55.26% | 2.75% | 52.87% | 8.50% |

## 49. What was inspected in the actual weights

Source: [runs/legs/gemma4-e2b/lookup-pattern-observation-2026-11-06/REPORT.md](../../studies/01-early-exploration/lookup-pattern-observation/run/REPORT.md), line 47.

| Query tensor | Actual stored type | Inspected output rows, zero-based | Input width |
|---|---|---|---:|
| `blk.29.attn_q.weight` | Q4_K | 2560–3071 | 1536 |
| `blk.27.attn_q.weight` | Q6_K | 1792–2047 | 1536 |
| `blk.24.attn_q.weight` | Q6_K | 768–1023 | 1536 |

## 50. What was inspected in the actual weights

Source: [runs/legs/gemma4-e2b/lookup-pattern-observation-2026-11-06/REPORT.md](../../studies/01-early-exploration/lookup-pattern-observation/run/REPORT.md), line 57.

| Comparison | Pair count | L29 H5 | L27 H7 | L24 H3 |
|---|---:|---:|---:|---:|
| Same value task, different vocabulary family | 4 | 0.659 | 0.650 | 0.750 |
| Same assignments, value versus position | 8 | 0.456 | 0.515 | 0.573 |
| Same assignments, value versus literal copy | 2 | 0.824 | 0.783 | 0.641 |
| Same assignments, different queried key | 2 | 0.735 | 0.661 | 0.775 |
| Same assignments, table versus prose | 4 | 0.930 | 0.920 | 0.883 |
| Same assignments, changed row order | 4 | 0.888 | 0.829 | 0.828 |

## 51. Why this is still short of identifying the intended path

Source: [runs/legs/gemma4-e2b/lookup-pattern-observation-2026-11-06/REPORT.md](../../studies/01-early-exploration/lookup-pattern-observation/run/REPORT.md), line 72.

| Captured input at layer | Same value task, different family | Same assignments, value versus position |
|---|---:|---:|
| 29 | 0.664 | 0.497 |
| 27 | 0.708 | 0.506 |
| 24 | 0.889 | 0.769 |

## 52. Instrumentation changes numerical scores

Source: [runs/legs/gemma4-e2b/lookup-pattern-observation-2026-11-06/REPORT.md](../../studies/01-early-exploration/lookup-pattern-observation/run/REPORT.md), line 92.

| Split | Mean per-case RMS logit difference | Range of per-case RMS differences | Largest absolute logit difference |
|---|---:|---:|---:|
| Discovery, n=40 | 0.4102 | 0.0144–1.7005 | 3.8499 |
| Confirmation, n=20 | 0.4546 | 0.0340–1.4319 | 4.0916 |

## 53. 2. Vocabulary and intended meaning

Source: [runs/legs/gemma4-e2b/distributed-pathways-reference-2026-11-06/REFERENCE.md](../../studies/01-early-exploration/distributed-pathways/run/REFERENCE.md), line 67.

| Term | Meaning in this framework |
|---|---|
| **Weights / conductors / rails** | The persistent numerical substrate. An isolated value does not independently identify the knowledge or capability expressed by a distributed whole. |
| **Parts / bits of weights** | Marcos's description of portions of the numerical encoding jointly participating across many weights. This does not assign one meaning to a particular bit or claim that a literal binary-bit decomposition has already been discovered. |
| **Horizontal / distributed** | The pattern spans portions of many weights, potentially much of the model, and can overlap with other patterns. This has the same meaning in all five examples. It is not a single layer or a memory layout. |
| **Path** | The proposed frozen, distributed encoded pattern constituting information or a capability. It is not a list of layers, an activation trace, or the downstream footprint of an edit. |
| **Journey / execution** | A particular expression of the model's learned organization during inference. Earlier writing sometimes conflated this with “path”; this reference now distinguishes them. |
| **Gravitational pull / strength** | Relative influence of learned patterns under a context, and their reinforcement across training examples. It is not a semantic quantity deposited into individual words. |
| **Geometry** | The proposed organization of the overlapping distributed paths, supporting their stability, distinguishability, and use. No specific shape, dimensionality, or metric has been established. |
| **Ghost** | The input-dependent expression of the frozen organization. It changes during inference while the stored parameters remain fixed. |
| **Observation** | A context and its resulting behavior/internal activity that may expose an aspect of the encoded pattern. The question is processed during prefill before answer generation. |
| **Cylinder / rigid connection** | An earlier analogy for coupled effects within shared organization. It does not require an actual rod, a sparse chain, or a guaranteed uniform propagation law. |
| **Dotted line / missing jigsaw piece** | A missing or defective functional relationship that prevents a task from being completed correctly. In the physics example, it refers to the control behavior needed to complete the task. |

## 54. The connection across all five

Source: [runs/legs/gemma4-e2b/distributed-pathways-reference-2026-11-06/REFERENCE.md](../../studies/01-early-exploration/distributed-pathways/run/REFERENCE.md), line 151.

| Setting | What recurs across changing examples | Why that matters to the theory |
|---|---|---|
| CPT | A factual relationship | Training repeatedly reinforces access to the relationship. |
| RL | A useful procedure or relationship contributing to success | Training can reinforce and combine reusable computations. |
| Small specialized network | A constrained task organization | Limited overlap can make the organization easier to recognize. |
| Physics/control repair | The behavior needed for a successful task trajectory | A constrained failure can expose a specific functional repair. |
| Qwen looping | A failure process across different completions | Varied observations can help isolate a recurring mechanism. |

## 55. Saved Gemma 4 E2B CPT evaluations

Source: [runs/legs/gemma4-e2b/distributed-pathways-reference-2026-11-06/REFERENCE.md](../../studies/01-early-exploration/distributed-pathways/run/REFERENCE.md), line 169.

| Saved run | Base | After CPT | Recorded setup |
|---|---:|---:|---|
| Pokémon lore, Kanto v1 | 89/300, 29.7% | 206/300, 68.7% | Gemma 4 E2B; FP4; forced-choice evaluation labeled `augmentation_holdout`; result `PASS`. |
| Pokémon lore, Kanto anneal | 21/77, 27.3% | 54/77, 70.1% | Gemma 4 E2B; FP4; forced-choice evaluation labeled `augmentation_holdout`; result `PASS`. |

## 56. A concrete before/after example: the repeated key Kiv

Source: [runs/legs/gemma4-e2b/path-inspection-2026-11-06/REPORT.md](../../studies/01-early-exploration/path-inspection/run/REPORT.md), line 65.

| Layer | Change relative to the original layer-output norm |
|---|---:|
| 29 | 14.30% |
| 30 | 13.26% |
| 31 | 12.73% |
| 32 | 12.60% |
| 33 | 11.41% |
| 34 | 4.93% |

## 57. Where the change is carried and transformed

Source: [runs/legs/gemma4-e2b/path-inspection-2026-11-06/REPORT.md](../../studies/01-early-exploration/path-inspection/run/REPORT.md), line 86.

| Layer | Incoming residual | Attention branch | Feed-forward branch | Per-layer embedding branch |
|---|---:|---:|---:|---:|
| 29 | 0.00% | 76.76% | 23.31% | −0.07% |
| 30 | 83.11% | 0.69% | 16.16% | 0.04% |
| 31 | 88.82% | 0.08% | 11.16% | −0.07% |
| 32 | 82.42% | 0.60% | 16.81% | 0.16% |
| 33 | 69.87% | 1.71% | 26.42% | 1.99% |
| 34 | 18.55% | 0.92% | 80.27% | 0.26% |

## 58. What we asked

Source: [runs/legs/gemma4-e2b/physics-trace-2026-11-06/REPORT.md](../../studies/02-physics-trace-observation/physics-trace-pilot/run/REPORT.md), line 26.

| Server pilot format | Correct YES/NO verdict | Reached output limit | Interpretation |
| --- | ---: | ---: | --- |
| Verdict first; 256-token cap | 9/18 | 18/18 | Every answer begins NO; no completed solutions |
| Calculation before verdict; 768-token cap | 11/18 | 1/18 | Most responses omit the requested numerical calculation |
| Explicit numerical calculation; 768-token cap | 6/18 | 10/18 | One complete correct calculation; one correct quantitative progression cut off before verdict |

## 59. The connected progression

Source: [runs/legs/gemma4-e2b/physics-trace-followup-2026-11-06/REPORT.md](../../studies/02-physics-trace-observation/physics-trace-followup/run/REPORT.md), line 44.

| Pair | Kind | Predicted sequence / eligible | Median FF14 ratio | Median FF15 ratio |
| --- | --- | ---: | ---: | ---: |
| cart | physics | 94/94 | 0.559 | 1.506 |
| sled | physics | 104/104 | 0.567 | 1.515 |
| bead | physics | 4/4 | 0.506 | 1.510 |
| coaster | physics | 223/223 | 0.556 | 1.497 |
| block | physics | 4/4 | 0.493 | 1.405 |
| trolley | physics | 251/251 | 0.574 | 1.513 |
| cart-paraphrase | paraphrase | 17/17 | 0.535 | 1.504 |
| coaster-paraphrase | paraphrase | 16/16 | 0.552 | 1.530 |
| block-paraphrase | paraphrase | 40/40 | 0.545 | 1.531 |
| cart-arithmetic | arithmetic | 25/25 | 0.563 | 1.625 |
| coaster-arithmetic | arithmetic | 34/34 | 0.556 | 1.690 |
| block-arithmetic | arithmetic | 34/34 | 0.568 | 1.682 |

## 60. The connected progression

Source: [runs/legs/gemma4-e2b/physics-trace-followup-2026-11-06/REPORT.md](../../studies/02-physics-trace-observation/physics-trace-followup/run/REPORT.md), line 76.

| Layer | Absolute separation ratio | State-length growth | Update alignment cosine | Before/after difference cosine |
| --- | ---: | ---: | ---: | ---: |
| 14 | 1.721–1.942 | 3.284–3.387 | -0.067–0.029 | 0.514–0.552 |
| 15 | 3.508–4.041 | 2.382–2.547 | -0.041–0.090 | 0.232–0.360 |

## 61. What the model actually did

Source: [runs/legs/gemma4-e2b/physics-trace-followup-2026-11-06/REPORT.md](../../studies/02-physics-trace-observation/physics-trace-followup/run/REPORT.md), line 93.

| Kind | Cases | Correct explicit verdicts | Complete correct solutions | Capped |
| --- | ---: | ---: | ---: | ---: |
| physics | 12 | 3 | 0 | 7 |
| paraphrase | 6 | 2 | 0 | 3 |
| arithmetic | 6 | 3 | 1 | 0 |

## 62. Connecting concrete calculation moments

Source: [runs/legs/gemma4-e2b/physics-trace-followup-2026-11-06/REPORT.md](../../studies/02-physics-trace-observation/physics-trace-followup/run/REPORT.md), line 125.

| Execution | Observable progression | Recorded decision indices (zero-based) |
| --- | --- | --- |
| sled-base | Adds downhill gain; obtains 90 J; rejects 50−40; later uses the rejected 10 J | 272, 283, 331, 538 |
| cart-paraphrase-changed | States addition of downhill gain; later substitutes subtraction | 387, 466 |
| cart-arithmetic-base | Calculates 60 and 40 correctly; answers NO incorrectly | 120 |
| cart-arithmetic-changed | Calculates 60 and 80 correctly; answers NO correctly | 79 |

## 63. Calculation anchors

Source: [runs/legs/gemma4-e2b/physics-trace-bf16-2026-11-06/CALCULATION-ANCHORS.md](../../studies/02-physics-trace-observation/physics-trace-bf16/run/CALCULATION-ANCHORS.md), line 7.

| Case | Moment | Decision (zero-based) | Input position | Pair history still matched? |
| --- | --- | ---: | ---: | --- |
| block-base | descent-addition | 166 | 338 | yes |
| block-base | descent-result | 177 | 349 | yes |
| block-base | loss-subtraction | 222 | 394 | yes |
| block-base | available-result | 233 | 405 | yes |
| block-changed | descent-addition | 166 | 338 | yes |
| block-changed | descent-result | 177 | 349 | yes |
| block-changed | loss-subtraction | 222 | 394 | yes |
| block-changed | available-result | 233 | 405 | yes |
| block-changed | wrong-required-height | 290 | 462 | no |
| bead-base | descent-result | 299 | 470 | no |
| bead-base | available-result | 341 | 512 | no |
| bead-base | extra-subtraction | 401 | 572 | no |
| bead-base | incorrect-zero | 412 | 583 | no |
| cart-paraphrase | descent-result | 326 | 505 | no |
| cart-paraphrase | loss-subtraction | 406 | 585 | no |
| cart-paraphrase | available-result | 425 | 604 | no |
| coaster-paraphrase-changed | descent-result | 567 | 749 | no |
| coaster-paraphrase-changed | loss-subtraction | 614 | 796 | no |
| coaster-paraphrase-changed | available-result | 625 | 807 | no |

## 64. What the model did

Source: [runs/legs/gemma4-e2b/physics-trace-bf16-2026-11-06/REPORT.md](../../studies/02-physics-trace-observation/physics-trace-bf16/run/REPORT.md), line 22.

| Observation across the same questions | Q4_K_M | BF16 |
| --- | ---: | ---: |
| Correct explicit final verdict, all cases | 8/24 | 13/24 |
| Complete correct arithmetic calculation and verdict | 1/6 | 4/6 |
| Complete correct physical numerical chain and verdict, including paraphrases | 0/18 | 2/18 |
| Physically consistent completed explanation | 0/18 | 0/18 |
| Responses reaching the 768-token cap | 10/24 | 8/24 |

## 65. A correct partial calculation with a controlled comparison

Source: [runs/legs/gemma4-e2b/physics-trace-bf16-2026-11-06/REPORT.md](../../studies/02-physics-trace-observation/physics-trace-bf16/run/REPORT.md), line 64.

| Observable calculation moment | Decision, zero-based | FF14 relative ratio | FF15 relative ratio |
| --- | ---: | ---: | ---: |
| Adds downhill gain: 54 + 30 | 166 | 0.493 | 1.471 |
| Produces 84 J | 177 | 0.546 | 1.425 |
| Subtracts friction: 84 − 24 | 222 | 0.527 | 1.425 |
| Produces 60 J | 233 | 0.583 | 1.281 |

## 66. How broadly the earlier progression persists

Source: [runs/legs/gemma4-e2b/physics-trace-bf16-2026-11-06/REPORT.md](../../studies/02-physics-trace-observation/physics-trace-bf16/run/REPORT.md), line 95.

| Pair | Q4 progression / eligible | BF16 progression / eligible | BF16 median FF14 ratio | BF16 median FF15 ratio |
| --- | ---: | ---: | ---: | ---: |
| cart | 94/94 | 297/297 | 0.510 | 1.537 |
| sled | 104/104 | 306/306 | 0.502 | 1.519 |
| bead | 4/4 | 20/20 | 0.482 | 1.530 |
| coaster | 223/223 | 270/270 | 0.502 | 1.547 |
| block | 4/4 | 253/254 | 0.526 | 1.492 |
| trolley | 251/251 | 307/307 | 0.503 | 1.489 |
| cart-paraphrase | 17/17 | 35/35 | 0.510 | 1.613 |
| coaster-paraphrase | 16/16 | 50/50 | 0.497 | 1.606 |
| block-paraphrase | 40/40 | 47/47 | 0.515 | 1.565 |
| cart-arithmetic | 25/25 | 72/72 | 0.541 | 1.682 |
| coaster-arithmetic | 34/34 | 25/25 | 0.563 | 1.714 |
| block-arithmetic | 34/34 | 25/25 | 0.585 | 1.657 |

## 67. Only after qualification validation exited0:

Source: [runs/legs/gemma4-e2b/physics-intervention-2026-11-07/COMMANDS.md](../../studies/02-physics-trace-observation/physics-intervention/run/COMMANDS.md), line 43.

| Command | Outcome |
| --- | --- |
| `prepare.mjs` | exit 0; sources and arms from declared eligibility, no inference |
| `validate-baselines.mjs` | exit 0; four ordinary/capture matches, 2555 decisions |
| `validate.mjs qualification` | exit 0; six exact controls, 2790 decisions |
| `validate.mjs donors` | exit 0; four exact interventions with before/after/consumer and provenance checks |
| `summarize.mjs` | exit 0; all four complete token sequences unchanged, exactly one score-hash difference per run |

## 68. The observable target

Source: [runs/legs/gemma4-e2b/physics-intervention-2026-11-07/REPORT.md](../../studies/02-physics-trace-observation/physics-intervention/run/REPORT.md), line 30.

| Recipient | Correct available / required / verdict | Recorded baseline |
| --- | --- | --- |
| 1 m climb, `block-base` | 60 J / 30 J / YES | Correct numerical chain and final triple; top/bottom prose labels are reversed |
| 3 m climb, `block-changed` | 60 J / 90 J / NO | 60 J / **120 J** / NO; it incorrectly adds the earlier descent to the final climb |

## 69. The additional numerical variants

Source: [runs/legs/gemma4-e2b/physics-intervention-2026-11-07/REPORT.md](../../studies/02-physics-trace-observation/physics-intervention/run/REPORT.md), line 61.

| Pair | Correct available energy | Required energy, base / changed | Shared generated prefix | Observation |
| --- | ---: | ---: | ---: | --- |
| 2 kg, speed 6 m/s, loss 16 J | 40 J | 20 / 60 J | 18 tokens | Base gets initial kinetic energy wrong, later has 2 J, and hits the token cap; changed has the correct final triple but inconsistent descent accounting |
| 4 kg, speed 4 m/s, loss 12 J | 60 J | 40 / 80 J | 88 tokens | Both final triples are correct; changed nevertheless makes an extra 40 J subtraction and derives an incorrect speed |

## 70. What the swaps changed

Source: [runs/legs/gemma4-e2b/physics-intervention-2026-11-07/REPORT.md](../../studies/02-physics-trace-observation/physics-intervention/run/REPORT.md), line 75.

| Recipient | Swap time | Difference in substituted branch, relative to recipient branch | Changed vocabulary-score decisions | Final answer |
| --- | --- | ---: | ---: | --- |
| Correct 1 m case | After 60 J, prediction 235 | 1.777% | Exactly 235 | Unchanged: 60 / 30 / YES |
| Correct 1 m case | First prediction, 0 | 2.939% | Exactly 0 | Unchanged: 60 / 30 / YES |
| Failing 3 m case | After 60 J, prediction 235 | 1.776% | Exactly 235 | Unchanged: 60 / 120 / NO |
| Failing 3 m case | First prediction, 0 | 2.940% | Exactly 0 | Unchanged: 60 / 120 / NO |

## 71. What the swaps changed

Source: [runs/legs/gemma4-e2b/physics-intervention-2026-11-07/REPORT.md](../../studies/02-physics-trace-observation/physics-intervention/run/REPORT.md), line 91.

| Recorded boundary | Difference from the unmodified recipient, relative L2 |
| --- | ---: |
| Layer 15 exit, preceding residual | Exactly zero |
| Substituted normalized attention branch 16 | 1.7761% |
| Native attention addition 16 | 0.5605% |
| Layer 16 exit | 0.6164% |
| Layer 17 exit | 0.5449% |
| Layer 34 exit | 0.0471% |
| Final normalized state | 0.1038% |
| Each retained boundary at prediction 236 | Byte-identical to baseline |


## Connected-preservation candidate C

Source: [What was predicted and what happened](../../studies/03-round-1-single-pair-edits/C-connected-preservation/run/REPORT.md), line 13.

| Declared condition | Result under C | Outcome |
| --- | --- | --- |
| Keep target numerical repair | 54+30−24=60 J available; 90 J required; NO. All 413 tokens identical to A | Passed |
| Keep sled numerical repair | 50+40−10=80 J available; 40 J required; YES. All 533 tokens identical to A | Passed |
| Restore protected 4 kg final triple | 60/80/NO. All 665 tokens identical to original | Passed |
| Keep all 7 originally correct numerical chains | All 7 remain correct, with complete continuations token-identical to original | Passed |

Source: [The saved change](../../studies/03-round-1-single-pair-edits/C-connected-preservation/run/REPORT.md), line 34.

| Stored matrix measurement | C |
| --- | ---: |
| Tensor | blk.16.attn_output.weight |
| Shape | 2,048 inputs × 1,536 outputs |
| Changed stored values | 523,778 / 3,145,728 |
| Changed bytes | 533,239 |
| Realized change norm / original matrix norm | 0.0948835% |
| Declared ceiling | 4×A's realized norm, about 0.118624% |
| Original checkpoint bytes compared | 9,311,304,896 |
| Changed bytes outside selected tensor | 0 |

Source: [Every population, without pooling incompatible scores](../../studies/03-round-1-single-pair-edits/C-connected-preservation/run/REPORT.md), line 49.

| Population | Numerical chains + conclusion, original→C | Correct literal verdict, original→C | Caps, original→C | Whole continuations identical |
| --- | --- | --- | --- | --- |
| Old 28, including 6 arithmetic cases | 7/28→9/28 | 16/28→17/28 | 9→9 | 19/28 |
| Previous 8 confirmation questions, now development | 3/8→4/8 | 3/8→4/8 | 4→3 | 5/8 |
| Earlier controls: 4 arithmetic + 4 language | Arithmetic 4/4→4/4; language 4/4→4/4 | Arithmetic 4/4→4/4 | 0→0 | 8/8 |
| Fresh 8: 4 physics + 2 arithmetic + 2 language | Physics 1/4→1/4; arithmetic 2/2→2/2; language 2/2→2/2 | Physics 3/4→3/4; arithmetic 2/2→2/2 | 1→1 | 7/8 |

Source: [Connected native capture](../../studies/03-round-1-single-pair-edits/C-connected-preservation/run/REPORT.md), line 66.

| Case | Original/A eligible decisions | Original/C eligible decisions | A/C eligible decisions |
| --- | ---: | ---: | ---: |
| Target block | 270 | 270 | 413 |
| Sled | 34 | 34 | 533 |
| Preserved paired block | 450 | 450 | 450 |
| Protected 4 kg block | 439 | 665 | 439 |
| Total for each pairing | 1,193 | 1,419 | 1,835 |

Source: [Connected native capture](../../studies/03-round-1-single-pair-edits/C-connected-preservation/run/REPORT.md), line 76.

| Native observation | A projection change | C projection change | A final-state change | C final-state change |
| --- | ---: | ---: | ---: | ---: |
| Target design input 235 | 0.9191% | 0.8858% | 0.0580% | 0.0574% |
| Target first changed output 269 | 0.7212% | 0.6623% | 0.1522% | 0.1379% |
| Sled first changed output 33 | 0.8620% | 0.8115% | 0.1807% | 0.1722% |
| Protected 4 kg, A's changed output 438 | 0.8609% | 0.2132% | 0.6315% | 0.1088% |

Source: [Connected native capture](../../studies/03-round-1-single-pair-edits/C-connected-preservation/run/REPORT.md), line 87.

| Shared-history decision; first token minus second | Original | A | B | C |
| --- | ---: | ---: | ---: | ---: |
| Target 269: gained − reached | −0.006393 | +0.008394 | −0.006218 | +0.008017 |
| Sled 33: math opener − newline | −0.000462 | +0.003975 | −0.001232 | +0.004982 |
| Protected 438: energy − total | +0.022707 | −0.007727 | +0.021198 | +0.015263 |


## Background sensitivity and stored candidate D

### Question and result of the background check

Source: [runs/legs/gemma4-e2b/physics-background-sensitivity-2026-11-07/REPORT.md:7](../../studies/03-round-1-single-pair-edits/D-background-sensitivity/run/REPORT.md).

| Measurement | A | B | C |
| --- | ---: | ---: | ---: |
| Intended read-key cosine with calibration mean | 0.423637 | 0.146152 | No single read key |
| Mean component / total squared action, eight calibration requests | 93.1599% | 78.1742% | 91.9102% |
| Same fraction, token-weighted sensitivity | 93.1130% | 77.9040% | 91.9185% |
| Squared action remaining after analytical subtraction of calibration mean, four held-out requests | 11.3931% | 30.9511% | 13.9278% |
| Range of per-request median native projection changes, six arithmetic controls | 0.5009–0.5376% | 0.4018–0.4290% | 0.4977–0.5213% |
| Same range, six language controls | 0.2364–0.5646% | 0.0613–0.2934% | 0.3776–0.5328% |

### Stored candidate D: quieter background, preserved repairs

Source: [runs/legs/gemma4-e2b/physics-background-sensitivity-2026-11-07/REPORT.md:48](../../studies/03-round-1-single-pair-edits/D-background-sensitivity/run/REPORT.md).

| Stored edit measurement | Result |
| --- | ---: |
| Changed BF16 values | 331,670 / 3,145,728 |
| Realized edit norm / original matrix norm | 0.0452194% |
| Size relative to A | 1.5248× |
| Size relative to C | 0.4766× |
| Remaining donor distance at the design input | 47.6960% (A: 44.4103%) |
| Original-input projection change at design decision 235 | 0.91851% (A: 0.91914%) |

### Actual rounded footprint

Source: [runs/legs/gemma4-e2b/physics-background-sensitivity-2026-11-07/REPORT.md:59](../../studies/03-round-1-single-pair-edits/D-background-sensitivity/run/REPORT.md).

| Population | D's RMS absolute projection action relative to A | Reduction |
| --- | ---: | ---: |
| Four held-out background requests | 0.14497× | 85.50% |
| Six language controls | 0.30348× | 69.65% |
| Six arithmetic controls | 0.65523× | 34.48% |
| Four complete original physics traces | 0.76460× | 23.54% |

### Native behavioral outcomes

Source: [runs/legs/gemma4-e2b/physics-background-sensitivity-2026-11-07/REPORT.md:72](../../studies/03-round-1-single-pair-edits/D-background-sensitivity/run/REPORT.md).

| Evaluation population | Original → D | Comparison with C |
| --- | --- | --- |
| Original 28 | Numerical chains 7→9; explicit verdicts 16→17; caps 9→9 | Same aggregate correctness as C; 20/28 original continuations retained |
| Previous eight confirmation questions | Numerical chains 3→4; verdicts 3→4; caps 4→3 | Same aggregate correctness as C |
| Four physics questions introduced with C, excluded from D's fit | Numerical chains 1/4→1/4; verdicts 3/4→3/4 | No gain |
| Twelve additional arithmetic/language controls | All correct; all original continuations retained | Same as C |


## 88. First incorrect calculations in the earlier fresh questions

Source: [runs/legs/gemma4-e2b/physics-multitrace-2026-11-07/FAILURE-CLASSIFICATION.md](../../studies/03-round-1-single-pair-edits/multitrace-observations/run/FAILURE-CLASSIFICATION.md), line 5.

| Question | Correct calculation | First incorrect step | Class / subtype | Relationship to old repairs |
|---|---|---|---|---|
| new-trolley-4 | 25 + 60 − 15 = 70 J; 2×10×4 = 80 J; NO | After correctly deriving both quantities, redefines required energy as 80 − 70 = 10 J and uses that deficit in the requested final field | Wrong formula/quantity definition: shortfall substituted for required energy | Different from both old first errors; arithmetic and verdict are correct |
| new-sledge-1 | 40 + 50 − 20 = 70 J; 5×10×1 = 50 J; YES | Energy after descent = initial kinetic − mgh = 40 − 50 = −10 J | Wrong formula: descent gain subtracted | Same first numerical error as old sled-changed. Both original and D then hit their 768-token caps; no complete verdict is available |
| new-sledge-2 | 40 + 50 − 20 = 70 J; 5×10×2 = 100 J; NO | Treats energy at foot as initial kinetic alone: ½mv² + 0 = 40 J. It later adds the already traversed rough-section loss to required energy | Wrong formula: descent omitted / energy reference misplaced | Different from old sled sign error and block extra-height error. Arithmetic and final NO are correct for its incorrect comparison |
| block-changed (old) | 54 + 30 − 24 = 60 J; 3×10×3 = 90 J; NO | Adds the earlier 1 m descent to the stated 3 m climb: final height = 4 m | Wrong formula: descent double-counted in final height | A/C/D repaired the final numerical chain; explanatory labels remained inconsistent |
| sled-changed (old) | 50 + 40 − 10 = 80 J; 4×10×1 = 40 J; YES | Kinetic after descent = initial kinetic − potential lost = 50 − 40 = 10 J | Wrong formula: descent gain subtracted | Same selected subtype as new-sledge-1 |


## 89. Fresh procedure-availability result

Source: [runs/legs/gemma4-e2b/physics-multitrace-2026-11-07/FRESH-RESULTS.md](../../studies/03-round-1-single-pair-edits/multitrace-observations/run/FRESH-RESULTS.md), line 5.

| Criterion | Greedy | Sampled pass@1 | pass@2 | pass@4 |
|---|---:|---:|---:|---:|
| Correct requested calculation and conclusion |10.0%|18.75%|33.33%|52.5%|
| Also coherent physical explanation |5.0%|5.0%|10.0%|20.0%|
| Correct final triple only |12.5%|20.0%|35.0%|55.0%|
