# From a recorded decode to an investigation

Purpose: let Marcos and an AI point to the same operation, inspect its relationships, distinguish competing explanations, and propose a measured change. The output is an investigation plan with evidence requirements, not an automatic bottleneck verdict.

## Behavioral and weight relationships

The optional `weightInvestigation` member extends the report with the
`qgre.weight-case.v1` graph, documented in [WEIGHT-INVESTIGATIONS.md](WEIGHT-INVESTIGATIONS.md).
Its reading order starts at a selected output-token span and follows recorded
reply/request hashes, separate diagnostic token correspondences, activation
boundaries, named matrices and their checkpoint storage planes, then a proposed
intervention. Every graph endpoint is included in the export.

This graph preserves two different links: `compared_token_ids_with` is an observed
comparison across executions, while `expected_input_boundary` and
`expected_output_boundary` are architecture relationships. Neither means a weight
was causally responsible for the output. A matrix's stored coordinates may be
rotated relative to activation coordinates. Untimed captures and shape mismatches
remain visible. The case can be shared, exported, restored and supplemented with
paired outcomes, but it cannot execute or apply an edit.

## Profiling relationships

The `profiling` member of an inspection contains an optional
`qgre.profiling-evidence.v1` bundle. [PROFILING-LENSES.md](PROFILING-LENSES.md)
defines its eight visual lenses and producer requirements. The original inspection
remains the source of capture, operation and worker identities.

The new subgraph is:

`instrument ← measured_by — observation — observes_operation → recorded operation`

For readiness, `producer task — publishes → dependency — required_by → consumer task`.
For experiments, `experiment — investigates → evidence row`. Dependencies carry the
specific readiness generation; buffer rows carry the allocation generation. Every
observation names its collection pass through its instrument. Unmeasured relationships
remain absent. An imported claim is not independently authenticated by a matching tape hash.

The visual progression is a proposed investigation: find a late prerequisite, inspect
instruction supply and useful work, check supply and residence, then compare token
behavior under matched measurement conditions. This is a navigation aid, not an automatic
causal verdict. Numerical distributions add boundary evidence. An experiment retains its
hypothesis, proposed change, prediction, outcome and supporting row IDs, including refuted
outcomes, so another investigation does not silently repeat the same failed explanation.

Example bundles use `origin: illustrated` throughout and have an independent manual or
timed clock. They demonstrate how to read a richer capture without manufacturing missing
observations in the two supplied recordings.

## The visual story

| Question | Representation | Interaction |
| --- | --- | --- |
| Where am I? | Recording → layer → operation → selected worker; an enclosing-operation branch for nested stamps | Keep selection synchronized with the machinery |
| What occupied this worker? | Chronological strip of resolved work, wait, mixed and unassigned spans; the selected observation outlined | Select a span to seek; use the existing clock |
| What comes before and after? | Connected operation nodes; solid lines for recorded order, enclosing bracket for nesting, dashed missing-producer branch | Follow a node without moving time |
| Do workers arrive together? | One selectable dot per valid observation, positioned by exit time; first/median/last landmarks | Select a worker; explicitly seek to the last exit |
| Does this appear in both recordings? | Aligned median-to-last ranges for the same source record in both captures | Inspect either capture; do not label this A/B |
| What is measured and what is missing? | Connected architecture areas with measured, derived and unavailable evidence states | Select an area to reveal the distinguishing measurement |
| What should we do next? | Observation → alternative explanations → required evidence → conditional experiment → validation | Choose an architecture area; export the same investigation |

Use position and length to encode time, connection to encode relationship, and filled versus outlined shapes to encode evidence availability. No idle animation, speedometer, or invented utilization. The machine remains the spatial view; these diagrams expose the recorded relationships behind it.

## Entities and identity

The export is `qgre.decode-inspection.v2`. It is a relational document with normalized tables, typed links and a reading order. Every entity has a globally unique document ID. Raw source record IDs remain as attributes, namespaced by capture in entity IDs; the same `l4:qtail:2` in two captures is not the same observation.

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

`relations` contains directed `from` / `to` foreign keys. `story` contains ordered questions and entity references. `focus` points to the exact run/layer/operation/worker, architecture area, time and window. An observation is stored once; chapters and relationships refer to it. The selected investigation area survives a shared link and is the same area used in the exported investigation chapter.

## Relationship vocabulary

- `contains`: run → layer, layer → operation, run → worker. Recorded hierarchy, not execution order.
- `nested_in`: child → enclosing operation. Nested observations must not be summed with their parents.
- `recorded_before`: predecessor → successor, sourced from the capture. This is program order, not proof of a data dependency.
- `observed_operation` and `observed_worker`: observation → operation / worker.
- `derived_from`: measurement or segment → its input observations/segments.
- `candidate_area`: operation → architecture area. This is the initial heuristic based on role/kind, not a measured hardware binding.
- `investigates_area`: operation → architecture area explicitly chosen by the analyst. This replaces the heuristic link for the current selection without claiming a hardware cause.
- `investigation_checkpoint`: architecture area → a conceptual checkpoint. `inspect_next` links the suggested inspection sequence, not recorded execution order. For TMA this draws issue → completion → acquire → release, with every event still marked missing.
- `analogous_record`: same record ID in a second capture. Structural correspondence only; no controlled-comparison claim.
- `motivates`: an observation/measurement → hypothesis. Does not mean supports or proves.
- `requires`: hypothesis → evidence requirement.
- `tested_by`: hypothesis → conditional experiment.
- `informed_by`: area, requirement or experiment → source document.

Unsupported relations are absent. In particular, no `causes_wait`, physical `runs_on_sm`, transfer-completion or exact-build-source relation is manufactured. Unknown hardware/model-policy identities are null with the missing evidence explained.

## Export scope and accounting

The report includes all operations in the selected layer for both available recordings, direct neighbors and enclosing operations, plus every operation needed to reconstruct the selected worker's visible window. It includes all raw observations of these operations, including nonpositive intervals. Foreign keys must resolve within the document. `scope` reports included and available counts; excluded layers are not silently treated as empty.

Time remains in nanoseconds in each capture's clock domain; negative globaltimer offsets are valid. Cross-run plots compare median-to-last durations on one zero-based scale and never subtract absolute clocks. The worker arrival axis starts at that operation's first observed exit. Zero-length and reversed observations are preserved but excluded from duration and exit statistics. Non-finite values would also be invalid. Missing observations are not zero time.

Worker time is the union of the existing resolved innermost intervals, clipped to the selected window. Parents and nested observations are not added together. Work, wait, mixed and unassigned sum to the window duration. Mixed stays mixed. A software wait interval is not a hardware warp-stall sample. Worker time is not GPU utilization or token latency.

Exit statistics are computed over valid raw observations of one operation. Median-to-last exit is not recoverable latency. Early entry, unequal assigned work, different roles, and later completion can all produce a spread. Two-pass Q-tail stamps may describe the last visit; attention interior stamps may describe only the final tile. The original capture semantics remain attached.

## Observation → explanation → action

1. Establish identity and measurement scope. Bind the capture to a binary, policy, checkpoint, workload, GPU and instrumentation settings before attributing a change. The present capture has tape/table hashes and separate source anatomy, not every binding.
2. Locate the interval. Inspect its worker distribution, enclosing operation and recorded neighbors. Preserve unusual observations and missing coverage.
3. Enumerate alternatives. A visible wait can come from late input, skewed assigned work, or coordination cost. A long operation can be compute throughput, data delivery, insufficient independent work, spills, or instrumentation effects.
4. Request the least additional evidence that distinguishes those alternatives. Use the domains below. The current recordings alone cannot advance a candidate to a confirmed hardware bottleneck.
5. Specify one conditional experiment. State the exact policy/code region, predicted primary change, companion metrics that must not worsen, and what would refute the hypothesis. Do not recommend blindly enabling a mechanism.
6. Compare controlled runs on the production megakernel path, preserve correctness, then re-profile. A lower stall count or higher cache hit rate without better end-to-end behavior is not a successful optimization.

## Architectural domains and discriminating evidence

| Area | Visible clue, not diagnosis | Evidence to acquire | Conditional action and refutation |
| --- | --- | --- | --- |
| Host and submission | Gaps in whole-request progress; current layer-only capture cannot see them | Nsight Systems CPU/CUDA/NVTX timeline, launch/queue/execution split and request boundaries | If host round trips dominate, change host scheduling or graph preparation. Refuted if wall time does not improve or the gap is GPU queueing |
| Coordination and assignment | Long software waits or separated worker exits | Producer/consumer IDs, readiness generation, equal-work grouping, all barrier arrivals, aligned phase clocks | If unequal assigned work delays a required producer, change assignment in policy. Refuted if critical completion is unchanged or another group slows |
| Memory and reuse | Long weight-consuming region | Phase-local DRAM/L2 bytes and throughput, demand-only hit rate, sectors, working set, traffic class and clock | If redundant demand traffic is proven, change reuse/layout. If bandwidth is saturated, remove bytes. Refuted if bytes fall but wall time does not or read amplification rises |
| TMA and staging | Waits near a transfer-consuming region | Issued transfer ID, descriptor/box, bytes, completion generation, consumer acquire, release and slot reuse | If exposed delivery latency is proven, change overlap/depth in policy. Refuted by more occupancy pressure, later completion, incorrect slot reuse or increased wall time |
| Warp issue and execution | Long arithmetic or mixed region | SchedulerStats, SourceCounters, WarpStateStats, per-pipe throughput, PC/SASS correlation and assigned work | If missed issue slots dominate, investigate the producer of the stalled operand. If a pipe is saturated, remove work on that pipe. Refuted if only samples move while duration does not |
| Registers and shared memory | Uneven progress or inability to keep independent work ready | Launch limits, compiled register/SMEM use, actual spills and bank-conflict wavefronts, eligible/issued warps | If resource pressure is limiting, shorten live ranges or change staging/geometry. Refuted by extra spills, lower issue, or slower completion |
| Numerical correctness | A faster candidate changes the emitted answer or state | Existing exact cases, equation invariants, deterministic replay and the production coherence gate | Keep the approved low-precision carrier and megakernel path. A correctness regression rejects the optimization; wider recompute is not the oracle |

Requirements are reported as `missing` in this viewer unless that exact evidence exists in the capture. The viewer does not run profilers, launch GPU experiments, or infer counter values from animated machinery.

## What the NVIDIA folder contributes

The source folder is [`.dev/research/nvidia-pro`](../research/nvidia-pro/README.md). It is a historical local reference, originally written against September 2026 tools. Its `[V]`, `[S]`, `[R]`, `[D]`, `[H]` labels are retained in citations, not upgraded to freshly verified facts.

- [07 — profiling toolchain](../research/nvidia-pro/07-profiling-toolchain.md), §§1–3: match the instrument to the question; separate application timing from counter collection.
- [08 — metric cookbook](../research/nvidia-pro/08-ncu-metric-cookbook.md), §§2–9,12: clock and metric scope, sectors and demand traffic, spills, scheduler statistics and source correlation. Query supported metrics on the target installation; names and derived sections can change.
- [16 — optimization playbook](../research/nvidia-pro/16-optimization-playbook.md), steps 1–6: classify then act, and reclassify after a change. Its 40/80-percent splits are heuristics, not universal boundaries; a large occupancy gap does not uniquely prove load imbalance.
- [02 — memory hierarchy](../research/nvidia-pro/02-memory-hierarchy.md), §§1–3: scope of storage and the resource cost of latency hiding. Historical capacities/latencies are not assigned to an unidentified captured device.
- [04 — async movement](../research/nvidia-pro/04-async-movement-tma-clusters.md), §§2–3: distinguish issue, completion and consumption; retain slot generation and dependency identity.
- [19 — L2 prefetch](../research/nvidia-pro/19-l2-prefetching-on-sm120.md), §§3–4,6: distinguish stale front, insufficient issue, and a gate that competes with demand traffic. Its rig outcomes are conditional, not a directive to enable the issuer. Chapter 16 records production regressions and disabled issuers; those historical observations must not be dropped.

## Official cross-checks and conflicts

Consulted November 5, 2026:

- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/): active, eligible and issued warps are different; stall samples matter in the context of missed issue opportunities. Replay and counter collection can perturb execution.
- [Nsight Compute Triage Guide](https://docs.nvidia.com/nsight-compute/ComputeTriage/): use multiple signals, trustworthy scope, source correlation and measured duration. Samples at a consumer do not identify the causal producer automatically.
- [Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html): compute capability 12.0 has different resource limits from 10.0. The guide's broad cluster discussion and the local chapter 04's negative DSMEM probe are not silently reconciled; verify the exact device/toolchain before proposing cluster-memory use.

Generic vendor advice to split kernels, widen precision, or assume datacenter Blackwell mechanisms is not transplanted into QGRE. Current project requirements keep the single production megakernel, policy-owned geometry, narrow carriers and existing correctness checks. This document defines investigation semantics; it does not install a new policy gate.
