# Experiment map

There are 22 indexed studies under `runs/legs/gemma4-e2b/`, plus the supporting `long-context-capture-7263027c5-gpu1-2` directory and the incomplete `signal-trace-repair-2026-11-07` recorder work. The latter is not a completed hypothesis test. This index is an ordering of the work, not a claim that each earlier method implemented the current hypothesis. File totals enumerate regular files without following links or dependency packages; byte totals sum logical file sizes, not unique disk blocks. The footprint, connected-preservation, background-sensitivity and many-trace studies followed the original 18-study relocation.

| Study | Place in the work | Reading note |
| --- | --- | --- |
| [decode-observatory-30k-2026-11-05](../../studies/01-early-exploration/decode-observatory/run/ANALYSIS.md) | Recorded workload and visual observatory | 30,000 new tokens across 187 continuing-conversation turns; replay and profiling evidence. |
| [ghost-inspection-2026-11-05](../../studies/01-early-exploration/ghost-inspection/run/analysis.md) | Early weight instrumentation | Historical weight/activation inspection; does not by itself establish a residual trace. |
| [ghost-retrieval-2026-11-05](../../studies/01-early-exploration/ghost-retrieval/run/PLAN.md) | Early intervention experiment | Original plan, prediction, amendment, captures and outcomes; predates the clarified trace direction. |
| [individual-edit-2026-11-06](../../studies/01-early-exploration/individual-weight-edit/run/PLAN.md) | Early individual-edit investigation | Historical candidate and measurement artifacts; keep separate from the later connected-trace study. |
| targeted-lookup-2026-11-06 | Targeted lookup attempts | Selection and three predictions with native-edit records; historical approach, not the final trace method. |
| [mechanism-study-2026-11-06](../../studies/01-early-exploration/mechanism-study/run/COLOR-PROBE.md) | Lookup mechanism investigation | Mechanism and color probes before the engine comparison and capability-focused physics studies. |
| [lookup-failure-family-2026-11-06](../../studies/01-early-exploration/lookup-failure-family/run/RESULTS.md) | Repeated lookup failure family | Question-family observations and full completion examination, including limits of the initially assumed pattern. |
| [behavior-pattern-20-2026-11-06](../../studies/01-early-exploration/behavior-pattern/run/STOPPED.md) | Twenty-question observation plan | Retains questions and the stopped-run account; do not count planned work as completed. |
| [llamacpp-lookup-comparison-2026-11-06](../../studies/01-early-exploration/llamacpp-lookup-comparison/run/RESULTS.md) | Inference-path comparison | Same lookup questions on the llama.cpp fork; pipeline and checkpoint distinctions retained. |
| [llamacpp-preference-edit-2026-11-06](../../studies/01-early-exploration/llamacpp-preference-edit/run/REPORT.md) | Native Q4_K preference edit | Actual fixed-scale Q4 edits and discovery/withheld/control outcomes; predates the current trace method. |
| [lookup-pattern-observation-2026-11-06](../../studies/01-early-exploration/lookup-pattern-observation/run/REPORT.md) | Lookup observation and weight products | Historical observation record and predictions. It is not automatically evidence for the later residual-trace definition. |
| [distributed-pathways-reference-2026-11-06](../../studies/01-early-exploration/distributed-pathways/run/REFERENCE.md) | Earlier theory reference | Contains supersession notices. Current meaning is in the full residual-trace theory and November 7 clarification. |
| [path-inspection-2026-11-06](../../studies/01-early-exploration/path-inspection/run/REPORT.md) | Superseded interpretation | The frozen-pattern-only interpretation excluding inference traces is superseded. Original measurements are preserved as history. |
| [physics-trace-2026-11-06](../../studies/02-physics-trace-observation/physics-trace-pilot/run/REPORT.md) | Physics pilot and recorder qualification | Six situations in three forms; prompt-format and capture-method amendments are part of the record. |
| [physics-trace-followup-2026-11-06](../../studies/02-physics-trace-observation/physics-trace-followup/run/REPORT.md) | Q4 connected-trace follow-up | 24 questions, 12 pairs, 846 eligible decisions; relative FF14/FF15 progression and behavioral limits. |
| [physics-trace-bf16-2026-11-06](../../studies/02-physics-trace-observation/physics-trace-bf16/run/REPORT.md) | BF16 connected-trace follow-up | 24 questions, 1,708 eligible decisions, matched comparisons and calculation anchors. Completed November 7 CT. |
| [physics-intervention-2026-11-07](../../studies/02-physics-trace-observation/physics-intervention/run/REPORT.md) | One-time activation substitution | Changed one contribution once. No emitted tokens changed. This was not an actual weight edit. |
| [physics-weight-edit-2026-11-07](../../studies/03-round-1-single-pair-edits/A-weight-edit/run/EVIDENCE-INDEX.md) | Persistent native BF16 weight edit | Actual edited checkpoint; 28 comparisons, two numerical repairs, one verdict regression, and downstream traces. |
| [physics-edit-footprint-2026-11-07](../../studies/03-round-1-single-pair-edits/B-selective-footprint/run/REPORT.md) | Connected footprint and selective candidate | Four-case native comparison, A on32 new prompts, B on28 previous plus8 confirmation cases. B's protection restores the damaged final triple but loses the target repair; one confirmation numerical repair. |
| [physics-connected-edit-2026-11-07](../../studies/03-round-1-single-pair-edits/C-connected-preservation/run/REPORT.md) | Preserve connected passages in saved edit C | Selective conjunction passed; known repairs retained and known verdict regression avoided.52 C cases,8 fresh baselines,4 qualified captures. No gain on4 fresh physics prompts. |
| [physics-background-sensitivity-2026-11-07](../../studies/03-round-1-single-pair-edits/D-background-sensitivity/run/REPORT.md) | Background sensitivity and saved edit D | Mean component dominates A/C background action. One adjusted read reduces held-out RMS action85.5% versus A while retaining known repairs and controls; no gain on four excluded physics questions. |
| [physics-multitrace-2026-11-07](../../studies/03-round-1-single-pair-edits/multitrace-observations/run/REPORT.md) | Retained D observations; E withdrawn | D discovery, classification, raw captures and 40 greedy plus 160 sampled responses remain. No E overlap or repair claim is retained. |


## Latest reading order

The invalid parallel round and E have both been withdrawn. [Removal of E](../../journal/2026-11-07-e-withdrawal.md). Round two is tracked in [ROUND-2.md](../../studies/04-round-2-overlap/ROUND-2.md); no round-two edit built from the overlap exists yet. Original A–D and the independent D observations remain.

Current clarification → full theory → Q4 follow-up → BF16 follow-up → activation intervention → persistent weight edit → [footprint and selective edit](../../studies/03-round-1-single-pair-edits/B-selective-footprint/run/REPORT.md) → [connected-preservation candidate C](../../studies/03-round-1-single-pair-edits/C-connected-preservation/run/REPORT.md).

## Historical interpretation

The [path-inspection report](../../studies/01-early-exploration/path-inspection/run/REPORT.md) still contains the frozen-pattern-only interpretation. Preserve it as the record of that investigation, but do not use it to define the present research target. The earlier [reference](../../studies/01-early-exploration/distributed-pathways/run/REFERENCE.md) and [handoff](../../studies/01-early-exploration/distributed-pathways/run/MODEL-HANDOFF.md) already identify their supersession.

## Structured navigation

[studies.json](../../records/studies.json) joins each study to its reports, figures, top-level data files, run-local source files, and on-disk totals. [source-tables.json](../../records/source-tables.json) retains table text with the original file and line. Neither flattens incompatible denominators into one score.
