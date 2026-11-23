# Evidence map

Start with [REPORT.md](REPORT.md). All paths below are relative to this study unless stated otherwise.

| Question | Source |
| --- | --- |
| What was proposed before running? | PLAN.md, design.json, SELECTIVE-PLAN.md |
| Which new prompts and budgets? | cases.json (A32), confirmation-cases.json (B8), candidate-cases.json (previous28+B8) |
| What was actually changed? | selective-applied.json, selective-dry.json, selective-conditioning.json, selective-key.f32, editor-compatibility.json |
| What did native storage rounding retain? | stored-protection.json |
| What did every response say? | RESPONSES.md, CANDIDATE-RESPONSES.md; original result.json files under each arm |
| How were responses scored? | family-assessment.json, candidate-assessment.json, REVIEW.md |
| Where did the computations differ? | connected-traces.json (A four cases), candidate-traces.json (B two cases) |
| How do paired states connect to raw evidence? | Each trace measure has graph node, capture event, tensor byte offset, width and source directory; raw graph.json, records.jsonl and tensors.bin remain in those directories |
| Was capture faithful? | Exact token and full-score-hash assertions in analyze.mjs and candidate-traces.mjs; fidelity records in both trace exports |
| Which programs and binary identities? | provenance.json, instrumentation/, COMMANDS.md |
| Where is the implementation? | ../../../../code/physics-edit-footprint-2026-11-07/ and ../../../../apps/decode-machine/src/trace-* |
| Where is the figure? | ../../../../figures/exports/physics-edit-footprint-2026-11-07--two-edits.svg and .png |

The native run phases are original-capture2, edited-capture1, original-plain32, edited-plain32, confirmation-original8, candidate-plain36 and candidate-capture2. Existing A source/edited captures and old28 original outputs are reused from ../physics-weight-edit-2026-11-07, with their source identities retained. These phase sizes are executions, not independent tasks.

The original checkpoint, A and B live in the established model store, reached through the research root's models link. Raw large tensors remain local and are not duplicated into Git. The prepared evidence exports, complete responses, assessments, source snapshots and provenance are versioned explicitly. The app export is A's four-case view, not a claim to cover B or all new-family responses.
