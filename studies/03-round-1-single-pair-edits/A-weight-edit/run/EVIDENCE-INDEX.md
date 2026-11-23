# Actual weight edit: completed evidence

The edited native BF16 checkpoint repaired the target calculation and a sled calculation, while introducing a wrong final verdict in the 4 kg variant. Complete correct numerical chains: 7 to 9 of 28. Correct explicit verdicts: 16 to 16. No training or runtime activation replacement.

## Begin with the connected account

- [Report](../../../../.dev/decode-machine/PHYSICS-TRACE-WEIGHT-EDIT.md): method, rounding, all-case outcomes and native propagation.
- [Plan](PLAN.md): original target and exact declared edit; hardware history appended.
- [Independent review](REVIEW.md): instrument, editor, analysis and full-response assessment.
- [Commands](commands.json): successful commands and exit codes.
- [Provenance](provenance.json): original/edited checkpoint identities, engine, settings and hardware history.

## From the observed execution to changed weights

1. [Cases](cases.json) gives all 28 exact prompt-token sequences, expected calculations and previous baseline locations. [Source cases](source-cases.json) selects the two original block cases.
2. original-plain/ holds all 28 ordinary original-model responses. All 14,578 decisions reproduce earlier token and complete-vocabulary score hashes.
3. source-capture/ records the native input, weight, projection, normalization and downstream graph for the original pair. All 930 decisions match ordinary inference.
4. [Conditioning](conditioning.json) joins each of six source vectors to its case, prediction235, native node, event, source byte offset and SHA256. vectors/ contains these activation rows.
5. [Dry edit](edit-dry.json) records native-rounding realization. [Applied edit](edit-applied.json) identifies the actual matrix, changed native values, full-file comparison and both model hashes.

Checkpoint: models/gemma4/gemma4-e2b/gguf-bf16-physics-weight-edit-2026-11-07/attention-output-16.gguf, relative to project root. projection-before.bf16 and projection-after.bf16 beside it contain actual native weight bytes; no weight files are stored in this runs directory. Original checkpoint remains at models/gemma4/gemma4-e2b/gguf-bf16-bartowski/google_gemma-4-E2B-it-bf16.gguf.

## From changed weights to ordinary behavior

- edited-plain/ holds all 28 native edited-model continuations, every vocabulary-score hash and top eight candidates per prediction.
- [All-case comparison](comparison.json) joins each case to both complete responses, first differences, matched-history lengths, changed-score counts and capture fidelity.
- [Original judgments](original-assessment.json) and [edited judgments](edited-assessment.json) separate the numerical chain, final verdict, explanation inconsistencies and truncation. Ten changed responses were independently reviewed in full.
- [First divergence](first-divergence.json) preserves the native reached/gained candidate scores at target prediction269 with matching incoming history.

## From changed weights through connected internal states

- edited-capture/ holds all six cases enumerated by [capture cases](capture-cases.json). All 3,396 decisions match edited ordinary inference exactly.
- [Paired trace](paired-trace.jsonl): 67,754 matched recorded boundary comparisons for the two original source cases, joined by case, input position, native name and original/edited graph IDs, events and byte offsets. Includes prompt positions. After a generated-history divergence, paired comparisons stop.
- [Trace summary](trace-summary.json): progression across all 720 comparable generated predictions in those two cases; raw output, normalized contribution, later layer exits and final state. The coefficient is the declared unrounded update's algebraic input response, not a capability probability.
- The four numerical variants have edited captures and plain/capture fidelity checks, but no paired internal comparison in this report.

## Retained unsuccessful attempt and instruments

- [GPU1 qualification](gpu1-qualification-failed/COMPARISON.json): failed cross-card comparison, seven completed cases; stopped before source capture or weight edit. Excluded from conditioning and original-versus-edited comparisons.
- instrumentation/ contains the exact C++ observer/editor, launch script, extraction/analysis scripts and binaries. [Manifest](instrumentation/MANIFEST.json) gives byte lengths and SHA256.
- Phase stdout/stderr files retain native loading, cache, launch and completion logs.

All completed comparisons use RTX5080 device0, the same original engine binary and inference settings. No new candidate was tried after these outcomes.
