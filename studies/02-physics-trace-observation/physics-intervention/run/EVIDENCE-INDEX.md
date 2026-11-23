# Evidence relationships

Experiment: a single recorded attention contribution transplanted between two
actual BF16 executions. No checkpoint changes. The tracked report is
`../../../../.dev/decode-machine/PHYSICS-TRACE-INTERVENTION.md`.

Read the evidence in this order:

1. **Question → expected calculation → observed behavior.** `PLAN.md` sets
   eligibility and endpoints before inference. `baseline-cases.json` holds four
   new questions, exact values and their expected calculations. `original-cases.json`
   holds the reused block pair. `baseline-behavior.json` separates final triples,
   explanatory contradictions and truncation. New actual outputs live in
   `baseline-plain/<case>/result.json`; original block outputs live in the prior
   `../physics-trace-bf16-2026-11-06/study-capture/<case>/result.json`.
2. **Behavior → eligibility → shared prediction.** `eligibility.json` records
   all three pairs, prefix lengths, exclusions, and original block prediction 235.
   It was written from ordinary responses before donor-vector inspection.
3. **Prediction → recorded source → transplant.** `sources.json` joins case,
   prediction, actual graph node/event, position, tensor-file byte offset, exact
   row checksum and generated-prefix checksum. `sources/*.bin` are the four
   immutable recorded rows. `arms.json` declares eight arms; four self-copy arms
   plus two no-intervention cases are in `qualification-cases.json`; four donor
   cases are in `donor-cases.json`.
4. **Transplant → actual next operation → downstream changes.** Every run under
   `qualification/` or `donors/` has its exact `case.json`, source-linked
   `graph.json`, `records.jsonl`, `tensors.bin`, and `intervention.json`.
   `before.bin`, `after.bin`, `residual-before.bin`, and `consumer.bin` preserve
   the intervention and the native immediate ADD. `connected-change.json` joins
   intervention and baseline node/event/byte offsets at matched input histories.
   It includes all retained boundaries at the affected and following prediction.
5. **Downstream changes → selected tokens → final outcome.** Each run's
   `result.json` keeps complete continuations, token IDs/exact byte pieces,
   complete-vocabulary score hashes, top eight scores, stop state and settings.
   `summary.json` links all four donor outcomes, one affected score decision per
   run, and the unchanged final triples. Hash equality proves equality without
   reconstructing logits. Numerical distance is not semantic importance.
6. **Persistence question → architecture → interpretation.**
   `cache-dependencies.json` preserves exact lines from `donors.stderr.txt`:
   20 shared-KV layers, earlier producers, and later reuse of 13/14. The native
   source snapshots `instrumentation/engine-gemma4.cpp` and
   `instrumentation/engine-llama-hparams.cpp` match the pinned build commit.
   No runtime KV-slot byte dump was captured. The report explains why this
   one-time downstream intervention has no persistent cache route.

## Fidelity and limits

`baseline-validation.json`: four ordinary/capture comparisons, 2555 decisions.
`qualification-validation.json`: six controls, 2790 decisions, identical tokens,
all-score hashes and retained boundaries. `donors-validation.json`: four donor
runs, 1860 decisions, exact before/after readback, immediate consumption, unchanged
upstream boundaries and pre-intervention histories. Associated stdout/stderr
files preserve validator outcomes.

Selected payloads cover all layer exits and final normalization, plus selected
internal boundaries in layers 12–16. Fused attention internals, individual weight
contributions, and runtime cache-slot lineage remain gaps. The new factory UI
does not import this research recording format. No general physics capability
score, recovered geometry or successful model edit is claimed.

`provenance.json` binds the prior unchanged checkpoint/engine to the new instrument
and preserved analysis sources. `COMMANDS.md` records how the study ran.
`REVIEW.md` records independent reads and distinguishes them from runtime checks.
