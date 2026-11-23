# One original full-turn reference

Completed November 15, 2026 at 8:41:13 PM CT, in 638.61 seconds. The
original-only CPU compiler read all fifteen declared fitting successes and wrote
fifteen original turn arrays. No answer was generated, no originally wrong trace
was evaluated and no model weight changed.

The selected observed reference is `expanded-004-h1`: 560 events, all 105 merges,
1,536 coordinates. Selection score: 0.03100051941122387. It is an actual original
trace, not a population mean. Its nonself component RMS tolerance and complete
correspondence are fixed by `reference.json`, sha256
`1be12cf319a120c009917218f3567a2fbed1fa48b5fc7dc1f90b1dbc3c5cc80d`.
The declaration binds the complete compiler import closure; commit `ab41f610`
holds the source used. Later loader preparation does not relabel those source
bytes or rerun the compilation.

## What the coverage actually shows

`original-coverage.json` records every nonself witness under both the frozen
tolerance and a tolerance recomputed without that witness. Selection and paths
still use the full original inventory; the latter is not independent validation.

All fourteen nonself successful traces have at least one outside component.
Together, 8,246 of their 8,695 distinct events are outside at some merge/component,
under either version. The closest witness, `discovery-01-channel-h1`, has 312 of
560 outside events; every member's exact counts and excess objective are in the
coverage artifact. There are 2,096,640 zero-tolerance components among the
reference's 90,316,800 components; zero and near-zero bounds remain literal.

This is observed evidence that the complete-zero-excess condition is stricter
than any nonself original successful trace in this inventory. It does not prove
mathematical unreachability, nor label every outside operation incorrect:
outcome-correct answers are not certified flawless computation, and chronological
turn correspondence is not proof that two events perform the same reasoning.
The reference itself has zero discrepancy. No multiplier, rank criterion,
alternate reference or tolerance is chosen from these results.

## Scope and provenance

This is a completed reference compilation and diagnostic, not capability closure,
a native correction admission or a transfer result. The trace is
`unit(merged)-unit(incoming)` from original native observations, with full turn
amount. The same fixed component boundary is intended to supply direct requests
and fresh-trace closure in a separately declared one-prompt experiment.

The final source check before execution passed 831 pipeline tests and 108 subtests,
with one Torch deprecation warning, plus focused Pyright. Eighteen focused tests
covered the turn definition, reference arithmetic and compiler/coverage connection.
Eli's independent mathematical assessment and its corrected assumptions are
recorded in `records/retrospectives/2026-11-15-turn-pair-binding-astra.md`.

The final `result.json` and complete `reference.json` are authoritative.
`extraction.json` and `selection-progress.json` are incremental progress records;
their incomplete flags are not final run status. All outputs and all original
inputs are retained.
