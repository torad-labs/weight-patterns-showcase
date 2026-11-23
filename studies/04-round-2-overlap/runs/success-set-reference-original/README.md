# Original-only successful-set reference

Prospectively declared November 15, 2026 (CT), in `26e0e322`.
The scientific choice is `source-records/objective.md`; the exact source-only admission
is `source-records/admission.md`. This run executes no model and reads no no-fit rows
into arithmetic.

## Phases

1. `reference_turns.py` materializes the thirty fitting members' complete prompt turns
   and verifies their original answer exports. Only the fitting slices of the retained
   incoming-state files enter arithmetic; whole-file hashes bind storage provenance.
2. A dated declaration revision binds the produced `turns/turns.json` hash, without
   changing any mapping rule, parameter, membership or prediction.
3. `compile_success_sets.py` compiles the immutable native-point memberships and
   normalization. It writes `reference/manifest.json` and `result.json` only on completion.

Source inventory: 30 fitting members, 15 successful donors, 19,751 native rows,
5,159 prompt rows, 105 merges. These counts came from the original native reading,
not an expected-output list.

## Declaration correction before execution

The first verifier call refused four metadata paths outside the allowed data roots
and the not-yet-existing turn directory. No producing phase had started. The original
declaration is preserved as `declaration-before-input-scope.json`; revision 1 retains
identical-byte metadata under `source-records/`, names the original bindings, and
admits the run's own intermediate through its existing root. The verifier then passed.
No verifier rule or refusal was bypassed.

## Current state

The first declared turn-producer attempt was refused by `check_native_run` before
production because `state-attention-0.f32` required archive re-verification. The
prescribed restore completed, and the unchanged guard admitted the next attempt.

Turn production completed: 30 prompt files, 5,159 prompt rows and all 105 merges;
3,150 of 3,150 answer arrays match the original exports, and 671,534 of 671,534
retained layer-exit rows match the original reading. The producer manifest's SHA-256
is `af21cb435e5fa6610e7e0c72199682226b0ce1e121b417e569f0fc71b2549923`.
Its actual hash is now bound by the dated declaration revision, with the preceding
declaration retained as `declaration-before-turn-binding.json`.

The immutable membership compilation completed. Its manifest SHA-256 is
`370eba55f7a8dfc2d3f6418a21debbfce78246aeb0f45d024084f6c9fca7dd38`.
The artifact retains 317,773 mapped native points across all 19,751 original frames,
30 members, 15 donors and 105 merges. `result.json` records the source-counted outcome.
No nearest-point loss, new-loop starting measurement or native execution was made.

The turn arrays and reference arrays/path table are bulk data for the existing bucket
backup path, not Git; manifests, row metadata and declarations remain in Git. Local
files stay available for active consumers, and no release is named by this completion.
Performance reports the complete run backed up and read-back verified at 2:25 AM CT
November 15: 47 files, all local copies retained. No data was released.
