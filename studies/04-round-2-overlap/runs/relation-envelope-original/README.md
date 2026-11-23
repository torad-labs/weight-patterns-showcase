# Original-success relational envelope

Completed November 15, 2026 (CT), under declaration revision 1. Source-only,
original-data compilation: no model was changed and no engine ran.

## Result

All fifteen fitting successes have **zero direct hinge at every one of their
971,775 original cells**, including the five outside the donor bank. All fifteen
wrong members retain positive loss: their sum is **0.052655286273596835**, with
138,187 active cells out of 1,102,080. The full domain remains thirty recipients,
19,751 rows and 105 merges (2,073,855 cells).

Under the prior mean calibration the same wrong members' sum was
0.5320145427257861 and the successes' sum 0.2866945943729895. This is a change in
calibration, **not a model improvement**. Roughly a tenth of the old wrong-path
loss remains. The new calibration prevents direct requests on the original known
successful paths but does not prove that the remaining discrepancy is exclusively
capability or that the original nonself mates follow valid routes.

The ten donor members, all thirty assignments, descriptor and original mass stay
fixed. Each of the fifteen original successes supplies its mapped local own-mate
disagreement, its own observed row included; their maximum is the new tolerance.
The witness contributions are retained. This is an empirical fitting envelope,
not leave-one-success-out validation or a measurement on no-fit data.

## Frozen files

- `reference/reference.json`: `937ff3ec2f53b17e6a9a42bba0cc3f4edc7a6ec7b93c6fbd240486d881ebea50`
- `reference/tolerance.npy`: `d60f280c645bd15d70a00a6938c7e0e77fae6cc86fc53240da5889e1d68fb4d7`
- `reference/original-loss.npy`: `143d908b254ce4671d4439e1c0cd88a9818903cab7a33c01a6864b94308405ce`
- `reference/tolerance-contributions.npy`: its hash, shape and witness order are in
  `reference.json`; the local files remain here, not in Git.

The mean-calibrated reference and its trials are unchanged. Every wrong member's
loss and active-cell count, every successful member's zero, source bindings and
coverage are in `result.json` and `declaration.json`.

## Preparation and arithmetic disclosure

The five original synthetic tests passed before calculation. The first real-data
attempt refused negative entries in the stored squared-moment distance table,
before writing a reference or result. Revision 1 records that failure and the
source correction: map those squared distances to their exact mathematical range
at zero when forming witnesses, not to a chosen noise floor. A sixth test covers
that case; all six pass. `result.json.squared_distance_arithmetic` preserves the
negative-entry count and lowest stored value. The old mean-calibration comparison
uses its original stored arithmetic and is not retrospectively regraded.

The declarer's first invocation also exposed and fixed an incorrect repository-root
parent count before any declaration was written; the verifier then refused a prose
`measured_from` instead of a real path, corrected before the prospective commit.
No verifier was bypassed. The editor initially reported an unresolved package
import; the focused Pyright check from `code/signal-trace` with the declared
Python dependencies reports zero errors and warnings for the compiler and tests.
The six runtime tests import the real module.

## Boundary

A zero native reference hinge does not itself establish zero Torch-surrogate
requests: the two forwards differ numerically. The next declared proposal must
keep the measured native gap distinct from its approximate derivative and trial
readings. Shared weights and attention can still move a zero-request path when
other paths are corrected. No first step under this envelope, native loss change,
long-loop result, repair or closure is claimed here.
