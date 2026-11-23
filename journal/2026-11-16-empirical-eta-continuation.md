# Empirical eta continuation

Marcos directs continuing from the exact improved pilot, without another approval
or a perfect-gradient search. The fixed empirical rate is 0.26989262219171917,
derived in the preceding [experiment](2026-11-16-active-calculation-experiments.md).
It multiplies each newly measured response scale before BF16 rounding. The full
fresh computational trace supplies the next weight derivative; reference and
tolerance remain original and fixed. Neither output tokens nor their agreement
judge an update.

The pilot changed 28,578 stored values and measured held and fresh native loss
0.0002879419773841299, versus reconstructed5's 0.00028798285443879684. The
gradient was recomputed at the actual native5 anchor, so this is not an isolated
same-gradient proof that scaling alone fixed the earlier failure.

The first continuation stopped before its baseline/update at11:00:58CT: eight
CPU threads changed the scalar reduction by one FP64 ULP, while all560×105
weighted residual cells were bit-identical. The saved pilot, failed attempt and
generation0 exit checkpoint remain. V2 uses the actual pilot's one CPU thread,
checks the baseline cell map and the tensors returned by the actual reverse/JVP
calls exactly, and records scalar reduction differences. No epsilon, scientific
tolerance, derivative, response or finite-update formula changed. Five focused
CPU fixtures and independent source-bound review qualify this narrow correction:
`records/retrospectives/2026-11-16-empirical-eta-comparison-review.json`.

The sole active controller `weight-patterns-eta-continuation-2.service` launched
at11:09:10CT. Source `loop_v2.py` SHA256
`f2021014e2774822689287446cf9db0ea58b6f371346b9cdeb6ed240320b1a21`
is bound in the run. One resident model and RAM GGUF receive all70 matrices;
each complete fresh native trace is measured at all105 merges. Only exit saves
a sparse checkpoint. Historical working9 remains frozen and untouched.

Observed fresh losses through11:15:14CT:

| Local update | Native computational loss |
| --- | --- |
| 0, exact pilot | 0.0002879419773841299 |
| 1 | 0.00028793946003340923 |
| 2 | 0.0002879763997599066 |
| 3 | 0.0016194367826251345 |
| 4 | 0.0002870383597568799 |
| 5 | 0.0016111752750417937 |
| 6 | 0.0016036440665457056 |

All are complete naturally ended measurements. Update4 improves the low;
updates3 and5 show large excursions. This does not establish convergence,
stagnation, divergence or a scale-only correction. The controller continues
under Marcos's manual fresh-loss judgment, allowing variation. No rate change,
plateau threshold, reference change or text inspection is introduced.

The existing host observer publishes the complete curve as `native.objective`
in experiment6, parent `a3b7c18618284d8f91c8467d9ec8d5ba`. Current instrumentation
is record-based; it does not claim internal operation spans. The external
observer's script/declaration binding was repaired without changing training.

Weight-research tests the first feed-forward local response from saved native
inputs and exact BF16 matrices on GPU1. Same-weight held/fresh controls are
bit-identical across all captured inputs, outputs and residuals. Pilot native
response still differs strongly from the reconstructed derivative downstream
of merge0; native sensitivity is not qualified merely by an anchored baseline
or a Torch JVP/VJP identity. The ongoing local test separates observed input
movement, local smooth derivative/finite response and native execution.

The authoritative growing records are
`studies/04-round-2-overlap/runs/vector-chance-empirical-eta-continuation-2/loop/`
and `progress.jsonl`; this table is a time-bounded observation, not a final result.

## Acquisition boundary at11:19:36CT

Updates7 and10 reached new complete lows0.00028614789135137156 and
0.00028607873789714854; updates8/9 rose to0.0019934662331258885 and
0.0019928707426766345. Update11 was fully installed at11:19:12CT, but its
fresh acquisition reached the existing768-step recorder bound, with940 observed
computational events and no natural end. Its partial loss0.0035864955016763054
is retained explicitly as partial and is not a whole-path comparison.

The reviewed loop exited cleanly at11:19:36CT and saved its final sparse weights
(SHA25615bf11ab528050cd5889a82be162e0da8d90cb8a92a675d4d18682318cd706e7
for final-state.json). No stop signal was sent and no installation was pending.
This is an acquisition boundary, not a plateau, closure or a hypothesis verdict.
The current source fixes native context capacity at2048 and rejects input plus
acquisition budget above that capacity. The implementation peer is assessing a declared capture
of this exact retained state using the existing remaining context capacity; no
guessed widening, model reset, new update or changed reference is implied.

## Exact retained-state continuation

The context-capacity capture completed naturally at956events, only16 beyond the
old940-event observation. All940 prior incoming and merged events at105merges
match exactly. Complete native loss is0.0021828841491281573; this is a new full
measurement, not relabeling the earlier partial loss. No weight changed in that
capture. The context remained2048; only its declared acquisition budget changed
from768 to1875, derived as2048 minus the unchanged173-event input.

The independent reviewer checked the exact retained11/capture handoff and unchanged loop interface.
The implementation peer launched `weight-patterns-eta-continuation-3.service` at11:32:43CT, using
unchanged loop_v2.py and the same calibratedeta/reference/tolerance. Local0 means
retained11, not original weights or a rollback to the earlier low. Actual launch
binding: `runs/vector-chance-empirical-eta-continuation-3/execution.json`; review:
`records/retrospectives/2026-11-16-empirical-eta-third-handoff-review.json`.

## GPU residency correction

Continuation3 reproduced the full baseline exactly and completed its weight
derivative, then failed before directional response or installation: a46MiB GPU
allocation failed while obsolete frozen PID1984536 retained5.33GiB and the new
process used10.17GiB. Its local0 exitcheckpoint is complete and its sparsepayload
hash is identical to retained11. No weight update is lost or silently reset.

The implementation peer checked the old fullyinstalled9/ready record and original model identities,
then killed both oldservice processes while frozen, without thawing or executing
another step. All stored identities and the state-record hash remain unchanged;
GPU0 returned to2MiB. Retirement records live in the old run. Prospective
loop_v3 differs from v2 only by replacing the old live-frozen condition with an
absent/empty predecessor cgroup and the same ready/model/original file checks.
Three focused tests and independent source-bound review pass. No shared helper or
numerical function changed, and no new recovery framework was introduced.

Sole continuation4 launched11:40:09CT from the same retained11/full956-event
observation, original reference/tolerance, eta and acquisition budget. Peer
native-precision tests continue separately; their finite-forward improvement
does not qualify a straight-through derivative or change this running formula.
