# Explicit recovery of the interrupted native experiment

This run retains the original-weight native observation from
`vector-chance-loop-original` and restarts only its interrupted inverse solve.
Its source run is read-only. All original scientific settings and implementation
bindings stay fixed; the recovery entrypoint is separately bound and reviewed.

The first observation is reused and computationally remeasured, with its original
acquisition/declaration identity retained. It is not a new capture. Inverse
progress iteration 33 is not a checkpoint and supplies no weight update. The
existing controller starts at zero updates; after installing the first complete
BF16 proposal it requires a fresh native acquisition. No closure or repair is
claimed by recovery or by an inner objective decrease.

Status: peer and source-bound Eli review YES; admitted for explicit recovery. Actual execution and progress are recorded separately.
The first run's data, reference, tolerance and method are not changed.

Independent checks: root eight recovery tests and three subtests pass; peer
verification covers 25 focused tests and three subtests. Focused Pyright is
clean. The actual backend constructor passed with acquisition forbidden and
zero updates, retaining the original 4,386 outside cells. Peer review reproduced
and fixed a nested-source path bug before approval. Exact source identities
and findings are in the peer and Eli review records under records/retrospectives/.

The declared user-service launcher has no automatic restart. Its bound
ExecStopPost recorder writes launcher-exit.json independently of the Python
worker, so a killed worker still has an external exit record. The disk survey
continues every two hours under weight-patterns-disk-survey.timer.

Started under weight-patterns-vector-recovery.service at 11:30:01 PM CT
November 15; the entrypoint records 11:30:21 PM CT. The retained observation
remeasured exactly at 11:30:45 PM CT, and the inverse restarted at iteration 0
at 11:31:02 PM CT. This is verified runtime progress, not a completed weight
update or closure. See execution.json, launcher-start.json and loop/1/measurement.json.
