# Stop training; update one working model

Marcos explicitly directed stopping `weight-patterns-vector-gradient-2.service`
and removing repeated full-model copies and verification. The implementation peer stopped the sole
controller at 04:30:50.341209 AM CT November 16. Three complete updates remain;
the fourth proposal was ready with a writing manifest at 0/70 matrices and no
reached model. The run's `authorized-stop.json` records the exact boundary.
Completed states, traces and partial records are preserved. No restart is authorized.

Marcos subsequently rejected the proposed recovery/publication framework and
asked for ordinary resident training: update the weights, generate, measure,
repeat; checkpoint saving is separate. This supersedes the earlier uncommitted
redo-journal design. No such journal, tensor hash chain, per-step receipt or
checkpoint schedule is part of the implementation.

The concrete engine constraint is the currently bound recorder-workers program:
`dependencies/recorder-workers/observe_stream_workers.cpp` loads a GGUF path with
`llama_model_load_from_file` at startup and exits after the fixed cases. It has no
resident parameter-update command. Its activation patch is not a weight upload.
Thus the minimal supported handoff is one private working GGUF, copied once from
the original, written in place with the complete prescribed BF16 matrices, then
loaded by a fresh native recorder. Torch's training and measurement models remain
resident and receive those same values in place. Original weights are untouched.

A small ready/writing marker and run/update revision identify the mutable model;
the revision is explicitly not a content hash. An incomplete write stops before
capture and preserves its partial file. There is no automatic recovery promise:
restoration from an original or separately saved checkpoint is an explicit future
operation. No full-model copies, full-model scans or checkpoint writes occur per
optimization step. Existing historical checkpoints remain intact.

The derivative through computation, one measured response-derived scale, BF16
rounding, original reference/tolerance, whole path, and fresh native closure rule
are unchanged. No generation scoring or token comparison is added. The implementation peer is sole
controller; the peer handles compatible MLflow revision labels; Eli independently
reviews the minimal route with the goal and all four original papers.

Marcos narrowed the work again: finish the scientific experiment, not an automated
platform. Further registry/exporter/dashboard/recovery development is stopped.
Only the minimum native update/capture/measurement path and already useful logging
remain. This does not authorize restarting the explicitly stopped research run.

The latest direction explicitly authorizes execution after the necessary scientific
review and removes the earlier stop-to-fix restart hold. The continuation binds
`vector-chance-gradient-2-original/loop/3/model.gguf` (SHA256
`0dfed0e7f5e0430e693def81549785baf863e4dd9a4f337f8d584d709a6d2a1a`)
as the initial state of `vector-chance-working-from-update-3`. The minimal optional
start loads that already-retained state once, without a recovery framework or a
silent reset. The original correct reference and tolerance are unchanged.

The mathematical distinction remains explicit: the residual-space derivative is
propagated through the trace-producing graph into weights, giving
`g_W = 2 J^T r` for `L = ||r(W)||^2`. The parameter delta is the response-scaled
negative weight derivative followed by native BF16 rounding, never `r` or a
trace-space derivative installed directly into weight storage.
