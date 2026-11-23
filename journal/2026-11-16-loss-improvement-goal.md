# Learn as far as the computational loss can improve

Marcos explicitly clarifies the objective on November 16: reduce actual fresh
computational loss as far as learning can take it. Exact restoration, zero residual
and outside==0 are not required endpoints. A plateau may remain nonzero.
Continued loss improvement is progress; report achieved residual and supporting
measurements when improvement genuinely stalls, allowing step-to-step variation.

Oversight withdraws its own zero-tolerance hold. The implementation peer thawed the existing sole
controller, preserving its resident state and all prior results. The running
source bindings, numerical loss, fixed original reference and tolerance remain
unchanged. The zero-radius count finding remains valid descriptive evidence and
is not a blocker to continued learning.

No epsilon/floor, perfect-correction search, per-step checkpoint copy, automatic
plateau detector, guessed schedule, or new infrastructure is introduced. Manual
research judgment determines when the measured learning curve has genuinely
stalled. The existing closure flag describes exact closure only; it does not
classify every nonzero residual as failed training. Historical declared rules and
reported outcomes keep their original meanings.

The precise thaw record and source-integrity check are in
`studies/04-round-2-overlap/runs/vector-chance-working-from-update-3/loss-goal-resume.json`.
