# Restore the complete training update

Marcos's correction, November17,2026 CT: the current work must implement the
computational-loss training loop, with its loss propagated into every declared
trainable matrix. A first-projection conditioning investigation had become the
live training construction: all105 trace terms contributed to one projection's
request while the other69 fresh direction blocks were explicitly zero. That
restriction was our implementation choice, not part of the requested learning
goal. Small improvements under that restriction did not establish convergence.

The immediate work is to remove that restriction and resume from the complete
latest native weights and FP32 remainder, keeping the original computational
reference, tolerance and fresh-regeneration criterion. All70 currently declared
trainable matrices across the35 layers receive their own derivative of the
complete105-term objective through the existing computation graph. The corrected
loop retains the already qualified operand-precision adapters and response-scaled
step, installs the resulting native update and measures the newly generated path.
No layer is selected by this correction and no supplied gradient is manually
replaced by zero. A mathematically zero derivative remains zero.

Further narrow scientific probes are stopped. The prepared historical281
first-projection decomposition is retained without execution. Source checks and
independent review now serve the corrected full training path. The loss and the
method's hypothesis do not change. An actual successful update, its subsequent
fresh loss and continued behavior establish what the corrected loop does; source
checks alone do not establish stable training or capability repair.
