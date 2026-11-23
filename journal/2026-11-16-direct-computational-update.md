# A weight derivative followed by a fresh computational path

Marcos's later November 16 correction, conveyed directly in the execution chat
and through the authorized oversight chat `01a1e317-adbf-7983-bb8b-3df59ef44c23`,
supersedes the earlier requirement to finish a joint inverse before applying
an update. Compute the computational discrepancy, differentiate through the
functions producing it into the weights, apply the update, generate from the
edited model and measure its complete computational path again. An inner
search for a numerically complete correction is not the intended iteration.
Once the replacement is ready, stop the unsaved CPU-vector attempt without
waiting for its first installed update. Its scalar progress and all earlier
experiment records remain historical; its unsaved inverse is not resumed.

## Implementation decision

This numerical rule is the implementation peer's choice, independently reviewed
by Eli with the four original papers and the goal. It is not a formula supplied
by Marcos. For the existing complete, weighted computational residual `r(W)`:

- Differentiate `L = sum(r(W)^2)` once through the complete native-anchored
  graph to obtain the weight gradient, then set `S = -dL/dW`.
- Measure one directional response `q = J_r S` through that same graph.
- Calculate `alpha = -<r,q>/<q,q>` from that response, then round
  `W + alpha*S` once to the model's native BF16 format.
- Install every declared matrix, generate a fresh response from that reached
  model, and remeasure/rebuild the full computational correspondence. Repeat
  against the same original reference and tolerance.

This is response-scaled weight-gradient descent, not a full inverse solve.
There is no CGLS, search for a candidate, guessed learning rate, halving,
rounding enlargement, or held-history acceptance objective. The local linear
prediction does not guarantee a decrease on the next nonlinear native path.
Zero gradient, zero response or a BF16 no-op while divergence remains ends an
attempt as nonclosure; none is successful training or a reason to invent a step.

The path-edit paper section 5.4, equation 8 supports calculating a directional
step from the response. The iterative-learning document's repeated-update
step 4 explicitly permits differentiation through the editable layers as well
as a coupled solve. Earlier solver and nonlinear-search experiments keep the
rules they actually ran. The current authorization supersedes their use as a
requirement for subsequent updates.

## Scope and evidence

The fixed original correct reference, original predictive-vector tolerance,
one wrong training member, complete path and all 105 merges remain unchanged.
Inside-tolerance cells supply zero direct loss gradient; shared parameters can
still affect them, so the next whole path must be measured. Text agreement,
token divergence, prompt-position cuts and output scoring supply no criterion.
Only complete fresh native agreement after an installed update establishes
closure. The derivative holds the acquired history/correspondence fixed and
uses the existing native-anchored differentiable graph; it is not a derivative
through discrete generation or BF16 rounding.

The implementation peer owns the sole controller. Weight-research reviews instrumentation and
the graph fixture; Eli independently reviews the mathematics and actual native
write-to-fresh-generation route. A source review or fixture is not a native
result. The prospective run's declaration, execution and reached-state records
will name the actual implementation and installation status.

Sources supplied to Eli: `reports/fixing-a-reasoning-mistake-by-editing-the-weights.pdf`,
`reports/how-the-trace-derived-edit-works.pdf`, `reports/how-the-path-edit-works.pdf`,
and `reports/Iterative_Learning_from_Computational_Traces.docx`.
