# Follow the changing computation through the correction

November 15, 2026 (CT). At 5:41 AM CT, claude-research relayed Marcos's response
to the first Torch-step descriptions and asked about continuation, coverage and
step size. This records Astra's understanding, not a transcript or a new operating
law. The reference remains the relational construction fixed earlier that morning.

## What the interim observations do not decide

Marcos expects that changing a computation may initially destabilize its existing
route before it approaches the desired one; they have seen early training disruption
followed by recovery in their own work. That possibility is not a prediction that
this construction will recover. It does mean that an early loss rise, a large
gradient or a shorter generated history does not establish damage or the final
failure of the correction. The output is used to acquire the trace, not to judge
what the mechanism has achieved. The complete evolving trace remains the judge,
and a final repair verdict waits for closure rather than a favorable first step.

The completed plain and lattice Torch trials do not test a repeatedly refreshed
correction trajectory: `relation_step.py:410–417,498–507` writes the model's own
histories before and after a loop of fixed-history updates, not after each update
for use by the next one. No tens-of-steps, regenerate-every-step result is established
by those records. Their fixed-history/own-history loss reversal remains an observation,
not a claim that an on-policy correction loop failed.

## Continuation

Continue the intended correction from each reached model, acquiring and measuring
its complete own trace before constructing its next update against the same frozen
original reference. An own-history loss increase alone is not a checkpoint rollback,
restart, early failure verdict or reason to replace the reference. A favorable
Torch-written-history loss is not an admission prerequisite. This clarification
supersedes interpreting the bounded first native comparison as a final judgment
or an instruction to stop there because its loss rises.

The actual repeated loop, with refreshed histories, coverage and measured step
selection, still needs its prospective declaration and execution admission. This
relay does not launch a process or retrospectively declare the development trials.
A numerical stall remains a stall, not closure; no hand-chosen iteration count or
smaller-step multiplier is introduced here.

## A shortened history and reference coverage

The chosen loss retains every original row's mass, split across its current
counterparts, and takes the positive shortfall at each pair before aggregation.
It does not discard the reference suffix or shrink the denominator when a current
history is shorter. But the complete monotonic correspondence is required to cover
both endpoints: it can assign many reference positions to one current position.
The existing missing-row check therefore cannot establish that each reference
operation has corresponding computational support in the current trace.

Performance's `29221287`, `pairs_coverage.py`, makes that multiplicity visible from
bound pair records alone. On the historical stopped loop's first capture, 3,313
of 14,592 original answer positions share current positions with other originals;
the longest such span is 124 original positions on one current position. These are
historical mapping observations, not results for the new Torch proposals. Its
exclusive/shared categories describe correspondence geometry, not whether a
reasoning operation was performed or omitted.

Astra's decision: report that coverage beside every measured loss, bound to the
capture/model, and retain full pair mappings and original-row loss/mass/multiplicity
contributions. Keep each original row's existing full-mass relational comparison;
introduce no invented penalty for answer length, no window mask and no tolerance
that turns unknown support into agreement. Neither forced coverage nor a low loss
alone proves the entire computational progression was supported. A zero under
this correspondence can establish only what the declared observable actually
measures, not make an unsupported stretch a verified repair. The unresolved
support interpretation is not a new gate before trying the correction.

## What chooses the step, rather than a preferred small number

For descent direction p, with q = -grad(L) dot p > 0, the current proposal starts
with h = L/q. For plain descent q is the gradient's squared norm. A BF16-rounded
trial with measured loss L(h) supplies the quadratic curvature
`c = (L(h) - L(0) + q*h) / h^2`, and its next trial is `q/(2*c)` when c is positive.
The fixed-history search chooses its lowest observed trial as specified; this is
not a manual halving schedule and does not guarantee a small parameter movement
or an own-history decrease after regeneration.

The recorded plain trial tried 0.432949, 0.130649, 0.076518 and 0.057387, selecting
the third from their measured fixed-history losses. Those are historical values,
not constants for later steps: the next update derives its length from its own
current loss, derivative and trials. No claim about a long-run trajectory can be
read from these four length trials.
