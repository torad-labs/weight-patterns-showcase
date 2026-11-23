# Reference, tolerance and correspondence review

November 17, 2026 (CT). Marcos requested a review of the complete executed research code and pointed specifically to the loss. This is a source and existing-record review; no model execution, source substitution, new reference calibration or parameter change was performed.

## Corrected assessment after Marcos's scope clarification

Marcos corrected root's interpretation on November17. The agreed first training experiment deliberately chooses the most-correct available reference and corrects one incorrect response's computational divergence from it. Training all30 together had been unstable and unproven. Other responses labeled correct can contain wrong intermediate computations; they are not a collection of fully correct targets that this experiment must admit at zero loss.

Root wrongly made the other outcome-successful traces' positive losses and cross-answer rankings central to a diagnosis of the active objective. That inference is withdrawn. Those numbers do not establish a defect in this one-pair loss or justify changing its reference, adding targets, widening tolerance or returning to cohort training. They remain descriptive observations of the existing calibration.

The source review found that the implementation evaluates the declared fixed-reference objective. The independent derivative review found no sign, weighting or chain-rule arithmetic error in that objective. The appropriate current questions are whether this wrong trace's divergence is represented and aligned properly against the selected fixed reference, whether the gradient reaches the relevant weights through the connected computation, and whether installed updates improve the freshly generated path. The wider hypothesis remains the research purpose, not an extra admission test that replaces this first experiment.

## What the implementation actually constructs

1. `turn_reference_run.py:58` extracts all native input/output turns from the fifteen original fitting-success traces. These are observations, not a generated correct answer supplied as a token target.
2. `turn_reference.py:47` considers each observed successful trace as the reference. Complete computational cosine DTW matches the other traces to it. Mean squared unnormalized turn departures, averaged over reference events and merges with equal witness mass, select `expanded-004-h1`. There is no word-equality selector.
3. `vector_chance_reference.py:43` reduces the fourteen other successful traces to a scalar mean squared vector departure C at each reference event and merge. It retains within-witness, between-witness and reference-offset summaries. It does not retain directional variation in the active tolerance.
4. The active zero-loss region at that cell is the ball of radius sqrt(2*C) around the selected reference turn. `vector_chance.py` computes squared distance outside that ball. Repeated matches share the reference event's mass. Each cell contributes its own residual; the worst cell's loss is not copied to other cells.
5. `turn_correction.correspondence` recomputes a complete monotonic DTW path for each fresh trace. The comparison uses the cosine of flattened 105-merge turn fields. All events are covered, including events for which no sufficiently similar counterpart has been established.

## Recorded properties; not established defects in the one-pair experiment

### Successful-reference variation receives a training penalty

`vector-chance-original/original-coverage.json` already records the full frozen loss for each of the fourteen nonself original successful witnesses. All fourteen are positive, and none is wholly inside the zero-loss region. Five have a larger loss than the original wrong training trace's 0.00029234223350039747. The maximum successful-witness loss is 0.001992035640873451; the minimum is 9.791098848310278e-10. The accompanying JSON retains every witness and the exact comparisons.

These are outcome-successful traces, not certified correct computations. The active first experiment does not require all of them to lie inside its zero-loss set or to rank below the selected wrong trace. Root's use of these observations as evidence against the chosen one-pair objective was a scope error. The observations were available before training; they are not a new experiment or a newly discovered source-code regression. This correction does not assume that every penalized cell is faulty either; that attribution was not measured here.

### Zero observed variation becomes zero permitted movement

The bound `chance.npy` contains 1,365 exact-zero cells, exactly all 105 merges at reference events 0 through 12. Positive chance values range from 1.317790326434746e-11 to 1.9448615573776407. Thus the active loss requests exact equality of the turn observable at those zero-radius cells whenever their observed turn changes; it does not require equality of raw states, operators or weights. Identical calibration observations explain how the zeros arise; they do not independently establish that every later departure is harmful computation. This review does not invent a tolerance floor or exclude those positions.

### The tolerance discards directional information

The full vector of witness differences becomes one scalar C per cell. Variation in one direction can therefore admit a departure in another direction where the successful witnesses were stable. Conversely, an alternate successful route can lie outside the ball. This is a property of the formula, already disclosed in the original compilation; it is not fixed by adding more trainable matrices.

The factor of two is explicitly an adaptation of the path paper's signal-dominance convention. The current C is a single-execution mean squared departure around a selected observed reference, not the paper's uncertainty in the difference between two group means. It supplies neither a confidence guarantee nor a simultaneous whole-trace coverage guarantee. The active code implements its declared adaptation faithfully.

### Correspondence and loss use different criteria

`complete_overlap.distance_path` minimizes the summed cosine distance of flattened normalized event fields. The loss subsequently uses unnormalized per-merge turn differences, cell-specific radii and reference-event multiplicity weights. Thus DTW does not minimize the reported training loss. Its forced endpoint-to-endpoint path contains no unmatched-event option. Computational matching avoids a word-based criterion, but does not establish semantic equivalence of every matched event.

Within an update, the selected history and DTW pairs are fixed. On regeneration they can change. The derivative of the fixed-pair objective therefore does not promise monotonic decrease of the newly generated, newly matched objective. This is an approximation boundary, not a demonstrated explanation for a particular historical spike.

## Relation to the supplied research documents

The trace-derived guide defines the contextual turn `unit(output) - unit(input)`. Its turn magnitude is preserved in this loss; normalization of the two states is part of that observable, not an accidental normalization of the turn itself. The path paper's group comparison estimates a recurring gap using both groups and their variation; the present objective instead compares one current trace to one fixed observed successful trace. Its DTW is explicitly a proposal for correspondence. The iterative-learning document proposes tolerance-based squared excess, requires reference assignment to reflect the supported relationship, and warns that a single centroid can erase legitimate computational differences. Later user directions supersede that document's separate preservation and answer-text objectives; this review does not restore them.

None of these sources establishes that the current one-reference scalar-ball objective isolates only incorrect computation. Conversely, their support for partial observable traces does not require recovering every internal operation or an exclusive complete capability before useful research is possible.

## Coverage and limits

The accompanying JSON binds the nine source files and the original artifacts read for this portion of the review. Root inspected reference extraction, selection, tolerance construction/loading, correspondence and native objective records. Independent review covers loss derivatives and the executed capture/weight-installation/regeneration route in the companion reviews. The active declaration binds 165 implementation entries, including historical imports, tests, operational tools and native dependencies; this document does not claim a fresh line-by-line review of all 165 or all repository history.

The independent loss reviewer read the earlier version and its data bindings (`2026-11-17-loss-semantics-code-review.md`, commit `3ad33c77`). Marcos's later clarification exposed the interpretation error despite that agreement; the reviewer was asked to amend its account too. No objective defect follows from the witness rankings alone. The corrected scope and the four-document reread are recorded in `journal/2026-11-17-one-pair-scope-correction.md`. No replacement loss, new target, tolerance adjustment, native experiment or retrospective numerical change was made.
