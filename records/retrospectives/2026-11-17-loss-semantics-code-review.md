# Independent loss-semantics code review

November17,2026 CT. This is a read-only mathematical/source audit of the consumed loss, its gradient and directional response. Root separately owns the correct-reference/chance construction, acquisition history and closure audit. No source/runtime/GPU change, new experiment or repeated test was performed. The companion JSON binds the exact sources and supplied reference metadata.

**Amendment, November17,2026 CT:** Marcos clarified that the agreed first experiment deliberately trains one wrong response toward the selected most-correct fixed reference. The narrower experiment was chosen because training all30 together had been unstable/unproved. Other outcome-correct responses may contain wrong intermediate computation. I gave their nonzero penalties and cross-answer rankings too much diagnostic weight. They do **not** establish a defect in this one-pair objective, and no requirement to admit all other successful witnesses belongs to this experiment. Their observed numbers remain below as historical observations, with that inference withdrawn.

**The inspected arithmetic faithfully implements the agreed one-pair radial turn-distance objective; no loss-definition, sign, mass or chain-rule defect was established by this audit.** A lower fresh native scalar is progress on that declared one-pair computational divergence. It is not by itself a full capability-repair, preservation, transfer or stability result. The observable, tolerance, correspondence and finite-response limits below bound those broader claims; they are not independent evidence that the agreed objective is wrong. One exception-cleanup defect is identified separately; no occurrence or numerical corruption was observed.

The previous full-weight YES established execution of the complete declared gradient scope. It did not prove a native research outcome. The following assesses the specified one-pair calculation rather than imposing a multi-answer correctness classifier or a different target set.

## What the active loss actually is

The active artifact is `single-original-vector-chance/1`, not the older coordinatewise-tolerance branch. `turn_artifact.load` selects `vector_chance.Reference`. It consumes one selected original successful turn field (`expanded-004-h1`,560 positions ×105 merges ×1536 coordinates) and a fixed nonnegative560×105 chance array. The metadata names all15 original successes, but the runtime residual compares the current path against the selected field and its supplied chance, not against15 separate live reference functions. This audit does not redo how the selected field or chance was constructed.

For incoming vector x and merged vector y, the observable is

```
T = y/||y|| - x/||x||.
```

For fixed correspondence pairs (i,j), reference position count R=560, merge count M=105, multiplicity n_i of reference position i, reference turn T*_im, and original chance C_im, the implementation gives

```
delta_ijm = T_current(j,m) - T*_im
b_im = sqrt(2 C_im)
mu_ijm = 1 / (R M n_i)
L = sum_(i,j in pairs),m mu_ijm * max(||delta_ijm|| - b_im, 0)^2.
```

`vector_chance.outside` represents the residual as `sqrt(mu) * delta * max(1-b/||delta||,0)`. For C=0 it deliberately uses the identity residual, including at delta=0. Squaring/summing this residual gives the formula above. The `outside` flag uses the same strict distance>radius boundary. Thus zero loss means every represented current/reference turn pair is inside its supplied ball, not equality of full input/output tensors, all internal operations, or capabilities.

All reference positions and all105 merges have equal total mass. When i has n_i matches, their individual masses add to1/(R M); total mass is one. This is reference-position weighting, not equal current-event weighting. No extra division by coordinate count, current path length, merge count or pair count is applied by the gradient/reducer. The legacy `turn_pair.Reference` instead computes coordinatewise interval excess, with a different zero set; the active loader does not accidentally mix those two definitions.

## Retained witness observations, not a one-pair defect

Root identified `vector-chance-original/original-coverage.json`; I read every witness's retained frozen result. All14 nonself originally successful witness paths have positive objective and at least one outside cell. None is wholly inside the frozen boundary. Their objectives range from `9.791098848310278e-10` to `0.001992035640873451`; **5/14**, not10, exceed the original wrong-path objective `0.00029234223350039747`. The selected reference itself is admitted by construction. The record explicitly says `independent_validation: false`, `retuned: false`, and describes whole-trace coverage as descriptive.

The agreed target is the selected most-correct reference, not every other outcome-successful trajectory. Their final-answer labels do not certify their intermediate computation or order its quality. Neither a false-positive attribution nor an incorrect-computation-severity ranking follows from these labels alone. **The14 nonzero penalties and five larger scores are not evidence against the agreed one-pair target or its loss.** My earlier treatment of those observations as central to an objective diagnosis was an inference beyond the experiment's scope and is withdrawn. They do not justify expanding the target set, widening the boundary or replacing the reference.

A radius derived from a mean-square departure is a pointwise dispersion scale, not automatically a simultaneous coverage guarantee over58,800 reference cells. The empty complete-coverage list therefore cannot establish that the pointwise arithmetic or radius formula is erroneous. Requiring every witness's full trace to fall inside would add an acceptance condition that was not agreed. This review proposes no retuned radius or coverage threshold. Root owns the underlying construction audit.

## Interpretation limits within the agreed one-pair scope

The following mathematical properties remain true. None alone establishes a defect in this experiment or requires a new loss, target set, parameter scope or validation gate.

**1. A complete turn field is a compressed observable, not a complete operation comparison.** Both edges are used and differentiated, but unit normalization removes radial magnitude. Even with zero tolerance, a zero turn occurs for any nonzero x and y=a*x with a>0, regardless of x's direction or a. More generally, a turn difference does not uniquely identify its two endpoints or the operator that produced them. Different internal attention/gating computations can also yield the same residual merge. All105 merge terms therefore do not compare every internal operation of the model. This is an intentional observable choice consistent with the trace-derived guide's definition of T; its sufficiency for the intended capability is unproved. It is not a reason to require all raw states or operators to match: alternate valid computations must remain possible.

**2. The tolerance expands that ambiguity deliberately.** The zero set is a Cartesian product of balls around individual selected reference turns, restricted by the supplied correspondence. Distinct nonzero departures inside those balls receive zero direct loss gradient. This honors the stated inside-tolerance rule, but the code cannot establish whether a particular tolerated direction is a valid alternative computation or the targeted mistake. That depends on the scientific meaning of the supplied chance boundary. No claim that the boundary is wrong, correctly calibrated or attainable follows from checking its arithmetic.

**3. Complete DTW coverage is weaker than semantic correspondence.** The consumer checks an integer monotonic path that reaches and covers both traces. It permits one-to-many and many-to-one matches and compares the same merge index m on both sides; no cross-layer operation correspondence is discovered by this residual. Covering every event does not prove that each matched pair performs the same computational role, or that a short path executed all steps of a longer reference computation.

The alignment helper uses cosine similarity of normalized flattened105-turn fields, while the loss uses per-merge radial excess and original chance. These are distinct comparison functions. Changing turns can alter the chosen correspondence and multiplicities, then alter which pairwise discrepancies share each reference event's mass. In particular, a reference event's penalty is the average over its matched current events: a larger set of low-error matches can lower that average while an existing high-error event remains unchanged. This is a property of the intended weighting code, not an accidental missing denominator. It must not be confused with demonstrated repair of that unchanged event. Root owns the construction/alignment validity assessment; this describes the consequence at the loss consumer.

**4. A lower total permits local regressions.** The scalar sums over the complete represented wrong-response path and all merges. A lower total is a real decrease of that one-pair aggregate divergence; it does not require every cell to improve individually. Improvements in some cells can outweigh worsening in others, and the unweighted outside count can rise while weighted excess falls. Neither behavior is an arithmetic inconsistency or a defect relative to the agreed aggregate objective. Requiring every cell to improve monotonically, or treating untrained responses as additional preservation targets, would add conditions to the experiment. Broader preservation claims remain separate outcomes, not prerequisites silently inserted here.

**5. Fresh paths change discrete evaluation inputs.** A proposal differentiates at fixed integer history and fixed correspondence. It backpropagates continuously through the current connected operations, including shared causal caches, but not through greedy token selection or the DTW decision. Fresh generation and correspondence then re-evaluate the same scoring procedure on a new path. The target and chance stay fixed; it would be inaccurate to say they are refitted each step. Nevertheless, the fresh scalar can change through both altered continuous computation and changed discrete path/allocation. The held-history derivative is not the total derivative of the rollout distribution.

**6. Local response is not a finite native guarantee.** Inside a positive-radius ball the residual and its JVP are exactly zero. A finite update can move a cell outside the ball, an effect absent from that cell's interior linearization. At equality with positive radius, the squared loss has zero first derivative, while the residual map has different inward/outward directional derivatives; AD selects the inactive branch's zero JVP. The C=0 identity special case avoids incorrectly erasing the point-residual Jacobian at zero. These branch conventions are intentional, not evidence of a missing loss gradient, but they limit a finite prediction.

The same response model omits nonlinear residual curvature and native/discrete rounding effects beyond the chosen STE. A faithful calculation of this local model can therefore disagree with the next full native loss. Source checks or a favorable predicted scalar cannot clear that limitation.

## Gradient and reduction audit

For nonzero x,y, the turn's derivatives are the normalized-vector Jacobians

```
dT/dy = (I - yhat*yhat.T) / ||y||
dT/dx = -(I - xhat*xhat.T) / ||x||.
```

`turn_pair.turns` retains both through canonical CPU F64 conversion and normalization; it does not detach them or normalize the turn vector itself. Zero/nonfinite incoming/output vectors are rejected. The original reference, chance and integration masses are detached constants, as required for the supplied fixed target.

Outside the ball, the pairwise squared-loss derivative is `2 mu (1-b/d) delta`; inside and at its positive-radius boundary it is zero. The code differentiates the vector residual and then its squared norm, yielding this derivative rather than treating the radial shrink factor as constant. At zero chance the squared loss is `mu ||delta||^2`, with the correct derivative and residual Jacobian at zero.

`trace_gradient.weight_gradient` forms every term from the actual tensor residual, then a tensor sum of all105 terms. It does not differentiate a reporting float, average per merge a second time, or omit zero-loss terms before construction. One `autograd.grad` targets every declared F32 native-valued leaf. An unused leaf is represented as zero by the generic helper, which is distinct from proof that a parameter scope is sufficient. The current corrected runner supplies all70 declared leaves and has no selected-projection replacement. Each weight derivative contains all its scalar entries and all causally connected position/merge contributions; later weights cannot affect earlier losses by causality.

`model_trace_response.measure_model` gives every declared block its tangent in the same graph and restores the weight mapping on normal completion and errors originating inside its producer generator. `trace_response.measure` receives the same already-weighted residuals, computes q=J_r S, then sums `||r||^2`, `<r,q>` and `||q||^2` across all chunks. Matrix contributions meet inside q before its norm is taken, so cross-matrix effects are retained.

The resulting slope is2<r,q> and length is-<r,q>/||q||^2. There is no missing factor of two: at that minimizer, `L + 2 alpha <r,q> + alpha^2 ||q||^2` simplifies to `L + alpha <r,q>`, exactly the reported optimum prediction. For an exact consistent differentiable residual with S=-gradient(L), the local slope would be-||gradient(L)||^2; no new numerical experiment was performed to check that identity on the native run. The response curvature is the squared first-order residual response, not the complete nonlinear Hessian of L.

The current runner separately evaluates its prescribed eta-scaled continuous prediction and coefficient-one actual-BF16 displacement prediction. Those are not substituted for fresh native measurements. The native anchor reproduces recorded origin values while retaining current derivatives; it does not validate the reconstructed cache/STE derivative as the native discrete derivative. Fixed BF16/FP16 rounding, native kernel differences, omitted nonlinear terms and the frozen parameter scope remain distinct boundaries.

## Source-level exception cleanup defect

`model_trace_response.py:42–53` puts restoration in the `finally` of the generator `current`, then directly returns `measure(current, ...)`. `trace_response.py:65–77` can reject a yielded chunk or nonfinite residual/tangent in the **consumer**, while that producer generator is suspended. The consumer has no explicit close, and `measure_model` has no outer restoration `finally`. With the consumer traceback retaining the suspended generator, restoration can be delayed until generator finalization rather than occurring before the failed call returns control. The existing test at `test_model_trace_response.py:50–57` exercises a producer callback exception, which takes a different path.

This is a conditional exception-cleanup defect in the stated mapping-restoration contract, not a sign error, loss-definition error or observed explanation of decreasing loss. No new reproduction was run. The active training controller treats the exception as failure and checkpoints native bytes/remainders separately; this review found no evidence that this edge has occurred or corrupted native data. My earlier unqualified statement that every measurement failure restores the mapping was too broad. A caller recovering from a response validation error must not assume this implementation has already restored its Python weight mapping. No active source was edited.

## Independent critique of root's construction review, amended

I read the earlier version of `2026-11-17-reference-correspondence-review.md/.json`, bound in this review's JSON. The stored observations and mathematical descriptions remain:5/14 is the correct count; none of the14 nonself full witness paths is inside; isotropic tolerance discards direction; the DTW criterion differs from the loss; and a changed-pair mechanism is not a diagnosis of a particular historical spike. The factor of two is a declared adaptation rather than a statistical confidence guarantee. **My agreement with using the witness penalties/rankings as a central diagnosis of the one-pair objective is withdrawn.** Marcos's clarified scope does not require the chosen reference loss to admit or rank all the other outcome-correct answers. Root is amending its own account separately.

Two wording limits matter. The coverage evidence is about **outcome-successful traces**, not certification that all their intermediate computation was correct. Also, exact restoration at the1,365 zero-chance cells means exact equality of the **turn observable**, not restored raw states, operators or weights. Root's reported zero-cell inventory was not recomputed here. Neither point supplies evidence for replacing the agreed one-pair loss. No broader claim that every tensor or internal operation must be trained/matched follows.

## Evidence and conclusion

All four original documents and equations were read directly in this review sequence. The first edit report demonstrates that matching a local target need not preserve later behavior. The trace-derived guide defines T and warns that improving a selected trace objective can miss other damage. The path report documents the turn/state distinction and examples where trace proximity did not track complete repair; its withdrawn state target is not adopted here. The iterative DOCX's clipped repair equation is compatible with a tolerance-excess construction in form, but it requires the chosen relationship, supported events and target assignment to carry the intended meaning. Its preservation and efficiency terms are proposals, not automatically part of this code. Later journal clarifications keep computation rather than answer text as the judge; no token identity or desired-answer label enters this residual.

These functions faithfully optimize **the specified one-wrong-response divergence from the selected most-correct fixed computational reference**, using the supplied turn observable, tolerance and correspondence. The audit establishes no arithmetic/gradient defect that invalidates that agreed experiment. A decrease in the freshly measured objective is progress on that declared divergence; the experiment's subsequent native behavior determines how useful and stable that progress is. General repair, preservation and transfer still require their own evidence. The mathematical limits above prevent overclaiming those outcomes, but they are not grounds to reject the first experiment or broaden it to all30 responses. No replacement tolerance, extra loss, all-tensor training rule, source/runtime change or new experiment is proposed by this amendment.
