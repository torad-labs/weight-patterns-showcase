# Can an all-layer trace retain correct computations and expose the wrong descent?

Pre-run specification, November 18, 2026, CT. Marcos proposed applying the earlier input-conditioned measurement across the entire generation, then looking for a descent error that the one-pair loss admitted. The deliverable at this stage is a comparison using saved executions. No overlap, model generation, or weight update has been performed under this specification.

## What would count

A useful signal must do both jobs on the same frozen rule. It must accept the previously verified correct descent computations, including correct solutions that took different routes, and expose the incorrect descent in the previously missed problem. Report whether it retains the other verified operations of those answers separately. A right verdict can follow a wrong descent. An unfinished answer only supplies evidence through its last captured computation.

Use the existing labels for individual physical operations as evaluation annotations. A verified correct descent can supply a positive reference for descent even if a later step fails. An answer with an unverified intermediate step cannot supply a positive reference for that step. Keep a case-by-operation table with correct, wrong and unassessed entries. Label the known observations before inspecting the new scores. Wording, symbol choice, generated length and ordinal position supply no evidence of computational correspondence.

## Which values exist

| Saved executions | Data available | Scope |
| --- | --- | --- |
| Original fitting cohort | All 30 members, 35 blocks, 105 merges, width 1,536. All 30 have local generated turn vectors and both endpoint norms at every generated event. The full-state reading covers 5,159 prompt events and 14,592 generated events. Its 105 native post-merge arrays match the digest entries in the verified 12.95 GB R2 reading. | The reading reproduced each model-authored history, with zero changes in generated choices according to its manifest. One response, `expanded-001-h3`, reached the generation limit. The existing descent development comparison used 15 correct and 13 wrong members. Its two exclusions remain outside discovery. |
| Final continuation, cumulative 348–390 | All 43 fresh fitted generations retain a raw float stream and its index. Updates 389 and 390 have both files locally. All 41 earlier observations have matching R2 readback receipts. These include the wrong descent at 387 and its recovery at 388. | One fitted prompt throughout. The current 390 index covers the incoming, branch and outgoing labels of all 105 merges at every one of its 612 captured events. A 390 cohort-wide preservation or transfer score cannot be recovered from these data. |
| Different questions at fixed edited checkpoints | Fitted, originally correct reference and unfitted wrong-descent cases have paired observations at 175 and 189. The reference and unfitted case also have baseline and 348 captures. The separate standard-versus-directional comparison retained all nine original/standard/treatment × fitted/reference/transfer acquisitions. These stream/index pairs are local. | Use the same-checkpoint comparisons to distinguish fit from transfer. The unfitted problem is `discovery-10-channel-h1` in the checkpoint profiles. The previously missed reordered question, `discovery-02-reordered-h3`, belongs to the separate directional comparison. Its known outcome makes it a development challenge. |

The older 30-correct/30-wrong path-edit cohort was a separate study. Its report includes an original overlap and an edited overlap on all 60 answers. Its 29 repaired descents are historical evidence and do not count as 29 distinct successes of checkpoint 390. [Earlier 60-answer study](../studies/04-round-2-overlap/runs/a2-path/result.json)

The inventory checks headers, shapes, file sizes, case membership and recorded archive hashes. For 390 it also checks every event index entry for the required native labels and bounds against the float stream's length. The audit did not read the arrays into an overlap or redownload archived bytes. The local `turns.npy` and `norms.npy` suffice for the direction increment and log gain. Choosing a successful reference by the **incoming direction** additionally needs the incoming vector. The documented reconstruction uses the previous stored merge state, the actual block output scale at transitions, and the model's scaled input embedding at the first merge. Use the archived native states or the original stream where available. This establishes coverage for the proposed merge measurements. It does not inventory every possible attention operand, branch Jacobian or causal edge. [Capture-by-capture inventory](trace-capture-inventory.json) · Original native reading · Archive and recorded readback · [Reconstruction on the original](../records/development/2026-11-12-path-forward/README.md) · [Fitted arithmetic and recovery](training-and-stopping.md)

## 1. Represent each observed transformation

For every consumed event and every residual merge, keep its incoming state $h$, branch contribution, outgoing state $y$, block and operation identity, and the order in the native execution. Retain prompt context and mark the prompt/generation boundary. The primary comparison covers every observed decode event, including divergent and agreeing stretches. Handle a true zero norm as an undefined direction and report its coverage.

For a nonzero state $v$, write $u(v)=v/\|v\|_2$ and $s(v)=\log\|v\|_2$. The historical feature and its measured increment are

$$
\phi(v)=[u(v),\sqrt{\rho}\, s(v)],\qquad
\Delta\phi(h,y)=[u(y)-u(h),\sqrt{\rho}\,\log(\|y\|_2/\|h\|_2)].
$$

Keep the angular and log-gain components separately as well as jointly. A branch that doubles its input along the same direction has zero turn and log gain $\log 2$. A nonzero log gain alone does not establish a consequential downstream effect. The historical calculation used $\rho=44.23381971299944$, the recorded size-to-direction weight ratio. Retain that value for the historical comparison. An independently calibrated version must estimate its scales from its discovery fold alone. The historical value comes from previously examined cases. Earlier feature weights and their derivation

## 2. Align executions by computation and retain the successful witnesses

Start with the saved, monotone, many-to-many correspondence computed from the sequence of full merge fields. A 670-event and a 900-event execution may traverse one computational phase at different rates. Preserve each execution's native ordering, list every many-to-one match and its stretch, and let an unsupported or omitted phase remain unresolved. Do not linearly rescale a response by its length. A full-history alignment locates an interval retrospectively. A claim that a signal was available *before* the error also requires recovering it using only the computation available up to that point.

Keep each successful donor's trajectory intact along depth and decode. For a query merge, select successful computations that cover the same operation. Weight their matched incoming features by the earlier nearest convex mixture. Carry those same coefficients through that merge to define its successful transformation:

$$
\alpha=\arg\min_{a_k\ge0,\,\sum a_k=1}
\|\phi(h)-\sum_k a_k\phi(H_k)\|_2^2,\qquad
z=\sum_k\alpha_k[\phi(Y_k)-\phi(H_k)].
$$

Here the pair $(H_k,Y_k)$ comes from the same witnessed merge of the same successful execution. Store donor identities, coefficients, incoming fit error and non-uniqueness alongside each comparison. A sequence assembled by switching to unrelated donors can manufacture a path no correct answer traversed. Show each donor-specific overlap as well as the mixture. Where a proposed connected divergence depends on a donor switch, check the same segment with the entrance weights carried through it. Flag a conclusion that disappears. The historical precursor was highly sensitive to removal of one successful witness. [Historical comparison and support-deletion evidence](../records/retrospectives/2026-11-17-pre-sign-localization-history-review.md)

## 3. Locate divergence, its extent and its specificity

At each matched event and merge retain the signed vector $e=\Delta\phi(h,y)-z$, its direction component, its log-gain component and the incoming mismatch. Calibrate ordinary variation by scoring verified correct executions against *other* eligible correct executions through the same procedure. Exclude the scored question and its related configurations from donors and calibration. A mixture of nearby inputs can predict the wrong output even for a correct nonlinear operation. The correct-against-correct measurements expose that error.

Keep the whole event-by-merge map. Report the beginning, peak, recovery and any reappearance of excess beyond the calibrated variation, in the native coordinates of each case. Connect successive observed operations through actual data flow where recorded. Show temporal succession separately when causal edges are unavailable. An isolated high cell or a large inherited difference at the input is insufficient to attribute an error to that operation. For each candidate descent region, freeze its signed wrong-versus-correct direction and its registration on discovery cases. Score the other cases with that same region and direction. Undefined correspondence and unsupported coverage remain explicit outcomes.

## 4. Decide whether the extra information matters

Use the same cases, correspondences, exclusions and scoring procedure in four comparisons: directional turns with the fixed reference, turn plus log gain with the fixed reference, directional turns with incoming-conditioned successful references, and turn plus log gain with incoming-conditioned references. Include the recorded one-pair loss as its own historical benchmark with its actual tolerance. A change in correspondence, tolerance or bank membership must be reported separately. For each comparison show every case, every independently verified operation, false alarms on correct computation, missed wrong descents, and unresolved regions.

Use the 387-to-388 reversal as a within-prompt test, and all other labeled continuations as repeated observations of that same prompt. Check the known unfitted failure and the correct reference at matched weights. Keep the previously missed reordered case outside signal selection, witness selection and tolerance fitting. Test whether the same region marks its erroneous operation while leaving its verified correct operations inside their allowances. A cutoff chosen after inspecting these cases cannot establish that result. Report continuous scores and their ordering as well as any frozen decision.

Retention of all known correct computations plus detection of the missed descent, especially when the recorded turn loss admitted the error, supports the claim that the added measurement contains information the earlier readout missed. If only one witness produces the effect, if the signal also fires on valid alternative solutions, or if it appears only after the mistake, report that distinction. These are tests of the signal. Transfer of a weight correction needs a subsequent edited-model evaluation on unused problems.

## 5. How this could feed the iterative learner

This stage describes the mapping into the existing learner. Training is not authorized by this note. Keep the original successful bank intact. On an edited recipient, freshly register its own generation, calculate its context-conditioned targets, and freeze those assignments within that backward pass. At supported cell $j$, one candidate use of the existing loss form is

$$
\ell_j=w_j[\max(0,\|e_j\|_2-\tau_j)]^2,\qquad
g_\theta=\sum_j (\partial e_j/\partial\theta)^\top\nabla_{e_j}\ell_j .
$$

The allowance $\tau_j$ comes from correct-against-correct variation for that operation. The previously observed failure of a scalar distance to distinguish some wrong computations makes this loss form conditional on the comparison above. A reliable signed error component would call for a separately specified directional allowance. Each local excess produces its own derivative. A peak supplies no blanket request to the rest of its region. Normalize event weights within each response, split repeated alignment mass, and accumulate shared-weight effects through both incoming and outgoing branches of the graph. A matched region inside tolerance requests no direct correction. An unmatched region is reported as unmeasured.

After an authorized update, regenerate from the same failing prompt using the installed weights. Recompute all local losses and the connected regions, track preservation and test distinct problems at fixed checkpoints. Retain temporary regressions. A useful endpoint would combine reduced fresh divergence, correct computations remaining inside their observed range, and transfer. A numerical stopping threshold or learning rate for this changed measurement cannot be inherited from the old loss without calibration. The open decision before any training is which of these signals, if any, satisfies the case-by-case comparison above.
