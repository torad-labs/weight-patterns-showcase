# Independent mathematical review of FINAL-EXPERIMENT.md

Reviewed November17,2026 CT against Marcos's current selected-pair goal and the four original reports, including their equations. Reviewer scope is the mathematical specification and complete loop; the architecture/parameter inventory has a separate independent reviewer. No learner, run, scientific source or GPU action was taken. This record is not launch approval.

The final reviewed draft snapshot has SHA256 **`0c2e009d6dce579d32b7684c48857c4d5c67180b822300eae30fd7265601e000`**,429 lines. Section references below refer to that version, including the incorporated architecture review `711103f8`. Root may revise the document in response; this review does not claim to have reviewed a later version automatically.

**Assessment:** the selected-pair loss, reference-position mass, local derivative and joint chain-rule formulation are internally coherent. They specify the intended local computation without broadcasting a worst/global error. The document correctly fixes the architecture while enlarging eligibility of learned numerical weights. It is a sound specification framework, but it does not yet determine a complete numerical training algorithm: tolerance policy, mixed-dtype accumulation and step-rule transfer remain decisions, and substantial code gaps are explicitly acknowledged. The final draft now defines how newly eligible remainders initialize; that implementation extension remains to be built. Several details below should be made explicit before calling the specification complete.

## The agreed scope is preserved

One incorrect response is regenerated from its fixed prompt and compared with the selected most-correct fixed original reference. Other outcome-correct traces are calibration evidence rather than simultaneous targets or certified error-free computations. Their nonzero loss or ranking is not evidence against this one-pair objective. Nothing in this review restores that withdrawn diagnosis or requires all30 answers to train together.

The four documents support useful computational fragments and persistent weight changes without requiring a complete unique mechanism. Their historical word-position choices, post-merge state targets and separate answer-level objectives do not override the later one-pair/turn/computation clarifications. The distinction between this first experiment and later capability preservation/transfer is stated correctly in §§1,2,10 and14.

## Formulas that are correct

1. **Observable and coverage, §§3–5.** `T=unit(output)-unit(input)` retains both edges and the turn's vector magnitude. It is not normalized again for the training loss; flattened normalization is confined to correspondence. Including every captured prompt/generated computation position and all105 merges is consistent with the declared comparison. The document correctly declines to invent a hidden-state event for a terminal token never consumed by a forward pass.
2. **Local loss, §5.** For fixed pairs and tolerance, `ell=max(||delta||-tau,0)^2` is a distinct term at each matched position/merge. Its weighted derivative outside is `2 w (1-tau/d) delta`, with norm `2 w (d-tau)`. It is zero inside and at a positive-radius boundary. A small local excess therefore gives a small direct trace derivative; a large downstream weight derivative can still arise through the actual Jacobian. The scalar example, weights and derivative values in §6 are correct.
3. **Integration mass, §6.** With full coverage and `n_i` matches for reference position i, `w=1/(R M n_i)` gives each reference-position/merge total mass `1/(R M)` and total mass one. Squaring `sqrt(w)`-weighted vector residuals gives the same L. It is a mean of local squared excesses, not a squared mean or maximum. Its lack of equal current-position mass is explicit and mathematically consistent.
4. **Zero-distance conventions, §6.** At positive tolerance and d=0 the residual/Jacobian is zero. At zero tolerance the point residual must be `sqrt(w) delta`, including its identity Jacobian at zero. The historical special case is described correctly. A squared hinge loss is differentiable with zero gradient at the positive-radius boundary, even though the residual map used by the JVP has distinct inward/outward directional derivatives there; the document acknowledges this prediction limit.
5. **Chain rule, §7.** The normalization Jacobian `(I-u u^T)/||v||`, opposite signs for the two turn edges, `c x^T` weight derivative and `W^T c` input derivative are correct. Local contributions accumulate into shared parameter arrays through all causal uses. Differentiating one aggregate tensor scalar retains those local contributions. Computing every derivative at the same current state before one joint installation is essential and is specified correctly; memory chunking is allowed only if it preserves that derivative.
6. **Conditional parameter scope, §7.** A parameter is eligible through its continuous causal use, not because its tensor name contains `weight`, and an eligible tensor need not have a nonzero gradient everywhere. Excluding the final scale/output normalization from the held-history trace derivative while retaining the tied embedding's input use is coherent. That does not differentiate their effect through future discrete choices. KV reuse must send late losses to the actual earlier producers, as stated.
7. **Response-derived length, §8.** For `S=-g`, weighted residual r and `q=J_r S`, the local residual quadratic is `||r||^2+2 alpha<r,q>+alpha^2||q||^2`; its stationary length is `-<r,q>/||q||^2`. No factor of two is missing. Under an exact consistent differentiable surrogate, its initial slope is `-||g||^2`. This is not a safe-step theorem for finite native execution or a complete nonlinear Hessian. The document labels the STE/native-anchor distinction honestly.
8. **Fresh measurement and stopping, §§9–10.** Fresh native acquisition, cache reset, recomputed correspondence and a complete naturally ended path remain separate from held-history predictions. Exact-zero loss is descriptive rather than a required endpoint, and manual cause-based judgment replaces a guessed plateau schedule. This is consistent with the current goal. A nonzero loss does not erase a demonstrated decrease of the declared one-pair divergence.

The final architecture additions also make the reconstruction's RMS/GELU/RoPE, embedding scaling, shared projection scaling-before-normalization and per-layer feature equations explicit. They agree with the inspected `torch_forward.py` expressions; the q/key score is stated as the reconstruction's dot product without an inserted generic head-width scale. The538 learned-tensor inventory,278 BF16/260 F32 accounting and fixed rotary-buffer classification incorporate the separate architecture review. These mathematical descriptions do not claim that the missing broader native precision routes are already implemented or finite-native qualified.

## Decisions and omissions to resolve

### A. Zero tolerance is a policy/calibration decision, not an arithmetic error

§5 correctly identifies that the old zero-radius cells have point targets for their **turns**. However, requiring nonzero allowance at each of the1,365 cells is an additional numerical policy. Allowing a nonzero final aggregate loss does not mathematically imply that every local tolerance must be positive. The trace report equations and the one-pair correction alone do not derive such an allowance. If Marcos's latest permitted-movement clarification explicitly calls for it, identify that source as the reason for the proposed change; otherwise present this as a proposed choice rather than an already demonstrated code defect. No guessed positive floor or witness-admission requirement follows.

The draft explicitly leaves the allowance unmeasured. It therefore cannot claim to specify its value yet. A changed tolerance must be a new bound original-derived artifact/definition, preserving the old selected turn array and historical calibration record rather than mutating their existing hashes. The loader, residual, outside flag, local records and declaration must consume the same new allowance.

This also affects §10's comparison curve. If tolerance changes, original/baseline/best/current numbers from the old definition cannot all be presented as numbers on the new loss. Either keep the historical series explicitly versioned or recompute a comparable score from a retained sufficient trace under the new definition. The missing historical203 weight checkpoint does not authorize inventing a new-definition203 loss. Keeping the original tolerance avoids this particular comparability issue. This is a conditional requirement about truthful comparison, not a request for a new experiment.

### B. Specify accumulation arithmetic; remainder initialization is now explicit

§8.3 is a useful conceptual error-feedback equation, but it does not state the evaluation precision/rounding order of U and the stored precision of `E_next`. The stopped implementation computes with F64 temporaries, converts the wanted value to F32, then performs BF16 storage rounding and retains the F32 wanted-minus-stored remainder. A formula over exact reals does not fix those implementation details.

For a native F32 tensor, if U is rounded to F32 before `Q_k` and `Q_k` is F32 storage, `E_next` is identically zero. Such a path does not accumulate sub-F32 requested changes. This need not be prohibited; the specification must say what it intends rather than imply that the old BF16 accumulator automatically provides error feedback below every native dtype's resolution. Preserving the stopped BF16 arithmetic and choosing any different policy for native F32 are distinguishable decisions.

The stopped checkpoint has remainder state for the old70 tensors, not the entire enlarged eligible inventory. The final draft now correctly specifies the transition: preserve those70 native/remainder identities exactly and initialize newly eligible remainders to zero because that run never requested updates in those formerly frozen slots. It does not pretend to restore an unavailable historical538-tensor accumulator. Shared/tied storage must accumulate its gradient and maintain/install its native value and remainder once, not once per alias, as the new text requires. Its implementation still needs an explicit complete-state convention for sparse embedding and unchanged rows.

### C. Retain the response account for the actual installed displacement

§8 defines a response for S, but the installed displacement after carried remainder and native conversion is generally **not** `eta alpha S`. To specify the existing distinction completely, retain the actual-displacement account at the same source weights/history/pairs:

```
D_native = float(W_next) - float(W_current)
q_native_estimate = J_r D_native
predicted_loss_at_installed_displacement = ||r + q_native_estimate||^2
```

Its coefficient is one. A fitted optimum computed from that second product is not the installed coefficient and must not become a hidden gate. This is the second JVP already present in the stopped loop, not a proposed extra finite/native probe.

In §11, distinguish continuous-direction prediction, realized-displacement **surrogate** prediction, and fresh native measurement. The second JVP is not itself measured native response. If fresh history or correspondence changes, comparing a held-history prediction with fresh per-merge loss does not isolate derivative fidelity alone; the trace/path changes must remain visible. No additional evaluation of other answers is required by this distinction.

### D. The enlarged scope still needs an executable, typed state contract

The eligibility principle is clear, but this document correctly does not make the stopped70-tensor implementation complete by declaration. The successor needs a manifest of unique native storage spans, learned/buffer classification, shapes/dtypes, causal uses and explicit exclusions. It must support indexed embedding lookup and gradient accumulation, newly eligible norms/scales/projections, and dependency paths through actual KV producers. Merely adding names to `graph.leaves` will not make file-backed embedding lookups differentiable or make the BF16 matrix encoder accept F32 scalars/vectors.

Complete native/resident installation, source bindings, sparse-transition representation, restoration, checkpoint/remainder schema and exact storage reserves must cover that manifest. The source-level consumer-exception restoration gap in `model_trace_response` needs an outer call-boundary cleanup, as already acknowledged in §12. An installed partial state must remain non-resumable as a completed update. No change to architecture, parameter count per operation, mask or token-selection rule is implied.

### E. Eta is a remaining declared choice, not supplied by the local optimum formula

The provenance and formula for the inherited eta are now concrete. The stationary alpha rescales the current joint direction, but does not make a multiplier calibrated on an earlier70/limited-scope response universally valid for the enlarged parameterization or its estimator. The draft correctly marks this decision unresolved. Root can justify retaining that empirical initial choice as an experiment with its limits, or specify a supported replacement; this review does not demand an extra probe, guessed schedule or perfect-step search. The numerical rule must be stated before the successor is described as specified.

### F. Small mathematical/recording clarifications

- State what happens if the **flattened full turn field** used for cosine correspondence has zero norm, even when every incoming/output vector is nonzero. Its cosine is undefined. The existing implementation refuses it; do not silently invent an epsilon or match.
- Use separate notation for tensor index and training iteration. §7 uses `W_k` as the kth tensor and then `W_k -> W_(k+1)` for an iteration. For example, `W_a^(s)` avoids suggesting that the next tensor is the updated one.
- Preserve the useful §11 distinction between pair-level data and the current reference-aggregated map. Pair-level weighted losses, reference-position totals and current-position totals must each sum back to the same L. The existing `560×105` aggregate plus pairs does not reconstruct every individual pair residual. Keep the new numerical maps local with provenance; the existing MLflow exporter can expose summaries/artifact links without logging raw hidden or parameter arrays.

## What remains scientific uncertainty rather than a missing formula

The selected reference is operationally fixed, not certified perfect. The observable compresses computation; isotropic tolerance is a declared scale rather than a confidence guarantee; DTW is a comparison proposal rather than proof of causal equivalence. These are explicit limitations of the agreed first experiment, not a requirement to replace it with all30 answers or a universal correctness classifier. The draft handles them appropriately when it distinguishes decreasing one-pair divergence from general repair or transfer.

A continuous surrogate, even one with exact native origin primals, can mispredict finite BF16/F32 behavior. New history/pairs, nonlinear residual response, quantization and expanded gradient allocation can all matter. No particular historical rise is diagnosed by this specification review, and no new native experiment was performed.

## Evidence and review boundaries

The original documents were read themselves in this continuous review sequence, rather than using another seat's summary. Their unchanged hashes were checked for this specification review:

- `reports/fixing-a-reasoning-mistake-by-editing-the-weights.pdf`: `a5761c87f7760d13cfe38d754fb8d0921460fc85f2a4b00ffd4bceb33d81e080`.
- `reports/how-the-trace-derived-edit-works.pdf`: `10de7506371d4bc40f0c3dd255d450c60a949891513e5050e58870a9bda91410`.
- `reports/how-the-path-edit-works.pdf`: `fced267c17a61c5da09fe9bb2b0fe4cabb618fbbd2a8542c4c64e38be76aacc2`.
- `reports/Iterative_Learning_from_Computational_Traces.docx`: `71cc3dfcfbf386898f706c571ad5539ef2f20629e510abc5710b003d2eb07149`. The embedded combined-loss, clipped repair-loss and total-cost equations were viewed directly. Its proposed separate behavior/preservation losses are not added to this experiment.

Checkpoint metadata agrees with the draft's clean handoff: local21/historical317, complete native state/remainder, no pending installation/transition, normal stopped launcher and a complete naturally ended measurement at loss `6.447133821394787e-5`. Inspected metadata hashes are `16523d45286c4ba64e5c52bad8675b94c05ff62fb6717fcbb3b114ea821c4891` (`final-state.json`), `d3fb0c722c268b344f0aeb64bf9ea224c465b642d9b4310429bf83f6212773ad` (`launcher-exit.json`) and `058687dc8f4f684892d769097e0fa11361b542b5cb5afee4a1b4b0d02ee3046d` (`loop/21/measurement.json`). Large payloads were not rehashed here. Process/service inactivity is root's execution report, not a new process inspection by this reviewer.

No new numerical test, reference recalibration, model replay, source edit or launch occurred. The core equations are sound; the unresolved choices and implementation work above must remain explicit. This review approves neither a successor executable nor a claimed research outcome.


## Addendum: revised specification, November 17, 2026

This addendum reviews `FINAL-EXPERIMENT.md` SHA256 **`a357b49acdc3c4c105f6d1fb1f11222961a6c1a329b15dd075420a8e72806972`**, 472 lines. It supersedes the earlier open-specification findings above only where the revised text resolves them; it is not implementation or run approval.

The proposed FP32 extension in §8.3 is arithmetically coherent as compensated accumulation at the stated precisions. Forming `U64` from the current stored weight, retained remainder and scaled direction, then computing `U64 - f64(W_next)` **before** casting that difference to FP32, retains the representable conversion residual. Computing the remainder only after converting both operands to FP32 would instead erase that information. The proposed formula avoids that mistake. `E_next` is still a rounded approximation to this residual: it does not make `W_next + E_next` an exact real-arithmetic decomposition, preserve increments lost during the FP64 additions, or guarantee retention below FP32's representable range. The draft now says this explicitly. No new arithmetic defect was found in the formula.

The BF16 branch retains the existing sequence: ordered FP64 accumulation, the historical `wanted32` conversion, native BF16 rounding, and FP32 remainder relative to `wanted32`. Thus it intentionally retains the existing precision boundary rather than silently changing the old 70-tensor recurrence. Preserving those 70 stored weights/remainders and initializing only newly eligible parameters with their actual stored weights and zero remainder is consistent with a scope extension. Shared storage must have one update/remainder state; the revised text states that requirement. Implementation still has to establish native conversion equivalence and reject nonfinite accumulators, converted weights or remainders before any partial installation. The formulas alone do not establish those executable properties.

The second JVP is now correctly specified on the **actual installed displacement**, with coefficient one. It describes the already selected update rather than fitting a second optimum and mislabeling that fitted prediction as the installed update. Its declared displacement/cast precision and approximate derivative status remain material: this is a held-history, held-correspondence surrogate prediction, not a new native measurement. The draft correctly separates it from fresh regeneration and does not attribute changes in generated history or alignment solely to derivative error.

The step-size decision is now specified: retain the measured `eta = 0.26989262219171917` initially while recomputing the full-scope response-derived `alpha`. My earlier statement that eta itself remained undecided is therefore superseded for this snapshot. Transferring that empirical factor from the earlier parameter scope remains an explicitly acknowledged assumption, not a derivation of stability for the enlarged update. No additional schedule or successful-training claim follows from the formula.

The revised tolerance discussion now identifies the user-origin reason for permitting collateral movement in agreeing computation, requires a new bound artifact and loss identity, and separates historical curves from the new definition. This resolves the policy/provenance omission; it does **not** supply the still-missing measured allowance values or authorize an arbitrary positive floor. Zero-cosine refusal, explicit iteration/tensor notation, and versioned baseline/best/current curves also resolve the corresponding specification ambiguities. The agreed experiment remains one incorrect response against the selected fixed reference; the other successful witnesses' saved calibration losses supply no defect diagnosis for that objective.

The local mathematical specification is internally coherent at this level. The remaining boundary is concrete implementation and qualification of the full eligible dependency graph, mixed-dtype installation/remainders, unconditional restoration, complete bindings and local observability, together with the unresolved tolerance artifact. The draft accurately lists these as unfinished. This addendum performed no code change, numerical test, GPU work, recalibration or launch; only the required commit guard is run for the review record.
