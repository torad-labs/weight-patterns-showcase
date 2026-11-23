# One pair, local computational discrepancy, and whole-path closure

Marcos directly restated the intended experiment on November 16, 2026, while
asking the implementation peer whether the active implementation follows it. This records his
meaning at that time; it is not a change to that running declaration. The later
direct correction recorded in `2026-11-16-direct-computational-update.md`
supersedes the inverse-solver requirement below for the next run. The earlier
experiment and its actual review remain historical.

- Train one originally wrong answer against one fixed original correct
  reference. This is one pair, not multiple targets during training.
- The discrepancy is between their paths of computation. The original check
  required an inverse weight correction. **Superseded:** the later correction
  requires differentiating the computational loss through its actual functions
  into the weights, applying a response-scaled finite update, then regenerating
  and measuring again; no inner inverse/search must finish first.
- After installing an update, generate a new response from the edited model
  and use its computational trace for the next correction. Do not grade or
  compare response tokens. Keep the original correct reference fixed.
- A local computational relationship already inside the original overlap
  tolerance supplies no direct correction gradient. Divergence supplies it.
- Successful training ends when the complete fresh computational path agrees
  with the reference within that tolerance, leaving no computational
  discrepancy that requests correction.

## Historical check against the then-active implementation

The active declaration is
`studies/04-round-2-overlap/runs/vector-chance-loop-recovery-2-original/declaration.json`.
It trains `discovery-05-channel-h1` against the fixed `expanded-004-h1`
original trace. Reference selection minimizes mean full-turn departure from
the other original successful traces (`turn_reference.select`); this is the
operational meaning of strongest computational agreement, not a claim that
every part of that answer is proven correct. Original successes calibrate
the fixed tolerance; they are not additional training targets.

The implementation peer read the active reference, discrepancy, inverse, installation and
controller code. The independent reviewer checked the same question with all four
actual research papers and the goal, and checked seven relevant source
bindings against the active declaration. The experiment matches the stated
loop with these precise distinctions:

- `vector_chance.outside` supplies zero residual and zero direct squared-loss
  gradient inside the boundary, including exact agreement at zero radius.
  At zero radius its residual Jacobian remains present so that the joint
  inverse can account for a proposed update breaking that agreement. This is
  not a nonzero local loss gradient at agreement.
- The correction is local in where discrepancy is measured; parameters are
  shared across the computation. Zero local loss gradient does not freeze
  every weight that contributes to that relationship. Remeasure the whole
  fresh trace after each installed update.
- The current inverse is approximate and uses the held-history graph and
  fixed linear response described in
  `records/retrospectives/2026-11-16-review-joint-inverse-fidelity.md`. Its predicted
  decrease is not a guarantee of decrease on the next native trace.
- `pair_loop.iterate` accepts closure only after an actual update, a complete
  fresh native trace, and zero outside cells. A zero parameter gradient or
  solver stall does not establish closure. An incomplete acquisition or
  failed correction can end the attempt with `closure=false`.

A small CPU-only check of the actual `vector_chance.outside` implementation
confirmed zero squared-loss gradient at a positive-radius interior, exactly
on its boundary, and at zero-radius exact agreement; an outside point gave a
nonzero correcting gradient. No training data, model, source code, reference,
tolerance or active run was changed for this check.

The four sources supplied to the independent reviewer are
`reports/fixing-a-reasoning-mistake-by-editing-the-weights.pdf`,
`reports/how-the-trace-derived-edit-works.pdf`,
`reports/how-the-path-edit-works.pdf`, and
`reports/Iterative_Learning_from_Computational_Traces.docx`.
