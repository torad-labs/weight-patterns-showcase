# The next new sweep tests separate mean and row-displacement bounds

November 11, 2026 (CT). Authorization received by Astra through dispatch, relaying dispatch-93. Astra did not receive this answer directly from Marcos.

## The answer

Dispatch reported that Marcos answered its AskUserQuestion at **1:48 PM CT** by choosing to test the split, the recommended option.

Dispatch said the question presented the counterexample in Astra's proposal: a mean-preserving change can satisfy an average displacement budget while increasing the original spread. The approval is for the next **new** sweep to test the split in `records/retrospectives/2026-11-11-mean-and-row-preservation-astra.md`, as proposed at commit `7b37734e`.

## What was approved

Separate, at each registered answer cell for direction and log size:

- The protected group-mean movement M, limited by the original member-mean resolution V/N.
- The mean squared protected row displacement Q, limited by the original empirical row spread T=(N-1)V/N+A.

Here N is the original group's member count, currently 30, V is the sample variance of original registered member means, and A is the original within-member registered-row variation. Their formulas and preservation meaning are in the proposal; the method pages now state them prospectively. The registration and the definition of protected movement are unchanged, including removal of the wrong group's shared movement along the gap where a move is asked.

The existing original-only floor principle is applied separately to each statistic: b_mean=max(V/N,F_M), b_row=max(T,F_Q). F_M is measured as mean movement and F_Q as row mean-square movement on the existing original-only floor comparisons. This includes the existing size-expression floor, separately for both statistics; it does not promote the baseline rebuilt-versus-engine diagnostic into a new floor. Neither is inferred from the other, and no edited-model discrepancy enlarges them. The reference is frozen for the new policy; existing references are not overwritten.

Both bounds apply to each of the existing fitting-third feasibility readings and their pool, using original N rather than the number read in a third. They also apply to the existing unused-answer replay. This does not expand which answers differentiate a pull: the all-fitting restoration, solved/held thirds, external audit/pool checks and candidate selection keep their declared roles. A five-member group's mean is being held to an original-reference policy, not given a new confidence interval.

## What was not changed

- `controlled-bf16-3` and every earlier declaration/reference keep the policy under which they were made. No running sweep is switched to the split and no past result is regraded.
- The target, permitted shared wrong-group movement, question-position bounds, own-answer mean/spread budgets and their own-start floors remain unchanged.
- The actual BF16-written matrix and native computation remain authoritative. Candidate selection and both final acceptance readings retain their existing requirements.
- No-fit answers do not become tuning data. Reserved-40b remains sealed.
- This is an average-displacement policy, not per-answer preservation or a guarantee that the original distribution or spread is preserved. The counterexample is not removed by approval to test.
- The approval is not a blanket authorization for other changes to the method, including a simultaneous restore-and-pull phase. Changes to the method's steps still require Marcos's decision through the established route.

## Why test it

The existing Q<=V/N policy bounds both protected mean movement and heterogeneous row movement by the resolution scale of an original group mean. Earlier readings showed cells where heterogeneous movement dominated that quantity, and others where coherent mean movement itself exceeded the bound. The proposed split distinguishes those two mechanisms without changing the mean-resolution policy or introducing an arbitrary multiplier.

The experiment tests whether allowing heterogeneous movement up to the original rows' empirical spread permits stronger useful construction while the protected mean and unchanged own-answer checks hold. It would be weakened by inadequate repair, protected-mean violations, or failure of the unchanged final checks. The next run's declaration supplies its own evidence-based prediction and failure interpretation before outcomes are known.

Earlier repairs and transfer remain established evidence. This experiment concerns a reliable training configuration, not reopening whether the technique has worked.
