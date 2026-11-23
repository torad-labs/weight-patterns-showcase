# Explaining the historical repairs and transfer

November 9, 2026. This is an examination of saved records under Marcos's current request, not a new round-two experiment. No inference, fit, checkpoint restoration or weight intervention has run. The stopped overlap stays stopped and reserved-40b stays sealed.

## The question

Why did some of the discarded edits repair a calculation, and why did a related correction appear on questions excluded from the fit? Can the responsible parts of the connected computation be discovered from their trace, instead of being chosen by an unreliable solver, a borrowed threshold or an arbitrary cap?

The old run did not implement the hypothesis. It fitted individual contextual pair differences, and its norm cap changed the fitted directions as well as their magnitude. Its successful answers do not validate either choice. A useful observation from a flawed construction can still be investigated, but the construction does not become the explanation.

There are two distinct objects of study:

- The historical **effect of the edit**: the change between original and edited computation on a question. Explaining this requires both executions and causal checks of the stored changes.
- The hypothesis's **recurring departure between correct and incorrect computation**: discovered by independently overlaying the two groups. This remains the source of a future round-two repair target. An old edit-effect direction is not a substitute for it.

## What the saved evidence now narrows down

### A specific distinction in the calculation transferred

The procedure needs descent to contribute positively to available energy, while the required final-climb energy uses the specified climb height. Several historical changes corrected the latter while disrupting the former.

The new arithmetic classification in [mechanism-evidence.json](mechanism-evidence.json) uses the saved final fields and the problem's known energy quantities. It is an exact expression match, not a claim to have inferred an internal operation from an output number.

| On the 40 excluded questions | Early FK | Later full-path pass |
|---|---:|---:|
| Wrong required-energy fields made correct | 15 | 15 |
| Among those, original required energy exactly equalled climb plus descent energy | 7 | 6 |
| Of that subgroup, edited available energy is also correct | 0 | 5 |
| Of that subgroup, edited available energy equals initial kinetic energy minus descent minus loss | 5 | 1 |

The remaining two early subgroup cases have other incorrect available values. The subgroups are not identical; these counts do not isolate a single construction change.

For `fresh-00-h2`, the correct calculation is 25 + 20 - 8 = 37 J available and 2 x 10 x 2 = 40 J required. The original produces 37 / 60. Early FK produces -3 / 40: the climb is fixed, but descent is subtracted in the available-energy calculation. The later pass produces 37 / 40.

For `fresh-10-h2`, the correct calculation is 98 + 40 - 14 = 124 J available and 4 x 10 x 2 = 80 J required. Early FK produces 44 / 80; the later pass produces 124 / 80.

This identifies a concrete candidate explanation to investigate: a change in how the descent quantity participates at different stages of the procedure. It does not establish a dedicated descent feature or locate its cause. The later edit changed many things at once. The full responses also contain explanatory defects, recorded in [answer-readings.json](answer-readings.json).

The new speeds and climb heights in the 40-question set show that an identical fitted prompt was not simply repeated. The same template family, previously seen evaluation results, and occasional incoherent explanations limit the transfer claim. A broadly useful physics mechanism has not been demonstrated.

### The first visible change does not identify the repair

I compared the saved original and later edited token sequences for all 40 questions. Their first different generated token occurs at positions 0 through 31. Both models have already processed the whole prompt at that point; an early change is not evidence that physics-related processing has not happened.

One identical early wording change occurs in 14 responses: after “Here is the”, the original continues “compact numerical calculation”, while the edit continues “calculation to determine if the block reaches the destination”. Those 14 include:

- 3 repaired answers;
- 4 regressions;
- 4 answers that remain wrong;
- 3 answers that remain correct.

Another change, from the heading “Point A” to “Start of Descent”, occurs in four repairs and one regression. It is therefore not sufficient to explain repair either.

The saved full-logit hashes already differ at the first prediction in every case, but a hash disagreement supplies neither a meaningful effect size nor a causal location. The branch records retain the actual first-choice margins where both choices appear in the saved top-eight scores.

This gives a useful negative result for automatic discovery: selecting the earliest changed word, a shared heading or any changed score would also select unsuccessful changes. We need to follow the connected internal influence into the relevant calculation and distinguish where successful and unsuccessful continuations separate.

### The transfer runs have no internal trace

All 80 original/edited result records for this 40-question comparison are plain natural generations with zero capture events and zero tensor bytes. They contain text, token IDs, score hashes and selected alternative scores. They do not contain the internal states needed to locate a transferred correction. Their source paths, content hashes and capture fields are in [mechanism-evidence.json](mechanism-evidence.json).

The old construction cohort does have internal measurements and component ablations. They show, within their limited scope, that:

- the layer-11 feed-forward change and layer-14 K/V change have different repair and damage profiles;
- removing the layer-0 and layer-20 attention-output changes together loses repairs, even though their local prediction scores were unfavorable;
- the old full-path distance can get worse while final fields improve.

Those observations constrain explanations. They do not provide the missing internal trace of generalization, identify either omitted block independently, or show that the same internal mechanism caused repair on both populations.

## How to investigate automatic discovery without reviving the old recipe

The historical study should first follow the **actual changes that led to repaired calculations**, including the cases where the useful climb correction coexists with descent damage. Its output should be a connected explanation with competing cases, not a list of large weights or a ranking by answer improvement.

1. **Locate a repeatable change in the connected execution.** For a repaired calculation, follow the original and edited natural executions from before the first erroneous operation through the relevant energy passages. Compare those changes across repaired construction cases and transferred repairs, and with nonrepairs and regressions that underwent similar early wording changes. Keep card arithmetic consistent within every comparison; do not pool coordinates across cards. A location is interesting because a comparable influence repeatedly enters the relevant computation, not because its matrix norm or isolated activation differs.
2. **Test responsibility using the existing stored change.** Once the traces nominate a contributing set of transformations, remove that part of the old stored edit while keeping the rest identical, and restore it. This is a diagnostic intervention, not a new fit or a search over arbitrary caps. Check whether the corresponding trace change and repair disappear and return together, on both the original repaired questions and transfer questions. Test interacting groups where a component's effect depends on the rest; a lone local score cannot establish responsibility.
3. **Carry forward the discovery criterion.** If a recurring connected feature nominates components whose interventions account for repair and transfer, that feature becomes a candidate for automatic localization in the new round two. It must then be found from the approved original success/failure overlaps before a new edit is built. The old layer list, pairwise targets and cap do not become defaults.

These are proposed next scientific operations, not completed tests. They introduce no step size, selection threshold or acceptance cutoff. This retrospective does not select the new round-two population. The old 40 are already inspected and can serve as historical diagnostic cases, not as new untouched confirmation.

## What is established, and what is still missing

The evidence establishes that some stored interventions changed a reusable numerical procedure on excluded inputs, sometimes correcting the climb while damaging available energy. It also establishes that obvious early text changes are shared by repairs and failures, and that the historical trace-distance metric did not explain the successful answers.

It does **not** yet establish why the transferred repairs happened internally or provide a validated automatic discovery mechanism. The missing evidence is specific: original-versus-edited internal executions for the transfer cases, followed by interventions on trace-nominated parts of the saved change. Those require recovering the exact historical checkpoint or verified stored patches and reproducing the saved natural generations before treating new captures as records of the old effect. No such recovery or capture has been performed here.

Source snapshot: `67559f4a2fc2b61eda99200b185d98e91d44f2ff`. New computation: saved-token first-branch comparison and exact arithmetic-expression classification. This is retrospective reasoning; no prediction is presented as having preceded the historical outcomes.
