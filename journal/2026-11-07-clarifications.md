# The trace, the repair, and what we have actually found

Discussion notes, November 7, 2026 (CT). This captures Marcos's intended meaning and the clarification reached after the persistent weight-edit experiment. It supplements the [full framework](../hypothesis/RESIDUAL-TRACE-THEORY.md); it is research context, not an operating law or a demand for agreement.

## What the path is

The research target is the pattern of information being transformed through a connected computation. Each participating weight contributes to part of that unfolding. What its contribution does depends on the incoming signal and what happened before and around it. The same frozen weights can therefore participate differently in different contexts.

“Horizontal” means that the relevant contribution is distributed across many weights and operations in an order and context. An isolated numerical value is not being treated as a labeled container for “physics,” a color, cohesion, or a procedure. Similar numbers in disconnected locations do not establish a shared path. Neither does simply finding that many questions use the same layers.

The signal changes; the parameters stay fixed during an ordinary inference execution. The proposed pattern is in the connected, context-dependent changes. Measurements let us observe those changes. The research is not a search for a special scalar value or a visually attractive arrangement of weights.

“Residual trace” means the observable effects left by an influence whose full origin may be hidden. It is not limited to the Transformer's residual stream, nor does it mean that one subtraction of two activation vectors is automatically the sought pattern.

## Why partial traces are enough

In the forest analogy, travelers may use stars, a lake, or other cues to find their way. The routes change with circumstances. Repeated marks can still reveal a familiar influence without revealing the cue, its exact rule, or the entire route.

The hidden-z analogy makes the same point: varying computations may retain observable effects of a common influence even if we never isolate z or recover a formula that reproduces it. Fibonacci, universal approximation, electricity, gravity, cylinders, and observation were explanatory analogies. They were not proposed as literal Transformer mechanisms, quantum dynamics, or literal moving weight matrices.

A useful fragment is already a finding. We do not move the threshold to recovering every part of a capability, an exclusive signature independent of all other tasks, a closed geometric object, or a complete function that deserves the name “physics.” The useful fragment may overlap with several capabilities.

The experimental opportunity is to ask questions that strongly and repeatedly require a capability while varying the situation. That supplies some of the narrowing a small dedicated network has inherently. It does not mean arbitrary prompts followed by a search for common numbers. The comparisons follow actual computational connections and retain the context in which each change occurs.

## The car/physics example: the distinction we resolved

Marcos's correction was that the neural network in the car example was limited, closed and trained only to resolve that physics simulation, so mechanical repair and completing the diverging branches of the physics pattern collapse into the same thing.

That is the intended functional comparison. In the dedicated car task, backing up, gaining momentum, clearing the obstacle, and returning to normal control belong to one coordinated behavior. Describing a repair as a conditional control path identifies how the repair was implemented. It does not, by itself, contradict describing the repair as completing the missing portion of the successful behavior.

The published example's limited environment, rules, observation layout, network, and recorded failure supplied strong constraints. Our interpretation is that these constraints made the missing progression legible. We should not turn that interpretation into a claim that the authors explicitly recovered a complete physics function or used our proposed trace-extraction method. Both the implementation account and the functional analogy can be retained without treating them as rival explanations. [Primary experiment](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp4_fix_car/README.md).

In an LLM, many more capabilities and contexts share the parameters. A family of relevant executions can expose a partial influence where inspecting the full weight array alone would not make the behavior legible. This is the reason for the observations, not an argument that the larger case must be impossible.

## What the Gemma experiment contributes

We observed a recurring partial progression in the connected computation: the normalized separation between paired executions decreased across FF14 and increased across FF15 on the eligible recorded decisions. This was **846/846** in the Q4 follow-up and **1,707/1,708** in the BF16 follow-up, with the exception retained. These are decision counts within paired executions, not that many independent problems. Absolute separation grew at both stages; the measurement is specifically relative separation.

Those observations guided investigation of the connected region and a later edit to **layer 16's attention-output projection**. They did not identify one scalar as “energy bookkeeping.” The edited matrix was read, changed, rounded into its native BF16 storage, and used by the unchanged inference program on every invocation.

The declared edit used the failed case's recorded projection input and the difference between the paired successful and failed raw projection outputs at prediction 235. Its immediate purpose was to move that contribution toward the successful case at that input. After native rounding, the remaining projection distance was **44.4103%** of the original: about **55.59% closer**. The ordinary response then used the recipient's own 3 m climb and computed **90 J**, rather than copying the donor's **30 J** answer.

This supports the practical description: **we recognized recurring partial computation, used connected observations to choose and construct a persistent parameter intervention, and observed a useful numerical repair together with its downstream propagation.** That is a meaningful result under the partial-trace objective we established.

The edit did not change the upstream FF14/FF15 progression. It changed the later layer-16 contribution selected for investigation. We did not show that an entire failing trajectory moved closer to a successful trajectory at every point. “Alignment” here has a measured, local meaning at the captured projection input, followed by observed downstream effects and a behavioral change. This precision describes the finding; it does not erase it or demand a complete capability as the new success criterion.

## What changed, and what did not

- The targeted numerical chain changed from **60 J available / 120 J required / NO** to **60 / 90 / NO**.
- The paired successful response stayed exactly **60 / 30 / YES**, including all generated tokens.
- A sled calculation also improved, from **0 / 40 / NO** to **80 / 40 / YES**.
- A different block problem acquired an incorrect required energy and verdict, changing **60 / 80 / NO** to **60 / 40 / YES**.
- Complete correct numerical calculations increased **7/28 → 9/28**. Correct explicit verdicts stayed **16/28**. The repaired responses still contained inconsistent explanatory labels.
- The first visible target divergence was prediction 269, “reached” versus “gained.” That is an observed branch in the text, not proof that one word alone caused the repair.
- Identical outputs did not mean untouched internals. All 11,053 comparable full-score hashes changed across the 28 cases.

The earlier one-prediction activation replacement was a different experiment: it changed scores once, without changing any emitted token. It was not a weight edit and did not expose the model to the same persistent change. Keeping the two experiments separate is essential to the comparison the user requested.

The remaining questions concern which observed fragments are useful for predicting or guiding an edit, how that generalizes to new cases, and how much interference the edit causes. They do not require pretending we have observed no recurrence or no repair. The current paired source difference also contains a real context change (the climb height), so it has not been isolated as a pure success/failure direction.

## How the motivating cases connect in Marcos's account

| Case | The common explanatory idea | Evidence status here |
| --- | --- | --- |
| EDC Las Vegas CPT | Repeated factual content across varied surrounding text reinforces its later accessibility; its influence can appear through varied uses | Marcos reports successful local training. This organization did not recover or rerun an independent CPT evaluation |
| Reinforcement learning | Successful procedures recur across varied completions; training can reinforce characteristics that make those procedures recur | A proposed interpretation, not a claim that shared rewards give equal gradients to all parameters |
| Small-network recognition | A limited set of tasks and capabilities makes coordinated functional organization easier to recognize | Motivating published examples; not proof that each weight has an intrinsic semantic label |
| Car/physics repair | Strong task constraints make an incomplete successful progression identifiable enough to repair | Published weight edit, interpreted through the functional analogy above |
| Qwen loop reduction | Different problem completions can expose a recurring unproductive tendency; altering a contributing direction can reduce it | Published experiment motivates partial intervention; “pruning looping branches” is an analogy, not a recovered literal graph |
| Gemma trace and weight edit | Connected observations make a partial region and contribution inspectable; changing parameters changes later ordinary computation and outcomes | Local measured results, including a repair and a collateral regression |

The shared interpretation is why the theory is worth investigating. The explanatory fit is not being presented as proof that every example used one identical algorithm or that all geometry has been recovered.

## Reading this in a future session

Start from these definitions and the actual evidence. Do not substitute a search for isolated weight magnitudes, a dump of layer names, or activation-only steering for the persistent weight-edit objective. Do not require the complete hidden function before recognizing a useful partial observation. Do retain negative results, eligibility limits, native precision, context differences, and the actual source of each claim.

The [full weight-edit report](../studies/03-round-1-single-pair-edits/A-weight-edit/STUDY.md), [consolidated findings](../publications/findings/FINDINGS.md), and [experiment map](../publications/findings/EXPERIMENTS.md) connect these notes to the records.
