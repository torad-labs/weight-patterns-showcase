# Earlier handoff: superseded by the residual-trace clarification

**Start with the current
residual-trace handoff (`~/Documents/dev/projects/qgre/qgre-main/.dev/decode-machine/RESIDUAL-TRACE-HANDOFF.md`)
and full reference (`~/Documents/dev/projects/qgre/qgre-main/.dev/decode-machine/RESIDUAL-TRACE-THEORY.md`).**

The passage below preserves an earlier interpretation. Its instruction to exclude
the inference trace from the target is no longer Marcos's intended meaning.
The current objective is to recognize recurring portions of the connected changes
as a capability is exercised. A useful partial trace need not reveal its full
cause, belong exclusively to one capability, or reproduce the complete mechanism.
Capability-focused requests are part of making that influence observable.

## Historical handoff; do not paste as current context

Read the full research reference (`~/Documents/dev/projects/qgre/qgre-main/runs/legs/gemma4-e2b/distributed-pathways-reference-2026-11-06/REFERENCE.md`) before proposing an experiment or debating the theory. It contains the five motivating examples, the intended vocabulary, local evidence, and the current editing results. This is research context, not a requirement to agree with every hypothesis.

The latest clarification from Marcos (`~/Documents/dev/projects/qgre/qgre-main/runs/legs/gemma4-e2b/distributed-pathways-reference-2026-11-06/sources/marcos-path-clarification-2026-11-06.md`) is the source note behind this summary; reread it if this summary loses a distinction.

## Pasteable context

I am investigating whether we can recognize the frozen distributed patterns that constitute information and capabilities in a model. I call those patterns **paths**.

The path is distributed through parts of many weights' numerical encoding. An individual weight does not independently contain the meaning of the whole. Portions or bits of many weights combine horizontally into the informational pattern, and the same substrate can support many overlapping patterns. A path may span almost everything; it does not have to be a small circuit.

The weights and their learned encoded organization are frozen during ordinary inference. An input makes an expression of that organization observable. **An inference trace is evidence through which we might recognize the path; it is not itself the path.** Do not redefine the target as a sequence of layers, a list of high-importance weights, or the downstream changes caused by an edit.

My neuron/electricity and electromagnetic-field comparisons are analogies for the distinction between a substrate and a distributed informational pattern. Engage that concept instead of treating the analogy as a literal mechanism to correct. “Horizontal” means the same distributed organization across all my examples. “Ghost” is the input-dependent expression of the frozen organization.

The five connected examples are:

1. **CPT:** I successfully installed knowledge in Gemma 4 E2B with four-bit training, repeating relationships through varied phrases and articles. My point is cumulative reinforcement of the recurring relationship across contexts.
2. **RL:** different successful completions can repeatedly exercise and reinforce a procedure. I am not claiming equal numerical gradients for all tokens or weights.
3. **Small specialized networks:** relatively few values and capabilities make their collective encoded pattern easier to recognize. Large models obscure such patterns through far more overlapping information, so multiple inference observations may be needed to distinguish them.
4. **Physics/control repair:** the dotted line or jigsaw piece is the functional behavior needed to complete the task. It is not a literal geometric hole.
5. **Qwen looping:** different completions with a recurring failure may help triangulate the same underlying encoded pattern within the overlap. The completions are observations, not the path itself.

I propose that repeated learning establishes reusable distributed patterns, and varied observations can help expose them. My stronger conjecture is that the organization of these overlapping paths helps make capabilities stable. That is the geometry I want to investigate.

The current task is recognition and investigation, without training. Use actual completions, internal activity, architecture, and correctly decoded native weights to orient it. The discovery method remains open; do not impose a favorite editing recipe or assume a particular vector or bit mask is the answer. Any later edit must follow an identified target and use the actual quantized representation.

We use the local **llama.cpp fork**. The earlier QGRE position-versus-value failure did not reproduce on its twenty-question comparison. A later Q4_K edit shifted a lookup preference, but neither that successful edit nor the analysis of its downstream effects identified the path I mean. The earlier claim of finding a path segment was withdrawn. Do not restart QGRE debugging as the goal.

The full reference links surviving FP4 CPT evaluations: 89/300 to 206/300, and 21/77 to 54/77, on their recorded forced-choice probes. These are empirical starting points. The proposed encoded paths have not yet been identified experimentally in our local work.

The subsequent original-model observation study (`~/Documents/dev/projects/qgre/qgre-main/runs/legs/gemma4-e2b/lookup-pattern-observation-2026-11-06/REPORT.md`) completed 40 discovery and 20 withheld questions, all answered correctly with exact answer-token agreement across ordinary, plain-observer, and captured runs. Selected attention diagnostics and native-quantized weight/input product profiles showed repeatable task contrasts on the withheld questions. Copying overlapped, and the contrast also existed in the inputs alone. The capture changed numerical logits despite preserving answers. No frozen encoded path was identified and no edit or training was performed; do not describe the inspected heads or product profiles as the paths meant here.

If you disagree, engage the proposition I actually made. Preserve the difference between my hypothesis, the analogies used to communicate it, and the evidence we have. Do not substitute a different claim and then correct that instead.
