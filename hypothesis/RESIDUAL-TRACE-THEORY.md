# Residual traces of capability

> Round two's status and method are in [ROUND-2.md](../studies/04-round-2-overlap/ROUND-2.md). Marcos's later clarifications of what the method means are in [the trace, the repair, and what we have actually found](../journal/2026-11-07-clarifications.md) and [overlap of connected capability traces](../journal/2026-11-07-multitrace-purpose.md). Round one's measured findings are in [FINDINGS.md](../publications/findings/FINDINGS.md).

Marcos's research framework, consolidated November 6, 2026, after the forest,
hidden-influence, partial-trace, and capability-focused prompt explanations.

This is a record of the agreed meaning and research direction, not a project law
or a requirement to agree with the hypothesis. The short entry point is
[RESIDUAL-TRACE-HANDOFF.md](RESIDUAL-TRACE-HANDOFF.md). This account supersedes the
earlier reference's instruction to search for a frozen encoded pattern while
excluding the inference trace from the research target.

## Start here

The object of interest is the connected trace of information being transformed
as a model exercises a capability. Each participating weight contributes to part
of that unfolding; its effect depends on the incoming signal and the surrounding
computation. Across appropriately varied executions, a capability's influence
may leave recognizable recurring portions of a trace. Those portions need not
have identical numerical values, occupy identical coordinates, or reveal the
complete mechanism that produced them.

The proposed discovery is recognition of this residual influence. A useful
fragment can be recognized before we can isolate its origin, describe a complete
function, reproduce the capability, or separate it from everything else the model
does. Carefully chosen requests can make a capability contribute strongly and
repeatedly, potentially making enough of its influence visible to understand or
repair part of the behavior.

The threshold, as Marcos sees it: the full signature may never be recoverable,
but parts of it definitely can be.

His confidence comes from seeing a common explanatory pattern across knowledge
installation, reinforcement learning, small-network recognition, physics/control
repair, and Qwen editing. He explicitly distinguishes that explanatory fit from
proof. The framework is recorded here with that distinction intact.

## 1. What a trace means

Weights remain fixed during ordinary inference. What changes is the signal
being transformed through their combined operation. The same parameter can
contribute differently as its input and computational context change.

"Horizontal" describes the distributed participation of many weights in that
unfolding. It does not mean one horizontal layer, a memory layout, or individually
meaningful weights joined by semantic labels. Marcos's emphasis is on the
function enacted through many contributions in an order and context: the
information-bearing pattern is in that coordinated operation.

The observation must therefore retain continuity: what information reached an
operation, how it changed, what that result contributed to next, and how the
developing computation reached its outcome. This can branch, combine, and reuse
state. It need not be a thin, single-file chain.

An execution trace is an observation of the unfolding. The recurring pattern
sought across executions is a residue of an influence within those traces. A raw
dump is not automatically a recognized pattern. Neither is an architectural
itinerary such as "attention, MLP, output," which many tasks share.

The measurements can be numerical. The methodological distinction is what we
look for in them: connected, context-dependent changes and their recurrence,
rather than treating an isolated magnitude or a similarity between disconnected
weights as the semantic object. The recording is evidence of an unfolding, not
an answer supplied by a visually interesting arrangement of numbers.

"Residual" here means the observable traces left by an influence. It is not
restricted to the Transformer's residual stream or to a subtraction of two
activation snapshots.

## 2. Partial recognition is already useful

We may never recover a single coherent function that deserves the name "the
physics function." The useful computation may depend on prior transformations
and context so extensively that no such isolated description is available to us.
Recognition of some of its effects can still be informative.

Potential discoveries include a recurring portion of successful computation, a
repeated departure from that progression, or a tendency that repeatedly returns
computation to an unproductive state. A fragment can participate in several
capabilities and still be useful. Exclusivity is not a prerequisite for observing
or investigating it.

There is no requirement to recover every path, close every loop, find the hidden
cause itself, or draw the complete geometry before making progress. An edit is
one possible later use of recognition. An output improvement by itself does not
establish what was recognized; conversely, an informative partial trace does not
need an edit to count as a research result.

Marcos describes Qwen in a branching-path analogy: a recurring branch can lead
the computation back into a route that returns to the same branch. Reducing the
influence that favors that return may let useful computation continue. This
analogy explains the behavioral effect without asserting that a literal cyclic
graph was extracted or that the model's entire reasoning capability was repaired.

## 3. The prompts are part of the instrument

In a small specialized network, a limited task and known input/output meanings
can constrain the possible useful computation. For a large language model,
deliberately chosen observations may provide some of that narrowing.

Choose a family of requests in which a particular capability or a few related
capabilities contribute strongly. Vary the situations while keeping the required
capability well represented. In a physics example, objects, wording, quantities,
and initial conditions can vary while a particular physical dependency keeps
being needed. A broad topic label alone is less informative than repeated use of
the relevant capability.

Each execution supplies another expression of its influence. Successful cases
can expose a progression; characteristic unsuccessful cases can expose where
that progression departs. The repeated observations may constrain what a repair
needs to accomplish even when the full mechanism remains hidden.

This is the meaning of "completing the dots": observations can make a useful
fragment legible enough to investigate an intervention. It is not a requirement
to recognize a predefined curve, isolate one vector, or fill a literal hole in
geometric space.

There is no magic number of prompts. The proposal is that relevant variety can
increase observability, not that merely increasing sample count guarantees a
recognizable pattern. Different executions may expose different portions or
families of traces. "Overlap" or "superposition" does not prescribe averaging
all raw vectors together, and alignment by a layer number or matching word alone
does not establish functional correspondence.

An arbitrary easy task selected because it is convenient to capture, a set of
unrelated questions, or a list of similar weight values would change the research
question. The capability and its use must guide what is observed.

## 4. The analogies and the concepts they convey

### The forest

Many people cross a forest and escape. Some navigate by stars, which occupy
different positions at different hours; others encounter and follow a lake.
Their routes need not coincide. Their footprints can nevertheless reveal
recurring traces of influences that helped them leave.

An observer may never discover that stars were the cue, reconstruct each
traveler's strategy, or identify one trail shared by everyone. A recognizable
portion of the marks could still reveal something useful. Several families of
trails can coexist. This is the central analogy for observable residue without
full recovery of its cause.

The advantage in our experiment is that we can shape the requests to make a few
influences dominant, instead of accepting an unconstrained population of forest
journeys.

### The hidden influence z

Marcos imagines quantities produced by changing, complicated expressions that
share an influence called z. We might not isolate z, know its value, or recover
the formulas. Across varied calculations, the way its influence changes the
intermediate results may still leave a recognizable trace.

The analogy is about recognizing an influence through connected effects. It
does not posit one real scalar z inside the model, require a constant activation,
or prescribe a ratio calculation as the discovery method.

### The other analogies

| Analogy | Concept it conveys |
|---|---|
| Physics as the trace | A learned capability is recognizable through coherent transformations across changing situations; the correct numerical answer alone is only an endpoint. |
| Dotted line or jigsaw | Enough surrounding functional context can make a missing or defective continuation understandable. |
| Sine-wave approximation and the golden ratio | The organization of changes can reveal a pattern that individual numbers do not make evident. These are explanatory examples, not claims about a Transformer's training algorithm. |
| Neurons, electricity, conductors, and bits | Distinguish the physical/numerical substrate from the information-bearing operation distributed through it. No literal binary-bit traversal or biological storage mechanism is asserted by the analogy. |
| Gravity, cylinders, roads, and geometry | Express influence, coordination, and coupled effects. No sphere, rigidity law, metric, dimensionality, or guaranteed safe propagation rule has been established. |
| Observation or quantum collapse | A request conditions which effects become expressed and observable. It does not invoke quantum hardware or require one predetermined continuation. |

Engage the conveyed concept before assessing a hypothesis. Treating an analogy
as a literal implementation claim and correcting that substitute was a repeated
source of misunderstanding in this conversation.

## 5. Why the examples belong to the same argument

The following are applications of Marcos's interpretation. The reported evidence
and the proposed explanation are kept distinct. The published Astra experiments
were performed by the cited researchers, not by this local session.

### CPT and EDC Las Vegas

Marcos has used continued pretraining to install knowledge in Gemma 4 E2B with
four-bit training. His example is repeatedly expressing an artist-stage fact,
such as Martin Garrix at Kinetic Field, through different phrases and articles.
That example describes corpus construction, not a claim about a current event
schedule.

The recurring fact is surrounded by varying content. His account is that
repeated training reinforces the computations that make the fact accessible
through those different contexts. The common influence may later be observable
through how the computation develops when that knowledge is used.

The [local CPT working report](../../../projects/kandi-cpt/kandi-cpt-main/docs/cpt_installation_best_practices.md)
records better installation with varied wording than with a small set of
repeated templates. It also records a stage-retrieval decline from 0.711 to 0.506
after downsampling to equalize coverage. These are surviving report entries,
not new measurements made for this document. They are not measurements of a
semantic fact's gradient mass or a recovered internal trace.

The point to preserve is cumulative reinforcement across varied examples. Do
not replace it with a claim that every token receives the same numerical
gradient. Nor should a new collaborator demand that Marcos first re-prove the
behavioral installation before discussing how to investigate its internal
expression.

Controlled research also reports that varied phrasing in pretraining can help
later knowledge extraction: [Physics of Language Models, Part 3.1](https://arxiv.org/abs/2309.14316).
This is related evidence, not a validation of the proposed trace representation.

### Reinforcement learning

Across varied successful completions, useful procedures can recur: checking a
result, reconsidering an assumption, or organizing intermediate steps toward a
goal. Marcos proposes that reinforcement makes the coordinated transformations
supporting such procedures more reliably expressed. Procedures can participate
in larger procedures, making their influence dependent on context.

The [DeepSeek-R1 report](https://arxiv.org/html/2501.12948v1) describes
self-verification and reconsideration emerging during reasoning-focused RL.
Those behavioral observations fit the proposed account; they do not directly
measure the internal trace. Written reasoning is an observable behavior and is
not automatically a faithful recording of every internal operation.

The proposal concerns reinforcement of useful coordination. It does not require
equal gradients, an optimizer that explicitly understands a procedure's meaning,
or knowledge of the historical training steps that produced a later trace.

### Astra's recognition experiments

The [authors' report](https://rikiyat.github.io/blog/weight-space-programming/)
describes recognizing four games, diffusion shapes, and characteristics of small
language models from weights. These were recognition exercises, not edits. Shape
recognition was incomplete: the response drew outlines for models that generated
filled shapes. In the sycophancy experiment, weights-only attempts failed, while
selected attention observations enabled identification.

Under Marcos's interpretation, constrained tasks can make functional organization
more legible, and partial recognition can precede recovery of complete behavior.
Where overlap obscures the interpretation, a request can expose an influence
through its expression. Parameter count alone does not prove a network has only
one mechanism or one possible interpretation.

### Astra's physics/control repair

The [released car-on-ramp record](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp4_fix_car/README.md)
states that Astra received weights, scene and task descriptions, and one recorded
episode. It introduced a small control path across several layers, using
position and velocity to back up, accelerate, then return control to the original
policy after clearing the obstacle. The reported edited result was 1,024
successes in 1,024 test episodes.

The relevant coherence is between actions and their consequences: backing up
creates room for acceleration, and acceleration creates the conditions needed
to pass the obstacle. The repair makes that progression work. This fits the
functional "dotted line" interpretation. The record establishes a conditional
control repair, not extraction of a complete general physics mechanism.

### Astra's Breakout, Seaquest, and Asterix policies

The [three-game record](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp5_write_from_scratch/README.md)
describes three independent constructions from zeroed weights using game rules,
source, and input/output definitions, without training or gameplay feedback.
All three beat the reported 100-million-step PPO baselines; comparison with the
longer-trained baselines varied by game.

In each case, weights were written to establish coherent transformations from
changing observations to useful actions. These show construction guided by known
behavior. Our discovery proposal goes in the other direction: observe existing
computation to recognize portions of the influence that makes it useful. The
connection does not imply those two directions are equally easy.

### Astra's Go policies

The [Go record](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp6_go_from_scratch/README.md)
describes separate runs with zero, 10, 25, and 50 observation budgets, starting
from zeroed weights. All four beat the tested smaller PPO opponents; the
50-observation version also beat the tested equal-sized PPO networks.

Different positions express useful play through different coordinates and
choices. Feedback can expose where an intended progression fails. In Marcos's
account, observations constrain a functional repair without requiring a complete
description of everything the network can do.

### Astra's Qwen loop interventions

The [Qwen record](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/README.md)
describes three independent editing runs. The 10-pass run suppressed a
repetition-associated direction across multiple output matrices; the 25-pass run
disabled one attention head; the 50-pass run targeted selected MLP output columns
in two layers. All three improved the reported math scores. The 50-pass result
raised MATH-500 from 7.9% to 48.1% and reduced answers reaching the token limit
from 93.0% to 33.1%.

Different problems and answers exposed a recurring unproductive behavior. The
intervention altered a contribution associated with that behavior. This supports
the usefulness of a partial handle on a recurring influence; it did not require
reconstruction of the complete mathematical capability. Marcos's branch analogy
describes interrupting a tendency to return to an unproductive route. It is not
a claim that researchers recovered or deleted literal graph branches.

These interventions also leave the discovery representation open: a head, a
distributed direction, or selected contributions can offer different handles on
behavior. None is prescribed as the answer for Gemma.

### The unifying interpretation

Training, recognition, construction, and repair concern related aspects of
coordinated computation. Varied training situations can reinforce useful
influences; varied inspection situations can expose how those influences act.
A repair may need only enough partial recognition to change a recurring tendency.

Marcos considers the fit across these cases a strong reason to investigate this
direction. The fit does not by itself demonstrate a shared extraction method,
establish a unique geometry, or guarantee that every capability will yield a
recognizable trace with our instruments.

## 6. Implications for the next experiment

The immediate direction is observation of a strongly represented capability,
without training. No new experiment or weight edit is authorized merely by this
document. The already selected local engine for this research line is the
llama.cpp fork; investigating QGRE correctness is a separate task.

Useful instrumentation would preserve actual computational dependencies,
intermediate transformations, prompt and generation positions, relevant cached
state, and enough context to compare how executions unfold. A few isolated
snapshots or only the final output would limit what can be inferred. Missing
coverage is reported rather than replaced by an invented connecting path.

The form of comparison remains open. It should respect the connected
computation and the capability exercised, preserve potentially different families
of traces, and avoid defining success as a preferred visual shape or a statistic
that does not answer the research question.

For the quantized Gemma model, measurements must respect each actual tensor's
encoding, packing, scales, and any transformations. A later edit must be realized
in the deployed representation and assessed after that realization. An arbitrary
floating-point proposal is not the actual four-bit edit. This numerical care
supports faithful observation; it does not make a stored value a semantic label.

Existing lookup studies, parameter rankings, and perturbation captures remain
historical measurements. They must not be relabeled as discovery of this trace
because the vocabulary has changed. Similar values at disconnected locations,
overlapping active components, or a changed answer alone do not establish the
connected recurring influence being sought.

## 7. Misunderstandings to avoid

| Misreading | Meaning to retain |
|---|---|
| Find values that contain a capability, then connect similar values. | Observe connected transformations during relevant executions; seek recurring portions of their influence. |
| Recover a full signature uniquely belonging to one capability first. | Partial, shared, context-dependent residue can already be useful. |
| The path is defined to exclude the inference trace. | The present target is recognizable residual pattern in how information unfolds; do not restore the earlier frozen-pattern-only requirement. |
| All examples must use identical weights, states, or directions. | What recurs may be how changes develop together through varying circumstances. |
| Any common activation or architectural route is the pattern. | Common machinery is a starting observation; it does not explain how the capability contributes. |
| More arbitrary prompts will reveal it automatically. | Shape requests so the relevant capability contributes strongly and repeatedly. |
| An edit must reconstruct or improve the whole capability. | It may be useful to alter a tendency toward an unproductive branch while much else continues to work. |
| The analogies specify literal physics or a training algorithm. | They communicate distributed influence, continuity, and partial observability. |
| The same reward means equal numerical gradients. | Marcos explicitly rejects that as a description of his argument. |
| A polished visualization or a changed answer completes the research. | The intended result is meaningful recognition of the unfolding influence; representation and intervention serve that purpose. |

## 8. Provenance and supersession

This synthesis preserves Marcos's explanations and the understanding he confirmed
in the conversation on November 6, 2026. Sources linked above were inspected in
that discussion. No new training, inference, model edit, or reproduction was
performed to write this document.

The older [REFERENCE.md](../../../../projects/qgre/qgre-main/runs/legs/gemma4-e2b/distributed-pathways-reference-2026-11-06/REFERENCE.md)
and its source material preserve the development of the idea and historical
local results. Its distinctions that make the research target exclusively a
frozen encoded pattern, exclude the trace, or require identification of a full
separable path are superseded by this account. The original empirical results
remain subject to their original scope and limitations.

The working question is:

> Can appropriately varied executions of a strongly represented capability make
> recurring portions of its influence recognizable in the connected trace of
> computation, useful enough to understand or repair part of the behavior even
> while the full mechanism remains hidden?
