# Beyond the Weights: Marcos's Research Framework

## Current clarification: read this first

**The frozen-pattern-only account below is superseded.** Marcos's later forest,
hidden-influence, partial-trace, and capability-focused prompt explanations are
consolidated in the versioned
Residual traces of capability
and its
short handoff.

The current target is recognizable recurring portions of how information is
transformed through connected computation. Partial residue can be useful even
when the complete underlying mechanism cannot be recovered or uniquely isolated.
Carefully chosen prompts can make a capability contribute strongly and repeatedly.
Do not use this older account's exclusion of inference traces, or its demand for
an identifiable encoded path, to redefine that target. Empirical records below
retain their original limitations; they have not become trace discoveries.

## Historical account preserved below

**Frozen distributed patterns, their expression during inference, and the geometry of learned capabilities**

Consolidated November 6, 2026, from Marcos's Version 2 research notes, his subsequent explanations, and surviving local experimental records. Revised after his explicit clarification that a path is the frozen distributed encoded pattern, not the inference trace or the route through the architecture.

This document preserves Marcos's position for future collaborators. It is a research reference, not a new project law or an instruction to agree with every hypothesis. Its purpose is to let a model begin at the actual argument instead of requiring Marcos to reconstruct it through repeated corrections.

The companion MODEL-HANDOFF.md is a short entry point. The original Version 2 notes are preserved separately. Later explanations consolidated here take precedence when describing what Marcos currently means.

Read Marcos's latest clarification if any editorial phrasing below obscures the intended meaning.

## 1. The argument to understand first

**The path is a frozen pattern distributed through parts of many weights' numerical encoding. Inference provides observable expressions of that pattern; the inference trace is not itself the path.**

This is Marcos's intended research object. He describes portions or bits of different weights as combining horizontally into a pattern that constitutes a piece of information, a capability, or an ability to perform an operation. A single weight does not independently contain the meaning of the whole. The informational object is the collective pattern across the encoded numerical substrate.

The same weight can participate in many such overlapping patterns. A path may extend through much of the model, potentially almost everything. There is no requirement that a capability occupy one attention head, a few high-magnitude weights, or a compact isolated circuit.

The frozen weights already support the learned organization. During ordinary inference, the current input makes an expression of that organization observable. Inference does not newly install the learned path, and a record of the resulting activations is an observation through which the path might be recognized.

Four things must remain distinct:

1. **Substrate:** the architecture and persistent numerical parameters.
2. **Encoded path:** the proposed frozen pattern distributed through portions of many weights, overlapping with other patterns in the same substrate.
3. **Expression:** what happens when a particular input is processed by that model.
4. **Observation:** completions, activations, probabilities, or intervention effects that may help distinguish the encoded pattern responsible for a capability.

The ordering and coordination within the pattern matter. Replacing it with the order of layer execution does not preserve that idea. “Layer 29 → attention → feed-forward → output” describes common machinery; it does not identify the encoded pattern that makes a particular capability possible.

Marcos also rejects describing the target primarily as a combination of relationships. That description can suggest individually meaningful weights joined together. His emphasis is on the horizontal, distributed whole: portions of many encoded values jointly form the informational pattern.

His geometry hypothesis concerns the organization of these overlapping paths. His stronger position is that this organization enables learned capabilities to remain stable and usable. The research question is whether that explanation can be made identifiable and predictive. Reducing it to the claim that a plot of activations has a shape would lose the proposition he is pursuing.

### The electricity and brain analogy

Marcos uses neurons and electricity to express the difference between a substrate and the distributed informational pattern it supports. In that analogy, weights are like the neurons, while the portions or bits participating across weights are compared to electrical activity. The crucial difference he emphasizes is that the model's encoded organization is frozen.

His electromagnetic-field comparison locates the informational whole in a distributed pattern rather than in an individual component. It is an analogy used to convey his concept, alongside his intuition about biological information. The document does not treat it as evidence establishing where biological memory resides.

Do not turn this explanation into a claim about literal electricity inside weight files, physically moving weights, binary switches being traversed one at a time, or proven electromagnetic storage in brains. Equally, do not discard the intended concept simply because the analogy is not a literal mechanism.

There is not yet an established mathematical extraction of this proposed pattern. That is what the experiments are intended to investigate. The document preserves the object to be investigated without pretending that a particular activation vector, bit mask, or circuit has already been shown to be it.

## 2. Vocabulary and intended meaning

| Term | Meaning in this framework |
|---|---|
| **Weights / conductors / rails** | The persistent numerical substrate. An isolated value does not independently identify the knowledge or capability expressed by a distributed whole. |
| **Parts / bits of weights** | Marcos's description of portions of the numerical encoding jointly participating across many weights. This does not assign one meaning to a particular bit or claim that a literal binary-bit decomposition has already been discovered. |
| **Horizontal / distributed** | The pattern spans portions of many weights, potentially much of the model, and can overlap with other patterns. This has the same meaning in all five examples. It is not a single layer or a memory layout. |
| **Path** | The proposed frozen, distributed encoded pattern constituting information or a capability. It is not a list of layers, an activation trace, or the downstream footprint of an edit. |
| **Journey / execution** | A particular expression of the model's learned organization during inference. Earlier writing sometimes conflated this with “path”; this reference now distinguishes them. |
| **Gravitational pull / strength** | Relative influence of learned patterns under a context, and their reinforcement across training examples. It is not a semantic quantity deposited into individual words. |
| **Geometry** | The proposed organization of the overlapping distributed paths, supporting their stability, distinguishability, and use. No specific shape, dimensionality, or metric has been established. |
| **Ghost** | The input-dependent expression of the frozen organization. It changes during inference while the stored parameters remain fixed. |
| **Observation** | A context and its resulting behavior/internal activity that may expose an aspect of the encoded pattern. The question is processed during prefill before answer generation. |
| **Cylinder / rigid connection** | An earlier analogy for coupled effects within shared organization. It does not require an actual rod, a sparse chain, or a guaranteed uniform propagation law. |
| **Dotted line / missing jigsaw piece** | A missing or defective functional relationship that prevents a task from being completed correctly. In the physics example, it refers to the control behavior needed to complete the task. |

There is no assumed one-to-one correspondence among a vocabulary token, a scalar weight, and a node in the visualization. The physical picture helps express the concept; finding a faithful measurable correspondence is part of the work.

The earlier quantum-collapse comparison expresses **conditioning by context**. It is not a claim that transformers perform quantum measurement, or that the first word of an unfinished prompt determines all future answers. The later notes explicitly describe successive conditional token predictions.

## 3. The five examples and the common argument

### 3.1 CPT: installing a repeated relationship through varied contexts

Marcos has successfully used continued pretraining to install knowledge into Gemma 4 E2B with four-bit training. This is an experimental starting point, not merely a proposed future test.

His illustrative relationship is Martin Garrix performing at Kinetic Field. Phrasings might include:

> Martin Garrix will play at Kinetic Field on Friday.
>
> Friday's Kinetic Field lineup includes Martin Garrix.
>
> The article describes the festival and mentions Martin Garrix's Friday performance at Kinetic Field.

These are examples of corpus construction, not a statement about a currently scheduled real-world performance.

Across hundreds or thousands of phrases and articles, the surrounding content varies while the relationship recurs. Marcos's explanation is that repeated exposure repeatedly reinforces that relationship, making it accessible from varied contexts. He describes the accumulated result as greater relative pull toward the relationship and more ways to reach it.

Here “relationship” describes the fact being taught, such as the artist-stage association. His proposed internal representation of that fact is the distributed encoded path. It is not an artist weight joined to a stage weight, or a separately meaningful value stored in one component.

The point is about the cumulative effect of repeated relational training. An answer about one token receiving a particular numerical gradient on one step does not address it. Nor is he proposing that gradients literally add a semantic quantity to a word.

The empirically successful method and the proposed account of its internal organization are connected but distinguishable. Existing success need not be re-proved before investigating that organization. A precise mathematical claim about accumulated gradient directions or magnitudes would need its own definitions; it should not be substituted for his relational argument and then presented as a refutation of it.

### 3.2 RL: reinforcing procedures across varied successful completions

Marcos sees the same organizing principle in reinforcement learning. Completions differ, but successful completions may repeatedly use a useful procedure, intermediate relationship, or way of reaching an answer.

His proposal is that training reinforces distributed encoded patterns supporting useful procedures across those different trajectories. More complex tasks combine and reuse learned procedures. In a sufficiently capable model, he conjectures that some recurring patterns become central to navigating and composing its capabilities. Different successful traces provide expressions of the learned organization; the traces are not definitions of the encoded paths themselves.

**He is not claiming that a shared reward gives every token or parameter an identical gradient.** His reference to procedures being reinforced together concerns their learned functional organization. A correction about gradients not being numerically equal answers a claim he has expressly rejected.

The connection he wants investigated is how repeated successful computation becomes reusable structure, including how several procedures become coordinated in more complex reasoning.

### 3.3 Small neural networks: less overlap makes the organization easier to see

In the weight-space research that informed him, an AI inspected small specialized networks and identified their tasks or games. Marcos interprets this as a case where the available behavior was sufficiently constrained for the network's functional organization to be recognizable.

There are relatively few numerical values and relatively few capabilities or pieces of information in this case. Marcos proposes that their collective encoded organization is easier to recognize because less overlapping information obscures it. He also suggests that an interpreting model may recognize a structure related to patterns it has learned itself; that is his proposed explanation, not direct evidence about the interpreter's internal experience.

“One shape” expresses that coherent task organization. It does not establish a unique mathematical embedding, and it does not require every small network to have only one possible internal organization.

In a larger model, the relevant path may run through almost everything while many other paths overlap it. The problem is distinguishing the collective pattern within that overlap, not simply finding a larger list of influential parameters.

### 3.4 Physics/control improvement: completing a functional relationship

Marcos interprets the physics/control example from that research as a constrained repair problem. Given the network, the environment, the intended behavior, and the failure, it became possible to identify a missing or poorly configured part of the task's computation and change it.

The “dotted line” is the behavior required to finish the job successfully. The “jigsaw piece” is the functional relationship that makes the surrounding behavior work together.

Do not recast this as a claim about a literal hole in geometric space. That was an assistant-introduced interpretation, not Marcos's argument.

### 3.5 Qwen: multiple failures expose a recurring mechanism in a larger model

The Qwen reasoning-loop example supplies the larger-model counterpart. The completions vary, yet an unproductive looping behavior recurs. Marcos's interpretation is that these different observations help isolate a shared computational influence behind that behavior.

Different questions and completions provide different observations of the model. Marcos proposes that a common error across them can help distinguish the frozen distributed pattern supporting that failure from the other overlapping patterns. The varied completions are observations used for triangulation; they are not themselves the path being sought.

The functional target is a tendency to enter an unproductive process. It is not a particular answer token to suppress or replace.

The small-network, physics, and Qwen descriptions above preserve Marcos's reading of the cited weight-space research. They are not claims that this local session performed those published experiments. Exact claims about the researchers' access, sequence of attempts, and edit construction require the primary paper and experiment records, rather than an invented reconstruction.

### The connection across all five

Marcos proposes the same distributed organization throughout:

| Setting | What recurs across changing examples | Why that matters to the theory |
|---|---|---|
| CPT | A factual relationship | Training repeatedly reinforces access to the relationship. |
| RL | A useful procedure or relationship contributing to success | Training can reinforce and combine reusable computations. |
| Small specialized network | A constrained task organization | Limited overlap can make the organization easier to recognize. |
| Physics/control repair | The behavior needed for a successful task trajectory | A constrained failure can expose a specific functional repair. |
| Qwen looping | A failure process across different completions | Varied observations can help isolate a recurring mechanism. |

**Training and diagnosis use variation in complementary directions.** In Marcos's account, varied training examples reinforce a distributed encoded pattern. Varied inference observations may help expose that pattern within the overlap. The proposed common object is the frozen path; what changes between examples is its observable expression.

This complementary framing summarizes the discussion. It does not assume that identifying a pattern is equivalent to recovering the exact sequence of historical training updates.

## 4. What survives on disk

These records were read on November 6, 2026. No training run was repeated to prepare this reference.

### Saved Gemma 4 E2B CPT evaluations

| Saved run | Base | After CPT | Recorded setup |
|---|---:|---:|---|
| Pokémon lore, Kanto v1 | 89/300, 29.7% | 206/300, 68.7% | Gemma 4 E2B; FP4; forced-choice evaluation labeled `augmentation_holdout`; result `PASS`. |
| Pokémon lore, Kanto anneal | 21/77, 27.3% | 54/77, 70.1% | Gemma 4 E2B; FP4; forced-choice evaluation labeled `augmentation_holdout`; result `PASS`. |

Sources: Kanto v1 evaluation, Kanto anneal evaluation.

These are recorded behavioral improvements under the stated probes. They are not being relabeled as unrestricted free-generation performance or as direct measurements of a particular internal geometry.

### Festival installation reference

The CPT installation working reference records:

- A Gemma 4 E2B-class NVFP4/FP4 installation baseline, with loss falling from 6.84 to 0.26 over 200 steps. Loss is a training measurement; the behavioral findings below supply separate evidence about retrieval.
- A finding that varied surface forms of the same facts installed better than a small set of repeated deterministic templates.
- V4 versus V5 results after downsampling facts to an equal exposure floor: question evaluation 17/19 to 16/19; cloze stage retrieval 71.1% to 50.6%.
- An explicit conclusion that uniform exposure was the wrong objective for that comparison; confused relationships needed additional distinguishing examples.

These festival results are recovered from a working report. They have a different provenance from the raw saved evaluation JSON files above. The report itself is evidence of what the team recorded, not a substitute for every original checkpoint or measurement artifact.

The EDC corpus import also records 3,208,704 packed tokens from the Kandi producer. That total includes replay material and is not a count of new festival facts or, by itself, proof of learning.

## 5. The current local editing experiment

The goal is to learn whether observation-guided editing can change a reusable behavior and reveal its organization. Fixing one particular wrong answer is not the purpose, and repairing QGRE is not a prerequisite to pursuing this experiment.

Marcos chose **the local llama.cpp fork for this line of work**.

An earlier position-versus-value failure occurred through QGRE. On the same twenty question strings, the llama.cpp comparison answered all twenty correctly. Consequently, that failure is not established as an intrinsic behavior of the Gemma model used through llama.cpp. The comparison did not isolate whether QGRE's conversion, numerical path, or another implementation difference caused the discrepancy. Comparison record.

The subsequent experiment instead targeted a real, observable preference: when a key appears twice with different values and no tie-breaking instruction, increase the model's default preference for the **first occurrence** while preserving explicit instructions and unique-key lookup.

The saved quantized-edit report records:

- An activity-guided edit to layer 29, attention head 5's query weights, stored in Q4_K. It changed 42,377 represented scalar weights while retaining existing quantization scales and minima.
- First-only choices on discovery prompts increased from 5/16 to 10/16; on withheld keys and values, from 7/16 to 10/16.
- Reversing the proposed direction favored the last occurrence: discovery last-only choices rose from 6/16 to 12/16.
- All 84 tested control answers remained unchanged under the selected forward edit, including four existing explicit-instruction errors. Unchanged is not the same as universally correct.
- Withheld examples changed keys and values but retained the templates. Wider generalization remains unmeasured.
- Quantization materially altered the proposed edit. The stored change was not an exact rank-one update.
- Instrumented captures disagreed with ordinary-server behavior on some cases. Behavioral scores came from the ordinary server; captures were diagnostic evidence, not assumed to be behavior-neutral.

This is a concrete starting result for mechanism-oriented editing. It is not a conclusion that the full proposed geometry has been found, that the chosen edit is the only possible one, or that unchanged test answers establish universal stability.

## 6. The experiment's purpose and method

**Current phase: investigate whether the encoded paths can be recognized from capability patterns and existing observations. No training is needed to state or pursue that question, and further edits are not a substitute for identifying the target.**

The earlier offline inspection followed the effects of an existing edit through recorded operations. Its claim to have found a segment of Marcos's path was incorrect and has been withdrawn. Its measurements describe perturbation propagation. They do not identify the frozen encoded pattern constituting the capability. Collecting more stages of the same propagation trace would not, by itself, fix this conceptual error.

The subsequent original-model observation study completed 40 discovery and 20 withheld questions without editing or training. All 60 answers were correct and their exact token sequences agreed across ordinary, plain-observer, and captured execution. Attention diagnostics and products formed from native-quantized query weights and captured inputs showed repeatable question-dependent contrasts on the withheld examples. Copying overlapped with retrieval, the contrast also existed in input activity alone, and instrumentation changed numerical logits despite preserving answers. The study therefore records useful observations but does not identify a frozen encoded path; neither its heads nor its derived product profiles are such an identification.

The research approach must preserve the following distinction between evidence and target:

1. Choose a capability or recurring behavior that provides a recognizable pattern across examples.
2. Use actual completions, architecture, internal activity, and correctly interpreted native weights to orient the investigation.
3. Look for a coherent distributed pattern in the encoded substrate that could account for those multiple expressions. Do not assume it is compact or sparse.
4. Explain why a candidate captures the capability across examples, rather than merely naming common machinery, frequent activations, or consequences of a chosen edit.
5. Develop predictions from that candidate. The discovery method remains open; no particular vector, subspace, bit mask, or editing recipe is assumed to be the answer.
6. If a later experiment tests an intervention, evaluate effects on the behavior family, new examples, and other capabilities. An output change alone does not establish that the proposed path was identified.

A useful discovery would distinguish a particular encoded pattern from other overlapping patterns using its observable expressions. A one-off substitution from `2` to `blue`, a successful edit, or a list of affected layers would not by itself demonstrate that discovery.

Changing a distributed encoded pattern would still numerically change participating weights. Marcos's distinction concerns the informational object of the change, not a denial of the parameters' role. Its implementation must honor the model's actual numerical encoding.

A model described as four-bit may contain mixed tensor formats. Identify each edited tensor's real format, packing, scale, and represented values. Measure the realized stored edit rather than treating a floating-point proposal as if it had been applied unchanged.

## 7. What the theory predicts, and what is still open

Marcos's proposed unification is that learning establishes overlapping distributed patterns in the numerical substrate. These frozen paths support information and capabilities, and inference provides expressions through which they may be recognized. The hypothesis concerns the encoded organization, not merely statistical similarity between execution traces.

The open questions are substantive:

- What mathematical description identifies a proposed encoded path without substituting its activation trace for the object itself?
- Can multiple observations distinguish one path within the overlap, including when it spans much of the model?
- Does a candidate account for a capability across examples beyond shared wording?
- Can its organization predict the direction and extent of an intervention's effects before they are measured?
- Can paths overlap in the same parameters while remaining distinguishable and selectively editable?
- Does the organization of those paths explain stability in the stronger sense Marcos proposes?
- How do quantization, model size, and overlap among capabilities constrain that organization?

His longer-term extension is a reusable map or library of computational mechanisms, potentially allowing patterns to be selected or composed separately from a shared execution substrate. This is an architectural research direction, not a claim that such a system has already been demonstrated here.

The geometry is to be discovered by studying the encoded substrate together with its observable expressions. It should not be forced to resemble the ghost prototype, a sphere, a cylinder, or any preferred rendering. Its exact numerical or bit-level characterization has not yet been established locally.

## 8. Reading notes for a future collaborator

These distinctions prevent repeating the misunderstandings that prompted this document:

- Keep the encoded path, its expression during inference, and the recording of that expression distinct. **An inference trace is evidence through which the path might be recognized; it is not itself the path.**
- Do not relabel an architectural route, a list of influential weights, or a perturbation's downstream footprint as a discovered path or path segment.
- Preserve the possibility of a broadly distributed pattern spanning almost everything and overlapping many other patterns. Smallness and sparsity are not assumptions of the theory.
- Treat the electricity/brain example as an analogy expressing the distinction between substrate and distributed information. Do not replace the intended concept with a literal claim and then critique that replacement.
- Begin from the successful CPT observations and the saved records. Do not require Marcos to rediscover that his installation method can work before discussing its explanation.
- Address accumulated reinforcement of a relationship across varied examples. Do not replace it with a claim about identical gradients, isolated token updates, or semantic values being stored in words.
- Keep “horizontal” consistent across all five examples.
- Read the dotted-line and jigsaw examples as functional repair.
- Read gravity, cylinders, roads, and observation in light of the latest distinction between frozen encoded organization and its input-dependent expression.
- Treat small-network recognition, physics repair, and Qwen editing as parts of the same proposed explanation. Do not present them as unrelated anecdotes and lose the comparison.
- Use real completions and native weight information to orient interpretation. Neither arbitrary weight magnitudes alone nor a familiar editing recipe substitutes for discovering the mechanism.
- Keep the target at the level of a reusable behavior. Preserve the choice of llama.cpp for the local experiment.
- If disagreeing, identify the actual proposition at issue and the evidence bearing on it. Acknowledging the framework does not require endorsing every conjecture; criticizing a different claim does not advance it.

The central research question is: **Can observations of a recurring capability help us recognize the frozen pattern distributed through portions of many weights that constitutes its path, distinguish it from overlapping patterns, and eventually make predictions about changing it?**

## Source provenance

- **Authorial source:** Marcos's pasted *Beyond the Weights*, Version 2.0, and his subsequent messages explaining CPT, RL, horizontal organization, the small-network example, the physics jigsaw, Qwen triangulation, and especially the frozen path through parts of the weights' encoding. His latest clarification governs this reference. The original pasted text is preserved byte-for-byte in the linked `sources` document as historical context, not silently revised to match the later explanation.
- **Local empirical sources:** the Kandi evaluation records and installation reference, QGRE corpus-import record, llama.cpp comparison report, and quantized-edit report linked above. This reference summarizes those artifacts; it does not claim new runs or independent reproduction.
- **Published-experiment interpretations:** preserved from Marcos's account and Version 2.0. No new literature review was performed to prepare this document.
- **Editorial synthesis:** the organization, glossary, and training-versus-diagnosis comparison were written to preserve the discussion. They are not presented as verbatim quotations from Marcos.

## Correction record

The previous edition incorrectly defined the path as an ordered computational traversal and called the offline perturbation analysis a discovered path segment. Marcos rejected that interpretation. This edition replaces it throughout with the distinction between the frozen distributed encoded pattern and its observable expression during inference. The empirical measurements are retained; their earlier path-discovery interpretation is withdrawn.
