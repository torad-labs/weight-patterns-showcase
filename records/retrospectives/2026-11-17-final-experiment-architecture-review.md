# Final experiment: architecture and learned-parameter review

**The revised full learned-parameter backpropagation is implementable within the unchanged architecture. The present 70-tensor trainer does not implement it.** The reviewed `FINAL-EXPERIMENT.md` correctly specifies one selected fixed reference, one freshly regenerated incorrect case, all captured positions/all105 merge losses, and shared weight derivatives through the contributing graph. Its revised inventory of **538 eligible learned tensors plus one fixed causal rotary buffer** is correct for that conditional merge-loss graph. This is a specification assessment, not approval to launch or evidence that the broader implementation is complete.

The draft reviewed here has SHA256 `009e5758f08b88b0add82dd9ea1959eaf6159e98bed6564a5178e02f1d4ca17b`. Root's §7 revision incorporates the rotary-buffer classification and538-family count. I read the complete draft, the actual model/reconstruction/gradient code, native Gemma4 architecture and cache-reuse construction, and the native converter. No code, model, run, service or declaration was changed; no native acquisition or GPU computation was performed. A narrowly targeted1,024-byte read of the existing original-model rotary buffer supplied a scalar classification check, not a new model execution.

## The selected experiment remains one pair

I read `STATE.md`'s goal, the November16 direct-gradient/one-pair clarifications, and the November17 selected-pair scope correction. The deliberate first experiment uses the fixed `expanded-004-h1` reference and regenerates `discovery-05-channel-h1` with current weights. Outcome-successful witnesses calibrate original variation; they are not simultaneous training targets or certified error-free computations. This review does not require all other original successful traces to lie inside the selected reference's loss boundary, and does not use their positive loss as a reason to replace the agreed objective or restore all-answer training.

All four original reports were read directly in this review sequence, including their equations; this task rechecked their unchanged identities, reread the directly extracted architecture/derivative sections and complete DOCX text, and visually inspected all three DOCX equation images again. Their recorded hashes are:

| Original source | SHA256 |
| --- | --- |
| `reports/fixing-a-reasoning-mistake-by-editing-the-weights.pdf` | `a5761c87f7760d13cfe38d754fb8d0921460fc85f2a4b00ffd4bceb33d81e080` |
| `reports/how-the-trace-derived-edit-works.pdf` | `10de7506371d4bc40f0c3dd255d450c60a949891513e5050e58870a9bda91410` |
| `reports/how-the-path-edit-works.pdf` | `fced267c17a61c5da09fe9bb2b0fe4cabb618fbbd2a8542c4c64e38be76aacc2` |
| `reports/Iterative_Learning_from_Computational_Traces.docx` | `71cc3dfcfbf386898f706c571ad5539ef2f20629e510abc5710b003d2eb07149` |

The trace guide's input/output turn and operation-specific chain rules support following the actual computation. The path paper's70-projection selection is a historical editable subspace, not a property of the architecture or a theorem that upstream weights cannot be differentiated. Its wording about an embedding discrepancy not being produced by a weight must not be turned into a claim that the model's embedding table is not learned. The iterative document explicitly allows joint derivatives with sampled history/alignment held and refreshed generation. Its multi-reference, keep/behavior and answer-evaluation proposals do not override the current selected-pair clarification. The reports do not require a nonzero derivative in every parameter or scalar entry.

## Exact anatomy and inventory

The bound architecture is35 blocks of residual width1536. Each block contains attention, gated feed-forward, and a per-layer-input branch, each followed by its residual addition:105 measured merges per captured computation position. Eight query heads use one K/V head; windowed heads have width256 and global heads512. Global blocks are4,9,14,19,24,29,34; other blocks use a512-position causal window. FFN width is6144 in blocks0–14 and12288 in blocks15–34. Per-layer input width is256.

The tensor metadata contain **601 stored tensors:318 BF16 and283 F32**. These are tensor counts, not scalar parameter counts. Structural accounting is:

| Scope | Learned tensors | Fixed stored buffers | Total stored tensors in scope |
| --- | ---: | ---: | ---: |
| Physical GGUF inventory | 600 | 1 | 601 |
| Used by native generation, excluding60 unused late K/V/K-norm tensors | 540 | 1 | 541 |
| Evaluated by current NativeGraph, which omits final output normalization | 539 | 1 | 540 |
| With a continuous path into the105 measured turns, also excluding the last block's output scale | **538** | **1** | **539** |
| Declared trainables in the stopped trainer | 70 | 0 | 70 |
| Other eligible learned tensors frozen by that trainer | **468** | 0 | **468** |

The previous539 causal/469 frozen figures counted the fixed rotary buffer among causal stored tensors. They must not be relabeled539 learned parameters or469 frozen learned tensors. The native generator additionally consumes the two output-side learned tensors even though this held-history merge loss supplies no derivative to them.

The538 eligible learned tensors consist of **278 BF16 tensors and260 F32 tensors**:

| Eligible family | Count | Native dtype and route |
| --- | ---: | --- |
| Token embedding table | 1 | BF16; initial residual lookup, tied to the native output head |
| Per-layer embedding table | 1 | BF16; token-row lookup supplies35 separate256-coordinate features |
| Shared per-layer-input projection | 1 | BF16; initial residual to all per-layer features |
| Shared per-layer-input norm | 1 | F32; learned scale of the shared features |
| Query projection / query norm | 35 + 35 | BF16 / F32; every block's attention query |
| Producer key projection / value projection / key norm | 15 + 15 + 15 | BF16 / BF16 / F32; only blocks0–14 produce caches |
| Attention output / input norm / postnorm | 35 + 35 + 35 | BF16 / F32 / F32 |
| FFN gate / up / down | 35 + 35 + 35 | BF16, gated branch |
| FFN input norm / postnorm | 35 + 35 | F32 |
| Per-layer-input gate / output projection / postnorm | 35 + 35 + 35 | BF16 / BF16 / F32 |
| Layer output scales0–33 | 34 | F32 scalars; outputs feed later measured blocks |
| **Total** | **538** | **278 BF16 +260 F32** |

These counts follow the physical metadata and executed architecture rather than the historical70-name training list. Eligibility is structural. A true zero caused by the current input, masking, activation, cancellation or an unused embedding row remains zero.

### Rotary-frequency factors are architectural data

`dependencies/llama.cpp-f089c9c/conversion/gemma.py:734–761`, `Gemma4Model.generate_extra_tensors`, generates `rope_freqs.weight` from configured proportional RoPE: ones for the rotated pairs and `1e30` for the pairs intended to remain unrotated. It is not loaded from a learned source parameter. The stored tensor is256 F32 values; the original model's exact range at offset15,818,848 contains64 values equal to1 and192 equal to the F32 representation of`1e30`, matching that construction. The `.weight` name is not a trainability criterion.

`torch_forward.py:160–168` and native `gemma4.cpp:192–235` consume these fixed factors for global attention. Derivatives still pass through the positional rotation into learned Q/K projections; the factors themselves remain fixed. Training them would change a configured positional rule rather than implement the stated learned-weight scope.

### Cache sharing is an actual dependency

Native `llama-model.cpp:2989–2997` constructs the Gemma4 reuse mapping. It agrees with `torch_forward.py:156–158`: blocks0–14 produce their own caches, and later blocks read block13 for windowed attention or block14 for global attention. The60 stored late K/V/K-norm tensors do not execute in this model. They are not60 missed derivative routes.

A late attention loss can flow through shared K/V values to the producer's K/V projections, normalization and input history. The producer's leaf receives the sum of its own and all consumers' contributions. It must not receive independent copied parameters for each consumer, and the caches must not be detached constants. Current NativeGraph reconstructs producer caches inside the connected forward graph (`native_graph.py:83–91`); retaining that connection while broadening leaves is essential. Its FP16 cache derivative remains a declared surrogate, as the draft states.

### Embedding/output tie and the final boundary

There is no separate `output.weight` in this GGUF inventory. Native `gemma4.cpp:44–50` creates the output projection from the token embedding when an independent output is absent. Both uses therefore share the existing physical learned tensor. Preserve that tie and its storage identity; adding a separate learned output matrix would change the model.

The token table is eligible through the initial embedding path. The per-layer table is eligible through the third branches. Their rows are indexed by ordinary model input IDs; those IDs are not targets, and repeated occurrences must accumulate into the same shared row. Unaccessed rows have zero derivative in the conditional trace calculation. Exact sparse row accumulation is therefore possible without restricting observed positions or selecting words.

The learned `blk.34.layer_output_scale.weight` is applied after the final measured third merge; `output_norm.weight` follows it. Neither has a continuous path into an already measured turn when token choices are held. The current native generator still uses them before its tied head, softcap30 and selection. Excluding their zero conditional derivative is justified by this loss boundary, not a claim that they never matter. There is no invented differentiable output-token loss to make them train. The final output-head use of the tied embedding likewise adds no token-selection derivative; its input-embedding use remains eligible.

## Mapping every local loss to weights

The draft correctly retains both edges of each merge turn. A local `(reference position,current position,block,branch)` contribution differentiates through its outgoing unit vector and incoming unit vector with opposite signs. Its pullback then follows the operations that produced both values. Loss is not pasted into the nearest output matrix and not broadcast as one maximum error.

Within a block, attention losses reach Q/K/V, their learned norms, softmax/value mixing, output projection and postnorm, with masks/rotations retained as fixed computation. FFN losses reach its norm, both gate and up branches, GELU/product, down projection and postnorm. Third-branch losses reach its input gate, shared/token-specific auxiliary feature, output projection and postnorm. All later losses can also reach earlier residual paths and learned scales; shared attention caches connect relevant earlier positions. Prompt positions participate like every other captured computation position.

Learned normalization gains and layer scalars are parameters even though they are not two-dimensional projection matrices. Their derivatives arise through multiplication and normalization rather than forcing the linear outer-product formula onto them. Shared per-layer projection/norm receives contributions from all35 feature bands and all uses. Embedding gathers require scatter-add back to shared rows, including repeated IDs. These are ordinary chain-rule routes within the existing model, not new layers.

The draft correctly requires the same current weight state for all contributions before a joint installation. Chunking, staging and exact sparse table-row handling may implement that derivative without creating separate models per token. Updating an early block while other gradients still use stale activations would change the specified algorithm.

## Concrete missing implementation routes

The draft is candid that these changes are not already complete. The remaining differences are concrete:

1. **Both table gradients are currently absent.** `torch_forward.py:71,120–125` explicitly omits the two embedding tables from `model.w` and `model.leaves`. `rows():147–151` reads them from `model.file` through NumPy/memmap and creates a fresh constant tensor. Merely adding their names to `leaves=` cannot make the private weights installed by `trace_gradient.weight_gradient():48–51` or the JVP appear in that lookup. The successor needs differentiable shared gathers whose evaluated rows come from the current graph's candidate/dual leaves, and a matching native row installation/readback/recovery route. Sparse accessed-row representations can implement the exact conditional derivative, provided repeated IDs share a leaf and existing state is retained.
2. **The installer and accumulator only support BF16 matrices.** `bf16_accumulation.py:13–24` rejects native types other than BF16, while the current native encoder expects BF16 storage. `run.py:120–147` reads/installs declared BF16 spans, and `:403–404` insists on70 matrices. This cannot install the260 newly eligible F32 norm/scale tensors as stated. Preserve each real scalar/vector/matrix shape, byte range and native dtype; do not cast learned F32 norms/scales into BF16 merely to reuse the old installer. Remainder arithmetic and complete checkpoint representation must cover the broadened inventory explicitly.
3. **The broadened graph needs explicit leaf coverage.** The current derivative only targets `tuple(graph.leaves)` and converts an unused `None` result to a zero direction. That is valid for genuinely disconnected parameters, but it can conceal an accidental missing route if the inventory is incomplete. Source/qualification must distinguish not declared, not executed, detached, connected-zero, native-rounded-away and actually changed. The draft's inventory promise is appropriate; the70-leaf fixture does not verify the new538-tensor routes.
4. **The shared per-layer-input projection is outside the projection adapter's interception.** The adapter wraps keys, attention and third-branch products (`adapter.py:148–158`); `per_layer_input` delegates to the underlying model's ordinary F32 calculation. That shared projection was previously frozen and its auxiliary output constant with respect to the70 selected leaves. It becomes a direct parameter route in the enlarged graph. Its actual native operator/backend/precision needs explicit treatment and qualification, including the native input-layer projection described in `gemma4.cpp:451–465`. Do not infer native derivative fidelity from the existing attention/third adapter qualification.
5. **Starting state must extend without changing the old70 state.** Preserve the stopped native values and their incoming70 remainders. Newly eligible tensors were frozen in that run; their initial state and any zero remainder must be stated from that fact rather than silently resetting old state or pretending an old full538 optimizer checkpoint existed. Untouched sparse table rows have current native values and zero prior update state; subsequently visited rows must accumulate normally.

NativeGraph can carry the additional norm/projection/scale derivatives once they are actual leaves. Its frozen anchor calculations must remain frozen baseline values while the current side reads candidate leaves. The embedding gather change is necessary to make this true for the tables. The existing exact-origin check remains useful but cannot prove full leaf coverage or native finite-response fidelity by itself.

The specification already identifies the unresolved eta transfer, precision-estimator limits, original-derived tolerance question, source-binding omission and error-cleanup issue. This architecture review does not resolve those independent numerical/implementation decisions or approve execution past them.

## Draft verdict and a wording correction

The reviewed §3 anatomy, §7 revised family inventory/exclusions, §8 dtype/estimator distinction, and §12 acknowledgement of missing routes are consistent with the actual sources. **Architectural/parameter-scope concept: supported and implementable. Existing implementation readiness: no.** The retained one-pair scope is the correct scope for this review; no broader population requirement is substituted for it.

One source-description correction remains in the reviewed draft: §7 says the current embedding reader loads rows “from the original GGUF.” `Model.rows` actually reads `self.file`, which the stopped runner sets to the **current working native model descriptor**. Those tables happen to remain original-valued because the70-block trainer never changes them. The precise defect is that file reads bypass the candidate/dual leaf mapping during AD; a newly installed native table update would be visible on a later file read. Describe it that way so a missing gradient route is not confused with a reader hardwired to the original file.

The final specification should keep these distinctions visible:538 eligible learned tensors are not538 necessarily nonzero updates; fixed RoPE is not a learned leaf;60 unused stored K/V tensors are not missing computation; output-side exclusions concern the held-history loss boundary; and implemented full learned-parameter backpropagation still uses the declared precision estimator rather than an exact derivative of discrete generation. Subject to that wording correction, I find no additional architecture contradiction in the reviewed draft. Code implementing the missing routes still needs its own independent source/qualification review before a native successor.

## Source identities and verification boundary

| Reviewed implementation / metadata | SHA256 |
| --- | --- |
| `records/development/2026-11-12-path-forward/torch_forward.py` | `ab0abc44eaa38fd28a215b7ec9f1060543eb9d878452c0e875a2d8df86ac241e` |
| `code/signal-trace/pipeline/weight_edit/native_graph.py` | `127de7e16bbdc48e1f3646dcb4f119ea832df80d11d70a799503f0025f838acf` |
| `code/signal-trace/pipeline/weight_edit/trace_gradient.py` | `3fa643bcbb2aa61f1a1f3b83c923f1ff10494dba3d0b2fdb82377285916ad32d` |
| `dependencies/llama.cpp-f089c9c/src/models/gemma4.cpp` | `8e19e703e3c0a7cfa54777a148108eccdbc670c690731aeb3448c2d21e4f7230` |
| `dependencies/llama.cpp-f089c9c/src/llama-model.cpp` | `d682e7f5d3a0136038feac699ba3fc89afd3258e13bde7134f1294e0e0042f34` |
| `dependencies/llama.cpp-f089c9c/conversion/gemma.py` | `872f3fea7496cefdcd39308ad02e2d30c9f507f68a18f8b4f6962995b09350f5` |
| `studies/04-round-2-overlap/runs/sent-overlap/wrong-inputs/tensors.json` | `e3f6750ce74ba78508c7f58cc9cde0e4fc0af1513fd40bdedaf5a36fb277d355` |
| `studies/04-round-2-overlap/runs/functional-layout-original/source-records/architecture.json` | `1905c24e43a943d2c20683f13c5538944281c2a9b939001cbfc5f746cbc5afde` |

Counts were recomputed from the601-entry metadata, the actual native cache-reuse rule, and the location of the measured final merge. I inspected source/control/data routes and the specified original-model buffer range only; no whole-model gradient, new model acquisition or finite native response was executed. This report does not certify all external Torch/CUDA/GGML implementations. The required round-two verifier is run before this review is committed.
