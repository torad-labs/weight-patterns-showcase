# Original Gemma lookup observations — completed

We obtained repeatable, question-dependent numerical observations on the unchanged model. The observations generalized to the two vocabulary families held aside. We did **not** identify the frozen pattern distributed through portions of weights that Marcos means by a path, and this experiment does not yet justify a weight edit.

The most informative limitation is empirical: the task contrast is also visible in the incoming activity before the inspected projection, and literal copying overlaps strongly with value retrieval. The weight/input product descriptor therefore has not established a distinctive encoded signature of the lookup capability. This limits the measurement, not the hypothesis.

## What ran

Original Gemma 4 E2B, the Q4_K_M GGUF, through the local llama.cpp fork. No training, parameter update, edited checkpoint, or intervention. Greedy generation, temperature zero, no repetition/frequency/presence penalties, no reused prompt cache. Each question was run through the ordinary server, the native observer without capture, and the same observer with capture.

Six families of three key/value assignments: colors, sports, animals, instruments, cities, and materials. The first four families supplied 40 discovery questions. Cities and materials supplied 20 confirmation questions. Each family varied table/prose wording, row order, and value/position instructions, plus one alternate-key question and one literal-copy question. The mappings were supplied in the prompt; this exercises an existing capability rather than testing facts installed in the checkpoint.

Before confirmation ran, the three inspected heads and the descriptive predictions were fixed in selection.json (`~/Documents/dev/projects/qgre/qgre-main/runs/legs/gemma4-e2b/lookup-pattern-observation-2026-11-06/selection.json`) and PREDICTIONS.md (`~/Documents/dev/projects/qgre/qgre-main/runs/legs/gemma4-e2b/lookup-pattern-observation-2026-11-06/PREDICTIONS.md`). Confirmation did not change that selection or the comparison rules.

| Condition | Discovery correct / total | Confirmation correct / total | Three-way exact answer-token agreement, total |
|---|---:|---:|---:|
| Assigned value | 16 / 16 | 8 / 8 | 24 / 24 |
| Position | 16 / 16 | 8 / 8 | 24 / 24 |
| Different queried key | 4 / 4 | 2 / 2 | 6 / 6 |
| Literal copy | 4 / 4 | 2 / 2 | 6 / 6 |
| **Total** | **40 / 40** | **20 / 20** | **60 / 60** |

No cases were excluded by the predefined answer-agreement criterion. These are 60 unique prompts, each evaluated in three modes; they are not 180 independent task examples. The previously discussed erroneous position-for-value behavior was not reproduced here.

For example, `Zib = Oslo; Huq = Tokyo; Mor = Lima` produced `Tokyo` for Huq's value and `2` for its position. Moving Mor to the front preserved `Tokyo` and changed the position answer to `3`. Asking about Mor produced `Lima`. This is the desired dependence on the requested operation and binding.

## Observable contrast and held-aside confirmation

The discovery scan enumerated all 35 captured layers and all eight query heads per layer: 280 locations. It selected the three with greatest average diagnostic affinity mass on the requested body value in the discovery value questions. These locations were not inherited from the previous edit. They are inspection locations, not identified paths or a claim that the capability resides in three heads.

The following percentages are **offline Q/K diagnostic affinity mass on the requested value in the assignment body**, averaged over confirmation questions. They are not output-token probabilities, not the amount of knowledge in a head, and not a measurement of the attention kernel after cache rounding.

| Inspection location, zero-based | Ask for value, n=8 | Ask for position, n=8 | Ask about alternate key, n=2 | Literal copy, n=2 |
|---|---:|---:|---:|---:|
| Layer 29, head 5 | 86.94% | 0.80% | 93.34% | 43.61% |
| Layer 27, head 7 | 64.99% | 0.33% | 64.93% | 19.49% |
| Layer 24, head 3 | 55.26% | 2.75% | 52.87% | 8.50% |

At every selected head, all eight confirmation value/position pairs separated in the predicted direction. The requested value also received more than half of the affinity assigned to the three body values in all ten value/alternate questions per head. Mean conditional shares in the eight value questions were 96.81%, 92.59%, and 93.73%, respectively. Conditional shares must be read alongside absolute mass: a high share of a tiny allocation in a position question is not strong overall attention to that value.

The corresponding discovery value/position mean masses were 79.97%/1.59%, 64.30%/1.89%, and 48.07%/2.98%. The effect was not confined to the original words or a single row position. However, literal copy also engages the inspected locations, so this is not a lookup-specific detector.

## What was inspected in the actual weights

The native GGUF type determines decoding, including its scales and packed values. The model's Q4_K_M label does not make every tensor four-bit:

| Query tensor | Actual stored type | Inspected output rows, zero-based | Input width |
|---|---|---|---:|
| `blk.29.attn_q.weight` | Q4_K | 2560–3071 | 1536 |
| `blk.27.attn_q.weight` | Q6_K | 1792–2047 | 1536 |
| `blk.24.attn_q.weight` | Q6_K | 768–1023 | 1536 |

For each captured projection input, the diagnostic multiplies the native-decoded stored weights by that input and sums signed contributions in groups of 32 input coordinates. Those groups are an observer-chosen summary, not an inferred boundary of the proposed encoded paths or a claim about bit-level semantic units. Native decoding is transient in memory; no dequantized weight file or new checkpoint is written. Saved products are derived diagnostic values. This offline arithmetic is not asserted to duplicate the GPU's arithmetic bit for bit.

The table shows confirmation mean cosine similarity between these product profiles. Higher means the numerical profiles point in more similar directions. Counts are pairs, and pairs share prompts; they are not independent statistical samples.

| Comparison | Pair count | L29 H5 | L27 H7 | L24 H3 |
|---|---:|---:|---:|---:|
| Same value task, different vocabulary family | 4 | 0.659 | 0.650 | 0.750 |
| Same assignments, value versus position | 8 | 0.456 | 0.515 | 0.573 |
| Same assignments, value versus literal copy | 2 | 0.824 | 0.783 | 0.641 |
| Same assignments, different queried key | 2 | 0.735 | 0.661 | 0.775 |
| Same assignments, table versus prose | 4 | 0.930 | 0.920 | 0.883 |
| Same assignments, changed row order | 4 | 0.888 | 0.829 | 0.828 |

All fixed descriptive predictions held on confirmation, including the predicted copy overlap and the input-only contrast. The detailed clause results, observed numbers, and denominators are in confirmation-evaluation.json (`~/Documents/dev/projects/qgre/qgre-main/runs/legs/gemma4-e2b/lookup-pattern-observation-2026-11-06/confirmation-evaluation.json`). Its 25 clauses are correlated bookkeeping for five prediction groups, not 25 independent validations.

## Why this is still short of identifying the intended path

The same cross-content-versus-operation contrast exists in the projection inputs before the diagnostic applies any inspected weights:

| Captured input at layer | Same value task, different family | Same assignments, value versus position |
|---|---:|---:|
| 29 | 0.664 | 0.497 |
| 27 | 0.708 | 0.506 |
| 24 | 0.889 | 0.769 |

For fixed weights, the grouped product descriptor is itself a fixed linear transformation of the input activity: `p(x) = A_W x`. Its pairwise dot products are `xᵀ A_Wᵀ A_W y`. Stable similarity in this descriptor can therefore reflect structure already present in those inputs, reweighted by the inspected projection. The results have not demonstrated that this descriptor recovers the distributed encoded organization of a capability, or provides additional specificity over activity alone. They also do not show that using such observations for future recognition is impossible.

The experiment has further boundaries:

- Value/position prompts change both instruction wording and word/number answer type. Common prompt structure and output preparation remain possible explanations for some separation.
- Copy controls cover only one format and layout per family. Two confirmation families are a small generalization check.
- Captures describe prompt processing up to the first answer decision. Most intermediate arrays retain the last token in each prompt chunk; K/V retain all prompt tokens. This is not a record of every internal state across every generated answer token.
- The 280-head scan is an attention-oriented way of choosing where to inspect. The native weight inspection then covers three query-projection slices. It does not inspect the full distributed encoding across attention, feed-forward operations, embeddings, and all other weights, and it must not silently become the definition of the path.
- No causal intervention was performed, no encoded bit subset was identified, and no proposed edit was tested.

## Instrumentation changes numerical scores

Exact output-token agreement held for all 60 questions, but plain and capture first-token logits were not identical:

| Split | Mean per-case RMS logit difference | Range of per-case RMS differences | Largest absolute logit difference |
|---|---:|---:|---:|
| Discovery, n=40 | 0.4102 | 0.0144–1.7005 | 3.8499 |
| Confirmation, n=20 | 0.4546 | 0.0340–1.4319 | 4.0916 |

No nonfinite logits were observed. These measurements preserve the difference between behavioral agreement on this corpus and numerical transparency. Internal results are observations of the captured execution, not proven numerically exact replicas of the uncaptured execution. This run did not diagnose the cause of the instrumentation differences.

## Outcome and practical implication

The useful result is a reproducible corpus and a set of contrasts that survive withheld vocabulary and presentation changes. There is a clear question-dependent observable effect in the unedited model. The stronger goal—recognizing a frozen pattern across portions of encoded weights—remains open. This experiment supports no conclusion for or against the existence of that pattern and supplies no justified surgical edit.

Before treating a future signature as a candidate for that stronger goal, the analysis needs to distinguish it from copying, answer type, shared wording, and the incoming activity itself, while retaining the possibility that the encoding is distributed widely. Enlarging this same head-ranking table alone would not address those gaps. A useful next experimental contrast would hold the requested operation and answer type constant while varying which supplied binding leads to the answer; that is a proposed next step, not a new run performed here.

## Provenance and saved evidence

- Model: `~/Documents/dev/projects/qgre/qgre-main/models/gemma4/gemma4-e2b/gguf-q4_k_m-bartowski/google_gemma-4-E2B-it-Q4_K_M.gguf`.
- Verified source SHA256: `923c4c86177d2ee173a7f5b4fa3d0ac65f5962ab15e6d6a5bc250aec4fd7bf7e`.
- Local llama.cpp checkout: `f089c9cb4bc285a11e510e1c4336b95aca449b08`; engine directory `~/Documents/dev/infra/rig/repo/local/engine-builds/f089c9c-sm120/`.
- Device 0: RTX 5080, UUID `GPU-b6d3869b-4794-4ff4-0398-47df517a8c82`, under `gpu_card_run 0`. No clock or power changes. The owned server exited; its port 17583 is no longer listening. Device 0 had no remaining compute process at cleanup.
- Plan (`~/Documents/dev/projects/qgre/qgre-main/runs/legs/gemma4-e2b/lookup-pattern-observation-2026-11-06/PLAN.md`), all prepared cases (`~/Documents/dev/projects/qgre/qgre-main/runs/legs/gemma4-e2b/lookup-pattern-observation-2026-11-06/cases.json`), and input hashes (`~/Documents/dev/projects/qgre/qgre-main/runs/legs/gemma4-e2b/lookup-pattern-observation-2026-11-06/inputs.sha256`).
- Discovery raw records and derived observations: `~/Documents/dev/projects/qgre/qgre-main/runs/legs/gemma4-e2b/lookup-pattern-observation-2026-11-06/discovery/`.
- Confirmation raw records and derived observations: `~/Documents/dev/projects/qgre/qgre-main/runs/legs/gemma4-e2b/lookup-pattern-observation-2026-11-06/confirmation/`.
- Both contain exact tokenized requests, server responses, plain and captured answers, first-token logits, internal capture metadata, `scan.json`, native product profiles with tensor provenance, and `product-comparisons.json`. Logs are alongside the phase folders. Total saved study size at completion: approximately 1.2 GiB.
- Native diagnostic source: weight-products.cpp (`~/Documents/dev/infra/rig/repo/local/research/lookup-pattern-observation-2026-11-06/weight-products.cpp`). The study folder retains request, scan, comparison, evaluation, and run scripts.
- Both `run.sh` phases, both scans, both native weight-product runs, both profile comparisons, and the fixed-prediction evaluation exited 0. These are experiment completion results, not a claim about the QGRE production test suite or model performance.

No checkpoint was modified, and no changes to the visualization were made in this experiment.
