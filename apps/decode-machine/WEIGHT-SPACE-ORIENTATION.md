# Gemma weight-space orientation: first mechanism investigation

**Historical experiment.** The target and QGRE execution arrangement below record
this earlier investigation. The current research meaning and the subsequent
choice of llama.cpp are in [the residual-trace handoff](../../hypothesis/RESIDUAL-TRACE-HANDOFF.md)
and [full reference](../../hypothesis/RESIDUAL-TRACE-THEORY.md). Static similarities and derived
products below are not claims to have found the connected trace now sought.

## Goal and method

Change the original quantized model's failing table lookup into the correct answer, while preserving its correct answers. Understand the relevant computation before selecting a weight edit. The immediate target remains discovery-01: the table says `Lom = blue`, but the model answers `2`.

The research comparison is the [original Qwen task given to Astra](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/runs/p50c/turns/t001/prompt.txt), with the [paper's account](https://rikiyat.github.io/blog/weight-space-programming/) and [experiment notes](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/runs/p50c/explanations.md). It supplied architecture, actual parameters, practice examples, execution tools, and internal observations. We reproduce that evidence-oriented working arrangement in the native QGRE path. The Qwen edit formula is not a recipe for Gemma.

This round completed original-model observation and an adaptive behavioral contrast. It did **not** establish a causal circuit or produce a successful weight edit.

## Immutable inputs and outputs

Run directory: `runs/legs/gemma4-e2b/mechanism-study-2026-11-06/`.

- Original checkpoint: `models/gemma4/gemma4-e2b/tq4-e2m1-nvme`.
- Original root: `a4e166e6798bfc833a99ee08460b58f0c305695ca077bc7c810f89646a7caab2`.
- Saved inference executable: `runs/legs/gemma4-e2b/decode-observatory-30k-2026-11-05/bin/qgre-infer`.
- Executable SHA256: `7bf8f6a1151b41887c6a9da6ec45a6db16b7a0ea91d9e27c8105c4be12dd1524`.
- `capture-join.json`: identity and output comparison for all 12 original diagnostic reruns.
- `captures/`: 12 original prompts, final prompt position, all 35 named KV views each.
- `analysis/`: attention relations, actual weight/readout relations, and final-MLP column/activity analysis.
- `COLOR-PROBE.md`: prediction recorded before the adaptive run.
- `color-cases.json`, `color-results.json`, `color-captures/`, `color-analysis/`: six paired wording contrasts and their observations.
- `orientation.json`: relational index of the architecture, examples, evidence and conclusions.

Every inference invocation used the saved original engine, GPU 1, greedy decoding, an eight-token cap, and its own prefill. All 12 original reruns matched the earlier stdout, prompt IDs, tap code bytes and tap scale bytes exactly. All six wording variants were checked against their native prompt token counts before treating their selected tap as the answer boundary. No unused confirmation questions were consumed.

## Architecture and numerical orientation

The actual E2B configuration has 35 layers, width 1,536, eight query heads and one KV head. Sliding layers have head width 256; every fifth layer is full attention with head width 512. Layers 0–14 produce their own KV. Later sliding layers read layer 13's KV; later full-attention layers read layer 14's KV. These relationships were joined from the recorded launch offsets, not inferred from screen position.

MLP width is 6,144 in layers 0–14 and 12,288 in layers 15–34. Gemma uses a GELU-tanh gated MLP, post-attention and post-MLP normalization, and a 256-wide per-layer input branch reinjected after each MLP. These branches matter when explaining changes between a post-attention tap and a layer-output tap.

The weight file contains 317 text projection matrices, including the output head and global per-layer projection. Forty stored K/V projections belong to shared-KV layers and are not independent KV producers in this decode graph. A stored tensor inventory must not be confused with active computation.

For inspected projections, the metadata declares the plain E2M1 domain, block size 32, high-nibble-first packing and zero WHT seed. Weight analysis uses the canonical production UE8M0-to-UE4M3 scale bake, including its underflow/saturation behavior. Norm gains and capture values use their actual E4M3/scale pairs. The source configuration's BF16 label is not the converted checkpoint's numerical format.

## What became observable

1. **Queries to remembered tokens.** `inspect-attention` relates each recorded post-RoPE query to every named key row, preserves source-layer sharing, and labels positions with the real tokenizer vocabulary. It also compares recorded attended vectors with value rows. Q/K ranks are algebraic affinities, not recorded softmax probabilities; value cosine is not a contribution fraction.
2. **Internal vectors to actual output weights.** `inspect-readout` reads selected real `lm_head` rows and compares them with recorded states. It exposes both the raw relation and a static direction incorporating the stored final-norm gains. Intermediate alignments are uncalibrated diagnostics, not native logits or counterfactual predictions. No RMS normalization or model forward is reconstructed.
3. **Final MLP connections and activity.** `inspect-last-mlp` reads every column of the last layer's real down projection, its post-MLP gains, final gains and the selected output-weight contrast. It joins these connections to the final layer's recorded activation row for every case.

The earlier KV dump was an 8 MiB prefix, not a complete cache. The explicit named-layer capture closes that gap for these short prompts. The ordinary GeGLU tap contains only 6,144 channels. For the final layer, the saved `Sp2X` and `GegluXScale` buffers preserve all 12,288 channels; their first 6,144 decoded values agree exactly with the corresponding tap in every inspected case. This recovers the final layer's missing half without modifying the engine. It does not recover the omitted halves of layers 15–33.

The complete final activation row is the E4M3 **prepacking companion**. Production down projection consumes its packed FP4 twin. The local column/activity diagnostic does not account for that repacking, normalization denominators, rounding, residual low terms or the PLE branch. It is useful for forming a hypothesis, not declaring an individual weight causal.

## Observations and the first test

Original discovery set: **9/12 correct**. Every red and green answer was correct. Three of four blue answers failed. This is a property of this small development set, not a general claim about the model and blue.

In discovery-01, layer 29's query heads 0, 1, 2, 4 and 5 rank the table's `blue` token first among all 46 prompt positions. Head 3 ranks green first. The model still emits `2`. This challenges the simple explanation that no relevant entry was matched; it does not prove that the model represented the correct answer faithfully throughout the network.

The adaptive test replaced only the two instances of `value` with `color`, keeping each table and requested key fixed. Prediction: repair at least two of the three failures and preserve all three successful controls.

| Pair | Expected | Original wording | Color wording |
| --- | --- | --- | --- |
| 01 | blue | 2 | green |
| 03 | blue | 3 | blue |
| 10 | blue | explanatory text, cap reached | blue |
| 00 | red | red | red |
| 02 | green | green | green |
| 08 | blue | blue | green |

Result: **two rescues and one regression; 3/6 → 4/6 correct. The declared selective-repair prediction failed.** Wording affects the behavior, but replacing it is neither a selective repair nor a weight edit. The prompt change may affect upstream retrieval as well as downstream processing. The useful result is a set of matched examples showing both directions of behavioral change.

## What the weight inspection rules out

The largest negative local blue-versus-2 product in discovery-01 belongs to final-MLP channel 4302. Its actual static weight coupling is approximately -5.008 in the stated diagnostic coordinates. Its recorded activation is 1.5 on the failure, but 3.5 on discovery-05, a correct red answer, and 1.125 on discovery-08, the correct blue answer. This is not a failure-specific channel.

Channel 10884's coupling is positive, approximately +5.379, but its activation is negative: -1 on the failing discovery-01 and -2.75 on the successful blue case. Merely selecting the largest opposing products would again confuse shared computation with a fault. Some smaller channels distinguish this small set more sharply, but that alone does not establish their role or justify amplification.

The evidence currently supports investigating the competing blue/green readout and later processing with the paired examples. It does not support calling layer 9, an entire head, or one of the largest final channels “the bad weights.” No new checkpoint was published in this round.

## Remaining limits and next experimental boundary

Full native teacher-forced probabilities and all-position internal traces are not part of this dataset. The existing E2B scoring-path gap is not disguised by treating host dot products as probabilities. Earlier late-layer GeGLU tails and separate per-layer PLE effects also remain unresolved. The individual-edit publisher currently supports attention-output edits; it should be extended to another matrix only after a mechanism actually selects that matrix.

A subsequent weight experiment needs a specific input condition, the connections proposed to detect or change it, a predicted effect on the original prompt and on the paired controls, and the actual representable FP4 changes. Evaluation stays on the original wording; changing a prompt is diagnostic evidence, not success at the original editing goal.

Validation for the reader changes: `cargo test -p qgre --test qgre_suite ghost_trial:: -- --nocapture` exited 0: eight passed, one native-GPU quantizer test intentionally ignored. This is the ghost-trial component, not the complete package suite. The new integration test checks real CLI output for query/key rank, exact dot values, token labels and shared-KV ownership. No decode-path source changed.
