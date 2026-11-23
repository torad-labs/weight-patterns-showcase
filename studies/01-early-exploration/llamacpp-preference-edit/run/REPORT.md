# A quantized edit of repeated-key lookup preference

We changed an existing behavior of Gemma 4 E2B through a real Q4_K weight edit, using the local llama.cpp fork. The selected edit increased first-occurrence choices on both discovery prompts and withheld keys/values. It did not fully impose a first-occurrence rule. All tested control answers remained unchanged, including the original model's existing errors.

## The behavior

The prompt gives a table containing the same key twice with different values, then asks for that key's value. Without an explicit tie-breaking instruction, the original model sometimes chooses the first value, sometimes the last, and sometimes returns both. The desired change is a stronger default preference for the first occurrence while preserving explicit first/last instructions and unique-key lookup.

Eight discovery families and eight withheld families each have two table layouts and three instruction modes: default, explicit first, and explicit last. Twenty unique-key questions are additional discovery controls. There are 116 distinct prompts in `cases.json`. The withheld set changes keys and values but retains the prompt templates; it does not establish generalization to arbitrary wording or unrelated tasks.

## What changed

Only `blk.29.attn_q.weight`, head 5, output rows 2560–3071 was eligible for editing. Layer/head indices are zero-based. The tensor is stored as Q4_K, with input width 1536 and total output width 4096. A Q4_K_M file contains mixed tensor formats; we checked the actual tensor type instead of treating the whole file as uniformly four-bit.

The native weight matrix and captured activity supplied the edit:

1. Across the 48 discovery instruction variants, layer 29 head 5 showed the strongest first-versus-last matching contrast among the measured heads.
2. The proposed query direction is `d = W_head × (mean x_first − mean x_last)`, where `x` is the final prompt position's input to the query projection.
3. The input gate `v` is the mean default input projected away from a span of 144 observed control/earlier-chunk inputs (rank 116), then normalized to have mean default response one.
4. The proposed edit is `delta W = alpha × d × vᵀ`. This is a coordinated query-projection edit. No output vocabulary bias or value-word embedding was edited.
5. Each proposed scalar was mapped to the nearest value representable by its existing Q4_K block. All original block scales and minima were retained. Native llama.cpp dequantization defined the represented values. Zero strength produced an exact identity.

The selected strength is alpha 2. It changes 42,377 represented scalar weights, stored in 38,914 changed bytes. The edited head contains 786,432 weights. The realized change has norm 2.2420 versus the head's original norm 30.3930, and cosine 0.7948 with the proposed change. Quantization therefore changes the proposed direction materially; it is not an exact rank-one update after storage.

The original GGUF's SHA-256 was checked against its recorded original hash. A whole-file comparison verified zero changed bytes outside the selected head and zero changed block-header bytes. Both candidate models are separate files; the original was not overwritten.

## Ordinary-server evaluation

All behavioral scores use the ordinary llama.cpp server with capture disabled: greedy decoding, temperature zero, no repetition penalties, no prompt-cache reuse, context 1024, batch 128, microbatch 32, flash attention on, and a 16-token answer cap. The local fork is `f089c9cb4bc285a11e510e1c4336b95aca449b08`, built with CUDA 13.3. Runs used the RTX 5080 through the shared GPU runner.

| Measurement | Original | Alpha 1 | Selected alpha 2 |
|---|---:|---:|---:|
| Discovery: first value only | 5/16 | 7/16 | 10/16 |
| Discovery: last value only | 6/16 | 4/16 | 1/16 |
| Discovery: both values | 5/16 | 5/16 | 5/16 |
| Discovery: explicit first correct | 14/16 | 14/16 | 14/16 |
| Discovery: explicit last correct | 16/16 | 16/16 | 16/16 |
| Discovery: unique-key correct | 20/20 | 20/20 | 20/20 |
| Withheld: first value only | 7/16 | Not evaluated | 10/16 |
| Withheld: last value only | 5/16 | Not evaluated | 2/16 |
| Withheld: both values | 4/16 | Not evaluated | 4/16 |
| Withheld: explicit first correct | 15/16 | Not evaluated | 15/16 |
| Withheld: explicit last correct | 15/16 | Not evaluated | 15/16 |

Alpha 2 was selected and its hash recorded in `selection.json` before any withheld evaluation. There was no tuning on withheld results. The original's 68 discovery outputs were then repeated through the ordinary server and matched token for token.

The withheld changes were:

| Case | First value | Last value | Original answer | Edited answer |
|---|---|---|---|---|
| confirmation-1-1-default | Ruby | Opal | Opal | Ruby |
| confirmation-4-1-default | Tennis | Golf | Golf | Tennis |
| confirmation-6-1-default | Circle | Square | Square | Circle |

All 84 control answers across discovery and confirmation were unchanged. Four of the 64 explicit-instruction answers were already wrong and remained wrong. The twenty unique-key answers were correct in both models. Returning both values was not resolved: all nine such default cases still returned both.

### Reversed-direction control

After selection and confirmation, we made one additional model with the same proposed edit negated (alpha −2), quantized independently into the original blocks. This was a signed-direction check on discovery prompts only, not another tuning candidate. It changed 42,458 scalar weights and retained the original scales.

| Discovery default answer | Reverse, −2 | Original | Toward first, +1 | Toward first, +2 |
|---|---:|---:|---:|---:|
| First value only | 2/16 | 5/16 | 7/16 | 10/16 |
| Last value only | 12/16 | 6/16 | 4/16 | 1/16 |
| Both values | 2/16 | 5/16 | 5/16 | 5/16 |

All 52 discovery control answers were unchanged under the reverse edit too. For example, default `Oslo` answers became `Tokyo`; `Violin,Flute` became `Flute`; and `blue,green` became `green`. The direction therefore has a bidirectional behavioral effect in this experiment. This control does not test arbitrary directions or establish that this is the only useful edit.

The reverse model is `models/gemma4/gemma4-e2b/gguf-first-preference-2026-11-06/reverse.gguf`, SHA-256 `a0c4f6adc735c562dc4598b74b075b8056e8abf8b87effa9601a1bc3cddc7d27`.

## Internal propagation

These figures come from a separate instrumented forward pass. We derive softmax affinities from captured post-RoPE queries and keys before KV-cache rounding. They are diagnostic affinities, not directly recorded attention-kernel probabilities.

For default prompts at the selected head:

| Diagnostic | Original | Edited |
|---|---:|---:|
| First value's share of the two candidates' affinity | 59.09% | 88.66% |
| Absolute affinity mass on first value | 50.39% | 62.86% |
| Absolute affinity mass on last value | 32.45% | 7.48% |
| Total affinity mass on either value | 82.84% | 70.34% |

The total mass on the two values fell, so the conditional share alone is not a complete account. Both absolute masses and their denominator are retained in `propagation.json`.

Under explicit-first instructions, first-value share stayed near 90.6% → 90.3%. Under explicit-last instructions, it stayed near 4.79% → 4.93%. The captured key vectors were unchanged. Captured final-position layer outputs were identical through layer 28. The default-prompt output changed at layer 29 by mean relative L2 7.99%, then measured 6.85% at layer 34. This is an observed downstream response, not a claim of proportional or universally stable propagation.

Instrumentation is not behavior-neutral in this build. Baseline capture differs from ordinary-server answers on two of 48 variants; edited capture differs on three. The exact cases are in `experiment-results.json`. Native uninstrumented harness runs also differ from the server on some cases. These are preserved as limitations rather than repaired in this editing experiment. Ordinary-server scores are the behavioral result; captures are diagnostic evidence for the mechanism.

## What the result establishes

An activity-guided, weight-aware, quantized query edit can shift a shared lookup preference and transfer to new keys and values in these templates, while preserving the tested control outputs. Increasing positive strength from one to two increased the measured effect; reversing it favored the last occurrence.

It does not establish a unique geometric representation, a universally safe edit, unrestricted generalization, or a full account of the model's stability. The test is small, the control span was fitted from discovery inputs, and the remaining first/last/both behavior shows that the selected head is not the whole decision mechanism. A control output staying the same also does not mean its internal state stayed identical.

## Evidence and models

- `experiment-results.json`: settings, model hashes, selected edit, case-by-case original/edited outputs, repeatability, and capture disagreements.
- `cases.json`: all original prompt definitions and split membership.
- `attention-relationships.json`: measured candidate-head relationships used to select the edit location.
- `candidate-design.json`, `weights/query-direction.f32`, `weights/input-gate.f32`: the proposed direction and input gate.
- `quantization/`: proposed-to-stored edit measurements and realized delta arrays.
- `model-provenance.json`: full-model hashes and whole-file change verification.
- `propagation.json`: paired captured affinities, layer output changes, and first-token logit margins.
- `baseline/`, `discovery-alpha-1/`, `discovery-alpha-2/`, `baseline-validation/`, `confirmation-alpha-2/`: raw server requests and responses.
- Selected model: `models/gemma4/gemma4-e2b/gguf-first-preference-2026-11-06/alpha-2.gguf` under the QGRE project root.
- Native observation/editor source: `<external-engine-workspace>/local/research/lookup-preference-2026-11-06/`.

The native C++ helpers compiled successfully. The zero-edit identity, fixed-scale readback, model byte comparison, repeated baseline, and ordinary-server evaluations passed their stated checks. These are local experiment results, not a production-engine release or a broad model-quality qualification.
