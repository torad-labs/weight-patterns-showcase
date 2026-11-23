# The local llama.cpp fork answers all twenty association questions correctly

The fork returned the correct concise value in **20/20** cases. All eight numeric-substitution failures observed through QGRE disappeared. Every short answer stopped naturally in two or three emitted tokens, including the end-of-turn token. The same three 128-token follow-ups returned the same short answers and token IDs. No weights were edited.

| Case | Requested value | QGRE short answer | llama.cpp answer |
|---|---|---|---|
| known-01 | blue | 2 | blue |
| known-03 | blue | 3 | blue |
| probe-03 | mango | Mango | Mango |
| probe-04 | mango | Mango | Mango |
| probe-05 | mango | Mango | Mango |
| probe-06 | falcon | Panda | Falcon |
| probe-07 | falcon | 3 | Falcon |
| probe-08 | falcon | The question asks for the value of " | Falcon |
| probe-09 | violin | Please provide the table you are referring to | Violin |
| probe-10 | violin | 3 | Violin |
| probe-11 | violin | 3 | Violin |
| probe-12 | silver | Copper | Silver |
| probe-13 | silver | 24 | Silver |
| probe-14 | silver | Silver | Silver |
| probe-15 | oslo | Tokyo | Oslo |
| probe-16 | oslo | The question asks for the value of " | Oslo |
| probe-17 | oslo | Lima | Oslo |
| probe-18 | train | Train | Train |
| probe-19 | train | 1 | Train |
| probe-20 | train | 1 | Train |

## What was controlled

- The same twenty question strings from the prior frozen corpus.
- Greedy generation: temperature zero, only the temperature sampler, repetition/presence/frequency/DRY penalties disabled.
- An eight-token cap, followed by the same three selected 128-token observations.
- The recorded QGRE conversation wrapper, passed as explicit token IDs to the completion endpoint; no automatic chat-template substitution.
- All 20 prompt lengths match QGRE. Both saved Silver prompt arrays match exactly. All 88 distinct prompt-token IDs have the same text in the QGRE vocabulary and llama.cpp tokenizer.
- One server slot with prompt reuse disabled; all 23 responses report zero reused cache tokens. QGRE used a fresh process for each question; llama.cpp reused loaded weights while evaluating each complete prompt independently.

## Interpretation and remaining differences

This comparison moves the investigation toward the QGRE-specific converted weights and execution pipeline. It does not support treating the observed numeric answers as an intrinsic, established Gemma lookup mechanism. The same task, framing and greedy choice succeed in the external pipeline.

It does not isolate an implementation bug: QGRE uses TQ4 E2M1 weights and its native low-precision numerical path, while this comparison uses the published Q4_K_M GGUF and llama.cpp numerical path. The GGUF publisher identifies the source model as google/gemma-4-E2B-it; exact upstream source-revision equality with the earlier QGRE conversion has not been established. Q4_K_M is a mixed tensor quantization, not a byte-equivalent copy of TQ4.

The initial QGRE short pass has 5 concise correct answers, 8 numeric substitutions, 4 wrong listed values and 3 truncated explanations. The longer QGRE follow-ups explicitly identify the correct value in two explanations, while the third asks for the already supplied table; do not label all fifteen non-concise short answers as proven wrong final answers.

The fork loader also recorded warnings for control-looking tokens 50 and 212 and repaired their special-token treatment. Neither token occurs in these prompt arrays or outputs; all responses ended at token 106. The raw warnings are preserved.

## Artifacts

- [Relational comparison and identities](comparison.json)
- [Original frozen questions](cases.json)
- [Execution plan](PLAN.md)
- [Execution script](run.sh)
- [Model repository metadata](model-source.json)
- [Binary, model and corpus hashes](inputs.sha256)
- [Raw short requests and responses](short/)
- [Raw extended requests and responses](extended/)
- [Prompt tokenization and matching checks](tokenization/)
- [Server output](server.stderr.txt)

Model source: [bartowski/google_gemma-4-E2B-it-GGUF](https://huggingface.co/bartowski/google_gemma-4-E2B-it-GGUF/tree/81012ba3538e061d5ee003f11f25335b17f82e2d). Fork build: f089c9cb4bc285a11e510e1c4336b95aca449b08. The run script completed with exit 0 and its owned server was stopped. No performance claim is made.
