# Same-question comparison through the local llama.cpp fork

Purpose: determine whether the twenty recorded association questions exhibit the same failures with the local llama.cpp fork. The user explicitly requested this external diagnostic comparison. That request authorizes this bounded exception to QGRE's no-external-oracle/default single-engine experiment rule; it does not make llama.cpp a QGRE acceptance gate.

Use the existing fork build f089c9cb4bc285a11e510e1c4336b95aca449b08, and the published bartowski/google_gemma-4-E2B-it-GGUF Q4_K_M file at repository revision 81012ba3538e061d5ee003f11f25335b17f82e2d. No local E2B GGUF was found in the searched project/model locations. The GGUF SHA256 must be 923c4c86177d2ee173a7f5b4fa3d0ac65f5962ab15e6d6a5bc250aec4fd7bf7e.

Keep all twenty question texts, greedy selection, no repetition penalties, an eight-token limit, and the recorded QGRE conversation wrapper. Feed explicit token arrays to the completion endpoint. Compare token counts for all questions and exact IDs against the two saved Silver captures. Check tokenizer token meanings against the QGRE vocabulary for all prompt IDs. Save original requests and responses.

Use one local server with one slot and prompt reuse disabled on each request. Confirm the returned cache counts, preserving this difference from QGRE's fresh-process-per-case runner. Requests are independent, not a continuing conversation. Run on physical GPU 0 through gpu_card_run and stop the owned server at completion. Repeat the same three extended observations (probe-08, probe-09, probe-16) with a 128-token budget. If other short responses hit the limit, retain their incomplete classification and extend those separately.

Q4_K_M GGUF and QGRE's TQ4 E2M1 checkpoint are different quantizations. The engines also use different activation/cache arithmetic. This comparison can locate a discrepancy between complete inference pipelines; it cannot alone attribute that discrepancy to a QGRE kernel bug or establish equivalence of model-weight bytes. No weights or production source are modified. The existing twenty-case corpus is a development diagnostic, not an unbiased accuracy benchmark.
