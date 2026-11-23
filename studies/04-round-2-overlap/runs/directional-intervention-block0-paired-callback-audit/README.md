# Block 0 attention response at the first native boundary

**The mismatch is present by the unweighted post-attention RMS output.** The native residual **ADD reproduced bit for bit** from its captured F32 operands at both endpoints across all 1,775,616 scalar additions. The first stored attention branch, `attn_post_norm-0`, already differs in how the two engines respond to the update. The preceding native tensors needed to identify an internal projection or attention operation are absent. [Measured comparison](result.json)

The original and projected native acquisitions use identical callback layouts and hold the same input history. At each endpoint the adapter received the exact embedded input recorded natively for that endpoint. The six changed embedding values move sixteen original-to-endpoint input rows, seven within the target’s causal prefix. That movement is shared by native and anchored inputs at the corresponding endpoint. The first implementation required equality across weight states and stopped before adapter execution. The paired-operand comparison corrected that assumption. [First attempt](../directional-intervention-block0-callback-audit/result.json) · [Preceding anchored replay](../directional-intervention-anchored-replay/result.json)

## What the saved operands establish

| Response comparison | Matched event vector gap | Largest gap in the causal prefix |
|---|---:|---:|
| Weighted attention branch before residual addition | 0.00477961153 | 0.591654271 |
| Anchored versus native merged residual | 0.00477957881 | 0.591654309 |
| Normalized branch before the learned gain, inferred from the stored products | 0.000263870472 | 0.0226140444 |

The one-block adapter replay reproduced the previous full anchored result at the matched event. The exact GGUF/native layouts were checked for RMS gains and the four BF16 projection matrices. The Q, K, V and output matrices changed 6, 3, 2 and 10 stored values respectively, and the input attention-normalization gain had zero changes. Call metadata preserved 405 single-column groups, five 32-column groups and one 13-column group. Measurement source · [Exact weight and call inventory](result.json)

The final learned gain cannot account for the gap. At the matched event, 1,516 of 1,525 unchanged nonzero gain coordinates have a changed native product. Dividing each retained F32 product by its exact stored gain gives an estimate of the unweighted RMS output. For each endpoint and coordinate, half the larger neighboring-product ULP divided by the absolute gain bounds the inversion error under the native F32 multiplication. Summing the two endpoint bounds coordinate-wise and taking the vector norm gives **0.00000341443** at the matched event. The measured difference in the inferred pre-gain response is **77.3 times** that bound. This calculation uses normal F32 products and nonzero gains. The native unweighted values themselves were not retained.

## Native gaps in the record

The matched native records contain the embedded input, `attn_post_norm-0`, and the residual `attn_out-0`. They lack the native `attn_norm-0` output that enters Q/K/V, the raw `Qcur/Kcur/Vcur` and normalized/rotated tensors, the exact cache operands and mask seen by attention, the `kqv_out-0` value entering the output projection, and the raw output of that projection entering the final RMS. The auxiliary signal payloads are empty. The adapter outputs those projection operands and results on its side. Those cannot qualify an individual internal operation without corresponding native values. Native Gemma graph · [Executed projection adapter](../../../../records/development/2026-11-16-projection-gradient-adapter/adapter.py)

The recorded native context had Flash Attention enabled. The training path has separate Torch score matmul, masking, softmax, and value matmul after reconstructed FP16 K/V, and chooses its grouped projection primal by native callback geometry. Those source differences identify comparisons of interest. The current archives support localization through the unweighted RMS boundary and leave normalization, Q/K/V, cache/attention, and the output projection unresolved. No inference was generated and no weights were installed.
