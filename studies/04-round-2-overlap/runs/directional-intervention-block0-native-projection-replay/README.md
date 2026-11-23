# Block 0 attention with native projection callbacks

**The first measured discrepancy at the target event is in the Q path by the normalized and rotated query output, `Qcur_pos-0`.** The normalized input to the projection, `attn_norm-0`, was bitwise identical between the engines at that event for both endpoints. The Q boundary differed by approximately 0.001020 in vector norm at each endpoint, and the engines’ responses to the frozen update differed there by $4.79623\times10^{-7}$. Raw `Qcur-0` is outside the supported projection capture, so the evidence bounds the first discrepancy to the Q multiplication, query RMS/gain, or RoPE. [Numerical result](result.json)

The same hashed worker ran `capture projection stream` with its supported block-0 label filter. Both original and projected replays used the complete archived teacher input, and the candidate used the already verified native patch. At both endpoints all 578 previously recorded rows of `inp_scaled`, `attn_post_norm-0`, and `attn_out-0` matched the earlier native captures bit for bit. The retained callback shapes, strides, labels, and call positions also agreed. Each replay kept eight F32 block-0 boundaries and a 24,858,624-byte tensor payload. No fresh response, new candidate, or recorder modification was involved. [Declaration](declaration.json) · Original capture · Projected capture

| Block 0 native boundary | Original adapter/native gap at event 312 | Gap in the change caused by the update at event 312 | Largest change gap over the causal prefix |
|---|---:|---:|---:|
| Normalized attention input | 0 | 0 | $5.821\times10^{-11}$ |
| Normalized/rotated Q | $1.0200\times10^{-3}$ | $4.796\times10^{-7}$ | $1.966\times10^{-6}$ |
| Normalized/rotated K | $4.0046\times10^{-5}$ | $1.101\times10^{-7}$ | $1.582\times10^{-7}$ |
| Normalized V | $1.4564\times10^{-6}$ | 0 | $1.263\times10^{-6}$ |
| Mixed attention output, before O projection | $7.7084\times10^{-4}$ | $2.601\times10^{-4}$ | $2.233\times10^{-2}$ |
| Weighted attention branch, before residual ADD | $1.3592\times10^{-2}$ | $4.780\times10^{-3}$ | $5.917\times10^{-1}$ |

Across the full causal prefix, the earliest absolute discrepancy is already visible at `attn_norm-0`, up to $9.098\times10^{-5}$. The two engines’ changes from their own baselines at that boundary nevertheless agree to $5.821\times10^{-11}$. The paired normalized input is exactly equal only at the matched event, so conclusions for earlier events retain that distinction.

The update-response gap is much larger in the mixed attention output `kqv_out-0`. Its sources already include discrepant Q/K/V paths. The physical FP16 cache reads and mask are unrecorded, as are the raw Q/K/V projection outputs and the raw post-O product before RMS. The record therefore cannot split Q math into projection, normalization and RoPE or assign the larger gap specifically to Flash Attention. The earlier same-layout residual read independently reproduced the residual ADD exactly and placed the gap before its learned post-attention gain. [Paired residual-boundary audit](../directional-intervention-block0-paired-callback-audit/README.md)
