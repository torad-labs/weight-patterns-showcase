# Original-native K anchor on the held attention pair

**Original native K plus the adapter's finite K response recovered the decode motion as closely as the oracle which used actual edited-state K.** The prediction wrote its key tensor and mixed-output movement before the native endpoint was read. It matched all 22 native FP16 K transitions on the single-column calls through event 312 and 44 of 46 across the full causal prefix. [Numerical result](result.json) · Exact FP16 coordinates and codes

The frozen computation was

$$
K_{\mathrm{pred}}=K_{\mathrm{native,original}}+
   \left(K_{\mathrm{adapter,projected}}-K_{\mathrm{adapter,original}}\right)
$$

in FP32 before the existing FP16 cache conversion. The adapter used its own Q and V at both endpoints. It also reconstructed the projected embedding from the original native input and the six already frozen physical embedding edits. Those predicted embedded operands matched the projected native operands bitwise at all 578 positions when checked later. No native projected operand entered the prediction. Implementation · Saved K prediction · Saved mixer motion

| Held portion | Original adapter error | Original-only K anchor error | Oracle K error |
|---|---:|---:|---:|
| Matched event 312 | 101.972% | 2.64748% | 2.64776% |
| Single-column calls through event 312 | 140.057% | 0.30528% | 0.30509% |
| Entire causal prefix through event 312 | 141.986% | 44.97711% | 44.97713% |
| Entire held history | 134.335% | 35.87977% | 35.47545% |

The error is the norm of the discrepancy between predicted and native change vectors divided by the native movement norm. At event 312 the original-only anchor's cosine with native movement was 0.999676. The target row had no local FP16 K transition, so the recovered response includes its dependence on earlier cached keys. The sizeable prefill discrepancy is present in both the prediction and oracle.

| Scope | Actual native transitions | Predicted transitions | Exact coordinate and new code matches | Missed | Extra |
|---|---:|---:|---:|---:|---:|
| Single-column causal calls | 22 | 22 | 22 | 0 | 0 |
| All calls in causal prefix | 46 | 44 | 44 | 2 | 0 |
| Entire held history | 69 | 69 | 66 | 3 | 3 |

Both misses in the causal prefix were in 32-column prefill calls, at events 88 and 92. Shared transitions always matched the new FP16 code. This demonstrates that the original key baseline is sufficient to recover the needed decode code crossings on this held pair with the existing adapter's finite key movement. The whole-history mismatches leave some key-response error, and the test does not qualify gradients, physical native cache reads, or the full learner. It used the existing captured native endpoints without generation, retuning, or a new weight installation. [Declaration](declaration.json) · [MLflow run](http://127.0.0.1:5000/#/experiments/6/runs/1c9c062dbdf44784b62dffdcdec8b297)
