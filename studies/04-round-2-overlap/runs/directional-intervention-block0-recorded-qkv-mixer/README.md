# Recorded native Q/K/V through the block 0 Torch mixer

**The captured inputs recover the native mixed-output motion on the matched decode event and nearly all of it across the single-column decode calls.** At event 312 the relative error fell from 102.0% to 3.43% and cosine rose from 0.1000 to 0.99946. The same substitution leaves a much larger discrepancy during multi-column prefill. [Numerical result](result.json)

| Held portion | Relative error, original adapter | Relative error, captured Q/K/V | Cosine, captured Q/K/V |
|---|---:|---:|---:|
| Matched event 312 | 101.97% | 3.43% | 0.999464 |
| Single-column calls through event 312 | 140.06% | 0.299% | 0.999996 |
| 13-column prefill call | 105.92% | 96.45% | 0.264266 |
| 32-column prefill calls | 147.52% | 86.21% | 0.507383 |
| Full causal prefix through event 312 | 141.99% | 44.97% | 0.893163 |
| Full held history | 134.34% | 35.47% | 0.934969 |

Relative error is the norm of the difference between the model-change vectors divided by the native change norm. At the matched event, the native movement norm was 0.0002550435. The difference fell from 0.0002600733 to 0.00000875532 after supplying the captured operands. The original adapter replay reproduced the prior measured gap exactly.

Both native endpoints reported Flash Attention enabled and FP16 storage for K and V in both cache groups. The Torch replay supplied each endpoint's observed post-RoPE Q and rounded the observed post-RoPE K and normalized V through its existing FP16 cast. It used the same Torch score product, model causal/window mask, softmax, and value product. The code replaced Q at its actual score-product call and verified the complete layouts at both products. Recorded call group identities were used to report results. Replay source · [Matched native acquisition](../directional-intervention-block0-native-projection-replay/README.md)

| Cast of the saved stage tensors | K changes in native, whole / causal prefix | K changes in adapter, whole / causal prefix | V changes in either |
|---|---:|---:|---:|
| FP16 | 69 / 46 | 68 / 42 | 0 |
| BF16, hypothetical cast | 12 / 7 | 6 / 2 | 0 |

The FP16 code counts are derived from the saved pre-cache tensors. At the matched event, neither path crossed a K or V code boundary between endpoints. The original native and adapter tensors already differed at nine FP16 K codes on that row and at 2,755 across the causal prefix. Their corresponding V counts were zero and 110. The actual physical native cache reads, mask tensors, and padded Flash Attention launch geometry were not retained. The prefill discrepancy remains unresolved. The calculation reused existing native captures and performed no further acquisition, installation, or learner change. [MLflow run](http://127.0.0.1:5000/#/experiments/6/runs/cb500e73cf34438fab82067bdf591b6d)
