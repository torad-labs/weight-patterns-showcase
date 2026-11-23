# Which recorded attention operand recovers the decode motion?

**K is the evidence-backed implementation priority for this held decode case.** Substituting captured native K at both endpoints reduced the mixed-output movement error at event 312 from 101.97% to **2.65%**, and from 140.06% to **0.305%** across the single-column calls through that event. Replacing Q alone slightly worsened the matched-event agreement. V alone had negligible effect. Both the original mixer and full native Q/K/V substitution reproduced the preceding measurements exactly. [Numerical result](result.json) · [Earlier joint substitution](../directional-intervention-block0-recorded-qkv-mixer/README.md)

| Native operands supplied | Relative error at event 312 | Single-column causal calls | 13-column prefill | 32-column prefill |
|---|---:|---:|---:|---:|
| None | 101.972% | 140.057% | 105.924% | 147.516% |
| Q | 102.864% | 140.053% | 105.907% | 147.525% |
| K | **2.648%** | 0.305% | 96.475% | 86.210% |
| V | 101.972% | 140.057% | 105.924% | 147.516% |
| Q, K | 3.432% | 0.299% | 96.448% | 86.206% |
| Q, V | 102.865% | 140.053% | 105.911% | 147.525% |
| K, V | 2.651% | 0.305% | 96.480% | 86.210% |
| Q, K, V | 3.433% | **0.299%** | 96.447% | 86.206% |

Each cell uses the selected native operands at both original and projected endpoints and obtains the other operands from the corresponding Torch endpoint. The existing FP16 cast, Torch score and value products, and causal/window mask are held fixed. Relative error is the difference-vector norm divided by the native movement norm. At the matched event, K improves squared error in every one of the four possible Q/V contexts. Q worsens it in all four K/V contexts. The target-event vector contrast for K alone has norm 0.000257336. The largest pair interaction, Q×K, has norm 0.00000949167, and the triple interaction is 0.000000477423. The data retain those interactions and the corresponding conditional improvements. Factorial method

The earliest absolute recorded mismatch at event 312 was in Q after normalization and RoPE. This comparison identifies the operand that controls agreement in the **update response** on the held decode path. It supports prioritizing faithful K formation through the FP16 representation used in mixing. The captures leave the raw K projection output and physical cache reads unobserved, so they cannot locate the problem within K projection, normalization, RoPE, or cache handling. Mixed-source inputs are diagnostic combinations, and the large prefill error remains after every substitution. The comparison reused the two existing native endpoints without capturing further output, installing weights, or changing the learner. [MLflow run](http://127.0.0.1:5000/#/experiments/6/runs/969bfe5d3c3c40d9babfe04dd8a07891)
