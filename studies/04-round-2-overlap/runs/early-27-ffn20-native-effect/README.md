# Native FFN20 response on the original failing history

On the same consumed early-27 history, the rounded update moved the frozen FFN20 projection from `0.1466058673` to `0.0924677471` at event 312, merge 61. The native change was `−0.0541381202` toward the correct-witness target `−0.1370641310`. The squared residual fell by 34.53%. The finite forecast was within 0.154% of the measured displacement magnitude. [Native projection and byte bindings](result.json)

| Estimate | Change in projection | Absolute error against native |
| --- | ---: | ---: |
| Scaled continuous JVP | −0.2836699982 | 0.2295318780 |
| JVP on actual rounded displacement | −0.2592516590 | 0.2051135388 |
| Finite forecast on actual rounded displacement | −0.0542213711 | 0.0000832509 |
| Native readback on the same input | −0.0541381202 | |

At the next original prediction event, addition gained `2.4776239395` logits relative to subtraction. The native margin for token 900 (` +`) against token 753 (` -`) narrowed from `−2.5241632462` to `−0.0465393066`. Both scores were retained by the recorder at both endpoints, and the edited native greedy choice remained subtraction. Those scores were acquired in inference and remained outside training. [Sign-prediction scores](sign-margin.json)

The free edited generation changed at generated index 3, consumed position 176, and still subtracted the 40 J downhill gain on its own trajectory. This comparison therefore reused the baseline's saved native row and made one compact GPU 1 teacher replay of the exact original token history on the edited weights. The held recorder retained the raw FFN20 incoming, branch, and merged vectors at event 312. The residual addition reproduced bitwise at both endpoints. Held request · Capture receipt · Primary free-run outcome

The patch was the one written by the FFN20 scalar-target experiment. It changed 696,318 native codes in 337 tensors. The replay resolved compact table indices to physical rows, verified each replacement byte against the patch, and checked each tensor's change count and squared displacement against the completed primary installation. The restored typed origin and its zero remainder were read without changing the source. The private native descriptor was sealed before inference. These observations establish a fixed-history effect of that one displacement. The fitted free arithmetic remained wrong. [Prospective declaration](declaration.json)
