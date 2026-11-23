# Original native K on the earlier directional displacement

**The frozen K wrapper cut finite signed projection error by 9.94% on a second displacement and preserved the correct positive sign.** The old estimate captured 6.64% of the observed native movement. The wrapper recovered 15.92%, leaving most of the movement unexplained on this earlier update. [Numerical result](result.json)

| Fixed history and earlier archived directional bytes | Signed pre-decision movement | Complete capped loss |
|---|---:|---:|
| Original native state | 0 | 0.000292459960 |
| Old finite anchored graph | $+2.53615117\times10^{-5}$ | 0.000291703138 |
| Finite graph with original native K | $+6.07730382\times10^{-5}$ | 0.000291642393 |
| Archived native directional endpoint | $+3.81764087\times10^{-4}$ | 0.000291613082 |

Absolute target error fell from $3.56402575\times10^{-4}$ to $3.20991049\times10^{-4}$. Complete capped-loss error fell from $9.00556998\times10^{-8}$ to $2.93104663\times10^{-8}$, a 67.45% reduction. Standard loss was 0.000291700203 in the old forecast, 0.000291639442 with K anchoring, and 0.000291609986 in the archived native observation.

This reused the same source-hashed wrapper and original-only post-RoPE K capture as the first full-path qualification. The earlier directional intervention changed 31,564 stored values, with displacement norm squared $4.24193041\times10^{-9}$. Both finite graphs reproduced every original incoming and merged state bitwise across all 105 merges. The same 578 consumed events, 584 computational pairs, exact original native inputs and endpoint bytes were used throughout. The forecast files were saved before the earlier edited-native observation was opened for evaluation. Saved forecasts · [Original keys](../directional-intervention-original-producer-keys/README.md) · [First qualification](../directional-intervention-full-native-k-qualification-retry/README.md)

This is numerical replication on a second existing weight displacement of the same fitted problem. Its improvement is smaller than on the first projected endpoint. The work used no new capture, generation, installation, gradient, or tuning. [Declaration](declaration.json) · [MLflow run](http://127.0.0.1:5000/#/experiments/6/runs/a8f63c6db70144b792736cf5192181d3)
