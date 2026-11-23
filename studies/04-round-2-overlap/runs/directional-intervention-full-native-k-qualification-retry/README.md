# Full held-path qualification of original-only native K anchoring

**The shared K wrapper restored the direction of the finite pre-decision forecast and reduced its absolute error by 90.66%.** The old finite anchored graph predicted a signed decrease. With the original native K for every producer, the finite graph predicted the observed positive movement on the same frozen candidate. The rounded-displacement JVP remained negative. [Result](result.json)

| Same held history and exact endpoint bytes | Signed pre-decision projection change | Complete capped loss |
|---|---:|---:|
| Original native, reproduced by the new graph | 0 | 0.000292459960 |
| Old rounded-displacement JVP | $-4.91365422\times10^{-6}$ | 0.000291745221 |
| New rounded-displacement JVP | $-4.92596239\times10^{-6}$ | 0.000291745211 |
| Old finite anchored graph | $-7.43178444\times10^{-5}$ | 0.000291666883 |
| New finite anchored graph with original native K | $+1.27643319\times10^{-4}$ | 0.000291708508 |
| Observed native finite endpoint | $+1.10394653\times10^{-4}$ | 0.000291695782 |

Both JVP totals square the linearized residual with coefficient one on the exact rounded displacement. The finite projection error fell from $1.84712497\times10^{-4}$ to $1.72486657\times10^{-5}$. The finite capped-loss error fell from $2.88992445\times10^{-8}$ to $1.27262696\times10^{-8}$. The new finite standard loss was 0.000291705528 against native 0.000291692809. The new JVP changes the earlier local prediction by only $-1.23\times10^{-8}$. Its absolute error remains about $1.1532\times10^{-4}$.

The replay used the same shared `native_key_anchor.wrap` implementation prepared for the controller. It was installed as a graph wrapper before the original merge anchors were computed. All original incoming and merged F32 arrays across 105 merges reproduced the native baseline bit for bit, and the complete standard and capped baseline losses matched exactly. The origin-only key capture supplied all 15 KV producers under the same original history and verified physical K layouts and call positions. [Original key capture](../directional-intervention-original-producer-keys/README.md) · [Exact source binding](declaration.json) · Measurement source

The held graph used the same 578 consumed events and 584 registered pairs. Its issued endpoint arrays recovered all 31,552 changed native values and the exact archived displacement norm. The script wrote its forecasts before reading the observed native projected endpoint. It then verified the original and projected native capture bindings and unchanged history, geometry, and patch. There was no edited-state K input to the forecast, further weight installation, new optimization, or generation. Saved forecasts

The first attempt was refused by the content-hash guard when the wrapper changed before launch. Its declaration and stopped receipt are preserved. This run bound the frozen shared wrapper at SHA-256 `d5c0326eb7121375e9c87ddea18a37bbb16f1ae3801072216e908134614c5325`. The result qualifies one held finite endpoint and does not establish a training or gradient improvement. [Refused attempt](../directional-intervention-full-native-k-qualification/result.json) · [MLflow run](http://127.0.0.1:5000/#/experiments/6/runs/0afe45db6fb54fe284116242a9b3c83b)
