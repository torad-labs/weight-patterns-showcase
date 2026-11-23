# Measured reference preservation at frozen348

The completed native check finds a substantial **increase in single-reference trace distance** on the fixed correct prompt, `expanded-004-h1`: original-model GPU1 control loss **1.3002743385995776e-7**, frozen348 loss **9.352029951136279e-5**, change **+9.339027207750283e-5**. All105 merge totals increase. This measures trace proximity for the correct prompt, separate from the fitted wrong-prompt metrics; it does not establish a loss of answer quality.

Both generations ended naturally. Original560 events become584 at348; computational correspondence changes560→597 pairs. Every560 original reference event and every current event remains covered at all105 merges, with total original normalization mass one. The common-reference loss maps reproduce both objectives to floating summation error. The original same-card control is measured nonzero, not assumed zero or used to refit tolerance.

The first attention merge contributes2.0349792392873042e-5 of the increase; all feed-forward merge increases sum4.704206999317077e-5. The fresh histories differ, so these are complete-path descriptive changes, not a fixed-input causal allocation among layers. The result shows less preservation of proximity to this one fixed computational reference. It is not an answer-text verdict or a population preservation estimate; this is the selected reference/calibration prompt.

The exact frozen state was restored using FullState.restore with complete538-tensor byte readback. Both paired native captures usedGPU1 and the same bound engine. The diagnostic completed at20:15:36CT on November17, exited0 and released its host model before the crossed-history task. The active learner, objective, reference, tolerance and optimizer were unchanged.

Complete assessment · [Native result](../../../studies/04-round-2-overlap/runs/vector-chance-reference-preservation-348/result.json) · Full coverage audit · Source-bound receipt.

The separately selected development case is still outside fitting and has its own prospective declaration. It was selected before either score was consulted and does not replace this preservation finding.

## Interpretation correction — November17

Marcos pointed out that distance from one reference does not establish task quality. Direct inspection of both saved answers confirms the same correct arithmetic:16J initial kinetic energy,40J gained in descent,56−12=44J available,20J required,YES. Frozen348 therefore preserves the task calculation in this case despite larger trace distance. Any earlier wording implying an answer-quality regression is withdrawn. The metric and native records remain unchanged; this separate arithmetic readback never enters fitting.
