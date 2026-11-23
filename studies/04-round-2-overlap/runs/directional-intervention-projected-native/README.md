# A finite native test of the archived projected update

**The fitted projection increased in the native model even though the rounded first-order forecast predicted a decrease.** The original archival direction had moved farther in the same adverse direction. The projected change reduced that observed adverse movement by 71.1% on the same history, while still increasing the excess above tolerance by 0.842%. [Numerical result](result.json) · [Prior gradient analysis](../directional-intervention-gradient-competition/README.md)

| Fitted readout | Signed projection change | Full capped loss change |
|---|---:|---:|
| Frozen first-order forecast of the rounded projected step | $-4.91336936\times10^{-6}$ | $-8.25464586\times10^{-7}$ |
| Projected step measured natively on the held history | $+1.10394653\times10^{-4}$ | $-7.64177742\times10^{-7}$ |
| Original historical one-step intervention on the same history | $+3.81764087\times10^{-4}$ | $-8.46877418\times10^{-7}$ |

The projected candidate's held fitted loss fell from 0.000292459960 to **0.000291695782**, a 0.2613% decrease. Its projection increased from 0.3932914975 to **0.3934018921** against a 0.3801804580 allowance. The complete forced replay on the original physical GPU and native engine reproduced the archived input history, full callback index, qualified merge geometry and computational correspondence. All 61,320 cells across 105 merges were read. This rules out a changed causal history or registration as the explanation for the local sign reversal in this control. The experiment does not separately resolve finite nonlinearity, native internal rounding and differences in the anchored derivative reconstruction. Held acquisition

## Fresh generation

- **Fitted response.** The candidate naturally ended on a different history, with its first different selection at zero-based output index 72. The complete sequence matched the archived standard control. It subtracted 40 J of downhill work to give 24 J at the bottom, subtracted friction correctly to give 11 J, then also charged the 20 J climb against available energy and reported −9 J, 20 J required, and NO. Fresh capped loss rose to **0.00162233602**, 5.547 times the original. The distinct fresh correspondence placed the historical reference event at a later current row. Trace measurement
- **Correct response.** The candidate naturally reproduced the original complete generation, including 44 J available, 20 J required, and YES. Its fresh capped loss rose from $1.30027434\times10^{-7}$ to $2.42273267\times10^{-7}$ while its historical cell stayed inside tolerance. Trace measurement

The run verified the prior forecast's unrounded and rounded scalar products individually for all 538 tensors before applying the update. The native installer verified the typed byte changes at 31,552 stored values on an anonymous copy, then sealed it for all three acquisitions. The diagnostic cell and direction were inherited from the historical intervention. No training selector or production learner change was involved. [Declaration](declaration.json) · Native-byte receipt · [MLflow run](http://127.0.0.1:5000/#/experiments/6/runs/4938fb56715f44daa07568f48f10c7df)
