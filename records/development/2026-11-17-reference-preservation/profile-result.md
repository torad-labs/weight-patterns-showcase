# Frozen-state evaluation profile:175, 189 and 348

**The correct calculation is retained at every tested state. The selected development answer already gains its observed partial repair by 175; 189 and 348 do not repair the remaining descent-sign error.** The complete native trace losses below are separate from that arithmetic observation.

| Native weight state | Correct reference prompt: fixed-reference loss | Development prompt: fixed-reference loss |
|---|---:|---:|
| original | 1.3002743385995776e-07 | 0.00032040364720995428 |
| 175 | 0.00010016395934887741 | 0.00016123181067475393 |
| 189 | 0.00010007528137387596 | 0.00016112863358353657 |
| 348 | 9.3520299511362786e-05 | 0.00052137735957128253 |

The four new free generations finished at**21:03:26 CT, November 17**, exit 0. Original and 348 values come from the earlier same GPU1 paired controls; source, prompt budget, complete cases, engine, GPU UUID/hardware, original reference/tolerance and full merge inventory were checked before reuse. Both new175/189states were restored from their exact70-matrix BF16 sparse checkpoints over the verified common seed, then sealed read-only. No optimizer or remainder state was required or altered for inference.

For `expanded-004-h1`, all four answers compute16 + 40 − 12 = 44 J available,20 J required,YES. All 105 trace-loss terms nevertheless increase relative to the original control at 175, 189 and 348. This is a direct example of preserved task arithmetic with a larger distance to one fixed trace.

For `discovery-10-channel-h1`, the original answer reports55 J available,60 J required,NO. Every edited state tested reports55 J available,30 J required,YES. The required climb energy and verdict are corrected by 175. All states still subtract the30 J descent gain from96 J, so their available55 J is wrong: the stated physical calculation is96 + 30 − 11 = 115 J. This is partial transfer, not a fully correct solution.

Under the fixed trace metric,175 and 189reduce the development case by**49.678535%** and**49.710737%** versus its original control (96 merges improve,9 worsen at each).348 instead increases it62.725164% (15 merges improve,90 worsen), without undoing the partial arithmetic correction. There is no monotonic correspondence here between this metric and the separately inspected task calculation.

Between175 and 189, each prompt has identical generated tokens, identical full answer text and identical computational pair arrays. Correct-prompt loss falls by8.867797500143311e-8 (**0.088533%**;77 merges improve,28 worsen), and development loss falls by1.0317709121733694e-7 (**0.063993%**;74 improve,31 worsen). Thus those 14 updates produce small complete-trace changes on unchanged histories, with no new visible arithmetic repair on either case.

The temporal prediction is mixed: the development trace-distance increase at 348appears after the earlier states, but the correct-reference trace-distance increase already exists at 175 and 189 and is slightly smaller by348. This profile therefore does not support calling later training universally damaging, nor does it establish an optimal checkpoint. It measures two selected prompts, with the reference prompt belonging to calibration and the development prompt outside this current one-pair fit but within the historical development cohort. The four evaluations never enter fitting, reference, correspondence or update selection.

Coverage was checked from every pair/map: both earlier correct traces have609 events / 622 pairs, and both earlier development traces have605 events / 611 pairs. Every560 reference event, every current event and all 105 merges are represented; normalization mass remains one. Full560×105 map sums and per-merge totals reproduce each saved loss to floating summation error. No scalar or event was removed because its loss increased.

Native result · Full coverage audit · Separate arithmetic readback · Source-bound receipt · Independent source-bound review.

Both private models and GPU1were released on exit. The existing run-close inventory finds all115 active learner inputs and all38 profile inputs present. No archival/deletion was needed:86,091,001,856bytes were free against a conservative71,378,364,688-byte live-state/capture reserve. The active learner remained untouched.
