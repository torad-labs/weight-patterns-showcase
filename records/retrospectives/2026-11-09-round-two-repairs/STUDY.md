# What the discarded round-two edits actually repaired

Historical review requested by Marcos on November 9, 2026. This request authorizes reading the retired experiments for their findings; it does not restart them or select the population for the current experiment. No model was run or edited. Reserved-40b outcomes were not read. The stopped `originals-overlap` remains stopped.

**Scope clarified by Marcos after this readout:** the retired attempt did not test the hypothesis. Its pairwise target mechanism and arbitrary cap are not validated by these repairs. The task now is to explain why particular stored changes repaired calculations and transferred to excluded questions, and whether the responsible locations can be discovered from traces. The outcome tables below are observations to investigate, not a construction recipe to carry forward. The follow-up is [MECHANISM-STUDY.md](MECHANISM-STUDY.md).

The source snapshot is `67559f4a2fc2b61eda99200b185d98e91d44f2ff`, immediately before the round-two restart. Historical paths and SHA-256 hashes of all 49 source records are in [sources.json](sources.json). [evidence.json](evidence.json) contains the reconstructed score tables, per-question records, checkpoint identities, and construction declarations. [answer-readings.json](answer-readings.json) contains the prompts and full original/edited text for all nine evaluation repairs made by the later edit, with a new reading of each edited response.

## The effects that need an explanation

There were real stored-weight repairs and transfer to questions excluded from fitting. The strongest existing-set repair was the first full-path pass: 14 of 18 targeted failures and 25 of 36 broader energy-accounting failures acquired the correct final available energy, required energy, and verdict. On the separate 40-question comparison set it repaired 9 of 31 wrong answers, while damaging 6 of the 9 originally correct ones.

The early two-layer edit supplied the cleaner held-out construction comparison: it corrected 15 of 20 wrong or missing climb-energy fields, including explicit calculations with the requested climb height. It fully repaired only 2 of 31 wrong answers because other parts of the calculation remained wrong or broke.

These are behavioral findings from the retired constructions. Their targets came from individual failures' compatible successful mates, not a recurring difference extracted from independent success and failure overlays. They therefore do not establish that the intended overlap method found the capability trace.

## What changed in the weights

| Candidate | Stored change | How its target was constructed |
|---|---|---|
| Early F | Layer 11 `ffn_down`; 8,929,954 BF16 values changed | A local inverse toward contextual successful-trace differences at five predictions around the climb passage |
| Early K | Layer 14 key and value matrices; 608,630 and 664,072 BF16 values changed | Shared-cache correction fitted after F, then separately evaluated without F; K's patches were not refitted for that ablation |
| Early FK | F and K together: three matrices in two layers | Five-position contextual trace target; natural-success zero-change term; original histories used for conditional measurements |
| Later full-path pass | Attention-output and feed-forward-output matrices in 23 layers: 46 matrices, 379,355,412 BF16 values changed | Contextual repair targets across supported erroneous passages, with zero-change targets on correct passages inside failures and on natural successes |

The 23 layers were 0, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, and 24. This was not every model layer. The location rule used a 1% gap-share threshold; the applied fit used a 7.2% relative matrix-change cap. Those were historical design choices, not measured universal repair scales or recommendations for the next experiment.

The later construction solved a block-structured inverse at fixed recorded inputs, then applied all 46 stored changes together. It did not solve the complete nonlinear multi-layer model exactly. Several dimensions changed between the early and later constructions, so their outcome difference cannot be attributed to layer count alone.

Checkpoint identities:

- F: `e90c411000f6b305f77c9201b4f01c10bf57a8d7e7f6c2ec77a38ac1d0e8ec2b`.
- K: `dd039cd8139d6f95c991e85e239478be713f5d8e8a20931c74a721c0d54e6f84`.
- FK: `30f8042e2fec8d2416a60ed152a6b85fcc6a45f47775473194206d732fb2fc25`.
- Full-path first pass: `c02e7dd39fc79d4b00feb83f07b42a000d7d5d1b51f5114a3b709fd378c35b57`.

These identities come from saved checkpoint records. This review did not restore or rehash the large model files.

## Repairs on the existing questions

The same evaluation set contains 67 physics/paraphrase questions and six arithmetic controls. The capability table excludes those controls. A repair below requires all three requested fields to be correct. That is not a claim that every sentence of the explanation is correct.

| Edit | Wrong capability answers repaired /36 | Wrong climb-energy fields repaired /26 | Previously correct capability answers damaged /31 |
|---|---:|---:|---:|
| F: layer-11 feed-forward change | 8 | 19 | 25 |
| K: layer-14 key/value changes alone | 6 | 9 | 6 |
| FK: early combined edit | 5 | 14 | 17 |
| Full-path first pass | **25** | **24** | **7** |

The full-path result separates into 14/18 targeted climb failures and 11/18 other capability failures. Capability correctness rises from 31/67 to 49/67 after repairs and regressions. Across the whole 73-question suite, capped responses fall from ten to seven.

Three targeted cases had no supported nonzero repair positions in the contextual mapping: `expanded-027-h3`, `expanded-083-h3`, and `expanded-088-h1`. All three were repaired. They still participated through zero targets on known-correct passages, so this is transfer beyond explicit nonzero repair targets within the observed population, not held-out confirmation.

Examples of the later edit, shown as available / required / verdict:

| Question | Original | Edited |
|---|---|---|
| block-changed | 60 / 120 / NO | 60 / 90 / NO |
| sled-changed | 0 / 40 / NO | 80 / 40 / YES |
| expanded-089-h1 | 194 / 100 / YES | 194 / 50 / YES |

The historical manual audit read all 25 repaired responses and found calculations supporting the final numbers. It also retained explanatory defects. In particular, two answers still verbally add descent and climb heights while their final climb equation uses the correct height. A correct final triple and a wholly coherent explanation remain different findings.

## What transferred beyond the fit

The 40-question set contained 20 underlying numerical situations, each asked at two destination heights. Its questions were excluded from both fits. They were not 40 unrelated physical problems.

| Measurement on the same 40 questions | Early FK | Later full-path pass |
|---|---:|---:|
| Originally wrong complete answers repaired /31 | 2 | **9** |
| Originally wrong/missing required-energy fields repaired /20 | **15** | **15** |
| Originally wrong/missing available-energy fields repaired /24 | 1 | 5 |
| Originally wrong/missing verdicts repaired /13 | 3 | 8 |
| Previously correct complete answers damaged /9 | 8 | 6 |
| Total complete final triples, original 9/40 | 3/40 | **12/40** |
| Token-limit cases, original 4/40 | 8/40 | 8/40 |

Thirteen of the 15 climb-field repairs are shared between the two edits; the equal totals do not mean exactly the same cases improved.

The early FK candidate was frozen before its edited evaluation outcomes were read. Its 15 climb-field improvements were checked with the corrected parser and manually read in the historical record. All 15 write a calculation using mass, the specified gravity, and the final-climb height. Thirteen correct wrong numerical fields; two finish fields absent in capped originals. Several retain earlier errors. Of six cases whose available energy was initially correct, five lose that correct available energy after the climb-field repair.

By the later full-path fit, the 40-question results had already been observed and had informed the research. Its 9/31 result is evidence of transfer outside the fitted questions, with net gain on this seen comparison set. It is not untouched confirmation. Reserved-40b remains sealed; this review draws no result from it.

The manifest comparison is concrete:

| Parameter | The 45 construction questions | The 40 evaluation questions |
|---|---|---|
| Mass | 3, 4, 5 kg | 2, 3, 4, 5, 6 kg |
| Initial speed | 4, 6, 8 m/s | **5, 7 m/s** |
| Final climb | 1, 3 m | **2, 4 m** |
| Descent | 1, 2 m | 1, 2 m |

There are zero identical prompts between these manifests. The evaluation uses the same energy-transfer template family, with 20 channel and 20 reordered wordings, all describing a block. This supports numerical transfer within that family, not broad physics or language generalization.

Every original/edited question remains on the same GPU. For the later pass, the 40-question repairs are six on GPU0 and three on GPU1; regressions are two and four respectively. The questions differ between cards, so these counts are not a hardware comparison. The hardware split's effects on the validity of trace pooling do not turn these same-card behavioral comparisons into cross-card comparisons.

## Reading the later evaluation repairs directly

All nine edited responses were read in full for this review. Their final fields agree with the saved scores. Examples:

| Question | Original fields | Edited calculation and fields |
|---|---|---|
| fresh-00-h2 | 37 / 60 / NO | 25 + 20 - 8 = 37; climb 2 x 10 x 2 = 40; NO |
| fresh-06-h2 | 32.5 / 57.5 / NO | 73.5 + 30 - 11 = 92.5; climb 3 x 10 x 2 = 60; YES |
| fresh-10-h2 | 289 / 120 / YES | 98 + 40 - 14 = 124; climb 4 x 10 x 2 = 80; YES |

The text is not uniformly clean. `fresh-17-h2` initially subtracts descent, then explicitly corrects itself before reaching the right answer. `fresh-16-h2` gives the right numbers and NO but incorrectly calls the inequality 115 < 120 false. `fresh-04-h4` reaches the right final fields through additional inconsistent height/kinetic-energy bookkeeping. Several other responses reverse an elevation description. The record supports final-field repairs and concrete numerical-procedure improvements, not nine flawless explanations.

## What the comparisons teach us

**The early changes affected a reusable part of the calculation.** The same FK edit led to correct final-climb computations over different masses, new speeds, and new climb heights. Its strong climb-field transfer and weak complete-answer repair show that correcting one part can coexist with damaging another. The component ablation associates much of the descent-sign damage with F; K alone has fewer observed regressions. This does not identify a unique internal physics mechanism.

**The natural-success term mattered for the successful large edit.** Removing only the 27 other-success zero-change rows, while retaining the successful target sources and correct-passage zero targets inside failures, produced 0/18 targeted repairs, 0/36 broader repairs, and damage to 31/31 originally correct capability answers. With the term present, the same first-pass construction produced 14/18, 25/36, and 7/31. This is evidence that the term shaped a useful correction in this fit, beyond merely reducing regressions. It is not a universal theorem about preservation terms or an independently confirmed optimum.

**Local gap scores did not determine a block's usefulness.** Removing the layer-0 and layer-20 attention-output changes because their own-merge predictions looked unfavorable reduced repairs from 14 to 12 targeted and 25 to 18 broader. That is a joint ablation of two blocks, not separate attribution. It shows why a block's connected downstream effect matters even when its local score looks bad.

**The next larger edit was not a better repair.** The continuation reduced targeted repairs from 14/18 to 1/18, often omitting descent energy. It changed the direction as well as the size: successive deltas had cosine 0.2174, and cumulative relative matrix change was 11.1857%. The half-cap restart also changed the fitted direction and omitted two blocks. Neither is a clean scalar dose-response experiment, so they cannot establish a universal optimal edit size.

**Answer repair and the old trace metric separated.** For the successful large pass, the repair-region conditional residual fell 13.20%, but zero-target drift made the aggregate failure objective 6.33% worse. In the natural three-way comparison, the destination passage became closer for only 3/14 matched failures, while final fields became closer for 6/8. Coverage was 6,453/8,169 correspondences across 15 failures; missing matches stayed missing. This prevents calling the successful answers proof that the measured full path converged toward the successful mates. It also prevents treating a worse aggregate distance as proof that no useful repair occurred.

## Cases worth investigating

The first 46-matrix pass supplies the largest set of observed repairs to explain. It is not a recommended construction. The early F/K/FK component comparison supplies limited intervention evidence about useful effects and associated damage. The 40-question text gives concrete cases where an excluded input acquired the correct numerical procedure, together with cases where a substep was repaired but the complete answer was not.

These are useful cases for a later, explicitly authorized comparison of the actual traces around repairs and regressions. This review does not choose them as the next experiment's population, recover an overlap target from them, or revive the old solver. The positive behavioral findings survive the method's rejection; the claim that those constructions implemented the intended recurring-overlap hypothesis does not.

## Verification performed in this review

The tables were reconstructed from per-question records rather than copied from aggregate prose. Recomputed repair and regression IDs match the saved primary and secondary lists. All nine later evaluation repairs were inspected as full natural responses, and their records identify greedy generation with zero supplied continuation tokens. The construction and evaluation manifests were joined by actual IDs and compared for prompt overlap and numerical ranges. No inference, training, native probes, checkpoint downloads, or sealed-outcome inspection occurred.
