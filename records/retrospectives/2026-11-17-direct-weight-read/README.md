# What changed in the weights as the fitted trace improved

November 17, 2026. The original selected reference (`expanded-004-h1`) and selected wrong response (`discovery-05-channel-h1`) used identical model parameters. Their different inputs produced different activations and different traces. To examine the weight changes, we compared that original checkpoint with reconstructed step 230 and the complete native parameter snapshots at 329, 340 and 348. Step 348 has an independently restored checkpoint and a fresh measurement. The reference, tolerance and one-pair objective come from the [Final Experiment](../../../FINAL-EXPERIMENT.md).

## Most of the loss reduction preceded the largest edit

The last column gives the Euclidean distance of the stored values of the first attention-output matrix from their original values. The loss measures ordinary generation under the weights at each checkpoint.

| Checkpoint | Fresh whole-trace loss | Distance of first attention-output matrix from original |
|---|---:|---:|
| Original | 0.00029234223 | 0 |
| 206 | 0.00008114998 | 0.007866 |
| 230 | 0.00007653474 | 0.008030 |
| 236 | 0.00007484178 | 4.069313 |
| 243 | 0.00007465837 | 4.069444 |
| 317 | 0.00006447134 | 4.146988 |
| 348 | 0.00006420801 | 4.146988 |

Step 230 already accounted for **94.60% of the net original-to-348 reduction**. These are selected checkpoints. An earlier observation at 203 reached 0.00004162764 and subsequently gave back gains. Its exact state was unavailable for this read. Checkpoint records and reconstruction · [Earlier low and phase history](../2026-11-17-full-weight-phase-review/ASSESSMENT.md).

At 230, 3,173,771 stored values differed across 70 matrices in 35 blocks. Attention-output matrices held 86.97% of the sum of squared changes, and feed-forward down projections held 13.03%. Block 12's attention output had the largest share at 20.06%. Block zero's attention output had 3.03%. The total Euclidean weight change was 0.04616 across all eligible parameters. The training phases up to 230 constrained which weights could move. These measurements describe the edit under that constraint. Per-matrix original-to-230 data.

At 348, 5,658,575 of 4,628,567,842 eligible physical weight values differed from the original across 497 of 538 arrays. Block zero's attention-output matrix now held **99.988%** of squared weight displacement. Its ten leading singular directions held 94.03% of its own squared displacement. In the later interval 329–348 that same matrix accounted for 9.77% of squared displacement between the two endpoints. Per-tensor checkpoint data and SHA-256 identities · First-matrix decomposition.

The intervening run explains when the concentration emerged. In updates 231–236, only the first attention-output matrix changed in the installed BF16 values. All seven measurements, including the starting state, had identical generated histories and computational correspondence. Whole-trace loss declined on each update, by **2.21%** altogether. This identifies the matrix as sufficient for that incremental gain from 230 on this pair. The six updates account for **0.74%** of the net original-to-348 decline. The run expressly selected this matrix. [Declared update](../../../studies/04-round-2-overlap/runs/vector-chance-conditioned-from-230/declaration.json) · Native transition payloads and fresh measurements.

## What the first matrix does on each history

We reconstructed the first attention operation on CPU using the original upstream weights. Within each row we held fixed the consumed history, upstream calculation, correspondence and tolerance. We replaced only the attention-output matrix. Each entry below is that merge's contribution to whole-trace loss, including its ordinary aggregation weight. The later history came from fresh generation under step 348. Feeding it to the original first block is a conditional comparison.

| History supplied | Events | Original matrix | Step-348 matrix |
|---|---:|---:|---:|
| Selected reference, original generation | 560 | 0.000000000023 | 0.000000174955 |
| Selected wrong response, original generation | 578 | 0.000103939301 | 0.000102890724 |
| Same failing prompt, step-348 generation | 612 | 0.000027957937 | 0.000026091032 |

The swap lowered this local loss by 1.01% on the original wrong history and by 6.68% on the later history. It also moved the reference computation slightly. Native reference self-loss is exactly zero. The original wrong native value is 0.000103939466, and the full step-348 native value is 0.000026090950. At the two native endpoints, all 105 merge-level contributions were lower on the later generation, and the first merge accounted for 34.12% of the total decline. Individual positions and alignments can differ. CPU swap · All 105 native contributions.

The original first block already scored substantially lower when fed the later history and its associated correspondence. The swap shows that the matrix changes computation on either history. It cannot identify the edit that made the later history occur in free generation. A large weight change, the location of a large loss contribution, and causal influence over a generated path are three different measurements.

## Implications for a repeatable update

Keep the original reference, local losses and evaluator fixed. Keep training all eligible weights through the trace derivatives. The six-update interval supplies a measured example for checking whether the rule correctly predicts a reduction on an unchanged history. The late checkpoint alone gives little guidance on which weights made the large early improvement possible.

For causal follow-up, prioritize an early observed transition where the history changed and loss fell. Where both states can be restored, measure each weight state on both recorded histories, using the same native acquisition settings and recomputing computational correspondence. Use fresh generation from both states to identify which history each one actually produces. If one weight group is implicated, transplant just its observed change into the earlier state and restore just that group in the later state, then regenerate. Interactions with the complementary weights can make both controls informative. Keep the chosen reference fixed for fitting and separately measure whether the updated model preserves correct computation. Test the resulting rule on another development pair before claiming repeatability.

The first-matrix swap above used only saved inputs on CPU. The full-model causal tests remain open. The running learner and its configuration were untouched by this analysis.

The scripts compare.py, early_weights.py, timecourse.py and first_merge.py reproduce the saved numbers from local native files. The streaming comparison verifies hashes over the original file and all three complete checkpoint payloads. The early-state reconstruction verifies the original working seed and sparse override payload. The first-merge reconstruction uses the project's CPU Torch implementation with an FP16 attention cache.
