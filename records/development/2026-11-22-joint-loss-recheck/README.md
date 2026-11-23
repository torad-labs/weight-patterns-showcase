# The joint loop's loss with a lost step counted

November 22, 2026 (CT). Marcos chose the next loop's loss that night: first make a lost step count, then run the November 16–18 whole-trace loop with the descent term added (`journal/2026-11-21-consolidation-takeover.md`). This page is the first half, done on records: it trains nothing and changes no code. Program: [`recheck.py`](recheck.py). Outputs: [`lineage.json`](lineage.json), [`climb-descent.json`](climb-descent.json), [`unbanded-27.json`](unbanded-27.json).

## Where the records came from

The joint runs' per-operation records (`loop/<n>/group-measurement.json`) left this disk on November 19 and are not in the bucket. The local MLflow artifact store kept a copy of each one under the run's `steps/<n>/`. The four of `joint-stock-prechoice-training` carry the SHA-256 that run's tracked `fresh-fit-readback.json` records for them; `recheck.py` also holds each file to its own total (its cells sum to its `measured_contribution`).

| Run | MLflow run | Revisions read |
|---|---|---|
| `joint-stock-prechoice-training` (stock, with the pre-choice term) | `e2c4836547314df3ac14e3296b16d521` | 0 to 3 |
| `joint-stock-prechoice-streamed-recovery-5` (from revision 3) | `93e31dd758b94f189ad29b149cac9f64` | 0 to 7 |
| `joint-stock-climb-descent-training` (stock, before the pre-choice term) | `6972d351a0f14afcbe8406fa875680df` | 0 to 3 |
| `unbanded-group-training-27` (from early checkpoint 27) | `3c4b24dbe08d4cada3b6d46d355238bb` | 0 to 2 |

Artifact root: `<mlflow-store>/artifacts/6/<run>/artifacts/steps`. The fifth start's revision 0 equals the stock run's revision 3 cell for cell, so the two are read as one lineage of ten updates.

## What is counted

The loop's loss is the sum, over the operations it registers on each freshly generated answer, of each measured cell's weighted squared error. A cell is one frozen donor location of one operation of one member (member, operation, source event, merge), with a mass fixed by the declared donor support. A cell the matcher does not find in the new answer adds nothing (`code/signal-trace/pipeline/weight_edit/group_contrast_objective.py`).

- **As logged**: the run's own number, the cells found in that revision.
- **Carried**: the cells found at their measured value, and every cell not found charged the value it last had when it was found.
- **Common**: the 3,845 cells found in all eleven revisions, so the same cells are compared throughout.

## The lineage, stock to ten updates

9,012 cells were registered at some revision; 8,899 at stock.

| Revision | As logged | Carried | Carried against stock | Carried: failing / preserved | Common cells | Cells found / lost |
|---|---|---|---|---|---|---|
| stock | 0.0271872 | 0.0271872 | | 0.02685 / 0.00034 | 0.0088873 | 8,899 / 0 |
| 1 | 0.0179418 | 0.0194170 | −28.6% | 0.01319 / 0.00623 | 0.0071092 | 6,842 / 2,059 |
| 2 | 0.0200033 | 0.0297364 | +9.4% | 0.01684 / 0.01290 | 0.0101127 | 6,707 / 2,305 |
| 3 | 0.0174012 | 0.0286672 | +5.4% | 0.01476 / 0.01391 | 0.0104518 | 6,182 / 2,830 |
| 4 | 0.0204714 | 0.0310614 | +14.3% | 0.01476 / 0.01630 | 0.0115269 | 6,834 / 2,178 |
| 5 | 0.0204625 | 0.0310525 | +14.2% | 0.01476 / 0.01630 | 0.0115181 | 6,834 / 2,178 |
| 6 | 0.0192227 | 0.0309633 | +13.9% | 0.01590 / 0.01506 | 0.0110164 | 6,359 / 2,653 |
| 7 | 0.0192118 | 0.0309523 | +13.8% | 0.01590 / 0.01505 | 0.0110060 | 6,359 / 2,653 |
| 8 | 0.0192020 | 0.0309426 | +13.8% | 0.01589 / 0.01505 | 0.0109968 | 6,359 / 2,653 |
| 9 | 0.0170373 | 0.0313133 | +15.2% | 0.01579 / 0.01552 | 0.0110706 | 6,161 / 2,851 |
| 10 | 0.0201945 | 0.0333207 | +22.6% | 0.01579 / 0.01753 | 0.0126170 | 6,638 / 2,374 |

Revisions 4 to 10 are the fifth streamed start's updates 1 to 7.

- **The first update was a real gain.** With lost cells charged, the loss fell 28.6%. On the common cells it fell 20.0%.
- **Every update after it made the carried loss worse than stock.** It ends 22.6% above stock after ten updates. The logged number ends 25.7% below stock.
- **The common cells agree.** On the same 3,845 cells in every revision the loss ends 42.0% above stock, so the rise is not an artifact of what is charged for lost cells.
- **What was gained and what was paid.** The failing operations' carried loss went from 0.02685 to 0.01579 (−41%). The preserved operations' went from 0.00034 to 0.01753, 51 times its stock value. The run traded the computations it was meant to keep for the ones it was meant to repair, and the logged loss hid part of the trade.
- **Largest charges at the end**: the 91 J question's pre-choice descent cell, 0.0073511 (lost since the second update); the 95 J question's climb, 386 cells, 0.0033295; the 91 J question's descent field, 331 cells, 0.0014039; the 95 J question's friction, 840 cells, 0.0010244.

The arithmetic of the regenerated answers is the independent check that a lost step was not a repaired one: at revision 3 all three answers got descent wrong (`studies/04-round-2-overlap/runs/joint-stock-prechoice-training/fresh-fit-readback.json`).

## The two earlier group loops

| Run | Revision | As logged | Carried | Carried against start | Common cells |
|---|---|---|---|---|---|
| `joint-stock-climb-descent-training` | 0 | 0.0183962 | 0.0183962 | | 0.0156284 |
| | 1 | 0.0183787 | 0.0183787 | −0.1% | 0.0156138 |
| | 2 | 0.0224179 | 0.0224216 | +21.9% | 0.0196651 |
| | 3 | 0.0165287 | 0.0192818 | +4.8% | 0.0165253 |
| `unbanded-group-training-27` | 0 | 0.0090201 | 0.0090201 | | 0.0086435 |
| | 1 | 0.0175708 | 0.0179473 | +99.0% | 0.0175708 |
| | 2 | 0.0175191 | 0.0178956 | +98.4% | 0.0175191 |

`joint-stock-climb-descent-training` logged a 10.2% fall at revision 3; with its lost descent field charged (331 cells, 0.0026661) it is 4.8% above its start. The unbanded loop lost few cells and its logged and carried numbers agree: its loss doubled at its first update.

## What this establishes

- In all three runs whose gradient held the labelled-operation terms alone, the loss that counts lost steps ends above where it started. None of them was held back by the number of updates.
- A loss threshold on the logged number could have been met by losing more cells. The declared stop (0.0013593606, 5% of stock) is a threshold on a quantity that falls when the matcher fails.
- The first update of the stock run is the one step in these records that improved the carried loss, by 28.6%.

## What it does not establish

- That charging the last measured value is the right price. It is the smallest charge consistent with "a lost step is no better than when it was last seen". A cell that was found again later is charged its newest measured value, whatever that is.
- Anything about the whole-trace loss with a descent term added, which has not been run.
- The cause of the preserved operations' rise. Shared weights moved them; which update terms did is in the proposals (`steps/<n>/proposal.json` beside each measurement) and has not been read here.

## For the next loop

1. The loss the loop minimizes and stops on counts every registered cell: found cells at their measured value, lost cells at their last measured value. `recheck.py` is the reference arithmetic for that rule.
2. The gradient can only come from found cells, so a lost cell cannot be pulled back directly. The term that kept the whole computation tied to the reference in the November 16–18 learner, the whole-trace loss against the fixed reference, is what the next loop puts back in the gradient beside the descent term (`code/signal-trace/pipeline/weight_edit/common_prefix_driver.py` already sums the ordinary whole-trace residuals with an added term for one answer).
3. The run is still to be declared. Its starting checkpoint, the descent term's weight and its stopping rule are the open numbers, each to be derived from a record.
