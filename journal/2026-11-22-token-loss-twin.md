# November 22, 2026: the token-loss twin is asked for

Times are CT. Written by the seat that did the consolidation (`journal/2026-11-21-consolidation-takeover.md`), from Marcos's direction of this afternoon.

## What Marcos asked

At about 3:35 PM he asked for the comparison the records have named and never run: a twin of the learner trained on the same question with the ordinary token loss. The November 12 plan put it this way: "Train a twin on the same answers with the ordinary token loss. It measures what the trace adds, which is the hypothesis's own claim" (`records/development/2026-11-12-path-forward/deep-think-analysis-2/1-deep-think.md`). The path-edit paper lists its absence as a limit, and `records/KNOWN.md` has held it as open since November 13.

No such run exists. That was checked on November 22 across every commit on every ref, a full listing of the research bucket, the local MLflow store and 474 session logs.

## His choices

- **Where it runs.** On this machine's RTX 5080 (card 0). A rented RTX 5090 was his first idea. The records say a 5090 computes different numbers from the cards here, enough that a rented card "can stand as a judge only against a reference recorded on a rented card" (`records/KNOWN.md`, the November 13 box records). Card 0 made the reference and checkpoints 175, 189 and 390, so a twin made on card 0 is read against them directly. No box was rented.
- **What it is compared with.** The existing checkpoints 175, 189 and 390 and the original, on the frozen panel of eight new formulations and three anchors (`studies/04-round-2-overlap/runs/frozen-energy-generalization/`). No fresh trace arm: from the original, the trace learner's loss went 0.000292 to about 0.000285 over its first 24 updates with the fitted answer still wrong, so a short fresh arm would show no repair.
- **Who builds it.** `CLAUDE.md` excludes Fable 5.2 from project code. Marcos said "let's do it" to this session, which runs on Fable 5.2. The trainer code of this run is written by a subagent on another model and reviewed by an independent one, as the gate change of this morning was. This seat writes the declaration, the brief and the records.

## What the twin is

The same loop as the one-pair learner, with one thing changed.

| | One-pair learner | Twin |
|---|---|---|
| Question | `discovery-05-channel-h1` (correct: 64 + 40 − 13 = 91 J) | the same |
| Start | the original model | the same |
| Weights that may change | the 538 eligible tensors | the same |
| What supplies the gradient | the whole-trace loss against the fixed reference | the next-token loss on a corrected answer |
| Installation | rounded once to native BF16, remainders carried | the same |
| After each update | the model writes its answer again, freely | the same |

**The corrected answer** is the original model's own: its 424-token answer to this question with one word fed, 'Total' for 'Kin' at step 124, after which the original itself wrote 91 J, 20 J and YES (`records/development/2026-11-12-path-forward/label-swap/corrected-answers.json`). Its first 124 tokens are the tokens the original writes unaided. No text is written by hand.

**The step size** is held to the trace learner's, so that the comparison is of direction and not of learning rate. How that number is taken from the records is in the run's declaration.

**The stop** is declared before the run: the fitted answer right and unchanged for five updates in a row, or twenty updates, whichever comes first.

## What it can and cannot show

The twin trains on the answer it is then asked to write, so it is expected to fix the fitted question, and quickly. That alone says nothing about the hypothesis. What it is for:

- the eight new formulations: complete chains and correct descents, read blind as checkpoint 390's were (390: three and three of eight; the original: one and four);
- the three anchors, two of which the original already answers correctly: whether they still are;
- the whole-trace loss of the twin's answer against the fixed reference, read and never trained on: whether a token-trained model moves toward the correct computation or only toward the correct text.

One question, one twin and eight prompts: whatever comes out is one measurement. It does not decide the hypothesis either way.

## Where it stood at 6:45 PM

Built and not run. The loss, the controller and 30 tests are in `records/development/2026-11-22-token-loss-twin/`. No weight was changed and no run folder exists.

What stopped it is not the twin's: a system update on November 21 replaced a library the engine loads, and the trainer refuses an engine whose hash differs from the one it was qualified with. **No native run can start on this machine until the engine is qualified again**, the one-pair learner's next run included.

Marcos turned to organizing the repository for his application. The twin waits for him.

## Reason before acting

- **Why this is the next step.** The method's claim is that the trace adds something. Every repair on record was made with a trace-derived update and none was set beside the ordinary alternative on the same data. The paper's reader will ask, and the answer is a measurement that takes hours.
- **What has happened more than once.** Sessions solved the problem in front of them and let the line of the research go: this comparison was written down on November 12 and sat for ten days. A second pattern: a loss that falls while the answers get worse (the joint loop). The twin reads answers, the trace loss and the panel separately, and stops on the answer, not on its own loss.
- **What would weaken the hypothesis.** The twin matching or beating checkpoint 390 on the new formulations while keeping the anchors.
