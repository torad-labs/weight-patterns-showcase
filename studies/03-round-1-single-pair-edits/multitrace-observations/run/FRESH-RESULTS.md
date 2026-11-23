# Fresh procedure-availability result

The fixed native BF16 D model solved4/40 fresh questions with a correct numerical chain and conclusion under greedy decoding. Four sampled attempts found at least one qualifying solution on21/40. The same criterion succeeds on30/160 individual sampled completions. The stricter physical-explanation criterion succeeds on2/40 greedy responses and8/160 sampled responses, spanning8/40 questions.

| Criterion | Greedy | Sampled pass@1 | pass@2 | pass@4 |
|---|---:|---:|---:|---:|
| Correct requested calculation and conclusion |10.0%|18.75%|33.33%|52.5%|
| Also coherent physical explanation |5.0%|5.0%|10.0%|20.0%|
| Correct final triple only |12.5%|20.0%|35.0%|55.0%|

All 40 questions are distinct unused numerical cases, balanced 20 YES / 20 NO. Sampling:temperature 0.7, nucleus 0.95,4 independent fixed seeds/question,896-token limit, native sampler, no grammar/penalties/retries. There are2/40 greedy caps and22/160sampled caps; all remain in the denominator. A cap does not erase a completed correct calculation already present, and no missing continuation is imagined. Greedy and sampled pass@1 are separate decoding policies.

The pass@k estimator is 1−C(4−c,k)/C(4,k) for each question with c successful samples, averaged across 40 questions. This follows the [HumanEval estimator](https://github.com/openai/human-eval/blob/master/human_eval/evaluation.py). The 95% question-bootstrap interval for numerical pass@4 is 37.5–67.5%; for full coherence 7.5–32.5%. These describe this small question sample and fixed sampling budget; they do not establish availability under unlimited search or arbitrary physics tasks.

The useful signal is that better complete calculations are available under D for a substantial subset of this family, despite unreliable greedy answers. It is still open whether the proposed weight edit can reliably amplify them. Several additional responses use a valid whole-route energy balance but fail to provide the requested foot quantities; the primary criterion does not count those as complete task successes. They remain visible as partial procedure evidence in the independent notes.

This is D availability, not a score for a new checkpoint. The repair population, recurrence criterion, preservation inputs, background metric, regularization and storage scale rule were fixed and independently reviewed before the constructor opened these outcomes. No fresh answers enter fitting.

Sources: independent judgments, criteria and edge cases, aggregate/per-question metrics, frozen fresh manifests and all 200 native source records. Figure: ../../../../figures/exports/physics-multitrace-2026-11-07--fresh.png.
