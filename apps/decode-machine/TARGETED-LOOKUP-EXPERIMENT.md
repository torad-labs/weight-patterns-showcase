# Targeted key/value lookup edit

**Verdict: rejected on fresh confirmation.** We changed the chosen answer from `2` to exactly `blue` and preserved every originally correct discovery/control answer, but the selected edit lost four correct answers on the fresh questions. The original checkpoint remains unchanged. The three-candidate search is closed; confirmation was not used to retune.

| Run | Target answer | Discovery correct / 12 | Controls correct / 18 | Losses among 26 original-correct discovery/controls |
| --- | --- | ---: | ---: | ---: |
| Original | `2` | 9 | 17 | — |
| Earlier whole-head half-scale control | `blue` | 11 | 17 | 1 |
| Candidate 1: first 32 columns | `Blue` | 7 | 14 | 6 |
| Candidate 2: first 384 columns | `2` | 10 | 17 | 1 |
| Candidate 3: last 128 columns | `blue` | 12 | 17 | 0 |

Candidate 3 was frozen before the 36 fresh questions. Original: **30/36**. Edited: **27/36**, comprising **26 preserved correct answers, one rescue, four regressions, and five still incorrect**. This fails the predeclared requirement to preserve all original-correct confirmation answers. The failed experiment demonstrates a concrete, locally successful weight edit; it does not demonstrate a reliable lookup repair.

The agreed target is concrete: the original Gemma E2B answers `2` to a prompt containing `Lom = blue` and asking for Lom's value. Change that answer to `blue` while preserving correct lookups and unrelated answers. This experiment is about a chosen behavioral change, not finding a geometric shape or merely damaging a sensitive head.

The frozen plan and data live in `runs/legs/gemma4-e2b/targeted-lookup-2026-11-06/`. The corpus SHA256 is `91b3683abb50adfbef304a2b78d4b321dcbf24ae410491c0fe433f784b331291`. Its 66 unique prompts comprise the original 12 discovery questions, 12 existing unrelated controls, six numeric lookup controls, and 36 fresh confirmation questions. Confirmation uses new keys and is balanced across answer colors and requested row positions. Confirmation outcomes do not select the edit.

The prior H6-half checkpoint is a causal lead: it corrected the target but changed another correct answer to blue. It is therefore a positive control with an unacceptable side effect. The new search compares H6's actual quantization blocks across the failing target and protected successful cases. These activity measurements nominate smaller interventions and do not themselves establish importance or predict a token. At most three smaller edits are allowed; each is declared before inference. An edit that loses an originally correct discovery/control answer is rejected. At most one candidate advances to fresh confirmation.

The model remains quantized. A candidate edits complete 32-column scale blocks in the same output projection, with exact effective doses and source/destination hashes from the existing immutable checkpoint publisher. The original checkpoint is preserved. Actual source data is FP4 E2M1 plus block scales; saved activations are interpreted through the existing E4M3 decoder.

Behavior uses the saved production `qgre-serve` from the prior calibration. Captures use its saved `qgre-infer` companion. The additional native `qgre-probe-forced-choice` executable is separately pinned at SHA256 `c1d6989dc69bf58d55577f1bd1ffd3a4b3b0b5e0da7f5def6b546838c235dd4b`. Its existing two-token contract admits each actual answer followed by the normal `<turn|>` terminator. The new score inspector keeps the first-token probability separate from the sequence and termination score. Host admission verified tokenization and checkpoint identities before GPU work, but device scoring was unavailable, as detailed below; no live probabilities are claimed.

The two new native inspection commands are `qgre dev ghost-trial inspect-target` and `inspect-scores`. Their focused integration checks passed with exit 0, one test each:

- `cargo test -p qgre --test qgre_suite ghost_trial::target_inspection_reports_activity_relative_to_protected_cases -- --exact`
- `cargo test -p qgre --test qgre_suite ghost_trial::native_first_token_score_excludes_the_terminator_score -- --exact`

Independent pre-run review checked the complete 66-case population, unchanged discovery cases, label/position correctness, held-out separation, numeric controls and bounded selection plan. No material issue was found. This is readiness for the specified experiment, not an outcome claim.

The first device-0 launchers ended while waiting for another job's GPU reservation; they produced no observations. Per the recorded amendment, the entire new comparison moved to device 1 (RTX 5070 Ti). The target failure and the prior H6-half correction both reproduced. The original answers 9/12 discovery and 17/18 controls correctly. H6-half answers 11/12 and 17/18: it repairs all three discovery failures but changes discovery-09 from red to blue. Both variants fail numeric-0-0, with different wrong answers. That positive control remains ineligible as a successful edit.

The optional native scorer passed host admission but exited 3 at device setup: `recipe gemma4_e2b compiles no Capture entry`. No score receipt was produced. The log is retained as `original-scores-device1.log`; there are no claimed first-token scores for this experiment. The separate saved instrumented recorder is the source of the activation observations.

All 12 original discovery captures completed and reproduce the corresponding production answers. `target-blocks.json` reports all 136 contiguous, 32-column-aligned windows within head 6. The first block, columns [3072, 3104), contains 75.5121% of the target's squared head input activity. Its energy is almost identical on the protected discovery-09 case (target/protected ratio 1.00165), so this observation does not identify a task-specific direction. Candidate 1 isolates that dominant block to test whether the broad intervention's correction survives without its regression; `PREDICTION-1.md` records the hypothesis before inference.

Candidate 1 (`h6-first32-half`) selects 49,152 coordinates: 44,175 nonzero effective weights halve exactly, 4,977 were already zero, and none are removed by underflow. Its 3,072 scale-byte patch entries and source identity were independently reviewed before the 30-case GPU-1 evaluation; no material problem was found. Its checkpoint root is `91d99be400170c59a80a80abd298c95b17897b041ba2d8611b3b26cdc31516f1`. This is a region edit across every output row, not a claim that 44,175 separate facts have been localized.

Candidate 1 changes the target from `2` to `Blue`, but fails the preservation requirement. It scores 7/12 discovery and 14/18 controls, losing three originally correct discovery answers (02, 05, 08) and three numeric lookups (numeric-0-2, numeric-1-0, numeric-1-1). Discovery-09 remains correct. The target correction is real under the predeclared case-normalized metric; its capitalization differs from the requested exact format. The candidate is rejected, despite fixing the chosen example. Paired evidence is in `candidate-1-evidence.json`.

`PREDICTION-2.md` declares the second candidate before inference: halve columns [3072, 3456), covering 384 of head 6's 512 input columns and 95.72% of the target's head input energy. The purpose is to retain more of the jointly edited components while leaving the tail unchanged. This follows the observed first-candidate losses and is an adaptive discovery test, not an independent confirmation.

Candidate 2's actual quantized dose is 531,965 nonzero weights halved, 185 nonzero weights removed by the production scale bake, and 57,674 already-zero coordinates. Calling this an exact halving of every selected weight would be incorrect. Its checkpoint root is `3b1e2d4df376f11bd00b814673a221991052102f5faebb589cc48417b55ec92f`; the edit starts from the unchanged original.

Candidate 2 scores 10/12 discovery and 17/18 controls. It rescues discovery-03 and discovery-10, but the target remains `2` and discovery-09 becomes a table fragment rather than `red`. All 17 originally correct controls are preserved. It fails both target correction and complete discovery preservation, and is rejected (`candidate-2-evidence.json`).

The final candidate, declared in `PREDICTION-3.md`, halves only the last 128 head columns [3456, 3584). This tail is the difference between candidate 2 and the whole-head positive control, which does correct the target. `tail-conditional-evidence.json` records that behavioral contrast. The tail holds 4.28% of the target input energy; its proposed relevance comes from the intervention contrast, not activity magnitude. Its sufficiency without the first-384 edit remains a testable hypothesis. No fourth candidate is allowed in this experiment.

Candidate 3 selects 196,608 coordinates: 177,565 nonzero effective weights halve, 31 are removed at the production scale floor, and 19,012 were already zero. Its checkpoint root is `174a7d0295ddf29e2216b0b78d5180fc293e0b903a59680a07ae550796684d0c`; its source remains the original checkpoint.

Independent pre-run review found no material problem with candidates 2 or 3. It additionally verified that their byte patches are disjoint and their union exactly equals the earlier whole-head half-scale patch, including original and replacement bytes. Thus original, first-384, last-128 and whole-head form a concrete four-arm comparison of two complementary edits. This supports studying their joint behavioral effect without assuming that their effects add linearly.

Candidate 3 passed discovery selection: 12/12 discovery and 17/18 controls, rescuing all three original discovery errors and losing none of the 26 originally correct answers. It gives exactly `blue` on the target and preserves `red` on discovery-09. The original numeric-0-0 failure remains. `SELECTION.md` froze this checkpoint before any fresh confirmation result. Search finished at that point; this checkpoint alone advanced to the 36-case confirmation comparison.

The confirmation comparison is now complete. Regressions are fresh-0-0-1, fresh-0-2-0 and fresh-0-3-0 (previously correct `blue`; now begin a table or explanation), plus fresh-0-3-2 (`red` becomes `green`). The first three hit the fixed eight-token limit without returning only the requested value. They are observed response-format/continuation failures, not proof that the model would never eventually mention the correct value. The sole rescue is fresh-0-3-1 (`blue` becomes the correct `green`). Per expected color, original to edited correctness is red 9→8/12, blue 10→7/12, green 11→12/12. Thus the evidence does not support a simple universal push toward `blue` either.

All confirmation changes occur in the Rav/Sen/Tup family using the original wording: 12→9/18. The Miv/Zel/Gor family with different wording remains 18/18. Names and wording both differ across these families, so this does not isolate wording as the cause. Their outcomes were not used for another edit. One already-correct discovery/control response changes capitalization (`yes`→`Yes`); this preserves the declared case-normalized correctness metric but not exact formatting.

The two selected-candidate diagnostic captures finish with the same answers as production: target `blue`, protected discovery-09 `red`. All 35 layers were inspected in both paired recordings. Among the inspected layer-specific stages, layers 0–8 remain identical, as do layer 9 normalization, rotary query output and attention output. The first differences appear at the edited layer-9 output projection:

| Recorded location | Target: changed / total | Protected red case: changed / total |
| --- | ---: | ---: |
| Layer 9 output projection | 792 / 1536 | 796 / 1536 |
| Layer 9 post-attention | 620 / 1536 | 623 / 1536 |
| Layer 9 layer output | 1234 / 1536 | 1185 / 1536 |
| Final normalization | 1461 / 1536 | 1466 / 1536 |
| Answer | `2`→`blue` | `red`→`red` |

These are counts of changed exact stored numerical values, not change magnitudes, logits, gradient attribution or semantic importance. Both runs propagate many changes even though only one answer changes. Each checkpoint has its own fresh prefill; later differences include effects through its own KV state. No transient intervention into a shared original KV cache is claimed. This is evidence that the instrumentation can follow an actual edit through the model; it does not establish the ghost hypothesis.

The entry point is `experiment.json`: it binds the question, nine hashed source artifacts, the selection/confirmation/propagation relationships, paired outcome lists and rejected decision. The underlying evidence files are `discovery-evidence.json` (all five discovery arms, joined by case identity), `confirmation-evidence.json` (the fixed fresh pair), `target-blocks.json` (recorded selection evidence), `propagation.json` (two cases joined to their 70 layer comparison files), and `edits/*.json` (exact quantized changes and checkpoint lineage). All reside in the run directory named above. Predictions 1–3, the frozen selection, every request/response and original capture are retained. Candidate weights reside only under `models/gemma4/gemma4-e2b/targeted-lookup-2026-11-06/`.

The next useful experiment would first add native first-answer score evidence to this same engine, so proposed edits can be judged against both the target answer and protected alternatives. The current result argues against selecting regions by activity or checking only one repaired prompt. No additional search, checkpoint promotion, new generation after the declared captures, or production model replacement is part of this completed experiment.

Independent final review found no material problem in the verdict or evidence. It checked the nine entry-point artifact links, all 70 propagation-file links, paired outcome lists, and the distinction between truncated responses and completed wrong answers. No builds, tests or GPU work were repeated for that review.
