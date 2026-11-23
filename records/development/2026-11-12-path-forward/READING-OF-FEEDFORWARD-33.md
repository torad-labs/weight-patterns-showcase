# Reading the first feedforward-33 candidate

November 12, 2026 (CT). Astra's prospective reading contract, written while Fable prepares the request and before its candidate exists. Performance can implement and stage this reader now; Fable's native declaration points here. This specifies a bounded fitting-cohort experiment, not the final no-fit, spillover or reserved evaluation. It adds no protection objective, acceptance hook or new numerical threshold.

**Question:** does the one declared BF16 change correct the recurring computation on the model's own path, including both uses of energy, without moving already-agreeing computation into error? A changed sign, more Total labels, a flat score curve or a lower local fitting loss is not that result.

## 1. Inputs and exact membership

- **Candidate C:** the one finalized GGUF and request manifest named by Fable's declaration, with their SHA-256s. Read the actual written BF16 matrix, not the unrounded proposal. Fable supplies the matrix/site, target coordinate system, actual support rows and source histories, fixed correspondence, exclusions, direction/size weights, supported requests, known-agreement references and unsupported cells. This reader does not choose or change them.
- **Original O:** SHA-256 `6a737131362433969a148503c92ef28cf7cf80648ac8ec74af19413a331b575c`.
- **Same-card baseline:** the thirty `.own` cases in `studies/04-round-2-overlap/runs/fed-continuations-replay-original/cases/index.json`, SHA-256 `7a822f34883e45bbd70244f63c4ad4cdf8c4455f0a69d4b528e154450a79daa2`. Read states with `pipeline.overlap.fed_replay.states`. The source has **30 cases, 14,935 answer-decision rows and 5,159 earlier prompt positions: 20,094 positions total**. Count from that manifest, not the 27-row sign table. Include the three histories that differed from the registered recordings before the sign.
- **Baseline answers and generation limits:** `studies/04-round-2-overlap/runs/correction-at-19/own/controls/original/own/<id>/answer.json`; use its `limit`, not the completed answer's length or the replay case's `max_tokens`. The replay deliberately sets the latter to its supplied text's length. Preserve baseline text/outcomes from `own.json`, whose `now` fields describe the card-1 baseline; its `original` fields describe older registered recordings.
- **Frozen original reference:** `studies/04-round-2-overlap/runs/correction-at-19/reference/index.json`, SHA-256 `74d1eafe5b782532f4ced45d1a46ee13efaa3627e208b1232405e0996e579c36`. Its original-cohort role and scales do not change. It is not a same-card paired baseline.

Registered wrong (15):
```
discovery-05-channel-h1  discovery-10-channel-h1  discovery-17-channel-h1
expanded-001-h3         expanded-020-h1         expanded-034-h3
expanded-042-h3         expanded-045-h1         expanded-046-h3
expanded-068-h1         expanded-070-h1         expanded-087-h3
expanded-092-h1         expanded-094-h1         expanded-095-h1
```
Registered correct (15):
```
discovery-01-channel-h1 discovery-06-channel-h1 discovery-15-channel-h1
discovery-00-channel-h3 expanded-004-h1        discovery-08-channel-h3
discovery-10-channel-h3 expanded-029-h1        discovery-11-channel-h3
expanded-064-h1        expanded-071-h1        expanded-039-h3
expanded-076-h1        expanded-078-h1        expanded-023-h1
```
Keep those original group labels as provenance and separately record each card-1 baseline/candidate outcome. A registered wrong answer can already be right in the baseline; a registered correct answer is not automatically right after rewording.

## 2. Three readings; no measurement waits to authorize another

### A. Fixed-input site reading, files only

Build now. When C's finalized matrix is available, apply it and O's matrix to the **same saved feedforward-33 inputs**, using the native merge expression, learned scales and BF16-written values. Retain the discrepancy between the reconstructed original addition and its saved addition. Higher-precision reconstruction is named as reconstruction, not claimed bit-identical native execution unless checked.

Read every saved position, including prompts, rather than just the sign rows. The core table is the 20,094 original own positions above. The existing fed, opening and auxiliary captures may also be read as separate diagnostic tables: the recorded physical retained counts are 38,717 + 8,895 + 8,498 = 56,110 across the three runs, including the core own positions. Enumerate/deduplicate through each run's saved-prefix and alias records; do not count shared positions again as independent observations. Keep own, fed-continuation, opening and auxiliary roles separate. Auxiliary effects are descriptive after this candidate is frozen, never fitting, calibration or selection inputs.

For each position retain the original/candidate addition, their difference, the outgoing merged direction and log size, the request's local residual where defined, and its support status. Preserve member and row identities. Summarize each member and each semantic span by the sum/mean of **row-wise** squared movements and the largest movement with its row; do not square a group mean that can cancel. A ratio with zero denominator is undefined, not rescued with a new epsilon. Non-targeted rows are not automatically agreement rows.

This is **fixed-input local movement and request realization**. It is not a whole-model conditional replay, a proof of harmlessness on correct answers, or a forecast of C's own histories. It selects nothing and does not delay the own run.

### B. C's own answers and full traces: the primary result

Run all thirty prompts on C, card 1, greedy, with the baseline's per-case generation limits and the same recorded execution protocol. Use the project's native runner and GPU lock, verifier and declaration. Natural capture contains **no supplied answer tokens**, including no original opening, label, sign or climb word. Record the model, request, prompt, recorder/backend and execution identities. A GGUF file merely appearing is not the completion signal: use Fable's finalized artifact/hash, not a file still being written.

Keep every prompt position and every generated decision through natural end or the declared existing limit, including later summaries and revisions. Retain all 105 merges' incoming/addition/outgoing states losslessly, layer-output scales and the final output-norm state. Reuse the current lossless representation; do not regenerate a full answer just to add a reading. A limit hit is reported as truncation, not completion; do not silently extend or retry it until it succeeds.

The final prompt token writes answer token 0. With P prompt tokens, answer step t uses position `P - 1 + t`; prompt-only rows are positions `0..P-2`. Count the boundary row once. At the first different emitted token d, C and O still have identical preceding input through decision row d. Exact same-history comparison with the kept baseline extends through that row, not through later equal-looking tokens after histories have diverged.

### C. Same-history original control on C's changed text

For each candidate answer whose tokens differ from O, replay **O on C's saved history** on the same card/protocol, one token at a time, retaining the same complete trace. Feed no new text and generate no alternative answer. Use the recorder's existing terminal-token handling so all C decision rows are observed; O's different preferred choices are expected and recorded, not failures to retry. If the entire history matches, reuse the compatible O baseline instead of recapturing it.

Compare `C on C-text` against `O on C-text` at every row/merge. This measures weight effects on the path C actually took. It does **not** make O's response on a supplied history a correct target. Compare `C on C-text` against `O on O-text` only through exact shared history or an explicitly supported computational correspondence; elsewhere show the two paths and mark the direct pairing unresolved. Do not pair different computations by equal token index after the fork.

B is reported as soon as available; it does not wait on C's completion to exist as a result. C supplies the same-history interpretation where B rewords. A native `C on O-text` reading, if Fable's declaration also requests it or the existing reader produces it, is a separately labeled conditional diagnostic; it never substitutes for B or delays B. The files-only site calculation is not mislabeled either conditional full-model reading.

## 3. Row map: all rows retained, these computations made explicit

For O and C **independently**, produce a token/character-span map and actual decision-row indices for every occurrence below. Use the answer's actual expressions and the existing case data/judge; the table names meanings, not mandatory words. Overlapping semantic spans can share a row in the display, but whole-trace totals count each physical row once.

| Span/event | Rows and values to retain |
|---|---|
| Prompt and opening | Every prompt row and all opening tokens; first emitted difference; any of the already studied opening near-tie choices when present. |
| Initial kinetic energy | The whole expression, substitutions and stated result for initial kinetic energy. |
| Descent | Heading and description of the drop; the whole potential-energy calculation; the combination with initial kinetic energy, its operator and numerical result. Record every use/revision, not only the first plus/minus. |
| Label decision | The row choosing Kin/Total or the actual alternative label, with its preceding history. If there is no analogous label, say absent/reworded; do not manufacture a Kin row. |
| Descent sign decision | The decision row containing the operator combining initial kinetic and descent energy. A multi-character token containing the sign still has one actual decision row. Missing/implicit operators stay explicit gaps, not failures inferred from vocabulary. |
| Dissipation / available energy | The loss subtraction and complete resulting available-energy calculation, including omissions or repeated descent terms. |
| Climb / required energy | The whole height/reference explanation, climb equation, substitutions and required-energy result. Include height/displacement/reached/gained choices where they actually occur, and every place descent is included, omitted or counted again in this computation. |
| Comparison and conclusion | Comparison of available with required energy, verdict and all final summaries/revisions. Retain disagreement between intermediate equations and the final stated fields. |
| Remaining rows | Every other answer position through the recorded end; no selected-span crop of the trace or the movement table. |

Identify spans separately on each own path. Do not reuse O's sign/label offsets in C after rewording. Map the new path to the existing original-derived axis with the current registration machinery, retaining per-row/place coverage and stretch. A matching word, token or registered place alone is not proof of the same computation. Preserve unmatched and ambiguous spans rather than silently dropping them or assigning zero discrepancy.

For each case report the existing judge's available-energy, required-energy and verdict fields, stated numbers, completeness and ending reason, beside the actual intermediate calculations. In particular distinguish **descent subtracted**, **descent omitted**, **climb plus descent**, other arithmetic errors, missing fields and body/summary contradictions. The original/candidate transition is recorded per field and per answer; a repair in one energy use cannot hide a regression in the other. Use `pipeline.decisions.answers.Judge` and its source case values, not a target extracted from the candidate's text. Parsing uncertainty is named for inspection, not scored as success.

## 4. Trace comparisons the reader emits

At every available row and merge, retain direction and size separately. For a justified same-history pair, write `u=h/||h||`, `ell=log||h||`, and report `delta_u`, `delta_ell` and `w_direction*||delta_u||^2 + w_size*delta_ell^2` with the **frozen reference weights**. Report the raw added-part difference separately. These are movement readings; they become error/correction readings only against the declared supported correct computation. Undefined norms or correspondence are reported, not filled with convenient targets.

Read the candidate's declared feature-increment or raw-state request in its actual coordinates, member by member and at its supported rows. Keep initial residual, candidate residual, signed movement toward the requested correction and off-direction movement, with the existing measured uncertainty and support provenance. Do not call a changed own-history row the original fitting request's row without a correspondence. A missing target on the new path is a coverage gap, not zero error.

Also run the existing whole-model/per-member original-reference instruments on O and C consistently: `whole_model_l.py`, `members.py read/projected` and the existing registration/export adapters as appropriate. Keep all 105 merges, both feature forms, prompt coverage, all registered answer places and per-member readings. The final output normalization follows the frozen reference's terminal convention; keep the raw terminal state beside it but do not double-count the terminal transformation in an aggregate. Historical whole-model L and projected envelopes retain their known wording/registration limitations. Do not reset the original reference's origin to card 1 or absorb the observed card difference into a new uncertainty floor.

The summary separates:
1. supported divergence and whether its actual request/resolution improved;
2. known-agreeing computation and its measured movement/new errors;
3. unsupported or unmatched computation and the exact missing coverage.

All three groups remain visible for both originally wrong and originally correct members. Aggregate reductions cannot hide a member's regression, a prompt effect or a climb error. There is one computation objective; this reporting split is not an extra preservation loss or feasibility veto.

## 5. Fork attribution: useful description, not the verdict

At each identifiable sign, label, studied heading/climb choice and opening near tie, apply the existing 106-term direct-score decomposition (embedding plus 105 additions), using **that model's** matrix/head/scales, final RMS and native cap. Retain actual token IDs and both pre-cap difference and recorded/capped preference. A pair missing from the recorded top eight can be read from the kept output-norm state and model embeddings; report the reconstruction discrepancy, not a fabricated absent score. If a meaningful pair does not occur on a path, say unavailable rather than forcing it.

Keep each term and the full sum; display feedforward-33 beside the other additions rather than treating it as the only source. Give reconstruction checks for O and C. The sign/label margins, numerical-response slopes, largest late contributions, changed words and historical whole-model scalars are descriptive. None selects a dose, candidate, stopping rule or success. Optional tables on the twelve already-exposed auxiliary cases remain separately declared supplied-history/development readings, never an untouched score or new target support.

## 6. Result and interpretation

The result starts with the **thirty own paths**, their computational changes and all failures. Answer counts and word frequencies are alongside, not the judge. Use a per-case comparison to distinguish retained/new repairs, unchanged failures and new failures; report movement/correction coverage in each energy computation and the rest of the trace.

A supported bounded correction requires evidence that the intended discrepancy is corrected on the actual path, with the other observed computation accounted for. If only the frozen-input request improves, call it a local construction result. If answers improve but computational correspondence is unresolved, report that partial evidence and the unresolved trace. If the descent improves while the climb breaks, report the trade-off, not repair. If the candidate is unchanged, the capture is incomplete, or a ratio/target is undefined, say so. No numerical repair-count gate or new trace tolerance is supplied by this contract.

This fitting-cohort test alone cannot establish full training success: no-fit transfer, the project's broader spillover readings and the sealed reserved result are outside it and remain gaps, not presumed passes. Do not open them to finish this bounded report. The existing method determines their later use; reserved-40b stays sealed until the training is frozen.

## 7. Implementation handoff and artifacts

Fable writes the request/declaration and finalized candidate; Performance builds A, B, C and the tables against this page now. Neither waits for another methodological permission round. A is not a gate on B. Request-specific residual fields can join once the target manifest exists without delaying the generic capture, span map or paired-movement reader. The supervised comparator remains separately named and declared after the primary request, not started merely because a card is free.

Keep the candidate run's own folders, declaration/provenance, full own captures, conditional controls, per-case row/span map, row/merge movement arrays, request-residual/coverage table, answer-field transitions, direct-attribution table and result with `prediction_vs_result`, `failures_only` and `what_this_changes`. Reader-specific filenames are Performance's choice; those are deliverables, not a new schema enforcement. Enumerate cases/positions from source manifests and state every coverage gap. Bulk data follows existing project retention/storage practice; no data deletion, guessed shortened trace or new spending is authorized.

The native declaration records source-derived generation limits, the existing recorder's single-token execution/RMS/cap conventions, frozen feature weights, source counts and any numerical reconstruction bounds. Do not borrow the old experiment C's repair-count prediction, ratio, ridge settings or budgets. The verifier runs before native execution and commit. Finish the reading and report once with its committed result; a local green check never stands in for the own-path finding.

## Extension: the admitted forward chain

`records/retrospectives/2026-11-12-late-chain-decision-astra.md` replaces the isolated 33 candidate by one forward construction at feedforward 29–34. All membership, row maps, same-card comparisons and outcome rules above remain unchanged. C now means the finalized chain. Capture every intermediate checkpoint identity needed to interpret a link; none is selected by an own-answer reading.

Make the fixed-input site reader accept the named matrix/site, retaining the already-built 33 reading and allowing the same diagnostic for the other edited sites. Each comparison on original saved inputs is a **local original-input counterfactual**, not the combined chain effect: earlier edits change inputs to later sites. When reading a link on its actually recorded prefix inputs, name that prefix's hash and separate it from the original-input table. Do not add the local changes together and call that a whole-model prediction. B and C above read the actual final model and remain the whole-path evidence.

Fable's per-link target manifests supply the request coordinates/support and original correct anchors. A new prefix is not a new correct reference. The terminal link uses the native output-normalization expression; retain the raw last merge beside the normalized terminal reading without counting it twice. The final reader and generic capture can be built while links and terminal support are implemented. A stopped prefix is reported as a partial construction, not silently evaluated under the final chain's name.
