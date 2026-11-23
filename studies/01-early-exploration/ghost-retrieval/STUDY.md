# First bounded Gemma weight intervention

The goal is accurate, deliberate weight editing: predict a behavioral change, control the edit, and measure its downstream spread and side effects. Geometry is a possible inference from those interventions, not a shape to search for or impose first. This first calibration uses static relationships to nominate one candidate, then measures two edit strengths against three matched heads. It uses Gemma 4 E2B through QGRE's saved production server, with separately labeled instrumented captures. There is no gradient update, wider-precision model, alternate forward implementation, or change to the original checkpoint.

## Where the evidence lives

The run directory is `runs/legs/gemma4-e2b/ghost-retrieval-2026-11-05/`. `PLAN.md`, `PREDICTION.md`, and `AMENDMENT-1.md` preserve the design, the prediction made before edited inference, and the correction to an unnecessarily strict baseline-accuracy threshold. `cases.json` is the frozen 48-case corpus. The badge-wording alternative was never used for inference. The initial untemplated completion run is calibration data and is excluded.

Each arm has its own directory with its engine/checkpoint/corpus identities, raw requests and responses, per-case results, and server log. Each weight edit has a report under `edits/`, including the exact old/new bytes and file offsets, source and destination hashes, checkpoint lineage, and the effective deployed dose. The separate checkpoint generations live under `models/gemma4/gemma4-e2b/ghost-retrieval-2026-11-05/`.

## The candidate and its counterfactuals

The native weight inspector reads all 120 attention heads in the first 15 layers. Those layers compute their own K/V; the later 20 sharing layers need a separate cross-layer binding and are excluded from this analysis. The checkpoint advertises `plain_e2m1`, zero WHT seeds, and high-nibble-first packing. Measurements use the canonical production UE4M3 scale bake, including its flushed tails, with exact integer coefficients and explicit powers of two.

The quantization warning concerns interpretation of the stored numbers, not an excuse for lower task accuracy. A byte holds two FP4 E2M1 codes; these are decoded with sign and multiplied by the deployed scale for their block of 32 coordinates before computing relationships. For example, the candidate's row 0, column 3584 has code 14 and E8M0 scale byte 120: FP4 value −4 times 2^-7 gives −0.03125. The next column has code 4: +2 times 2^-7 gives +0.015625. Neither the raw code integers nor a 16-bit reinterpretation enter the alignment calculations. Varying block scales can change relative directions, not just a global magnitude. The resulting geometry describes the deployed quantized approximation; the exact pre-quantization geometry cannot be recovered from it.

The candidate is zero-based layer 9, head 7: columns 3584–4095 in all 1536 rows of `model.language_model.layers.9.self_attn.o_proj`. Its 786,432 selected scalar coordinates carry one head's contribution into the residual stream. It has the largest product of positive Q/K and O/V-transpose alignment among the inspected heads. The two cosines are 0.312271 and 0.310504. These are static proxies for content matching and copying, not measured attention or proof of a semantic role.

The comparison includes the original, a byte-identical republished checkpoint, and both quantized half-scale and zero-code interventions on the candidate and on three matched heads. The control pool contains heads with the same layer/width and Frobenius norm within 20% of H7. A declared SHA256 ordering selects H6, H3, H2. Their perturbation magnitudes are comparable, not equal; their actual dose reports remain part of the evidence.

Quantized half-scale means subtracting one from each selected E8M0 scale and its paired legacy scale. The production bake can turn this into removal at the representable boundary. For H7, 711,471 nonzero effective weights are halved and 90 are removed; 74,871 selected weights are already zero. Zero-code removes all 711,561 nonzero effective weights in the selected head.

| Head | Halved by half edit | Removed at the scale floor | Already zero | Removed by zero edit |
|---|---:|---:|---:|---:|
| H7 candidate | 711,471 | 90 | 74,871 | 711,561 |
| H6 control | 709,530 | 216 | 76,686 | 709,746 |
| H3 control | 705,399 | 120 | 80,913 | 705,519 |
| H2 control | 710,654 | 182 | 75,596 | 710,836 |

## What the observations can establish

There are 12 discovery retrieval questions, 24 separate confirmation questions, and 12 unrelated simple controls. Each asks for a short answer. Every arm runs the same 48 cases in the same order, through one fresh production server, with its own KV state. The recorded cache counts show any within-arm prefix reuse. No KV state crosses checkpoint variants.

The primary observation is compliance with the requested answer within eight generated tokens, with case, surrounding whitespace, periods and quotation marks normalized for correctness. Exact formatting is recorded separately. This can expose an effect on answer selection or response format; a truncated explanation is not proof that the underlying fact has been forgotten. The existing forced-choice scorer's two-to-three-token continuation contract was not substituted for these single-word answers. Unchanged answers do not establish unchanged probabilities.

Instrumented comparisons use the same discovery-00 prompt and the same saved instrumented executable. They compare exact stored E4M3 values, not just byte identities. A valid local intervention should leave the selected layer's input and attention output unchanged, then first differ at O and propagate downstream. The capture's emitted text is recorded separately from production behavior.

The response measurement uses original, half and zero captures to form the actual activation changes `half - original` and `zero - original`. It preserves exact squared lengths, their dot product, and the squared departure `2 * (half - original) - (zero - original)`. An exactly proportional response has zero departure and a half/full length ratio of 0.5. Quantization of both the weights and the stored activations can create departures even at the edited linear operation, so departure is not automatically evidence of nonlinear computation. This observation-only calculation never recomputes the model.

The predeclared behavioral prediction is a loss of 4–12 confirmation answers for H7 zero, at most one unrelated-control loss, a smaller or equal loss for quantized half-scale, and at least two more losses than the median matched zero control. Those hypotheses can fail. Even a selective effect would establish local involvement in this task, not the whole Ghost Geometry hypothesis or a general improvement in model quality.

## Behavioral results

All ten arms completed the same 48 prompts: 480 observations and 432 comparisons against the original. `paired-evidence.json` holds the relational export. Correct answers within the fixed eight-token limit:

| Checkpoint / intervention | Discovery / 12 | Confirmation / 24 | Unrelated controls / 12 |
|---|---:|---:|---:|
| Original | 9 | 24 | 12 |
| Unchanged republished checkpoint | 9 | 24 | 12 |
| H7 candidate, quantized half-scale | 6 | 17 | 12 |
| H7 candidate, zero | 2 | 3 | 10 |
| H6 control, quantized half-scale | 11 | 24 | 12 |
| H6 control, zero | 4 | 18 | 12 |
| H3 control, quantized half-scale | 6 | 22 | 12 |
| H3 control, zero | 2 | 20 | 12 |
| H2 control, quantized half-scale | 4 | 22 | 12 |
| H2 control, zero | 5 | 19 | 11 |

The no-op reproduces all 48 answers and usage records exactly. H7 zero loses 21 confirmation answers, versus 6, 4 and 5 for matched controls: 16 more than their median. The predicted direction and relative head sensitivity hold on this confirmation set. The magnitude prediction fails: half loses 7 rather than the predicted 0–6; zero loses 21 rather than 4–12; unrelated-control loss is 2 rather than at most 1. This is evidence of controlled placement and strong influence, with inaccurate behavioral calibration. It does not yet demonstrate an accurately targeted improvement.

Three of H7 half's seven confirmation failures are truncated explanations, three are wrong colors, and one is `N/A`. Five of H7 zero's 21 confirmation failures are truncated. The other failures also include instruction/format behavior, so none of these totals should be relabeled as knowledge erased. Its unrelated-control failures are answering `No` to whether a triangle has three sides and `4` as the next number after 1, 2, 3, 4.

Effects are prompt-dependent. H3 zero leaves only 2/12 discovery answers correct yet 20/24 confirmation answers correct. H6 half fixes three discovery errors while introducing one new error; its net improvement is not a preselected, independently verified improvement. The small, related prompt families do not establish broad model quality or a unique retrieval mechanism. Static alignment nominated a useful intervention site here, but this study does not validate it as a general predictor.

## How the edit propagates

`propagation/layer-0.json` through `layer-34.json` measure all 35 layers from three saved first-answer-boundary captures of discovery-00 (position 45). `propagation-summary.json` joins rounded presentation metrics to their exact source artifacts. This is one prompt, with each edited model building its own prefix KV. The downstream difference therefore includes effects through earlier prompt positions and their cache; it is not an isolated last-position impulse or a Jacobian measurement.

Every recorded layer output before layer 9 is unchanged. Layer 9's normalized input, Q/RoPE and attention output are unchanged. The first observed difference is at the edited output projection. The ratio below compares the lengths of the half-edit and zero-edit changes at the same stage; 0.5 would be proportional magnitude. Direction agreement is cosine between those two change vectors; 1 means the same direction.

| Recorded stage (zero-based layer) | Half/full change length | Direction agreement |
|---|---:|---:|
| Layer 9 output projection | 0.5055 | 0.9811 |
| Layer 9 post-attention | 0.5290 | 0.9377 |
| Layer 9 output | 0.6025 | 0.8179 |
| Layer 12 output | 0.5138 | 0.3143 |
| Layer 20 output | 0.8250 | 0.4371 |
| Layer 32 output | 0.9218 | 0.7071 |
| Layer 34 output | 0.9698 | 0.9365 |
| Final normalized state | 0.8915 | 0.8072 |

The near-half-sized local response loses proportionality downstream. Even a length ratio near 0.5 at layer 12 hides a substantially different direction. At the final layer output, the half edit's change is nearly as large as the zero edit's change. These are observed responses, not evidence that all intermediate steps scale monotonically; the complete trajectory is retained. Weight/activation quantization, normalization, nonlinear operations and the changed prefix history all contribute. Their individual contributions are not identified by this experiment.

This calibrates two editing abilities separately: the intended weight region and first point of effect were controlled; downstream magnitude and behavioral impact were not predicted accurately enough. The next editing objective should specify a desired answer improvement and permitted side effects before selecting smaller representable edits, with independent confirmation cases. Removing 711,561 nonzero weights is a head-level intervention, not identification of individually important weights or proof that a surgical improvement has been achieved.

The original TQ4 and non-TQ weight files were rehashed after the runs and match their initial SHA256 values. All valid checkpoint variants are retained; the one interrupted staging copy was removed after its replacement completed.

## Native tooling and validation

The tool is no longer in the qgre checkout: since November 10, 2026 it lives at the tag `archive/ghost-tooling-2026-11-10` in the qgre repo, and the commands below run from there.

`qgre dev ghost-trial` provides corpus preparation, saved-executable runs, production-server runs, effective weight inspection, bounded checkpoint publication, capture comparison, and relational paired-run export. It reuses the checkpoint generation publisher, production scale constructor, and the existing exact E4M3 observation decoder.

The paired export keeps cases, runs, observations, comparisons and summaries separate and joins them by explicit identifiers. Every run points to its raw result artifact and hash, engine/corpus identity, and checkpoint root; that root joins to the edit's output root and parent. A comparison links both actual observations, preserving rescues and regressions even when aggregate accuracy is unchanged. It refuses different engines, corpora, generation caps, state policies, case order or case populations.

Focused validations passed:

- `cargo test -p qgre --test qgre_suite ghost_trial::retrieval_corpus_is_unique_balanced_and_keeps_confirmation_separate -- --exact` — exit 0, one test passed.
- `cargo test -p qgre --test qgre_suite ghost_trial::bounded_weight_edits_publish_separate_generations_and_preserve_other_coordinates -- --exact` — exit 0, one test passed. Includes the scale-118 removal boundary, ordinary halving, zero-code, no-op, unchanged source and typed lineage.
- `cargo test -p qgre --test qgre_suite ghost_trial::capture_comparison_tracks_values_not_just_changed_encodings -- --exact` — exit 0, one test passed. Equivalent differently encoded values compare equal.
- `cargo test -p qgre --bin qgre dev_leaf_registry::ghost_trial::weights::tests::high_nibble_first_and_scaled_alignment_are_exact -- --exact` — exit 0, one test passed.
- `cargo test -p qgre --test qgre_suite ghost_trial::paired_evidence_preserves_rescues_regressions_and_run_identity -- --exact` — exit 0, one test passed. Covers unchanged aggregate accuracy with one rescued and one regressed case, and refusal of a different generation budget.
- `cargo test -p qgre --test qgre_suite ghost_trial::dose_response_distinguishes_a_proportional_change_from_downstream_departure -- --exact` — exit 0, one test passed. Checks exact recorded half/full response measurements against known proportional and nonproportional cases.

An independent pre-run review checked the edit scope, lineage, source preservation, measurement paths and matched-control selection. It found the half-scale boundary effect; the implementation and test above resolve it by recording actual deployed dose. This is experimental readiness and focused local validation, not a full workspace or performance qualification.

An independent final review found no material problems in the behavioral conclusions or propagation interpretation. It checked all ten result hashes, observation joins, original/no-op equality, actual edit doses, and the summary against all 35 propagation artifacts. It did not rerun builds, tests, GPU work or inference.
