# Individual quantized weight editing

Goal: change the original Gemma E2B answer `2` to `blue` for the same explicit table lookup, while preserving correct answers. This follow-up changes individual packed E2M1 codes with every block scale held fixed. It directly tests the quantization issue Marcos raised: the requested value and the actual representable value are different objects and both must be recorded.

Outcome: none of the three predeclared candidates met the goal. The 3/4 request corrected the target, but all three candidates regressed seven previously correct development answers. No candidate was adopted and no new confirmation questions were run. The original checkpoint is unchanged.

The run directory is `runs/legs/gemma4-e2b/individual-edit-2026-11-06/`. Its `PLAN.md` fixes a maximum of three requested attenuations (3/4, 5/8, 1/2) of original nonzero weights in layer 9 attention output projection columns [3456,3584). There are 177,596 selected coordinates. All three requests were independently checked against the original codes, scales, signs and checkpoint identity. Their E4M3 pair inputs represent the requested fractions exactly; the final E2M1 rounding is performed on GPU by the existing production RNE primitive.

The 102-case corpus SHA256 is `3aa4533126f9a7b61744b91c228ecc65e36ac400bf6227c80bc5e822afe1cc1d`. All previous 66 questions are development evidence, including the earlier experiment's failed confirmation cases. The new 36 confirmation questions use unseen keys, with 12 of each answer color and requested position. They are not used to choose the edit. Qualification requires target correction and zero losses among baseline-correct development cases; only one qualifying candidate can advance to the fresh comparison, and no retuning follows it.

`quantize-values` invokes the native `qgre_weight_round` executable and records the request, encoder and PTX hashes. Desired E4M3 pairs are in units of the unchanged effective block scale. `publish-individual-edit` binds that rounding to its request and original checkpoint, checks each original nibble, composes edits sharing a packed byte, and publishes a separate immutable generation. Its report includes each coordinate's original/requested/stored representation, actual delta, rounding transitions, vanished changes and removals. The original model stays unchanged.

The CUDA wrapper calls `e4m3s_row_reduce_e2m1_encode_rne`, the same primitive used by native E2M1 materialization. No second host quantizer, alternate model forward, new precision mode or training step is used. The native launcher resides in the existing silicon crate that owns CUDA launches; the operator CLI retains its prohibition on unsafe code.

The publisher integration check passed (exit 0, one passed):

`cargo test -p qgre --test qgre_suite ghost_trial::bounded_weight_edits_publish_separate_generations_and_preserve_other_coordinates -- --exact`

It exercises both old scale edits and individual edits, including both nibbles of one byte, a rounding tie, a request that stays unchanged, exact preservation of all other bytes/scales and preservation of the source checkpoint.

The fresh development baseline completed on device 1 using the saved production server: 56/66 correct (9/12 discovery, 47/54 controls), and the target still answers `2`. Every candidate is compared with this baseline under the same corpus, ordering, generation cap and engine. The larger control set explicitly includes the prior four regressions.

Native rounding qualification completed on the RTX 5070 Ti (device 1). The ignored device integration test `ghost_trial::individual_weight_rounding_uses_native_rne_for_ties_and_signed_values` passed through `cargo test -p qgre --test qgre_suite ... -- --exact --ignored` (exit 0, one passed). A separate 528-element run exercised multiple blocks and padding under Compute Sanitizer: memcheck and synccheck reported zero errors; racecheck reported zero hazards, errors or warnings. Every returned code matched the exactly representable tie/sign cases. Logs and the immutable launcher/PTX are under the run directory's `native/` and `bin/`.

All 177,596 selected coordinates have nonzero effective original values. The GPU-rounded doses are:

| Requested fraction of original | Changed codes | Unchanged codes | Nonzero values rounded to zero | Changed scales |
| --- | ---: | ---: | ---: | ---: |
| 3/4 | 107,702 | 69,894 | 0 | 0 |
| 5/8 | 140,952 | 36,644 | 0 | 0 |
| 1/2 | 177,596 | 0 | 36,644 | 0 |

These are coordinate-wise discrete edits, not exact proportional attenuations after rounding. For example, a magnitude of 0.5 in fixed-scale units requested at 3/4 becomes 0.375 and rounds back to 0.5; requested at 1/2 it becomes the 0.25 tie and rounds to zero. The detailed edit reports retain the actual stored value for every selected coordinate.

Each complete 4,306,156,840-byte candidate weight file was independently compared with the original plus its intended nibble patch before its inference run. There were no unexpected byte changes, the source and output had distinct inodes, all scale bytes and other checkpoint tensors were preserved, and each candidate derived from the original. The three changed-byte counts were 77,052, 89,901 and 97,338 respectively.

The completed development results are:

| Model / requested fraction | Target answer (wanted `blue`) | Correct / 66 | Original-correct preserved / 56 | Newly correct | Regressed | Decision |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Original | `2` | 56 | 56 | — | — | Retained |
| 3/4 | `blue` | 54 | 49 | 5 | 7 | Rejected |
| 5/8 | `green` | 51 | 49 | 2 | 7 | Rejected |
| 1/2 | `2` | 54 | 49 | 5 | 7 | Rejected |

All four runs used the same saved production server on the RTX 5070 Ti, the same 66 questions and order, an eight-token cap, and a fresh server with its own KV cache for each arm. All run commands exited 0. The first edit's regressions included a green answer becoming a table, several colors becoming numbers or another color, and a correct numeric answer `3` becoming `2`. The three candidates regressed different sets of questions; their equal regression counts do not imply equal effects. A response that runs into the eight-token cap fails this short-answer task, but that alone does not establish loss of knowledge.

The experiment demonstrates a controlled change in generated behavior using actual quantized code edits, with fixed scales and an intact original. It does not demonstrate a sufficiently selective repair, generalization, the semantic meaning of an individual weight, or the ghost geometry hypothesis. Although each weight is individually addressable and recorded, these candidates are coordinated edits across one region of 177,596 selected weights. No new activation/Jacobian capture or first-answer probability data was collected in this follow-up.

The run's `experiment.json` is the relational entry point. It links the original and candidate model identities to requests, native-rounded outputs, published edits, actual runs, the paired evidence, and the rejection decision. `development-evidence.json` contains every paired answer and case identity; `edits/*.json` contains the exact before/requested/after representation and patch. The corpus's 36 new confirmation cases remain unused because no development candidate qualified. The sweep stopped at the planned three candidates, with no retuning.

The implementation adds native individual-value rounding and publication. The CUDA wrapper, host launcher, CLI integration and byte-preservation check were committed together; this report records the completed experiment rather than claiming a production model improvement.
