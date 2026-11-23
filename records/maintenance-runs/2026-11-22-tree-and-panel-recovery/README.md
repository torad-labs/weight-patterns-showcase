# What came back after the November 19 deletion, and how each file was checked

November 21–22, 2026 (CT). Record of the seat that took the consolidation over (`journal/2026-11-21-consolidation-takeover.md`). It restored files and deleted none. The deletion itself is accounted for in `records/maintenance-runs/2026-11-19-evening-studies-deletion/README.md`.

Nothing here was recomputed, graded again or typed in. A file was put back only when its bytes could be held to a hash that a record made before the deletion states for it.

## What came back

| From | Files | How each was checked | Log |
|---|---|---|---|
| The November 18 publication archive (`records/maintenance-runs/2026-11-18-publication-archive/restore.py`), 37 packs | 1,544 | `restore.py` checks every file against the SHA-256 in the archive's catalog and overwrites nothing; no pack failed | `logs/restore-run.jsonl`, `restore-run-2.jsonl`, `restore-run-3.jsonl`, `wide-source-restore.jsonl` |
| The bucket, under the path's own key (`torad-research traces download`) | 3 paths | the tool checks each file against its `.sha256` object | the same logs, `own-key` lines |
| The local MLflow artifact store (`<mlflow-store>/artifacts/6/<run>/artifacts/steps`) | 6 | each equals a SHA-256 a tracked record states for its path | `logs/placed-from-mlflow.json` |
| Byte-identical files under another path of this tree | 5 | each equals a SHA-256 a tracked record states for the lost path | `logs/placed-twins.json` |
| The original model | 2 | see "The stock-origin checkpoint" | |
| The grader's session logs and the archive | 8 | see "The paper's panel records" | `logs/panel-rebuild-provenance.json`, `logs/grader-judgments-2026-11-18.json` |

The earlier consolidating session's downloads of November 21 (135 GB, by its own account in the loss record) are not in these logs.

## The stock-origin checkpoint

`runs/joint-pristine-stock-origin/final-checkpoint/native.bin` (9,257,704,584 bytes) was rebuilt from the original model `models/gemma4/gemma4-e2b/gguf-bf16-bartowski/google_gemma-4-E2B-it-bf16.gguf`. The run's tracked `state.json` records, for each of the 538 tensors, its place in the payload and its place in the model file; `programs/rebuild_stock_origin.py` copies them in that order and places the result only if its SHA-256 equals the recorded one. It did. `remainder.f32` is the empty file the same record states (0 bytes, the SHA-256 of no bytes).

## The 60 answers of `runs/wide-overlap/source/original-gpu0`

The folder had 300 files (five per member, 60 members). No store holds it under that path. The tracked `runs/wide-overlap/source/labels-original-gpu0.json` states the SHA-256 of each member's `result.json`, and the archive holds those same bytes under two other folders (`runs/wide-overlap/capture/batch-*/<member>/` and `runs/controlled-bf16-1/own/original/raw-*/<member>/`). All 60 `result.json` are back in `source/original-gpu0/<member>/`, each equal to its stated hash (`programs/restore_wide_source_results.py`, `logs/wide-source-results-placed-2.json`).

The other 240 files of that folder (`case.json`, `graph.json`, `signal-data.bin` and `signal-events.jsonl` per member) are **not** back under it. No tracked record states their hashes, so nothing could hold a copy to them. The archive lists files of those names for the same 60 members under `runs/wide-overlap/capture/batch-*/`. The pipeline's readers of this folder (`pipeline/weight_edit/a2_pattern.py`, `pipeline/overlap/same_card.py`) read `result.json` only.

## The paper's panel records

The eight records behind the paper's transfer table (`runs/frozen-energy-generalization/{result,blinded-readings,blinded-responses,unblinding,paired-outcomes,matched-fits}.json`, its `summarize.py`, and `runs/frozen-energy-fit-390-gpu1/result.json`) were recovered as `journal/2026-11-21-consolidation-takeover.md` sets out, with these programs:

- `programs/recover_readings.py` reads the 35 judgments from the blind grader's own session log; `programs/check_recovered.py` and `programs/compare_with_logged_prints.py` hold them to what that log printed on November 18.
- `programs/rebuild_panel_readings.py` rebuilds `blinded-readings.json`; its SHA-256 `a37db3cb…919f07` is the one the grader printed when it froze its readings.
- `programs/finish_panel_records.py` writes the other records in the formats of the programs that first wrote them; `result.json` has the SHA-256 `e9529f48…c148f8` that `records/development/2026-11-18-joint-fit-panel/inventory.json` has recorded since November 18.

## What is still not on this disk

63 files are in no store. They are named one by one, with the hash their records state, in `records/maintenance-runs/2026-11-19-evening-studies-deletion/recorded-losses.json`. `logs/lost-final-table.tsv` is the working table that list was reconciled against.

## What is on this disk and not in git

`logs/restored-and-untracked-run-files.txt` lists the 824 restored files under `studies/04-round-2-overlap/runs/` that git did not track when this record was written. Since then:

- the paper's eight panel records were committed on November 22;
- five bulk files are ignored like the other raw types (`.gitignore`): `runs/controlled-bf16-4/model/controlled-bf16-4.partial` (9.3 GB, the bucket holds it under that name at the same size) and four 18.9 MB `.bf16` tensors of `vector-chance-native-first-ffn-replay` (in the archive);
- 811 stay untracked. The archive's catalog lists 588 of them under their own path and the bucket holds 156 under their own key. The other 67 have no store under their own path, and each has the same bytes elsewhere: the 60 answers above (in the archive under two other folders), the six files from the MLflow store, and one byte-identical copy.

They were untracked before the deletion too, hidden by the local exclude line of November 16. They are 217 MB of JSON, and whether they enter git is not decided here.

## Programs

`programs/` holds every program this recovery ran, as run. They were written in a working folder outside the repository and read their inputs from arguments. `plan_restore.py`, `plan_breakdown.py`, `run_restore.py` and `restore_until_settled.sh` planned and ran the archive and bucket restores from the gate's own output; `find_lost_by_hash.py`, `find_by_hash_in_stores.py`, `place_by_hash.py`, `place_twins.py` and `reconcile_recorded_losses.py` looked for the remaining files by content; `gate_without_list.py` prints what the gate says with no recorded loss taken into account; `mutation_check_recorded_losses.py` removes each protection of the recorded-losses check in memory and confirms a test then fails (run it from `code/signal-trace`).
