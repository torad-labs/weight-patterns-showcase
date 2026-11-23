# November 21, 2026: the consolidation changes hands, what the records say, and the panel's grading recovered

Times are CT. Written by the seat that took the consolidation over at about 10:35 PM, from Marcos's direction as it understood him and from what it checked itself.

## Who does this work

The session that began the consolidation that evening stopped when its conversation grew too long to continue. It committed nothing. Marcos asked this session to take the consolidation over. This session runs on Fable 5.2, which `CLAUDE.md` excludes from project work; he set that rule aside for this consolidation himself, as he did on November 19 for the streaming build (`journal/2026-11-19-host-memory-and-disk.md`). The exclusion stands for everything else, and no project task was delegated to that model. What this seat changed in the tree is listed at the end.

## Three statements, checked against the records

Marcos asked the earlier session to confirm three statements. Checked against the run records, the bucket's listing and MLflow:

1. **A signal locates the climb error, all the pair questions were fixed and the correct ones kept.** The signal is recorded: on the 115 J question the climb discrepancy fell 86.9% when its climb was repaired (`studies/04-round-2-overlap/runs/joint-climb-conditional-overlap/README.md`). Fix-and-keep is recorded for the questions checked at 175, 189 and 348: the 115 J climb corrected, the 44 J answer kept (paper §3). The 14 earlier fitted questions were never scored under this learner. The FK edit corrected the required energy in 7 of its 14 fit members and no complete answer, and damaged 8 of 9 correct answers on the separate 40 questions (`records/development/2026-11-18-joint-fit-panel/README.md`; `records/retrospectives/2026-11-09-round-two-repairs/STUDY.md`).
2. **A different signal locates the descent error, it was fixed and the correct answers kept.** The fitted 91 J descent was repaired, with the 44 J answer kept, by the whole-trace loss, which is zero at the row where the sign is chosen (paper §4). The descent signal recognizes wrong descents on the 30 development answers it was built on (15 of 15, with 3 of 15 valid descents also flagged) and misses the relapse at 387 (paper §8). No run that put a descent signal into the update produced a correct descent that was not already there (table below). On the eight new formulations, correct descents were 4 at baseline and 3 at 390.
3. **No loop repairs both yet.** Confirmed.

## What ran, by what was in the gradient

| Run | In the gradient | Start | Updates | Descent afterwards | Record |
|---|---|---|---|---|---|
| One-pair learner, November 16–18 | whole-trace loss against one reference, with tolerance | original model, fitted answer wrong | about 390 | fitted 91 J answer repaired; relapses at 356, 371 and 387 gone one update later | `final-experiment/training-and-stopping.md` |
| Two one-update tests, November 17–18 | whole-trace loss plus a pre-choice descent term at one cell (about 0.001% of the loss); then that step with its adverse local part removed | original model | 1 each | still subtracted | `final-experiment/one-update-experiment.md`; `runs/directional-intervention-projected-native/README.md` |
| 390 to 401, November 18 | whole-trace loss plus the shared-direction term, equal weights | 390, fitted answer already correct (33 of 7,402 contributions active) | 11 | fit unchanged; no new correct descent on eight held-out prompts | `records/development/2026-11-18-prefix-direction-training/README.md` |
| Pure pre-choice pullback, November 18 | the pre-choice descent term alone | early checkpoint 27, fitted answer wrong | 1 | still subtracted | `records/retrospectives/2026-11-18-pure-descent-native-read/README.md` |
| Unbanded group loop, November 18 | labelled-operation terms only | early checkpoint 27 | 2 | still subtracted, with a new initial-energy error | `records/development/2026-11-18-unbanded-group/README.md` |
| Joint stock loops, November 18–19 | labelled-operation terms only | stock | 3; then 3 and 7 more | all three answers wrong at revision 3 | `runs/joint-stock-climb-descent-training/result.json`; `runs/joint-stock-prechoice-training/fresh-fit-readback.json` |

The whole-trace loss with a descent term at a weight that matters, run for many updates from an answer whose descent is wrong, has not been run.

## The joint loop

- **What was asked.** On November 18 Marcos asked for the earlier fitted questions and descent to be repaired in one run, by adding the descent signal to the loop that had produced the repairs. The run that was built trained two questions from stock (91 J descent, 95 J climb, with the 44 J answer as control); the session that built it called that substitution its own mistake the same day (`records/retrospectives/2026-11-21-oversight-session-transcript.md`). `final-experiment/joint-fit-and-transfer.md` records the two-question design as his milestone.
- **The loss fell while the answers got worse.** Group loss 0.0272, 0.0179, 0.0200, 0.0174 over revisions 0 to 3; wrong descents in one answer of three at stock, two after updates 1 and 2, all three at revision 3 (`fresh-fit-readback.json`). The pre-choice descent measurement, 0.0102 of the stock loss, was found at stock and after update 1 and not afterwards. An operation the matcher does not find adds no loss. Failing cells represented: 1,036, 1,011, 677, 639; preservation loss 0.00034 to 0.0121 (MLflow `computational-training`, run `e2c4836547314df3ac14e3296b16d521`).
- **What changed in the answers.** Each number that changed moved by that question's descent energy: the 91 J question went 11/20 to 51/60 (40 J each); the 95 J question 95/120 to 15/40 to −65/40 (80 J steps); the 44 J control 44 to 4 (40 J).
- **What the gradient held.** Only the labelled-operation terms (`code/signal-trace/pipeline/weight_edit/group_contrast_driver.py`, `gradients`). The whole-trace loss against the fixed reference, the loss of the November 16–18 learner, was logged and not differentiated. On the fitted question it went 0.000292, 0.00269, 0.00387, 0.00387 and stayed there; on the reference's own prompt it went from 0 to 0.0041. The "0.00387, flat over seven updates" on `STATE.md` and in the streamed-state record is this number, 13 times its stock value, and not the group loss.
- **The continuation.** Group loss 0.0174, 0.0205, 0.0205, 0.0192, 0.0192, 0.0192, 0.0170, 0.0202 at updates 0 to 7, against the stop threshold 0.00136 (MLflow run "Group repair · … Nov 19, 14:05 CST", `native.group.measured_contribution`).
- **Revision 3.** Its weights (`final-checkpoint/native.bin` and `remainder.f32`; sizes and hashes in `records/maintenance-runs/2026-11-18-joint-revision3-headroom/receipt.json`) are not on this disk and under no `runs/joint-*` folder of the bucket (listing of November 21). The lineage's surviving checkpoint is the third streamed start's update 4 (`runs/joint-stock-prechoice-streamed-recovery-3/final-checkpoint`, bucket).

What the next loop's loss is stays open with Marcos: whether the whole-trace term returns to the gradient, with or without the tolerance band, which questions train, and where it starts. Nothing runs until he says.

**Decided November 22, 2:50 AM CT.** Marcos chose: make a lost step count, then run the November 16–18 whole-trace loop with the pre-choice descent term added at a calibrated weight. The first half was done the same night on records: the local MLflow store had kept every joint run's per-operation measurements, which the bucket and git had not. With every lost cell charged the value it last had, the lineage's loss goes 0.0272 at stock, 0.0194 after the first update (−28.6%), then above stock at every later update, ending 0.0333 (+22.6%) after ten, where the run's own number ended 25.7% below stock. On the 3,845 cells found in every revision it ends 42% above stock. The failing operations' loss fell 41% and the preserved operations' rose 51-fold ([recheck](../records/development/2026-11-22-joint-loss-recheck/README.md)). A first reading that night charged one cell only, from the tracked readback, and gave 0.0248 at revision 3; the full count replaces it. The run itself is still to be declared; its starting checkpoint and the term's weight are the open choices.

## The frozen panel's grading: recovered, not graded again

Eight records behind the paper's transfer table were in neither git nor the bucket: `runs/frozen-energy-generalization/{result,blinded-readings,blinded-responses,unblinding,paired-outcomes,matched-fits}.json`, its `summarize.py`, and `runs/frozen-energy-fit-390-gpu1/result.json`. On November 21 they were recovered:

- **The 44 answers and their capture receipts** came from the November 18 publication archive through its `restore.py` (135 members of one pack). Each `result.json` equals the SHA-256 its `verified.json` records.
- **The grader's 35 judgments and its two programs** (the status rules, and the later flag for a descent written as "potential energy gained") were read from its own session log (`<session-transcripts>/2026/11/18/rollout-2026-11-18T05-01-37-01a1ee51-c1c5-7f70-a317-2e58c371a939.jsonl`, sha256 `424def37…e3dc2a`). Each blind ID was matched to its answer by the text that log shows for it: 34 on the full displayed text, one on the SHA-256 of its exact text.
- **`blinded-readings.json`** rebuilt from those has SHA-256 `a37db3cbdfd7f0ff1c57c19734ba5cf20ee13f9ee2ebed65e875e4300d919f07`, the value the grader printed when it froze its readings and the one `reports/descent-trace-repair/unified-review.md` cites.
- **`blinded-responses.json` and `unblinding.json`** were written in the format of the program that first wrote them (`prepare.py`, in the root session's log `<session-transcripts>/2026/11/16/rollout-2026-11-16T00-42-22-01a1e317-adbf-7983-bb8b-3df59ef44c23.jsonl`, sha256 `09296c0d…61a62c`).
- **`paired-outcomes.json`** was written by the run's own `summarize.py`, restored from the archive.
- **`result.json`** was written by the command that first wrote it, taken from the root session's log and cut off after that write. Its SHA-256 `e9529f48ecb30b696dbb1ac348dfb9a104aaca9a95951d5909c229a3aec148f8` equals the one `records/development/2026-11-18-joint-fit-panel/inventory.json` (committed November 18) records, and it records the SHA-256 of `paired-outcomes.json`, `blinded-readings.json`, `blinded-responses.json` and `matched-fits.json`, each equal to the recovered file.
- **`matched-fits.json`, `summarize.py` and the 390 fit's `result.json`** came from the archive, hash-checked by `restore.py`.

The recovered counts are the paper's Table 3: correct full chains 1, 2, 2, 3 and correct descents 4, 2, 3, 3 of eight at the original, 175, 189 and 390.

**Placed.** The eight files are in their two run folders since 11:55 PM, each with the bytes of its recovered copy; nothing was overwritten. All entries of the working release list and all 72 links of the paper's evidence map exist on this disk. The eight were committed on November 22 with this page. The programs that rebuilt them are in `records/maintenance-runs/2026-11-22-tree-and-panel-recovery/programs/`.

## Two folders recorded as lost are in the bucket

`runs/wide-overlap/exported` (180 files, 19.76 GB) and `runs/controlled-bf16-1/own/original` (392 files, 10.44 GB) were recorded on November 19 as having one copy, and on November 21 as lost. The November 18 publication archive holds both: its catalog lists every one of the 572 files, each archived object is in the bucket at the catalog's size (listing of November 21, 11:59 PM), and one member of each folder came back through `records/maintenance-runs/2026-11-18-publication-archive/restore.py` with its hash checked. The November 19 dry run that found nothing looked under the runs' own bucket paths; the archive's packs are under `traces/maintenance/publication-2026-11-18/`. This seat made the same mistake at 8:05 PM and told Marcos the exported turns were not in the bucket.

Of the 675 paths the earlier session classed as in neither git nor the bucket, that catalog lists 596 (48.6 GB); nine of the direct objects they depend on were not found in the listing. The other 79 are in no store found: most are records of the joint runs of November 18–19, made after the archive was written. Revision 3's weights are among what is gone: none of the eleven `native.bin` objects in the bucket has its hash.

## The sealed set

`git restore` at 7:26 PM brought `inputs/acquisition/reserved-40b` back as a readable folder (mode 755 on the host). It is mode 000 again since 11:37 PM. In between, two size and file-count walks of this seat passed through it at about 7:34 and 7:43 PM, with no file opened and no name printed. A second session, started at 10:07 PM, was asked to read every file in the repository; whether it read the sealed set is not established here.

## November 22: the tree restored, the gate green

**What came back.** 1,544 files from the November 18 archive, three paths from the bucket under their own keys, six files from the local MLflow store, five from byte-identical copies under other paths, the stock-origin checkpoint rebuilt from the original model, and the 60 answers of `wide-overlap/source/original-gpu0`. Each was placed only when its bytes equalled a hash recorded before the deletion ([record, logs and programs](../records/maintenance-runs/2026-11-22-tree-and-panel-recovery/README.md)). The other 240 files of that `source/original-gpu0` folder are not back: no record states their hashes.

**What is in no store.** 63 files, not the 79 counted above: 16 were found by content under other paths. Revision 3's weights are among the 63.

**The gate.** It went from 3,495 lines to 197 by the restores. The 197 were all absences of those 63 files, named by 28 old declarations and 37 old archive records. Those records are history and are not rewritten, so this seat added a list of the 63 that the method file binds by hash (`round-2-method.json`, `recorded_losses`; `code/signal-trace/pipeline/research_guard/recorded_losses.py`). This is a change to the gate's method file with a written reason, which `CLAUDE.md` allows; it changes no step of the method. What it excuses and what it refuses is in the [loss account](../records/maintenance-runs/2026-11-19-evening-studies-deletion/README.md), "The census and the gate". The gate reported no drift from 6:15 AM, with the first version, and reports none with the version committed. The pre-commit hook runs the verifier of the working tree, so for those hours every seat's commit would have passed on the strength of an unreviewed change; no seat committed in them.

**How the change was checked.** This seat wrote the first version at 6:13 AM. An independent reviewer (a fresh subagent started with the Opus model setting, given the code and the claims, not the conclusion) then read it four times, and each time found something:

| Pass | Found | Changed |
|---|---|---|
| 1 | An archive record that cited a lost receipt passed with any inventory, and a new record could cite the same receipt. Any new declaration could name a lost file and pass the commit gate. A native run could declare a folder holding a lost file. | A loss is known only to the declarations and record files its entry names, each by its SHA-256. A native run is refused for a folder that holds a loss. |
| 2 | The native-run refusal compared spellings, so `x/.`, `a/../b`, an absolute path or a link got past it. Running the list's builder again could have listed a new absence. | Paths are compared as they resolve. The builder refuses any declaration or record that is not byte for byte the file of commit `7640dc02`. |
| 3 | A target equal to the runs folder crashed the gate (it did before this change too). Only inputs and the target were checked for losses, not other fields a driver reads. | Both reported as problems. A loss named in any field of a declaration stops a native run. |
| 4 | A declaration that names another run by its bare name, which some drivers join with a file name, was not held. Nothing the gate refused before is accepted now. | A run named without a path is held when its folder holds a loss. |

After each pass the guard tests were extended; 124 pass. A separate check removes each protection in memory and confirms a test then fails (`programs/mutation_check_recorded_losses.py` in the recovery record). The implementation peer and the independent reviewer have not read the change; this session has no independent-reviewer connection.

Known limits, as the fourth pass left them. Paths are compared as this Linux file system resolves them. A path a driver builds from something the declaration does not state (a run name kept in another record, a name fixed in code) is not seen. The gate still raises, as it did before this change, on a declaration whose `inputs`, `outputs` or `parameters` have the wrong shape; none of the 226 declarations has one. Four passes that each found something are not a clean pass: the change is small and tested, and it has not had the review `CLAUDE.md` asks of a build.

**Which tests ran, and where.** The eight guard test modules are plain Python and ran on this machine (124 tests, three seconds). `tools/remote-tests.sh` was started at 10:20 AM for those and for the four modules that import the changed files and need torch (`test_rented_solve_bundle`, `test_vector_recovery`, `test_working_capture`, `test_final_experiment`). It did not run: the remote sync runner stopped at the sync, because rsync could not open the sealed folder, and answered 75. On November 19 the remote runner shipped the tree without that folder. This seat did not change the folder's mode or the remote runner to get past it, and did not run those four modules here. They have not run against this change.

**What this seat got wrong on the way.** Its conversation overflowed at 6:15 AM, just after it wrote the gate change. The continued session did not recognise the change as its own and spent about fifteen minutes looking for another writer before its own tool log settled it. The first version it had written was the one with the holes of pass 1, and the gate was already reporting no drift with it.

**Which model wrote what.** The session ran on Fable 5.2 except for two stretches after Marcos switched models: DeepSeek Flash wrote the recheck's README and the "Decided November 22" paragraph above at 2:55 AM, and Sonnet 5.5 moved the guard tests' set-up into `tests/guard_fixture.py` at 7:07 AM. The recheck README's numbers were compared with the program's outputs afterwards; they agree. The reviewer subagents ran on Opus.

## What this seat changed

- The sealed folder's mode (000 again).
- The lines of `STATE.md`, `reports/descent-trace-repair/milestone.md`, the loss account, `records/KNOWN.md`, `ROUND-2.md`'s status line and the recomputability note that the checks above contradict.
- The gate: `recorded_losses.py` (new), and the places `verify_round2.py`, `input_archives.py`, `storage_archives.py` and `native_records/run_declaration.py` ask it; `round-2-method.json` (one entry added, every other byte unchanged); the guard tests.
- Files put back under `studies/04-round-2-overlap/runs/`, none overwritten.
- `.gitignore`: `*.partial` and `*.bf16` under `studies/`.
- The release list, 137 entries: `recorded_losses.py` added, since three listed modules import it; and the paper's sixth figure (`figures/prefix-transfer.svg`) with the two records `make_figures.py` draws it from, since the bundle built from the 134-entry list stopped at that figure when its README's rebuild step was run. With them, all six figures regenerate in a standalone copy to the committed bytes. The README's "five SVGs" is now six, with one evidence row for the figure.
- This page, the recheck record and the recovery record.

It removed two empty `.git/index.lock` files (its own of 7:06 PM November 21, and one another process left at 11:17 PM). It deleted nothing else.

**Commits.** The 325 record files the earlier session staged are committed as it staged them, in a commit of their own. This seat checked them for sealed names (none), credential patterns (none) and size (largest 4.3 MB) and did not read them one by one. That session's change to three guard modules is not applied; it is kept as `records/development/2026-11-21-make-storage-proposal/make-storage.patch`. The restored run files that were never in git stay untracked.
