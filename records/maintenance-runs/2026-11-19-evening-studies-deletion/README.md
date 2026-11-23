# The evening deletion of November 19, 2026: what left this disk, what the bucket holds, and what is lost

Recorded November 21, 2026 by the consolidating seat. Times are CT. This record accounts for the deletion
itself; the release decisions of that morning are `records/maintenance-runs/2026-11-19-finished-run-bulk/`
and `records/maintenance-runs/2026-11-19-memory-headroom/`. Nothing here was released by this record; the
restores named below brought bytes back and deleted nothing.

## What happened

Shortly after 7:30 PM CT on November 19 (directory timestamps say 19:40–19:47; no record names the seat or
session), the working tree lost, without a commit:

- every tracked file under `studies/04-round-2-overlap/` — 10,765 files: the round-two method
  (`ROUND-2.md`, `round-2-method.json`), every run's `declaration.json`, results, records and registries,
  including the whole `input-archives/` registry;
- every untracked bulk file under the same tree — the captures, traces, checkpoints and exports the release
  passes of November 13–19 had left on this disk, plus the working files of the streamed starts.

The last commit before it is `7640dc02` (the fifth streamed start's kill, recorded that evening). The
oversight session's transcript ends 10:30 AM CT November 19 with the verifier green ("Verifier: no drift",
research-optimizer, 10:22 AM CT), so the round-two gate's 3,574 drifts of November 21 trace entirely to this
deletion.

## What was verified where, before this record

- The tracked files were restored from git on November 21 (`git restore studies/04-round-2-overlap/`); git
  HEAD held them intact.
- The bucket holds 3.2 TB across 176 trace folders (`torad-research traces list`, November 21), including
  every round-two run folder's released bulk and the 154.1 GB of `inputs/observation`.
- No running `virtiofsd` held the deleted blocks (`held_deleted.py look`, November 21: only the VM's own
  memfds), so nothing was recoverable from held-deleted handles.
- The tree that reached the secondary test host holds tracked files and unignored untracked files only
  (`tools/remote-tests.sh`: "tracked files and untracked ones git does not ignore"); `runs/*/exported/` is
  gitignored, so no second copy of the exported turns ever left this machine. Marcos confirmed November 21
  that nothing of it is on the secondary test host.

## What is confirmed lost, and how it comes back

Probed per file against the bucket on November 21 (`traces download` of named paths; the full per-path
census is the restore log of that day, `/tmp/restore-log.txt`):

| Lost bytes | Size | Recompute or restore path |
|---|---|---|
| `runs/wide-overlap/exported/` — the wide cohort's exported turns | 180 files, 19.8 GB | **Not lost** (corrected 11:59 PM CT November 21): all 180 files are in the November 18 publication archive in the bucket and return with `records/maintenance-runs/2026-11-18-publication-archive/restore.py`; one member was restored and hash-checked. The storage-only backup refused on November 19 was not the only copy |
| `runs/controlled-bf16-1/own/original/` — the non-fitting members and auxiliary files | 392 files, 10.4 GB total; the fitting members are held by record `89316e5cc00c-original.json` and return with `archive_inputs.py restore 89316e5cc00c-original.json --fetch` | **Not lost** (corrected 11:59 PM CT November 21): all 392 files are in the November 18 publication archive in the bucket and return with its `restore.py`; one member was restored and hash-checked |
| Every run's kept `observability/events.jsonl` | ~2.67 GB | Operational logs; their contents were exported to MLflow before the deletion; not regenerable byte-for-byte without replaying the runs |
| 0.95 GB of raw streams in `joint-stock-prechoice-loss-continuation` and `joint-stock-prechoice-training` | 0.95 GB | Recompute by replaying the recorded updates (both runs' records are bucket-held) |
| `runs/joint-pristine-stock-origin/model.gguf` | 8.7 GB | Trivial: a verified copy of the published BF16 model, which is intact at `models/gemma4/gemma4-e2b/gguf-bf16-bartowski/google_gemma-4-E2B-it-bf16.gguf` |
| Smaller declared paths (a2-path answers, calibrated-descent cases, two capture-pilot event logs, and others the census names) | misc | Recompute from the original model, or accept the loss per path as the census reports it |

Not lost: `runs/same-prefix-135-original`'s no-fit folder was backed up and read back that morning
(`b4bdd3beadce-expanded-040-h1.json`), so it returns with `archive_inputs.py restore`. The original and seed
models, the 30 witnesses, checkpoints 397 and 401 and the third streamed start's update-4 checkpoint are
bucket-held or local as their records say. Revision 3's weights (`runs/joint-stock-prechoice-training/final-checkpoint/native.bin`
and `remainder.f32`, hashes in `records/maintenance-runs/2026-11-18-joint-revision3-headroom/receipt.json`) are lost:
they were kept on this disk only, and no `native.bin` in the bucket has that hash (listing of November 21).

Marcos's word of November 21: the code is safe in git, the session transcripts exist, and in the worst case
everything lost can be recomputed; he is not worried.

## Why the deletion took records as well as bulk

`.git/info/exclude` line 22 (`/studies/04-round-2-overlap/`, added 2026-11-16 05:42 CT by the independent review agent to
keep the desktop IDE diff from reading study data) made every new file under the round-two tree
ignored. Records seats committed by explicit path survived in git; every record created after that
moment and never named explicitly had one copy. Of the 800 paths the gate misses, a read-only bucket
listing found 118 in the bucket (106.1 GB) and 675 in neither git nor the bucket; the paper's own share of
the 675 is eight files of the frozen transfer panel (`records/retrospectives/2026-11-21-paper-recomputability.md`).
The census by download was stopped at 135 GB restored; the classification is in this session's
`missing-classified.json`.

**Corrected later that night.** The listing behind "in neither" looked under each path's own bucket key
and not in the November 18 publication archive, whose packs sit under `traces/maintenance/publication-2026-11-18/`.
That archive's catalog lists 596 of the 675 paths (48.6 GB), the two large folders of the table above among
them; nine of the direct objects those paths depend on were not found in the bucket. The remaining 79 paths
are in no store found, most of them records of the joint runs of November 18–19, made after the archive was
written. The paper's eight panel files were recovered from that archive and the grader's session logs
without grading again and are back in their run folders, untracked (`journal/2026-11-21-consolidation-takeover.md`).

## What is in no store, settled November 22

After the restores of November 21–22 (`records/maintenance-runs/2026-11-22-tree-and-panel-recovery/README.md`:
1,544 files from the November 18 archive, three paths from the bucket, 13 from local copies and the
original model, and 60 answers copied back to the folder that lost them, each held to a recorded hash) the
gate named 63 files that nothing holds: 35 capture
receipts, 14 small run records, four derived arrays, three captured streams with their three indexes, two
process logs, and revision 3's `native.bin` and `remainder.f32`. They belong to 15 runs, the joint runs of
November 18–19 most of all. Each was looked for under its own bucket key, in the November 18 archive's
catalog by path and by content hash, in every registry inventory by content hash, and on this disk by
content hash; none was found. The 79 of the paragraph above became 63 as files were found by content under
other paths.

They are listed one by one in [`recorded-losses.json`](recorded-losses.json), written by
`build_recorded_losses.py` from the gate's own output and
`stated-hashes.json` (every SHA-256 a tracked record binds to each path). Each entry
holds the lost file's recorded SHA-256 and the records that need it: 28 run declarations name 32 of the
files as inputs, and 37 archive records bind 32 of them as the receipt that stated an archived file's hash.

## The census and the gate

The per-path census and restore of the 800 missing declared-input and must-stay paths ran from the bucket on
November 21 (log: `/tmp/restore-log.txt`; each restored path satisfies the gate by its presence), and from
the November 18 archive on November 22.

**The gate is green again since the morning of November 22**, by a written reason in the method file
(`studies/04-round-2-overlap/round-2-method.json`, `recorded_losses`) that binds the list above by its
SHA-256. What the list does and does not do (`code/signal-trace/pipeline/research_guard/recorded_losses.py`):

- A listed file's absence is excused only to the declarations and archive records its entry names, and only
  while each is byte for byte the file commit `7640dc02` holds, the last commit before the deletion. A new
  declaration or record that names a lost file, or an old one that was changed, is refused as before.
- Nothing else is excused. Where an input comes from, parameters and their citations, targets and results
  are checked as they were.
- For the 37 archive records, the one check the lost receipt alone made possible is not made: that the
  receipt states the archived file's SHA-256. The record's own inventory is pinned by the record's hash.
- No native run may read a lost file, anything below one, or a folder that holds one, however the path is
  spelled and in whichever field of its declaration it is named. A run that needs revision 3's weights
  cannot start. Of the 226 declared runs, 31 cannot be started again as declared: the 28 the list names,
  and three paired evaluations whose learner folder holds a lost file (`joint-prechoice-paired-evaluation`,
  `joint-stock-paired-evaluation`, `joint-stock-threshold-paired-evaluation`).
- Paths are compared as this Linux file system resolves them. On a file system that ignores letter case, a
  differently cased spelling of a lost path would not be caught.
- A listed file that comes back is reported until it is taken off the list.

The change was read four times by an independent reviewer before it was committed, and each reading found
something that was then changed; what it found is in `journal/2026-11-21-consolidation-takeover.md`. The implementation peer
and the independent reviewer have not reviewed it.

The
bucket inventory of November 19 (97,288 completed objects, 2,559.12 GB, about $38.40/month at R2 Standard)
stands as the oversight session recorded it; this deletion added nothing to the bucket and removed only
local copies.
