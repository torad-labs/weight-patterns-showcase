# A: overlap acquisition with archival, and account for storage

November 8, 2026, 02:22 CT.

Marcos asked why A was taking longer than the earlier round and whether models
and builds were still scattered. He then pointed out that leaving the GPU idle
while independent work could run in parallel was wasteful. The serial acquisition
runner was our mistake: after capturing a response, it waited for qualification,
upload, and complete remote verification before starting the next response.

## Observed delay and the change

The progress log records a 5.16-minute capture followed by 1.36 minutes of
qualification and local hashing, 3.99 minutes of upload, and 31.32 minutes of
remote verification for `discovery-13-channel-h1`. For `discovery-16-channel-h1`,
the corresponding intervals were 4.58, 1.66, 3.28, and 20.62 minutes. Other
verifications completed in about four minutes. These are instrumented acquisition
and archive timings, not ordinary model decoding speed. The reason individual
remote verifications varied this much has not been established.

At 02:19 CT, the serial runner was replaced by four concurrent case workers.
Each retains the existing complete acquisition, qualification, full-byte R2
verification, and verified local release sequence. The existing GPU0 lock allows
only one response capture at a time; the other cases can qualify or archive
while it runs. The free-space check now occurs after taking that lock, immediately
before a capture starts. The active-case limit bounds how many large captures
can accumulate. A case failure stops further dispatch while already active cases
finish retaining their evidence. This does not promise continuous GPU use if
all workers are archiving or disk space becomes insufficient.

The serial runner was stopped during read-only remote verification of
`discovery-16-channel-h1-corrected`. Its complete local recording was kept. The
parallel runner reused it and repeated its archive checks rather than generating
that response again. The 64 remaining job objects in
`cohort-pipelined-jobs.json` are exactly equal to those in
`cohort-remaining-jobs.json`; already archived cases are skipped. A separately
interrupted archive for `expanded-089-h1` remains included in that list.

Observed overlap: `expanded-064-h1` began GPU capture at 02:19:16 CT while
`discovery-16-channel-h1-corrected` began upload at 02:19:57 CT and remote
verification at 02:20:30 CT. There is one acquisition coordinator in the owned
`weight-patterns-a` tmux session. The complete event-retention worker also runs
there, with a fix preventing it from racing the capture worker's own local
release transaction.

The next handoff occurred at 02:22:33 CT: `expanded-064-h1` entered qualification
and `discovery-14-channel-h1` started capture in the same second, while the
earlier corrected-history archive was still verifying. This verifies the
intended scheduling overlap without attributing a model-speed improvement to it.

The scheduling test first failed with serial dispatch, then passed with parallel
dispatch. Validation completed with exit 0:

- `python3 -m unittest discover -s code/signal-trace -p test_cohort_scheduling.py` — one test.
- `python3 -m unittest discover -s code/signal-trace -p test_event_retention.py` — two tests.

## Why the underlying acquisition is also larger

The earlier A source pair occupied about 0.664 GB in total. The full native
tensor and event recordings for the same two responses now occupy about
20.490 GB. The current cohort contains 68 complete recordings: 21 failures,
26 natural numerical successes, and 21 separately labeled supplied corrected
histories. Earlier instrumentation repairs, incomplete relocation, and our
scheduling error also cost time; the larger capture does not excuse those delays.

No scientific setting or eligibility rule changed with this scheduling fix.
The primary analysis remains the separate overlap of all failed and all natural
successful progressions through the connected sequence of weight applications,
operations, branches, and history, followed by their contrast. Paired context
comparisons remain supporting observations. Full group comparison, coordinated
stored-weight edits derived from that comparison, edited-cohort recapture, and
behavioral evaluation remain pending. No new candidate edit has been made.

## Storage: what was actually removed

Two completed supporting-comparison raw tensor copies were released after their
existing full R2 verification, fresh local hash checks, and checks for open
readers. This freed **18,269,260,980 bytes (18.269 GB)**. Full compressed event
records, comparisons, responses, manifests, and restoration paths remain.
The exact two paths, hashes, sizes, and object keys are recorded in
`records/A-completed-pair-raw-release-2026-11-08.json`.

## Additional cleanup candidates, not deletions

Sizes below are allocated decimal GB at the inspection, rounded. These are
location and ownership findings, not approval to delete another project's work.

| Location | Size | Finding and disposition |
| --- | ---: | --- |
| Research `models/gemma4/gemma4-e2b/ghost-retrieval-2026-11-05` | 43.34 GB | Nine old QGRE edited variants; not in R2. Archive and verify before any local release. |
| Research `models/gemma4/gemma4-e2b/individual-edit-2026-11-06` | 14.45 GB | Three old QGRE edited variants; not in R2. Same disposition. |
| Research `models/gemma4/gemma4-e2b/targeted-lookup-2026-11-06` | 14.45 GB | Three old QGRE edited variants; not in R2. Same disposition. |
| Research old TQ4 base | 4.82 GB | The two main tensors match the QGRE copies byte for byte, but are separate files. Preserve via archive before releasing our copy; do not make active research depend on QGRE again. |
| Research historical D full checkpoint | 9.31 GB | Present in R2, retained locally by the earlier retention choice. Potential local offload, not removed. |
| Research original Q4 GGUF | 3.46 GB | Present in R2; also retained earlier. Current A uses the original BF16 checkpoint. Not removed. |
| Rig `local-nvme/prebuilt` | 9.48 GB | Multiple engine source/build generations belonging to Rig. A now resolves its own engine within research. Future Rig consumers were not established; not removed. |
| Rig `local-nvme/engine-build-trees` | 1.07 GB | Another build location, not removed. |
| torad-llama.cpp `build-cpu` | 1.65 GB | An active speculative-inference test was using these libraries. Keep. |
| torad-llama.cpp `build-engine12` | 1.06 GB | No live user observed, but future consumers unchecked. Not established as disposable. |

The first three research rows total **72.23 GB across 15 edited variants**.
They are not interchangeable duplicate checkpoints. Historical A/B/C full
checkpoints are already R2-only; their small local projection records remain.
The active original BF16 model is about 9.31 GB and stays local. The research
engine source, source archive, and compiled dependencies together are about
0.46 GB, so those are not a major storage issue.

Other large directories include Rig's 16.12 GB research directory,
torad-llama.cpp's 20.46 GB `.agent-cache` directory, and QGRE's 82.02 GB model tree.
They contain other projects' work. Some QGRE generation paths are hard links,
so repeated filenames do not represent additional allocated model copies.
Three apparent 120 GB GGUF files in a metadata-fixture directory allocate only
about 1 MiB each; apparent size must not be counted as reclaimable storage.

`torad-research traces list` also reports older research evidence that has not
been archived, including the 40.4 GB physics trace run. That evidence is not a
disposable build. A large additional archival job has not been started alongside
A's active transfers.

The inspection covered the research, Rig, QGRE, and torad-llama.cpp locations,
with one unreadable Rig overlay work directory. It was not an exhaustive search
of the entire computer. No models or other projects' builds were deleted in
this cleanup.

Primary timing and progress evidence:
`runs/legs/gemma4-e2b/signal-trace-A-2026-11-07/cohort-progress.jsonl` and
`cohort-pipelined.log`. Live progress is acquisition progress, not a result of A.
