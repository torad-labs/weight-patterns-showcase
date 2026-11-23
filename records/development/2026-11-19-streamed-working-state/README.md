# The trainer's working state on disk, one piece in host memory at a time

November 19, 2026. Marcos asked for the training code to keep on the host only what the host has to
work on, with the card used fully, after the joint continuation took the machine into a memory emergency
on the night of November 18. This page says what held the memory, what the build does about it, what it
costs in disk, how the resumed run is held to the recorded one, and what is still not bounded.

## What held the memory

`joint-stock-prechoice-loss-continuation` died at 10:50:39 PM CT on November 18 with an X server IO error
during a diagnostics save, right after its fourth install. Its unit held 57 GB of RAM and 9 GB of shared
memory (peak 63.4 GB) while the model on card 0 took 11.5 GB. The first disk-backed start after it reached
34.7 GB of anonymous memory before any gradient
(intent).

The model was never the term. The trainer kept these whole and alive together on the host:

| Array | Bytes | Why it exists |
|---|---|---|
| FP32 copies of the weights the forward keeps on the host | 7,509,078,016 | the derivative reads FP32, the file is BF16 |
| The rounding remainder of every eligible dense tensor, FP32 | 7,509,061,632 | carried from update to update |
| The joint direction, FP32 | 7,782,797,312 | minus the summed member gradients |
| The displacement of the previewed step, FP32 | 7,782,797,312 | what the install writes |
| One fresh native trace per member, 105 merges in and out | 3,963,617,280 each at 2,048 events | the graph is held bit for bit to it |
| Each member's gradient before it was summed | up to 7.8 GB each | `autograd.grad` returned all 538 at once |

These, with three members and the copies the measurement makes of a trace, are what the 57 GB was made of;
that process was not measured array by array. The sizes are from the stored shapes
(`group_disk_state.file_limits`) and the review's recount. The plotting backend held a connection to the
display, and the process died when that connection failed.

## What the build does

- **Four files in place of four arrays.** `host_pages.Store` gives each array its own shared mapping of a
  file allocated ahead (`posix_fallocate`). A release takes the pages out of the process and leaves them in
  the kernel's page cache; a released array that is read again gets the same bytes back, from the cache while
  the machine has the memory free. The first three starts ran a build that also flushed every release to the
  file and dropped it from the cache; [what that did to the disk](#what-the-third-start-did-to-the-disk-and-the-change)
  is below.
  `group_disk_state.Working` holds the leaves, the remainder, the direction and the displacement this way.
- **Gradients arrive one leaf at a time.** The backward runs on the card with a hook per leaf; each
  gradient is negated or summed into the direction file as it arrives and is then let go
  (`group_contrast_driver.py`). No member's whole gradient exists on the host.
- **A fresh native trace has no array of its own.** Each merge is read from the capture's stream when it is
  asked for (`group_disk_state.native_traces`), and `turn_pair.Turns` forms one merge's turn at a time.
- **No display.** The launcher removes `DISPLAY`, `WAYLAND_DISPLAY` and `XAUTHORITY` and sets
  `MPLBACKEND=Agg`; the two render functions force Agg as well.

The frozen controller, the frozen forward, the capture and the objective are not changed. The computed
values, the order of every sum and every dtype are the same, by the independent review's reading
(review).

## What it costs in disk

The working files at their largest are **39,895,039,168 bytes**: the 9,311,304,896-byte model file plus the
four arrays of the table above (`group_disk_state.file_limits`, from the stored tensors' shapes, the cohort's
event caps and the table rows the checkpoint retains; every array in whole pages).

The launcher's `check` asks for 118,965,425,616 bytes free before it starts:

| Bytes | For |
|---|---|
| 40,000,000,000 | the floor this machine keeps free, never written |
| 39,895,039,168 | the working files |
| 22,030,528,512 | two cohort capture reserves of 11,015,264,256 |
| 9,257,704,584 | the next native checkpoint |
| 7,782,153,352 | the next remainder checkpoint |

An earlier form of this build also wrote each member's native trace to a file of its own, 3.96 GB a member,
and asked for 130.86 GB. Marcos asked why a check needs that much. Reading each merge from the capture's
stream removed those files. Of what is left, the capture reserve is counted twice by the frozen budget and
the native launcher, 22 GB against about 5 GB a cohort has taken on disk; that is the launcher's owners'
to change.

## The lineage guard, and how the run is declared

`final_experiment.prepare` refuses this build, because nine sources that
`vector-chance-final-experiment-installed-348-recovery` binds have changed. The guard is doing its work.
The run is declared by `declare.py` as an open revision of
`joint-stock-prechoice-disk-recovery-2`'s declaration: it names every changed source with
`group_disk_state.py` and `host_pages.py`, binds the review, and has to pass the launcher's own `check` and
the verifier. The frozen controller still holds every executed module to its declared hash. The reviewer
accepted that route only in this open form, and says the better fix is for the launcher to accept an
explicit reviewed source revision.

**One change after the review's addendum.** The launcher's `declare_disk` cited four records in a list as
the working-storage parameter's `measured_from`, and the verifier, which takes one record path there,
crashed on the declaration (`verify_round2.py:97`). The parameter now cites the tensor record and names the
other three in its derivation. `group_contrast_experiment.py` is `2a55d32a0371` with that change, where the
addendum read `5926b770a2f8`; `gate.py` was also rewritten after the addendum to close its findings 8 and 9.
Neither is on the path of a computed value.

**The gate.** The resume starts from revision 3 and recomputes updates the continuation recorded. As each
record lands it is compared with
`studies/04-round-2-overlap/runs/joint-stock-prechoice-loss-continuation/loop/`: for updates 0 to 4 each
group measurement and each member's pairs and answer; for updates 1 to 4 the proposal (gradient norm,
per-tensor norms and nonzero counts, continuous response, applied length, chunk hashes, installed response)
and the receipts. `gate.py` does it: it requires every record the continuation holds for those
updates (58, counted from its folders), compares values exactly, and leaves out only what two runs cannot
share (the time a record was written, the run's folder name in a path, the hash of a JSON record that holds
those). It runs beside the trainer as its own systemd user unit, reads each record as it lands, and sends
the trainer's unit SIGTERM at the first difference. Updates 1 to 4 retrace a recorded rise of the cohort
contribution; new information starts at update 5.

## The hook that had lost its target

Commit 800cfd81 removed `vector_chance.turns` while both observability inventories still named it
(`pipeline/observability/instrumentation.py` and the frozen `capture.py:58`), so the second disk-backed
start recorded a setup error for `computation.turns` and no measurement-side turn spans. The name is back:
`vector_chance.turns` returns a `turn_pair.Turns`, a sequence the hook can summarize, which still forms one
merge at a time.

## What is not bounded by this build

- **One member's whole trace, during measurement.** Correspondence makes a float64 copy of the trace with
  the stacked reference (about 2.6 GB at 2,048 events, `turn_correction.py:33-39`) and registration a
  float32 query (about 1.3 GB, `group_contrast_objective.py:262`), one member at a time. This page
  predicted a few GB of anonymous memory at its most from those two; the first start measured 7.5 to
  11.4 GB (next section).
- **The frozen reader's own arrays**, for the seconds it takes to read a capture.
- **Page cache of the capture's stream**, which the kernel can drop at any time and which is not memory the
  run holds.
- **Disk accounting assumes nothing keeps the previous direction, displacement or trace** when a store is
  reset (`Working.pending`). It holds in the frozen loop; a new reference would keep up to 7.78 GB on disk
  past the reserve, inside the 40 GB floor.
- **A store whose file was removed from outside raises during cleanup** (`close_scratch` in a `finally`),
  which would hide the first error.
- **NVMe writes.** Each update writes about 23 GB of direction, 7.8 GB each of displacement and remainder,
  and reads the 7.5 GB of host weights at least nine times.

## What the first streamed start measured

`joint-stock-prechoice-streamed-recovery` ran as a systemd user unit from 4:41 to 5:39 AM CT on November 19,
with no display. The samples, one every 30 seconds from the unit's cgroup, are in
`first-start-memory.log`.

**Same results.** The gate compared all 22 records of the baseline and update 1 with the continuation's and
found them equal, value for value: baseline objective 0.0038740613441862965 with 53,455 cells outside, the
install of update 1 changing 22,241 values, and update 1's measurement at 0.0038744651545817364 with 53,416
outside.

**Memory.** systemd reported a peak of 15.6 GB for the unit (16.31 GB in the samples), against 63.4 GB for the
continuation. It is not the few GB this page predicted. What made it up:

| Phase | Anonymous memory | File pages |
|---|---|---|
| Restore of revision 3 (4:42 to 4:53) | 2.9 to 3.7 GB | up to 8.9 GB, 8.6 GB of it mapped |
| Baseline measurement (4:53 to 4:57) | 4.4 to 6.2 GB | under 0.4 GB |
| Proposal of update 1 (4:58 to 5:26) | 7.5 to 7.7 GB, flat, once 8.4 GB | under 0.6 GB |
| Install and receipts (5:26 to 5:29) | 5.2 to 7.7 GB | 8.5 GB at 5:26:49, 6.6 GB at 5:28:49 |
| Measurement of update 1 (5:29 to 5:39) | 6.4 to 11.35 GB (the most, at 5:32:49) | up to 5.0 GB |

- **The peak is a whole working file resident at once.** The restore, the install and the receipts walk a
  whole store, and its pages stay mapped until the store's `release()` at the end of the pass. The `Dropper`
  thread of that build (removed since) could not help there: `posix_fadvise(DONTNEED)` does not drop pages
  that are mapped or dirty. These are file
  pages the kernel can write back and take, not memory the process owns, but they count in the unit's peak.
- **About 2.4 GB of the proposal's anonymous memory left right after the install** (7.65 GB at 5:26:49, 5.23
  GB at 5:28:19). That fits one member's native anchors held on the host for the derivative; it is an
  inference from the timing and has not been read from the process.
- **The measurement holds more than the two arrays listed above**: 11.35 GB at its most, where the float64
  correspondence copy and the float32 query account for about 4 GB.
- **At 5:22 the trainer held 5.08 GB in its heap and 1.83 GB in anonymous mappings.** glibc raises its
  mapping threshold up to 32 MB as large blocks are freed, after which arrays under that size are cut from
  the heap, and a heap gives memory back only from its top.

Volume: 691.6 GB read and 104.1 GB written by the unit in 58 minutes, 25 min 22 s of CPU.

**Why it stopped.** After update 1's measurement the frozen loop archives the consumed observation, and that
step works only for a run named in its declaration's `consumed_trace_archive.active_runs`. The launcher's
`prepare` adds the run to that list; `declare.py`, which declares this build in its place, did
not, so the step raised and the trainer wrote its exit checkpoint at 5:39:06 AM CT. The launcher's `check`,
the unit tests and the review do not exercise that step, and none of them caught it. `declare.py` now adds the
run as `prepare` does. Before the next start the rest of the step was read against the new run: the
observer's publication of updates 0 and 1 existed for the first start, the operator the trainer loads
(`records/development/2026-11-17-capture-context/archive_inputs.py`) computes the active inputs for the new
list with no problem, and the bucket tool is on the unit's path.

**What was removed.** The stopped run's `working-files/` (16.8 GB) and its twelve raw capture streams (11.17
GB, listed in `first-start-removed-streams.txt`), which a new start makes
again, to give the next start its disk. Its exit checkpoint and every record are kept.

## The second streamed start

`joint-stock-prechoice-streamed-recovery-2`, unit `weight-patterns-streamed-recovery-2`, started at 5:56 AM
CT on November 19 from revision 3 with the same trainer sources, byte for byte, the same review binding and
the same gate (`weight-patterns-streamed-recovery-2-gate`). Two things differ:

- its declaration names the run in the archive step's list;
- the unit sets `MALLOC_MMAP_THRESHOLD_=131072` and `MALLOC_TRIM_THRESHOLD_=131072`. Those are glibc's own
  initial values; setting them turns off the growth of the mapping threshold, so every array of 128 KiB or
  more is its own mapping and goes back to the kernel when freed. It changes where arrays sit and no
  arithmetic, and the gate holds the run to that. It also makes the live arrays readable from outside, as
  the sizes of the anonymous mappings in `/proc/<pid>/smaps`.

It does not change what the first start's table shows, because no trainer source changed: a declared run's
sources cannot be edited while it executes.

**Same results, as far as it went.** The gate compared the 18 records the run wrote (the baseline, the install
of update 1 at 7:06:38 AM CT with 22,241 changed values, and update 1's measurement for the first two members)
with the continuation's and found every one equal.

**Why it stopped.** At 7:13:41 AM CT, before the third member's acquisition of update 1, the trainer's own
disk reserve refused: `OSError: complete capture/checkpoint reserve needs 76100156688 bytes; 76049248256
available`, 51 MB short. The frozen loop makes that check before every acquisition, not once: the next
checkpoint (9.26 + 7.78 GB), the cohort's captures (11.02 GB) and the machine's 40 GB, to which the driver adds
the working files still to be written (8.05 GB). The launcher's `check` had found 124.41 GB free against the
118.97 GB the declaration asks for. In the 77 minutes after it the run wrote 25.6 GB of its own (working files
16.82 GB, capture streams 8.63 GB, records 0.16 GB), and the disk lost 48.36 GB, so about 22.8 GB went to other
use of the same disk; `/tmp/agent-scratch` held 42.6 GB of sessions' scratch at the stop, 0.4 GB of it this
project's. Two things were this seat's: it started the run 5.4 GB above the check's requirement, and it had
left the first start's 16.8 GB exit checkpoint on the disk. A check at the start says nothing about disk that
other work takes during the run.

**Memory.** The unit peaked at 14.87 GB (13.8G by systemd's count), against 16.31 GB for the first start. The
samples are in `second-start-memory.log`, the trainer's mappings in
`second-start-mappings.log`.

| Phase | Anonymous memory | File pages |
|---|---|---|
| Restore and baseline (5:56 to 6:22) | up to 8.37 GB | up to 7.6 GB, 7.4 GB of it mapped |
| Proposal of update 1 (6:22 to 7:06) | 6.2 to 8.53 GB (the most, at 6:30:40) | under 1 GB |
| Install and measurement of update 1 (7:06 to 7:13) | 4.1 to 7.2 GB | up to 8.7 GB, 8.55 GB of it mapped at 7:06:56 |

**What the two allocator variables did.** The proposal held the same anonymous memory as without them, and the
most the unit held outside it fell from 11.35 to 8.53 GB. The run was slower: 26 minutes to the baseline
against 16, and 44 minutes for the proposal against 29, with 29 min 11 s of CPU for less work than the first
start's 25 min 22 s. The machine's load average was 19 to 23 during it, so how much of that is the allocator
is not separated. They are not a way to a few GB, and the third start does not set them.

Volume: 673.8 GB read and 101.7 GB written by the unit in 77 minutes.

**What was removed.** Seventeen files, 59.0 GB, listed in
`second-start-removed-files.txt`: this run's exit checkpoint payload, working
files and ten raw capture streams, and the first start's exit checkpoint payload. Both were checkpoints of
update 1, which a start from revision 3 makes again; every record of both runs is kept.

## The third streamed start

`joint-stock-prechoice-streamed-recovery-3`, unit `weight-patterns-streamed-recovery-3`, started at 7:25:39 AM
CT on November 19 from revision 3: the first start's trainer sources byte for byte, the same review binding,
the same gate (`weight-patterns-streamed-recovery-3-gate`), and the first start's unit, with no allocator
variables. Its `check` found 132.99 GB free. A third unit, `weight-patterns-streamed-recovery-3-sampler`,
writes the unit's memory and the disk's free space and the run's own size every 30 seconds.

Whoever holds this run keeps the disk above 76.10 GB free until it ends. Resuming from a stopped start's exit
checkpoint was not taken: the gate compares a run from revision 3 with the recorded one, and an exit
checkpoint of update 1 is not a state the recorded run saved.

**It passed the step that stopped both earlier starts.** The baseline was acquired at 7:43:57 AM CT
(objective 0.0038740613441862965, 53,455 cells outside), update 1 was installed at 8:20:45 AM CT (22,241
changed values) and measured at 8:30:52 AM CT (objective 0.0038744651545817364, 53,416 outside), and the
consumed observation was archived at 8:33:38 AM CT (`trace-archives/0/release.json`, phase `complete`). The
gate had compared 22 records by then, the baseline's and update 1's, and every one equals the continuation's.
Updates 2 to 4 were still to retrace when the table below was made; [how the start ended](#what-the-third-start-did-to-the-disk-and-the-change)
is below.

What the unit used through that archive step, from the sampler (`third-start-memory.log`,
`third-start-disk.log`, 7:25:39 to 8:34:19 AM CT, one line every 30 seconds) and systemd's counters:

| Stretch | Anonymous | File pages (mapped) | Unit at most, sampled |
|---|---|---|---|
| Restore and baseline, to 7:43:57 | up to 8.47 GB | up to 7.11 GB (5.76) | 11.70 GB |
| Proposal, to 8:20:45 | 6.25 to 8.56 GB | 0.33 to 0.92 GB (0.35) | 9.33 GB |
| Install and measurement, to 8:30:52 | 4.97 to 10.60 GB | up to 7.00 GB (6.89) | 14.85 GB |
| Archive step, to 8:34 | 8.57 to 8.67 GB | up to 1.61 GB (0.10) | 10.40 GB |

- **Peak: 17.49 GB** (systemd's `MemoryPeak`, 17,488,068,608 bytes), first reached between two samples at
  8:21 AM CT, as the install began. The first start's was 16.31 GB and the second's 14.87 GB, so three starts
  of the same sources stand between 14.9 and 17.5 GB against the continuation's 63.4 GB. It is not the few
  GB the brief asked for; the section below says what holds it.
- **Disk:** 130.66 GB free at the start, 92.59 GB at the least (8:28:18 AM CT, during the measurement), which
  is 16.5 GB above the trainer's 76.10 GB; the run's own folder stood at 44.96 GB at most, during the
  proposal, and at 41.79 GB after the archive step.
- **Volume and time:** 730.0 GB read and 95.0 GB written by 8:37 AM CT, 29 min 53 s of processor time.

**Implementation-peer handoff review, November 19.** In `declare_disk` the single tensor record supplies the storage parameter's source. The derivation still identifies the other inputs, and the byte count still comes from `disk_requirements`. In `declare.py` the new folder is appended to `consumed_trace_archive.active_runs`; the third declaration contains that name and its first archive completed. The replay gate enumerates the predecessor's available records, compares array bytes, and retries an apparent difference before requesting a stop. Its comparison drops time fields, local run names and local JSON receipt hashes. It verifies the contents it enumerates, so a matching replay is evidence about the reproduced updates. The gate ignores the return code of its SIGTERM request. If it exits with a mismatch, verify the trainer actually stopped before treating it as stopped. The [evaluation declaration](../../../studies/04-round-2-overlap/runs/joint-stock-streamed-threshold-evaluation/declaration.json) selects this recovery on the original stock-loss criterion.

## What the third start did to the disk, and the change

**The third start was stopped on purpose after update 4, and the reason was this build.** Marcos reported on
November 19 that the machine was lagging under the trainer's constant reading and writing of the NVMe, and
directed that the rate be limited at once and the cause removed.

**What the unit did** (`third-start-io.txt`, its cgroup's counters): 2.01 TB read and 227 GB written by 10:56
AM CT, and 2.38 TB read and 258 GB written by 11:24 AM CT, four hours after it started, in 86.8 million reads.
That is 165 to 225 MB/s of reads for the whole run, with about half the machine's memory free.

**The cause was the release this build was made of.** To keep the unit's memory small, `host_pages` did three
things every time a phase let an array go: it flushed the array's changed pages (`msync`), took them out of
the process (`MADV_DONTNEED`), and told the kernel to drop them from the page cache (`POSIX_FADV_DONTNEED`). A
`Dropper` thread dropped the model, checkpoint and capture files from the cache once a second, and the model
copy flushed and dropped every 256 MB. So nothing the trainer had read or written was ever in memory when it
was needed again, and every phase read its 7.5 to 9.3 GB files from the disk again. The memory figures above
were bought with that, and the samples that judged the build measured memory only.

**The limits, and one mistake in them.** At 10:41:27 AM CT the unit was limited to 120 MB/s of reads and 25
MB/s of writes (`io.max`, as runtime properties of the unit; the NVMe has no scheduler, so weights and
`ionice` do nothing there). At 10:56:28 AM CT the trainer was asked to stop and the limits were raised to 300
and 200 MB/s so that its exit would be quick. That was wrong: the frozen controller honors a stop only between
updates (`records/development/2026-11-17-final-experiment/run.py`, the `while not stop` loop), so the trainer
went on through update 4 and read 371 GB more at about 225 MB/s. At 11:24:11 AM CT the limits went back to 120
and 60 MB/s and stayed until the unit ended.

**How it ended.** Update 4 was installed at 11:21:10 AM CT (21,443 changed values) and measured at 11:41:00 AM
CT (objective 0.00387460960589942, 53,419 cells outside). The gate had then compared every record of updates
0 to 4 with the continuation's and exited 0 at 11:41:02 AM CT: **the streamed build reproduces the recorded
run through update 4.** The archive step kept the observation locally, the exit checkpoint was saved at
11:49:22 AM CT (`final-state.json`, reason `stop_requested`, iteration 4), and the unit ended at 11:49:37 AM CT
with exit code 0. Its peak was 19.24 GB. The free disk stood at 81.11 GB at its least while it trained (9:47 AM
CT) and fell to 73.17 GB while the exit checkpoint was written, under the 76.10 GB the trainer asks before an
acquisition: one more update would have stopped it on the disk (`third-start-disk.log`,
`third-start-memory.log`, complete).

**The change** (`host_pages.py`, `group_disk_state.py`):

- A release only takes the pages out of the process (`MADV_DONTNEED`). They stay in the kernel's page cache,
  which keeps them while memory is free, drops them the moment anything else needs it, and writes the changed
  ones to their file in the background.
- `drop_cache`, the `Dropper` thread and its watch list are gone, and the model copy is an ordinary copy.
- A phase's end (`Working.settle`) waits for each working file's background writes (`Store.sync`,
  `fdatasync`), and for the working model's through a descriptor opened for that, so a write the disk refused
  stops the run where the earlier build's `msync` did. The model copy ends with the same wait.

**The review.** An independent review subagent (Opus) read the change against the four papers and the goal:
`2026-11-19-page-cache-release-review.md`, with an
addendum on the final bytes. Its answer is yes to a fourth gated start. No value any reader sees can differ:
a dirty page of a shared mapping stays dirty in the cache after `MADV_DONTNEED`, and every reader in the
build goes through the page cache. The unit tests for the fifteen affected modules passed on the secondary test host
(`tools/remote-tests.sh`, 94 tests, one skipped).

**What the change does not remove: about 59 GB written per update.** The reviewer counted it from the code,
and it is not measured yet: the direction file three times over the member passes (22.6 GB), the
displacement (7.5), the remainders (7.5, `full_state.py:421-424`), the leaves (7.5, `sync_parameter` in the
frozen `run.py`), the working model (9.26, `full_state.install` rewrites every eligible span) and the capture
streams (about 4.8). About 21,000 values change in an update, and 16.8 GB of model and leaves are rewritten
for them. Three reductions are possible and none is built:

- Leave a leaf span alone when the installed bytes equal what it holds (in `Working.sync`; about 7.5 GB an
  update). The comparison is of bytes, not values, because of negative zero.
- Install only the spans that changed (in `full_state.py`, which the lineage binds; about 9 GB).
- Keep the direction, displacement and remainders in memory instead of files (about 37 GB an update, for
  about 22 GB more anonymous memory).

**A low write limit makes the lag worse.** The machine's dirty limit is 1 GiB (`vm.dirty_bytes`), with
background writing from 256 MiB. A unit that dirties pages far faster than its write limit lets them drain
fills that 1 GiB, and then every writer on the machine waits in the kernel, whoever it is. So the patched
build runs with no write limit; a read limit is harmless. The released pages are also file cache charged to
the unit, about 48 GB that the kernel takes back on demand, so the unit's total reads higher than before while
its anonymous memory does not.

**Left on the disk and in the bucket.** The third start's records are kept. Its raw capture streams went
(`third-start-removed-streams.txt`, 12 files, 10.03 GB; a start makes them again). Its exit checkpoint of
update 4 is in the bucket, read back before the local payload left (`third-start-removed-checkpoint.txt`,
with the command that brings it back). It is not what the next start resumes from: the launcher's resume
route refuses this build's sources, and the gate holds a run from revision 3 to the recorded one.

## The fourth streamed start

`joint-stock-prechoice-streamed-recovery-4`, unit `weight-patterns-streamed-recovery-4`, started at 12:13:31 PM
CT on November 19 from revision 3, declared at revision 8 with the two changed sources and the review above
bound. The same gate holds it to the continuation's updates 0 to 4
(`weight-patterns-streamed-recovery-4-gate`), and `sampler.sh` (`weight-patterns-streamed-recovery-4-sampler`)
writes its memory, the disk and its reads and writes every 30 seconds (`fourth-start-memory.log`,
`fourth-start-disk.log`, `fourth-start-io.log`, copied here at milestones from `/tmp/wpprobe/streamed4-*.log`).
It runs with a read limit of 120 MB/s and **no write limit**; nobody adds one, for the reason above. The disk
had 135.43 GB free when it started.

**The baseline equals the recorded run's**: acquired at 12:30:29 PM CT, objective 0.0038740613441862965, 53,455
cells outside, and the gate found every record of update 0 the same.

**What changed on the disk, first 36 minutes** (to 12:49:37 PM CT, the restore, the baseline and half of the
first proposal):

| | Third start | Fourth start |
|---|---|---|
| Read | 165 to 225 MB/s for the whole run | 118.2 GB, 55 MB/s on average; 22 to 89 MB/s by five minutes, and one measured minute at the 120 MB/s limit |
| Written | 18 to 22 MB/s for the whole run | 69.4 GB: 39.2 GB to set the working files up, then 31 MB/s |
| Reads a second | about 6,000 | about 2,400 |

**The reads that remain are the kernel taking the pages back.** The reviewer expected about 48 GB of released
pages to stay in the cache. The unit kept 14 to 19.6 GB: by 12:46 PM CT, 69.6 GB of its 98.5 GB of reads were
pages it had held, lost and read again (`workingset_refault_file`, 17.0 million pages), and 139 GB of its
pages had been reclaimed. The machine had 59 to 70 GB in use by processes and 4 to 7 GB free when it ran, so
about 50 GB of cache is shared by every project on it, and the trainer's working files alone are 39 GB plus
the two models. **The build streams 39 GB of dense working state through files, and that state has to be in
memory or be read again each time a phase touches it.** The change stopped the trainer from throwing it out
itself; it cannot make room for it.

**What the machine felt.** The desktop session's own stall counters (`session.slice`, where the shell runs)
read 0.00% for I/O and for memory over the five minutes to 12:49 PM CT, in the middle of the proposal. The
machine-wide I/O pressure was 8 to 12% before the trainer started (other projects' work) and 8 to 13% with it;
the trainer's own waiting is part of that figure. Dirty memory in the unit reached 0.75 GB once and was under
0.25 GB otherwise.

**Memory.** Anonymous memory reached 12.07 GB (12:34 PM CT). The unit's total reached 29.09 GB, because the
cache pages it reads are now counted to it; that total is not comparable with the earlier starts' peaks, which
counted almost no cache. It stayed far under the host resource governor's 60 GB.

**Update 1 equals the recorded one** (measured 1:04:58 PM CT, archived 1:13:48 PM CT; every record the gate
compares was the same). From the baseline to that archive, 43 minutes against the third start's 55, the unit
read 97.3 GB and wrote 55.8 GB. By 1:16 PM CT it had read 170.6 GB and written 96.9 GB, 32.7 million of its
cache pages (134 GB) had been lost and read again, and the desktop session's I/O and memory stall counters
were still 0.00.

**Update 2 equals the recorded one too, and the start was stopped there on purpose.** It was asked to stop at
1:27:17 PM CT, during update 2, so that the build below could replace it. The controller honors a stop between
updates: it installed and measured update 2 (1:44:07 PM CT; the gate found all 34 records of updates 0 to 2
the same), saved its exit checkpoint at 1:49:09 PM CT (`final-state.json`, sha256 `9d3b1ff522de…`, iteration
2) and its unit ended at 1:49:18 PM CT with exit code 0. The gate then exited 2, its answer when the watched
unit ends before updates 3 and 4 are written. Update 2 took 30 minutes from the archive of update 1 to its
measurement and moved the same traffic as update 1: 98.5 GB read and 56.2 GB written. The exit checkpoint
read 30.9 GB and wrote 16.4 GB. Over its 96 minutes the unit read 268.3 GB and wrote 156 GB (systemd's
count; third start: 2.38 TB read in four hours), its anonymous memory reached 14.74 GB, its total with cache
38.99 GB, and the machine's available memory was never under 44.44 GB. Its records, capture streams and exit
checkpoint (16 GB) are on the disk as it left them; the checkpoint is a state the recorded run and the third
start's update-4 checkpoint in the bucket already cover.

**Where a step's reads and writes come from** (the unit's `io.stat` between the events of
`events.jsonl`, first update; each figure is one phase):

| Phase | Read | Written | What moves |
|---|---|---|---|
| Setup: `state.restore` (9.9 min) | 52.4 GB | 26.5 GB | The model copied to its working file, the FP32 weights and the remainders made |
| `model.load` | 2.3 GB | 5.8 GB | |
| `training.baseline` | 5.2 GB | 6.5 GB | |
| `proposal.weight_derivative` | 15.2 GB | 22.6 GB | The direction file (7.5 GB) written about three times: each cohort member's gradient is summed into it |
| `proposal.native_rounding` (the preview) | 12.1 GB | 7.2 GB | The displacement file written, the direction read back |
| The two responses of the actual displacement | about 18.4 GB | | Direction and displacement read back |
| `weights.install` | at least 14.6 GB | at least 14.1 GB | Every span of the model file and of the FP32 weights written |

**About 59 GB written a step is not inherent to the build; about 8 GB is.** The review bound to this start
opens by calling the 59 GB inherent, and this seat repeated that when it reported the fourth start; the
count in the section above, now measured, says otherwise. The four
recorded updates changed 22,241, 21,533, 21,185 and 21,443 stored values of 1,881,586,210, in 367 to 379 of
the 538 tensors (each update's `changed_values`). For those the install wrote the 9.3 GB model file whole
(`FullState.install` writes each span it previewed, changed or not) and the frozen `sync_parameter` copied
every span into the 7.5 GB weights file. Only the dense remainders really change everywhere each step. The
direction and the displacement are made again by every step; in files, each pass over them is disk traffic,
because the kernel takes the unit's cache back (the unit has no memory limit, and research units are never
given the host resource governor's 60 GB ceiling; the machine had 4 to 7 GB free and about 50 GB of cache shared by every
project). Which working files stayed in memory, sampled every 30 seconds (`fourth-start-resident.log`, GB
resident of GB in the file): the weights file is read back whole during a derivative and gone again after
it, the remainder file is almost never resident, and 0.7 to 0.8 GB of the model file is.

**The next build** is the [fifth streamed start](#the-fifth-streamed-start)'s:

- an install writes the pages of the model file whose bytes differ (`full_state._write_changed`);
- an installed span reaches the weights file as the elements whose bits differ (`Working.sync`);
- the direction and the displacement are RAM tensors again, as before the streamed build.

**Who held it.** `research-optimizer` started it, held it and stopped it.

## The fifth streamed start

`joint-stock-prechoice-streamed-recovery-5`, unit `weight-patterns-streamed-recovery-5`, started at 2:05:03 PM CT
on November 19 from the saved revision 3, with the same objective, matcher, step fraction and endpoint. It
exists because of what the fourth start still sent to the disk: 97.3 GB read and 55.8 GB written in an update
that changed 22,241 stored values. Marcos had asked for the trainer's disk rate to be limited and its cause
removed (November 19); the fourth start carried the limit and removed the forced eviction, and this build
removes the writes the method does not need.

**What changed, in four trainer sources** (`full_state.py`, `group_disk_state.py`, `group_contrast_driver.py`,
`group_contrast_experiment.py`):

- An install writes only the pages of the model file whose bytes differ from the bytes it has just checked
  the file holds (`full_state._write_changed`). Through the fourth start it wrote every span, 9.3 GB.
- An installed dense span reaches the FP32 weights file as the elements whose bits differ (`Working.sync`).
  Through the fourth start the whole 7.5 GB file was written. A sparse-row table keeps its frozen copy.
- The direction and the displacement are RAM tensors again, as they were before the streamed build
  (`gradients`, `propose`; the `Working.preview` hook is gone). They were two 7.8 GB files that every member's
  pass read and wrote: the direction about three times a step.

The working files are 24,329,444,544 bytes (model 9,311,304,896, weights 7,509,078,016, rounding remainder
7,509,061,632), where they were 39,895,039,168.

**The review.** An independent review subagent read the change against the four papers and the goal:
`2026-11-19-changed-only-install-review.md`,
YES with five conditions. No computed value changes: the model file's bytes after an install, the weights'
bits, the direction and the displacement are the same. Its conditions and what was done with each:

1. Apply only after the fourth start has stopped cleanly, and declare with `declare.py`. Done: the fourth
   start's unit ended at 1:49:18 PM CT with exit 0, and this start's declaration was written at 2:04 PM CT.
2. Hold the first start with the build to the recorded updates 1 and 2 with `gate.py`. This start is held to
   updates 0 to 4, and the gate stops it at the first record that differs.
3. Run the test modules that import the changed sources via the remote runner before committing. Done, below.
4. Correct two texts before declaring: the derivation of `working_storage_bytes`, which cited the resume
   checkpoint, and `declare.py`'s wording (the model file is never mapped; the remainder is the BF16 rounding
   remainder, not an optimizer's). Done.
5. Correct this page's and `STATE.md`'s numbers when the start is declared. Done: they are the numbers below.

The review is bound to the patch with sha256 `add4b84f…`. The patch that was applied has sha256
`4fe16bc2039f0be135c5b45c34a09b31a60778333bdc9d7c4f8fc80c17b0a6de`: it differs by the two texts of condition 4
and by one test of mine, corrected after its first run (below). No line of the trainer's computation differs
from what was reviewed.

**The tests** ran on the secondary test host through `tools/remote-tests.sh`, not on this machine: the seven modules that
import the changed sources and `test_host_pages`, 69 tests. 68 passed and one failed, a test written with this
change (`test_an_installed_span_that_keeps_its_bytes_is_not_written_again`). The failure was the test's: it
asked the BF16 tensors for 2^-25 beside 1.0 and expected a remainder, and the remainder of a BF16 tensor is
formed from the FP32 sum, where 1 + 2^-25 is 1.0, so there is none. Its other assertion, that nothing is
written, passed. The test now asks 2^-20 of the BF16 tensors and 2^-25 of the FP32 one, and its module passed
again (24 tests). The tests of the old `Working.preview` hook were removed with the hook, and tests of the
changed-only install and sync were added, as part of this change.

**What the reviewer corrected in the numbers, and what is predicted.** Each is measured by the sampler every
30 seconds:

| | Fourth start, update 1 | Predicted for this build |
|---|---|---|
| Written in an update | 55.8 GB | about 12.5 GB: the 7.5 GB rounding remainder, about 4.8 GB of capture streams, at most 0.2 GB of changed pages |
| Read in an update | 97.3 GB | 20 to 30 GB, a prediction and not a consequence of the change |
| Trainer's anonymous memory | 8 to 13 GB (most 14.74) | 24 to 28 GB from the preview to the end of the installed response |
| Free disk the trainer asks before an acquisition | 76.10 GB | 68.06 GB |
| Free disk the launcher asks at the start | | 103.40 GB (68,055,122,192 + 11,015,264,256 + 24,329,444,544 bytes); 156.0 GB were free |

An earlier estimate of mine, 8 GB written an update, left the capture streams out; 12.5 GB is the reviewer's
count. The remainder file is still written whole each step: every remainder changes at every step.

**The cost, stated before the start.** The direction and the displacement, about 15.6 GB, are anonymous
memory again, and they take that much from the memory the kernel can reclaim, so the weights file may be read
back from the disk more often. At 2:00 PM CT, with no trainer running, the machine had 57.6 GB available and
the other projects held 56.2 GB of anonymous memory, so 30 to 34 GB is expected to stay available in that
window. the host resource governor starts queueing other projects' new heavy work when memory is about three-quarters full,
30.25 GiB (32.5 GB) available on this machine: the window is expected at that line, and the retrace measures
which side of it. The declared rule: if the trainer's anonymous memory passes 28 GB by more than a tensor's
worth, or available memory is under that line while the trainer's memory is what holds it there, the start is
stopped at the next update and the memory is lowered first. The displacement is nonzero only at the stored
values an install changes, so it can be kept as those elements instead of whole arrays; failing that the
direction goes back to a file. No memory limit is put on the unit.

**What it runs with.** A 120 MB/s read limit on the project's disk and no write limit, as the fourth start.
If the machine lags, the read limit is lowered.

**The disk between the two starts.** From 1:49 to 2:05 PM CT no trainer ran. At 1:59 to 2:03 PM CT the machine's
I/O pressure was 46 to 67% (`/proc/pressure/io`, some, 10-second average) at a load of 35, and over 15 seconds
one unit of another project (a file search in the remote runner seat's session) read 1,090 MB/s; it had ended two minutes
later and the pressure was 19%. The desktop session's own stall counter read 0.00 throughout. This takes nothing
from what the trainer did to the disk, which is measured above; it is recorded because a reading of this
start's effect on the machine has to be taken beside what else is running.

**Who holds it.** `research-optimizer` started it and holds it until the research seat takes it on
`STATE.md`. The seat that holds it keeps the disk above 68.06 GB for the whole run, the exit checkpoint's
16.8 GB included. Updates 0 to 4 retrace the recorded run; new information starts at update 5.

**What updates 1 and 2 measured.** Every record the gate has compared equals the recorded run's: the baseline
(2:21:27 PM CT), update 1 (22,241 changed values, installed 2:44:08 PM CT, measured 2:48:41 PM CT) and update 2
through its install (21,533 changed values, 3:19:38 PM CT) and fresh measurement. From the unit's `io.stat`
(`fifth-start-io.log`), update 1 counted from the baseline to its archive (2:57:51 PM CT):

| | Fourth start, update 1 | Predicted | This start, update 1 |
|---|---|---|---|
| Written | 55.8 GB | about 12.5 GB | 14.6 GB |
| Read | 97.3 GB | 20 to 30 GB | 57.3 GB |
| Baseline to archive | 43 min | | 36 min |
| Trainer's most anonymous memory | 14.74 GB | 24 to 28 GB | 22.75 GB (24.17 GB in update 2) |
| Least available memory | 44.44 GB | 30 to 34 GB | 41.56 GB (45.39 GB in update 2) |

The writes are a quarter of what they were and near the reviewer's count. The reads fell by 40 GB and are
about twice my prediction, which was wrong: 57.3 GB over 36 minutes is 26 MB/s on average. By the phase
times, 9.7 GB were read between the baseline and the start of the derivative, 13.0 GB in the derivative,
14.4 GB from the proposal to the install, 8.1 GB in the fresh generation and measurement, and 12.1 GB in the
archive of the consumed trace, which reads the capture streams it sends. The residency log shows where part
of the rest comes from: the rounding-remainder file is almost never in the cache (0.06 to 0.21 of 7.51 GB),
so each use reads it from the disk, while the model file and the weights file stay resident (9.31 of 9.31 GB
and 7.3 of 7.5 GB). Which file each phase's reads belong to was not measured. Update 2 from its start to its
install read 31.6 GB and wrote 8.5 GB.

The memory cost was smaller than stated: available memory stayed 9 GB above the host resource governor's admission line at
its lowest, so the declared stop rule did not apply. The desktop session's stall counters read 0.00
throughout. The setup, unchanged, read 60.6 GB and wrote 39.1 GB before the baseline.

**The retrace is complete and equal.** The gate exited 0 at 4:31:31 PM CT: all 58 required records of updates
0 to 4 are there and the same as the recorded run's (update 3: 21,185 changed values, installed 3:57:25 PM CT;
update 4: 21,443, installed 4:27:39 PM CT, measured 4:31:22 PM CT). That meets the review's second
condition. From here the start runs with no gate, and what it writes from update 5 on is new.

| Update, archive to archive | Read | Written | Minutes | Most anonymous memory | Least available memory |
|---|---|---|---|---|---|
| 1 (from the baseline) | 57.3 GB | 14.6 GB | 36 | 22.75 GB | 41.56 GB |
| 2 | 48.4 GB | 14.5 GB | 35 | 24.17 GB | 44.96 GB |
| 3 | 50.3 GB | 13.8 GB | 38 | 24.37 GB | 42.33 GB |
| 4, to its fresh measurement | 27.5 GB | 13.8 GB | 21 | 23.28 GB | 42.56 GB |

The fourth start's updates 1 and 2 read 97.3 and 98.5 GB and wrote 55.8 and 56.2 GB. The desktop session's
stall counters read 0.00 through all four updates. The run's folder was 35.0 GB at update 4, 23 GB of it the
working files, with 115.5 GB free on the disk.

## What would bring the peak down, and is not built

Each of these changes a source the review is bound to, so it needs the review again and goes in at a clean
stop of the run, whose exit checkpoint a later start resumes from.

- **Return a store's pages during a pass that walks it**, not only at its end (`group_disk_state.page_state`,
  `paged_preview`, `paged_checkpoint`, and the per-leaf loops of the frozen `preview` and `install`, which
  the driver wraps). A release from a second thread needs a lock against `Store.reset`.
- **Keep a member's native anchors on the card or in a store** during the proposal.
- **Page the reference** that `group_contrast_objective.register` keeps whole for each member.
- **Form the correspondence one merge at a time.** This one changes the order of a floating-point sum, so it
  cannot be held to the recorded updates by exact comparison and is a decision for the peers.

## The same pass over research's other checks and jobs

The brief asked for the trainer's pass (host memory bounded by the piece being worked on, no display) on
what else research runs here. Read from the sources and from what the units have already used, with nothing
run to measure it except one call of the verifier:

| Job | What it holds | Display |
|---|---|---|
| `weight-patterns-tracking.service` (`tools/box-image/bin/wp-box-track`, the MLflow export) | 1.0 GB, 1.4 GB at its most, over 13.5 hours | Its environment carries `DISPLAY=:0`, but `wp_overlap.py` selects the Agg backend before it plots and the process maps no X or Tk library |
| Every plotting module under `code/signal-trace/pipeline` and `tools` | — | Each sets Agg before `pyplot` (`full_loss_maps._headless_pyplot` forces it) |
| The verifier (`verify_round2.py`), called by the pre-commit hook and before native runs | Reads the pipeline's and the runs' `.py`, `.json` and `.md` one at a time and skips any over 2 MB; 9.0 s wall and 7.1 s of one core per call at 6:03 AM CT November 19 | None |
| The bucket tool (`tools/torad-research`) | A hash by stream; an upload through Bun's part-by-part file upload (about 130 MB at any file size, four 64 MiB parts in flight); a read-back of two 16 MiB ranges at a time | None |
| `archive_inputs.py`, `pass_release.py`, `held_deleted.py`, `disk.sh` | Small JSON records; hashes by stream | None |
| `pipeline.observability` | Small event records to one file; no arrays | None |
| The git hooks | The 16 MB staged-file check, the verifier, and fence's three hooks | None |
| The pipeline's unit tests | Not run here: `tools/remote-tests.sh` (the pipeline's README now says so) | — |

Nothing in that list holds more than the piece it works on, and nothing opens the display. Three things
are noted and not changed:

- The pre-commit hook's comment says the verifier runs in well under a second; it took 9.0 s. It is a
  safeguard other seats set, so its cost is theirs to weigh.
- `tools/torad-research/README.md` still gives `bun test` as a command to run here. This build changed
  nothing in that tool and did not run it; the next seat that changes it sends its tests via the remote runner.
- `weight-patterns-joint-disk-recovery.service` is listed as failed. It is a disk-backed restart of
  11:40 PM CT November 18 that was stopped after 27 seconds (6.5 GB at its most); a failed unit holds no
  memory.

## Where the tests ran

Not on this machine. The affected unit modules ran in Linux in Docker on the secondary test host through
`the remote runner remote run`. the remote runner's credential scan first refused the tree for 4,707 lines that are array names, hashes
and measured numbers; they were checked by script and cleared on the President's decision
(`records/maintenance-runs/2026-11-19-remote-run-scan/`, commits 2150bddf and ed88eea3).

The 37 pipeline test modules ran there on this build before it was committed (Python 3.12, torch 2.15.0 for
CPU, numpy 2.5.5): 236 tests, 234 passed, one skipped, and one could not find its input,
`runs/sent-overlap/wrong-inputs/tensors.json`, a 113 KB record of the original model's tensor table that
was on this disk untracked, so the remote runner did not ship it. The record is tracked now and that module ran again
there: 6 of 6. The trainer itself needs this machine's cards, so what the unit tests cannot show (the
direction file and the remainders carried between updates) is held by the gate above.

## The fifth start was killed

Times are CT, November 19, 2026.

The fifth start ran from 2:05 PM to 6:35 PM and measured updates 5, 6 and 7 after its equal retrace
(objectives 0.0038739118, 0.0038746453 and 0.0038732241; baseline 0.0038740613; stop threshold
0.0013593606). Marcos reported bursts of lag from early evening, then that the training was destroying
the machine, and stopped it. This seat sent SIGKILL to the unit at 6:35:14 PM, while update 8's install was
writing the working files (their modification times were 6:35:09 and 6:35:10 PM). There is no exit
checkpoint. The three working files (24.3 GB) were torn and are deleted; the records of updates 0 to 7
stay in the run folder.

**What the trainer was doing to the machine.** Its working state was 24.3 GB of memory-mapped files and
the 9.3 GB model, beside 20 to 24 GB of anonymous memory. At 6:34 PM its cgroup held 10.8 GB of file
pages (7.1 active, 3.7 inactive) and had refaulted 92.8 million file pages since the start; over ten
seconds it refaulted 10,570 a second. The President's seat measured about 18,800 a second for it against
about 23,000 for all of `app.slice` a few minutes earlier, on a disk 94% full and 73% busy. The 48 to 57 GB
an update this page could not trace to files were these re-reads: pages the build released stayed in the
cache unmapped, the kernel evicted them first, and the next phase read them back. After the kill,
available memory went from 42.8 to 59.9 GB and card 0 from 11.4 GB to empty.

**What was wrong in this seat's reading.** From 5:55 PM a catcher sampled the desktop session's own
pressure counters and found CPU waits (45.2 s in 28 minutes) and almost no I/O waits (1.9 s), and this
seat told Marcos at 6:23 PM that the lag was other projects' sessions. A cgroup's I/O pressure counts its
own tasks' waits on its own reads; it does not show the machine's page cache being turned over by another
cgroup, nor a disk kept busy by it. The measure that showed it was the refault count per cgroup
(`workingset_refault_file` in `memory.stat`), which this seat did not read until the President's seat
named it. Marcos had pointed at the trainer each time.

**What follows for a next start.** The disk-backed working state is not run on this machine again. Moving
state from RAM to files lowered the trainer's anonymous memory and made the page cache and the disk pay
for it, on a machine without swap that thirty projects share. A next start keeps its whole working state
in RAM: on a rented GPU box via the remote runner, or with a trainer whose state fits in RAM here (BF16 leaves, one
tensor at a time on the card). Which one is the research seat's decision with Marcos; the rented box costs
money, which is his.
