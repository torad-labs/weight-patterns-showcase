# Can the code we have recompute what the paper claims?

November 21, 2026 (CT). Marcos's condition for the release: "we still need to recompute everything that we
claim on the paper. Or at least confirm the code we have can do it. I can't ship the paper with buggy
code." This record is the confirmation's map and protocol; it fills in as the census
(`/tmp/restore-log.txt`, the per-path restore log of this day) completes.

## The three classes

Every quantitative claim in the manuscript (`reports/descent-trace-repair/paper.pdf`, committed
`98e82e176`) falls in exactly one class:

- **A — the record is bucket-held.** The producing run's declaration, results and measurements are in the
  `torad-research` store; `archive_inputs.py restore <record>.json --fetch` or the `traces download` key
  on its line of a release journal brings the bytes back, and the claim is re-readable without recomputation.
- **B — the bytes are gone; the code recomputes them.** Decoding is greedy and the engine is repeatable
  (every historical recording reproduced its run's tokens and full-score fingerprints), so a lost capture
  regenerates bit-for-bit from the intact original model with the committed recorder at a named commit.
  The recompute cost is generation plus capture time per answer (the stream recorder ran about 15 seconds
  an answer; full captures are slower), on one GPU.
- **C — confirmed at fixture scale.** The claim's arithmetic is exercised by the pipeline's own tests
  (fixtures and focused tests in `code/signal-trace/tests/`); the full-size rerun is not repeated because
  class A or B already holds for its data.

## What is known today

- The original model is intact locally (`models/gemma4/gemma4-e2b/gguf-bf16-bartowski/`, 9.3 GB) and in
  the bucket (`torad-research checkpoints`, November 21: `gguf-bf16-bartowski` — in bucket). This is the
  root of every class-B recompute.
- The paper's core runs are class A today: the one-pair lineage
  (`runs/vector-chance-final-experiment-installed-348-recovery`, 110 files bucket-held), the frozen panel
  (`runs/frozen-energy-generalization`: its 44 answers and captures are in the November 18 publication archive; its eight reading records were in neither git nor the bucket and were recovered, see below), the prefix-direction interval and its panel
  (`runs/prefix-direction-training-390-retry`, `-397-continuation`, `runs/prefix-direction-generalization`,
  bucket-held), the signal studies (`runs/descent-landmark-route-comparison` bucket-held;
  `runs/descent-operand-identity` per the census), and the same-input comparison
  (`runs/same-input-operation-comparison`, bucket-held).
- Recorded as lost on November 21 and corrected that night: the wide cohort's exported turns and the non-fitting members of `runs/controlled-bf16-1/own/original` are in the November 18 publication archive in the bucket and return with its `restore.py` (`journal/2026-11-21-consolidation-takeover.md`). The list as first written (the [loss account](../maintenance-runs/2026-11-19-evening-studies-deletion/README.md),
  as amended by the census): the wide cohort's exported turns and its captures
  (`runs/wide-overlap/`), the non-fitting members of `runs/controlled-bf16-1/own/original`, the kept
  `observability/events.jsonl` set, 0.95 GB of joint-stock raw streams, `runs/joint-pristine-stock-origin/model.gguf`
  (trivially re-copied from the original), and the declared inputs the passes had deliberately left local —
  the census names each. The paper's claims do not rest on these; the earlier constructions they belonged
  to (the whole-gap sweeps, the reader protocols) are historical evidence the papers' §13 reports from
  their committed results.
- Code health: the guard and archive modules' focused tests pass with the November 21 `make-storage` patch
  (72 tests; the only failing guard test is the registry-whole check that the deletion's drift also trips).
  The torch-dependent modules error on this host for lack of torch — they run on the secondary test host through
  `tools/remote-tests.sh`; the November 21 remote run was lost to the secondary test host's load ("exec format error"
  after a connection loss) and is to be retried before the release ships.

## What the census and the git check found (November 21, evening)

**The census was stopped and replaced by a listing.** Downloading every missing path to learn whether it
existed had restored 135 GB (72 GB of it `runs/input-overlap/inputs`, a historical input no current claim
uses) before it was stopped; one read-only listing of the bucket (`bucket.objects`, the tool's own lister)
classified all 800 gate-missing paths without moving a byte: 7 are back on disk, 118 are in the bucket
(106.1 GB, restorable on demand), and **675 are in neither**. None of the 675 is under another bucket key.

**Why so much is in neither: a local exclude.** `.git/info/exclude` line 22, `/studies/04-round-2-overlap/`,
was added 2026-11-16 05:42 CT by the independent review agent so the desktop IDE diff would stop reading study data
("delete this line to undo"). It is local (`.git/info/exclude` is never committed) and it makes git treat
every **new** file under the round-two tree as ignored. Files already tracked stayed tracked, and seats
that committed with explicit paths kept adding records (2,026 on November 16, 197 on the 17th, 908 on the
18th, 406 on the 19th), but every run record nobody named explicitly after that moment — run
declarations included (`runs/vector-chance-descent-sign-swap/declaration.json` is an example) — had one
copy, on this disk, and left with the evening deletion. The bucket held only what the release passes
sent, which by design was bulk, not the small records.

**The paper's evidence.** Of the 175 files the paper's evidence map, the memo and the release manifest
name, 167 are in git and on disk. Eight are in neither, all in the frozen eight-formulation transfer panel
(the paper's claim of 3/8 correct chains at 390 versus 1/8 at baseline, and its matched-device fit):
`runs/frozen-energy-generalization/{result,blinded-readings,blinded-responses,matched-fits,paired-outcomes,
unblinding}.json`, its `summarize.py`, and `runs/frozen-energy-fit-390-gpu1/result.json`. The panel's
`declaration.json` and `panel.json` are in git. The bucket holds none of the eight. The release manifest's
uncommitted additions (`FILES.txt`) name them, so the release bundle, as edited, cannot build until they
come back.

**Where they can come back from.** The agent session logs of November 16–18
(`<session-transcripts>/2026/11/1{6,8}/rollout-*.jsonl`) contain the blinded readings' frozen SHA-256
(`a37db3cb…`, the unified review's) and the file names, so the files' contents or the commands that wrote
them may be recoverable from the transcripts. They were: all eight were recovered the same night without grading again, with `blinded-readings.json` equal to that frozen SHA-256 and `result.json` equal to the hash `records/development/2026-11-18-joint-fit-panel/inventory.json` recorded on November 18 (`journal/2026-11-21-consolidation-takeover.md`). The fallback that follows was not needed. Failing that, class B applies: the 44
generations regenerate bit-for-bit from the four frozen weight states (the original, 175, 189 and 390,
whose checkpoints are bucket-held) with the committed panel and recorder, and the blinded physics reading
is redone by blind readers against the frozen rubric.

## The protocol, per claim

1. Name the claim's run and record (the [claim-to-evidence memo](../../reports/descent-trace-repair/milestone.md)
   holds the map).
2. Class it (A/B/C above) from the census log and the bucket registry.
3. For A: name the restore command and confirm the restored bytes match the record's hashes.
4. For B: name the entry point (`code/signal-trace/TRAINING.md`, the recorder and learner commands), the
   commit, and the expected deterministic outcome; run it at fixture scale, and at full scale only where
   the class-A record is absent.
5. For C: name the test module and pass it.
6. Record the confirmation beside the claim in the release's evidence map before the package ships.
