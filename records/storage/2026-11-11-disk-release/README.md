# Research archive and local disk release

Requested by Marcos on November 11, 2026 (CT): upload research data worth keeping through the research CLI, then clear unnecessary local data. The initial root filesystem reading was 96% used with 79 GiB available; the checkout occupied about 519 GiB. This is an authorized, one-off archive/release operation, not a new retention rule or automatic cleanup service.

## What remains local

The research owners identified these dependencies; none is in the release batches:

- The original BF16 model, and the complete `controlled-bf16-4` run. Fable separately owns archiving its left model after its last inspection; this operation does not touch it.
- All read-only round-two inputs under `studies/04-round-2-overlap/inputs/`.
- `controlled-bf16-1/baseline` and `controlled-bf16-1/own/original`, which current and future sweeps count from.
- `input-overlap/inputs`, `wide-overlap/exported`, `wide-overlap/capture`, and `wide-overlap/members.json`.
- The recorder qualification and kept calibration rows in controlled-bf16-1/2/3; their references, declarations, results and build records also stay.
- `state-overlap` and `state-overlap-2`, and `trace-overlap-complete-registered` including its members and correspondence pairs.
- `sent-overlap/wrong-inputs/tensors.json` and every other individual file named by a declaration's `inputs`, `target_source` or parameter `measured_from`.
- The a2-path model. Its older declarations still name it, and its existing local-retention ruling is not changed here.
- `records/development/2026-11-10-speed-review`, including the kept matrices that reconstruct old states and their readings.
- Source code, Git-tracked files, uncommitted work, small metadata, declarations, answer summaries and human-readable research records. Required directory paths remain in place after bulk release.

Performance's current scratch reconstruction is in use and excluded. No session scratch tree, unrelated project, system directory, Git history or cache is deleted wholesale.

## Archive and release procedure

The existing `torad-research` CLI sends each selected inactive folder to the private `torad-research` R2 bucket. Model keys are `checkpoints/<models-relative path>`. For these studies the trace root is explicitly `TORAD_RESEARCH_TRACES=<checkout>/studies`, preserving existing keys `traces/<studies-relative path>`.

Each batch's `manifest.json` records its complete pre-transfer inventory, proposed release paths and retained paths. `upload.log` and `verify.log` record the CLI outcomes. Full remote-byte SHA-256 verification must succeed against the local files before any local release. The source identity, size and modification time must still match, and open descriptors/mappings are checked. Non-dumpable same-user system services are inspected read-only with privileged `lsof`; zombies have no remaining descriptors or mappings. No deletion is run with elevated privileges.

Only listed, untracked bulk files are released. The old round-one capture batches also release raw `records.jsonl` beside their `tensors.bin`, after both were archived and verified. They retain summaries and response logs. Directory symlinks are not traversed or removed; the full-overlap exports and derived arrays are named as separate real folders instead.

A completed manifest has status `verified local bulk copies released` and a `released` list. An upload/verification error, changed source or open file leaves that batch's source in place. The round-two verifier runs around releases to ensure named inputs and records still exist. A running or prepared batch is not a claim of space recovered.

## Restoring data

For a model folder:

```sh
torad-research download gemma4/gemma4-e2b/early-studies
```

For a study folder, from the checkout:

```sh
TORAD_RESEARCH_TRACES="$PWD/studies" torad-research traces download 04-round-2-overlap/runs/sent-overlap
```

Use the exact folder recorded in a batch's manifest. Download verifies each archived file and does not overwrite a differing local file. Complete inactive folders were uploaded, including the metadata needed to interpret their arrays, not just selected result numbers.

## Outcome

Confirmed completed batches:

- `round1-models`: nine checkpoint files archived and SHA-verified; 9,361,636,544 local bytes released. Root had 77 GiB available then, versus 79 GiB initially, because concurrent writes consumed more than the first release.
- `sent-overlap`: all 611 files archived and SHA-verified; 322 untracked bulk files totaling 14,443,256,576 bytes released, with 289 metadata/required files retained. Its initial release stopped before deletion at a non-dumpable-process visibility check; it resumed using the unchanged verified inventory and read-only privileged `lsof`, without repeating or skipping verification.

The following additional completed batches have full remote verification and recorded local releases:

| Batch | Released bulk files | Released bytes |
| --- | ---: | ---: |
| `full-exported` | 56 | 11,858,438,368 |
| `full-calculation` | 4 | 1,176,038,804 |
| `full-pre-error` | 4 | 1,176,038,804 |
| `a2` | 4 | 3,260,436,992 |
| `a2-pattern` | 301 | 1,836,800,310 |
| `a2-diverge` | 181 | 1,602,965,444 |
| `round1-background` | 162 | 10,111,911,763 |
| `round1-multitrace` | 142 | 9,401,288,642 |
| `round1-connected` | 8 | 1,414,336,741 |
| `overlap` | 152 | 14,620,979,958 |
| `originals-overlap` | 1,849 | 3,431,514,920 |
| `q4-checkpoint` | 1 | 3,462,680,032 |
| `early-models` | 30 regular files + 30 symlinks | 71,736,410,205 |
| `tq4-model` | 4 hardlinked file paths | 9,564,854,694 |

**All sixteen batches are complete.** Counting every manifest in this directory gives 6,664 archived inventory paths, 3,259 released paths and 3,405 retained paths within those batches. No incomplete batch or still-present released path remained in the final accounting. These totals cover this operation only, not other seats' releases.

The released paths comprise 3,229 regular-file paths and 30 symlinks. The table sums regular-file sizes by path: **168,459,588,797 bytes**. TQ4's four names refer to two regular files, so the distinct data total is **163,677,161,450 bytes (163.7 GB), across 3,227 regular-file inodes**. The 30 removed symlinks add 2,070 bytes of link text, not another copy of their targets. No released inode had an inventoried hardlink left outside the release list.

The distinct regular files had **163,486,679,040 allocated bytes** in their pre-release inventories. Neither that number nor the logical-data count is presented as an exact filesystem-space delta: sparse files, block allocation and concurrent work differ. At the final measurement, November 11 at 23:08 CT, root had **238,565,920,768 bytes available (222.2 GiB), 88% used**, compared with approximately 79 GiB available and 96% used initially. The checkout occupied **385,058,488,320 bytes (358.6 GiB)**, versus about 519 GiB initially. `completion-summary.json` contains the source-counted per-batch totals, deduplication method and disk measurements.

The round-two verifier passed after every release. The final completion/commit check also runs it; its output is kept in `completion-check.log`. Original inputs, the original BF16 model, active dependencies and the owner-confirmed exclusions above remain local.

The original early-models worker, like sent-overlap, finished upload and full remote verification but stopped before deletion because its older process check could not inspect a non-dumpable service. The corrected continuation used the unchanged verified inventory and read-only privileged `lsof`; it did not repeat or skip verification. It then completed TQ4 upload, full verification and local release.

A separate `uv cache prune --offline` exited 2 after its 300-second wait for the in-use cache lock. No cache release is claimed. Its in-use protection was respected: no `--force`, lock removal or timeout override. Model caches and active environments were not cleared manually. See `uv-cache-prune.log`.
