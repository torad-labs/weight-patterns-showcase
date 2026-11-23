# Archive superseded checkpoints and release unused local traces

**Later retention correction:** Marcos asked to keep only useful R2 artifacts.
The excluded supplied-answer tensor/event objects and the two reconstructible
full trial checkpoints described below were subsequently removed from R2.
Their source/identity records and small reconstruction patches remain. See the
[correction](2026-11-08-retention-correction.md) for current disposition; the
transfer and verification record below describes what happened before that change.

Recorded 2026-11-08T23:43:10.631302-06:00 (CT). Marcos asked to send checkpoints to Cloudflare and remove
round-two captures that are no longer used by the edit or analysis.

## What changed

| Material | Local copies removed | Recovery |
| --- | ---: | --- |
| Construction 01 combined checkpoint and Construction 02 full-strength step-01 checkpoint | 18.623 GB | Uploaded with `torad-research`; full remote byte readback passed before release. |
| 45 reconstructed original trace excerpts | 129.946 GB | Complete original recordings already had verified R2 copies. Retained exact request and segment maps reconstruct the same bytes. |
| Two excluded supplied-answer captures, discovery 16 and discovery 20 | 15.694 GB | Compressed native captures and their accompanying response/identity records uploaded and verified before release. |

The removed local paths total **164.262 GB**. This is not the same as
immediately recovered filesystem space. At this report, the desktop VM's shared-filesystem
process still holds cached handles to **145.642 GB** of their deleted
inodes. Those blocks become available when the handles close. No VM, other
session or other project's job was stopped. The two checkpoint releases had no
open handles and released their allocation immediately. Root reports
**237.5 GiB free** at this observation; other ongoing work also changes
that figure, so the global before/after difference is not attributed to this cleanup.

## What stays

The original BF16 model, the current step-01-scaled construction base, and the
new step-02 native probe model remain here.
All 37 compact current-checkpoint input sets, original trace targets, direction
patches, comparison maps, responses, scores, summaries and analysis outputs remain.
The newest prefix streams and selected conditional captures remain local while
Construction 02 uses them. Historical A–D results are retained. Supplied-answer
captures remain excluded from A; archival does not turn them into eligible data.

The 45 excerpt files were reconstructible working copies. Their deletion does
not remove the complete natural-response recordings from R2 or their connected
observations. The 90 request/segment maps and 46 release/source records are also
stored in a verified recovery bundle in R2, and the JSON originals remain local.

No broad run-directory deletion was used. The older 691 missing tracked files
reported by the reviewer predate this cleanup and were not included in its commits.
Other unarchived historical models and group arrays were not silently discarded.

## Evidence and restoration

- Checkpoint transfers, hashes and restore commands.
- Excluded-capture transfers, hashes and restore commands.
- Exact excerpt source and reconstruction map.
- Excerpt release and cached-handle record.
- Verified recovery bundle.
- Counts and filesystem observation.

For an excerpt, use its recorded `torad-research traces extract` command to a new
`native-restored` directory. Compare its source/segment map with the retained
`native/excerpt.json`, then put `chunks.bin` back at its recorded native path.
This preserves the distinction between a partial excerpt and a full recording.

The resumed A solve uses the retained compact inputs. Storage housekeeping does
not change its construction target, population, checkpoint precision or eligibility.
