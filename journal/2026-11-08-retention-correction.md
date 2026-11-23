# Retain useful research artifacts; discard invalid bulk data

Marcos clarified on November 8 that R2 should hold checkpoints and evidence we
need, rather than every intermediate file. Invalid data and artifacts that add
no useful evidence can be deleted. All durable research files belong under
`~/Documents/dev/research/weight-patterns`.

## Correction to the preceding cleanup

The preceding cleanup archived supplied-answer recordings even though they were
excluded from A. That was unnecessarily broad. This pass removes their large
R2 tensor/event objects, including duplicate compressed copies. Their small
cases, response records, identities and exclusion notes remain. The retained
original model and recorder/source inputs provide the regeneration route if an
audit ever needs to repeat those excluded executions.

Construction 01's combined full checkpoint and Construction 02's superseded
full-strength first trial also do not need standalone R2 copies. Their original
base, exact patches, coefficients, checkpoint identities and measured findings
remain. The last local Construction 01 working checkpoint was checked against
its recorded SHA and removed without uploading it. Its matching layer-24 patch
and reconstruction record remain.

This does not erase an unsuccessful result. Construction 01's weak fit and its
targeting mismatch remain in the notebook. Construction 02's first-step native
comparison remains useful calibration evidence. Their reconstructible full
model copies are redundant with the retained data needed to reproduce them.

## Research-folder confinement

Two hidden undo folders outside this project still held withdrawn round-two and
E artifacts. Their obsolete tensor/event bulk was removed. The remaining source,
records and compact checkpoint reconstruction were moved into
`provenance/withdrawn/` and compressed to a total of 194,544,061 bytes. Both archives
were compared against their source before releasing the extracted copies. They
are local audit provenance, not active research or R2 uploads.

The active signal-trace pipeline resolves its root and recorder inside this
project. Its current models, run tree and code have no symlinks pointing outside
the research folder. Historical paths in old records remain historical; they
are not destinations for new research output.

## What we retain

- The original model, current measured candidates and useful historical A–D data.
- Natural-response captures needed for connected-trace analysis and reproduction.
- Actual observations, comparisons, input/correspondence maps and small edit patches.
- Enough source and identity information to interpret negative findings honestly.

Archive candidates are chosen for that purpose before transfer. Reconstructible
intermediate full checkpoints can be represented by their base and exact patches.
Withdrawn, method-mismatched bulk captures do not become valuable merely because
we generated them. These are working retention choices, not a new gate or quota.

The exact deletions and retained reconstruction routes are recorded in
[the R2 record](../records/retention-correction-2026-11-08-r2.json) and
[the local record](../records/retention-correction-2026-11-08-local.json).

The current Construction 02 recovery material is now protected in R2 as well:
the two new K/V patches and the two checkpoint reconstruction records added only
**15,741,005 bytes**. Full remote readback verified those files and the existing
step-01 patch. Together with the previously archived original BF16 model, the
three patches and their coefficients reproduce the two current measured
checkpoints; both full current checkpoints remain local for ongoing work.
[Current recovery archive record](../records/retention-correction-2026-11-08-recovery.json).

## Completed disposition

- **75,484,617,553 bytes (75.48 GB) removed from R2 across 24 exact objects**,
  including checksum sidecars. Fresh listings confirmed the selected objects
  absent and all unselected objects in those prefixes retained.
- **21,416,329,393 bytes (21.42 GB) of obsolete local files removed**, across
  243 files, in addition to the preceding cleanup. This is logical file size;
  cached open handles can defer the filesystem's actual space reclamation.
- Withdrawn provenance was reduced to **194,544,061 bytes (194.54 MB)** in two
  local archives, both compared with their sources before removing those copies.
- **Five complete GGUF files remain local**: the original BF16 and Q4 models,
  historical D, and the two useful Construction 02 checkpoints. Historical
  TQ4/safetensors checkpoints are a separate retained collection, not additional
  GGUFs and not uploaded by this cleanup.

The research folder measured 342.47 GB allocated at 23:53 CT, including 223.22 GB
in runs and 117.98 GB in models. The user's approximate 200 GB was not used as a
deletion target. The whole filesystem had about 246 GiB available at this check;
other ongoing work and delayed handle release affect that number.
