# Organization and provenance

**Historical record of the first relocation.** The later November 7 consolidation moved checkpoints, the legacy replay and reference inputs here as well. Use the [current layout](INDEPENDENCE-PLAN.md), relocation record, [checkpoint catalog](../CHECKPOINTS.md) and verification for current locations. The account below retains what was true at the first move.

Organized November 7, 2026 (CT), at Marcos's request to keep the theory, evidence, custom code and Decode Machine together for a later blog post.

## Physical relocation

The canonical home is `~/Documents/dev/research/weight-patterns`.

Twenty-six directories were moved by same-filesystem rename:

- One complete Decode Machine application from QGRE's `.dev/decode-machine`.
- Seven standalone tool directories from Rig's `local/research` area.
- Eighteen study directories from QGRE's `runs/legs/gemma4-e2b` area.

Every original location now has a compatibility symlink to the new directory. The moves preserve the directories' filesystem identities; large evidence payloads were not copied or regenerated. [relocation.json](relocation.json) contains each old/new path and directory identity.

The 18 study directories contain **217,635 regular files**, totaling **74,582,504,693 bytes** (about 74.58 GB decimal / 69.46 GiB), counted without following external links or dependency packages. This is a file-length inventory, not allocated disk usage. These totals exclude the application, separate helper folders, and external model checkpoints. [Study inventory](../studies.json).

## Preservation and new material

- All 109 originally versioned application files matched their pre-move SHA256 values immediately after relocation. Original file list.
- Three application Markdown entry points subsequently received a dated navigation note. Their pre-organization versions are in `references/history/`. No application behavior changed.
- Historical study files, including failed runs, amendments, reviews, and superseded interpretations, remain intact. The reading guide labels superseded methods instead of rewriting their history.
- The two user-supplied theory attachments were copied into `references/originals/`; their original attachment locations remain intact.
- Seven research SVGs were copied byte-for-byte into `figures/exports/`, with SHA256 and source links in [figures.json](../figures.json). Eleven existing observatory screenshots are indexed at their retained locations.
- Sixty-one top-level study Markdown files are indexed. Seventy-one source tables were extracted verbatim from those documents and the application Markdown documents. Some tables repeat earlier measurements; they must not be added together.
- New writing consists of the reading guide, November 7 clarification, consolidated findings, experiment/code indexes, and blog preparation. It is distinguishable from the primary records.

## Local links and reproducibility

compatibility.json lists the links that preserve relative application paths and shared dependencies. Models, native engine checkouts/builds, QGRE-integrated instrumentation, the shared earlier replay library, and the NVIDIA reference folder remain in their original maintained locations. The files physically moved here are not a second checkout of either engine.

This collection is usable on this computer. It is not a portable, self-contained release: absolute dependency links and original run commands need deliberate adaptation on another machine. Source versions and model hashes recorded in the studies remain the identities for their results. A current dependency checkout is not automatically the historical binary's source.

The research repository retains code, reports, original theory texts, exported figures, compact indexes and selected small evidence summaries. Large raw captures, installed packages, model files and native executables remain local and are excluded from normal Git additions. They have not been deleted. The collection has no public remote configured by this task.

## Validation

From `apps/decode-machine`:

- `bun run test`: **71 passed, 0 failed; exit 0**.
- `bun run build`: **exit 0**. Vite retained its large-chunk advisory; there was no build failure.
- Local preview restarted on `127.0.0.1:4318` from the new directory.
- The existing browser view of recorded token 1067 loaded, including the machine, workers, L2/VRAM labels and diagnostic timeline. The separate 30k-run tab is checked after reloading its recorded files; final details are in validation.json.
- Compatibility paths and new document links were checked locally. No model inference, training, model edit, or production-engine rebuild was run during this organization.
