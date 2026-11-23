# Checkpoints to archive to R2

Inventory checked November 7, 2026, CT. This handoff identifies existing research checkpoints suitable for copying to private R2 storage. It does not upload or delete anything, and it does not rate the behavioral quality of the edited models.

The physical model directory is:

`<qgre-workspace>/models/gemma4/gemma4-e2b/`

The research directory `models/gemma4/gemma4-e2b/` resolves to these same files. Do not upload or count both paths as separate copies.

## Archive first

These six historical checkpoints total **35.69 GiB**. Their full GGUF files are not needed for the current multitrace experiment. They are candidates for reclaiming local space after the uploaded contents have been verified and current use has been checked again.

| Checkpoint | Path relative to the physical directory above | Size |
|---|---|---:|
| A — original physics edit | `gguf-bf16-physics-weight-edit-2026-11-07/attention-output-16.gguf` | 8.672 GiB |
| B — selective edit | `gguf-bf16-physics-selective-edit-2026-11-07/attention-output-16.gguf` | 8.672 GiB |
| C — connected preservation edit | `gguf-bf16-physics-connected-edit-2026-11-07/attention-output-16.gguf` | 8.672 GiB |
| Q4 preference alpha 1 | `gguf-first-preference-2026-11-06/alpha-1.gguf` | 3.225 GiB |
| Q4 preference alpha 2 | `gguf-first-preference-2026-11-06/alpha-2.gguf` | 3.225 GiB |
| Q4 reverse control | `gguf-first-preference-2026-11-06/reverse.gguf` | 3.225 GiB |

The three physics checkpoints alone account for **26.02 GiB**. Keep their `projection-before.bf16` and `projection-after.bf16` files locally: these are only 6 MiB each and remain useful for the ongoing experiment and provenance. Do not delete whole checkpoint directories when reclaiming a GGUF.

## Also suitable for backup; retain locally

| Checkpoint | Path relative to the physical directory above | Size | Why retain it |
|---|---|---:|---|
| D — whitened/background-aware edit | `gguf-bf16-physics-background-edit-2026-11-07/attention-output-16.gguf` | 8.672 GiB | Working baseline and source for the current multitrace experiment. |
| Original BF16 | `gguf-bf16-bartowski/google_gemma-4-E2B-it-bf16.gguf` | 8.672 GiB | Original research baseline. |
| Original Q4_K_M | `gguf-q4_k_m-bartowski/google_gemma-4-E2B-it-Q4_K_M.gguf` | 3.225 GiB | Baseline for reproducing the earlier quantized experiments. |

All nine full GGUF files together total **56.26 GiB**. Each BF16 file is exactly 9,311,304,896 bytes; each Q4_K_M file is exactly 3,462,680,032 bytes.

## Transfer information

- The machine-readable inventory is checkpoint-upload-inventory-2026-11-07.json. It includes absolute paths, byte sizes, SHA-256 values, provenance reports, companion tensors, and retention guidance.
- The checksum list is checkpoint-upload-SHA256SUMS-2026-11-07.txt. Its filenames are relative to the physical model directory above.
- Most full-file checksums were recorded when the checkpoints were created; this inventory rechecked their existence and sizes rather than rereading every large file. The reverse control was freshly rehashed and matched its recorded checksum.
- Verify source and uploaded content against the expected SHA-256 and byte size before any local removal. A multipart object ETag is not a substitute for a SHA-256 verification.
- Archive the companion before/after tensors and the referenced edit reports with the model. Keep these small files, the research reports, custom code, and captured evidence locally.
- The process inspection found no matching open GGUF files or mappings at the moment of inspection. That does not remove the current experiment's dependency on D or its original baseline. Check again before deleting a file.

## Exclusions

`gguf-bf16-physics-multitrace-edit-2026-11-07/` does not yet contain a completed new GGUF. It contains editor-validation artifacts; do not label it a completed checkpoint.

The separate `tq4-e2m1-nvme/` QGRE bundle is not part of this verified GGUF list. It contains multiple files and a checkpoint pointer, so its archive and retention decision needs a separate dependency check.
