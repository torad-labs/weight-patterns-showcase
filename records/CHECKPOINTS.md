# Checkpoints

Current filesystem inventory: 2026-11-09T06:48:30.188774-06:00. Counted from the model store, not from a hand-maintained list. **5 complete GGUF files are local** at this snapshot. Models and patches stay under this research repository. Exact local inventory.

| Checkpoint | Current role | Full GGUF status |
| --- | --- | --- |
| Original BF16 | Source for every current rung | Local; previously R2 verified |
| Selected full-path A2, step-01-bounded | Best measured A2; source of C2 positive actions | Local and R2 verified |
| Full-path B2, step-01 | Frozen first B2 candidate; natural preservation failed | Local for native traces and common comparison; exact codes in local patches |
| Historical round-one D | Earlier single-projection comparison | Local and previously archived; open handle observed at this update |
| Original Q4_K_M | Earlier low-precision reference experiment | Local and previously archived; open handle observed at this update |

C2 is fitting. D2 is declared and collecting original background inputs. Neither has a full checkpoint at this snapshot. The current experiment uses the original BF16 model with native stored BF16 edits; the earlier Q4 work is historical.

## Released full copies and retained evidence

| Artifact | Recovery and research status |
| --- | --- |
| Superseded anchor-only F, K and FK candidates | Full copies archived with torad-research, read-back verified and released locally. Their results remain superseded learnings. |
| Full-path A2 cumulative pass2 | Rejected after behavioral regression; full checkpoint archived and verified, local full copy released. Exact patches and results retained. |
| Full-path A2 half-cap restart | Rejected after behavioral regression; full checkpoint archived and verified, local full copy released. Natural compact traces archived and verified; local working vectors released. |
| A2 two-block omission attribution | Full checkpoint archived and verified, local copy released; exact attribution and scoring records retained. |
| A2 no-success first-pass ablation | Full local copy released. All46 patches archived and verified; exact historical reconstruction verified against each applied tensor. Use the explicit legacy materialization mode for this historical checkpoint. |
| Earlier round-one A, B, C and first-preference checkpoints | Historical archive and release records below retain identities and restore paths. |
| E and invalid parallel candidates | Withdrawn; not part of the ladder or a claim about the hypothesis. |

The unit materializer now writes the already rounded BF16 code arrays directly. Older checkpoints that were formed through float32 realized-delta reconstruction retain their own hashes and the explicit `legacy_realized_delta` recovery mode. Historical outcomes are not silently assigned to newly reconstructed code arrays. See the no-success native-storage-recovery record.

## Retention records

- November9 current-run cleanup and releases
- Cumulative release identities are retained in the dated storage-release records and its run folder.
- Restart full checkpoint
- Restart compact natural vectors
- Two-block checkpoint
- No-success checkpoint and exact recovery
- Historical checkpoint identities
- November7 archive verification
- November8 releases
- Historical retention correction

The historical TQ4/safetensors folders remain outside this GGUF count. No removal or upload of those older experiments is implied by this inventory. Small reports, hashes, declarations, response text and reproducible construction records remain in Git; bulk is transferred through torad-research and verified before local release.
