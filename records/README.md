# Research Records, Syntheses & Provenance (`records/`)

`records/` holds the cumulative empirical ledger of the project, cross-run mechanistic retrospectives, independent mathematical and whole-build audits, offline diagnostic investigations, and cryptographic provenance/storage manifests.

---

## 1. Master Empirical Ledgers

- **[KNOWN.md](KNOWN.md)** — **3,290+ structured empirical entries** (`4,200+` lines) cataloging verified measurements, negative results, falsified hypotheses, numerical tolerances, and corrected confounds across every phase of the research, each linked to its primary run or report artifact.
- **[CHECKPOINTS.md](CHECKPOINTS.md)** — Inventory and reconstruction lineage for local and archived GGUF checkpoints (`BF16`, `Q4_K_M`, Round 1 `A–D`, and early Round 2 rungs).

---

## 2. Key Mechanistic Syntheses & Reviews (`records/retrospectives/`)

`records/retrospectives/` contains `479` dated peer and independent (`Eli`) audits, mathematical checks, and post-experiment syntheses. Start with these six landmark analyses:

| Retrospective | Primary File(s) | Core Contribution |
| --- | --- | --- |
| **Historical Repairs Audit (Nov 9)** | [retrospectives/2026-11-09-round-two-repairs/STUDY.md](retrospectives/2026-11-09-round-two-repairs/STUDY.md) · [MECHANISM-STUDY.md](retrospectives/2026-11-09-round-two-repairs/MECHANISM-STUDY.md) | Audits what the early multi-layer edits actually changed in generated physics solutions, separating genuine sign-flip repairs from downstream prose/arithmetic breakdowns. |
| **What Made Past Edits Work (Nov 11)** | [retrospectives/2026-11-11-what-made-the-edits-work/README.md](retrospectives/2026-11-11-what-made-the-edits-work/README.md) · [PROPOSAL.md](retrospectives/2026-11-11-what-made-the-edits-work/PROPOSAL.md) | Comparative read-only synthesis across `A`, `C`, `D1`, and `a2-path`–`d2-path`, quantifying edit dose in overlap units and diagnosing why single-sweep constraints stalled. |
| **Whole-Build & Archive Contracts (Nov 13–15)** | [retrospectives/2026-11-13-declared-input-archives-astra.md](retrospectives/2026-11-13-declared-input-archives-astra.md) · [2026-11-14-storage-only-archives-astra.md](retrospectives/2026-11-14-storage-only-archives-astra.md) · [2026-11-15-independent-source-review-peer.md](retrospectives/2026-11-15-independent-source-review-peer.md) · [2026-11-15-eli-whole-build-review.md](retrospectives/2026-11-15-eli-whole-build-review.md) | Defines the SHA-256 declared-input archive contracts (`form 2` & `form 3`) and verifies the end-to-end native `vector_loop` build against all four foundational papers. |
| **Final Experiment Math Review (Nov 17)** | [retrospectives/2026-11-17-final-experiment-math-review.md](retrospectives/2026-11-17-final-experiment-math-review.md) | Independent equation-level audit of `FINAL-EXPERIMENT.md`, checking unit-turn differentiation, radial tolerance boundaries, response-scaled step sizes, and held-history vs. native-cache assumptions. |
| **Full-Weight Phase Assessment (Nov 17)** | [retrospectives/2026-11-17-full-weight-phase-review/ASSESSMENT.md](retrospectives/2026-11-17-full-weight-phase-review/ASSESSMENT.md) | Evaluates the `538`-tensor full-weight training progression through checkpoint `390` (`loss = 6.406e-5`), documenting exact `91 J / 20 J / YES` repair alongside remaining transfer limits. |
| **Pure Descent Native Readback (Nov 18)** | [retrospectives/2026-11-18-pure-descent-native-read/README.md](retrospectives/2026-11-18-pure-descent-native-read/README.md) | Analyzes an isolated weight update driven purely by the phase-checked correct-group target on `FFN20` (with zero transplant or token-logit supervision) on free native generation. |
| **Paper Recomputability Audit (Nov 21)** | [retrospectives/2026-11-21-paper-recomputability.md](retrospectives/2026-11-21-paper-recomputability.md) | Line-by-line verification of every figure, table, and numerical claim in the November 18 case-study paper (`reports/descent-trace-repair/`) against retained code, bucket archives, and checksummed records. |

---

## 3. Key Diagnostic & Engineering Investigations (`records/development/`)

`records/development/` houses `99` focused technical investigations, counterfactual readbacks, and qualification fixtures:

- **[development/2026-11-22-joint-loss-recheck/](development/2026-11-22-joint-loss-recheck/README.md)** — Rechecks `10` sequential joint-training updates from MLflow cell records (`8,899` stock cells), demonstrating that while *as-logged* loss fell `25.7%`, charging *lost* (unmatched) cells at their last seen error reveals a `28.6%` genuine drop on update 1 followed by net degradation (`+22.6%` carried loss, `51×` increase on preserved operations).
- **[development/2026-11-18-unbanded-control-readback/](development/2026-11-18-unbanded-control-readback/README.md)** — Tests the unbanded `27`-donor group-contrast loss on known positive weight repairs vs. a known unsuccessful edit before launching group training, verifying sharp separation on repaired downhill operations.
- **[development/2026-11-12-path-forward/](development/2026-11-12-path-forward/README.md)** — Multi-seat diagnostic suite (`native_events.py`, `torch_forward.py`, `states_correspondence.py`, [READING-OF-FEEDFORWARD-33.md](development/2026-11-12-path-forward/READING-OF-FEEDFORWARD-33.md)) isolating fixed-input vs. free-history movement, GPU arithmetic reproducibility, and correspondence stability.
- **Additional Mechanistic Deep Dives:**
  - [development/2026-11-17-crossed-history/README.md](development/2026-11-17-crossed-history/README.md) — `2×2` weights-by-history factorial between checkpoints `175` and `189` (`99.66–99.80%` of diagonal loss change driven by trajectory history shift).
  - [development/2026-11-17-descent-context/README.md](development/2026-11-17-descent-context/README.md) & [heading.md](development/2026-11-17-descent-context/heading.md) — Pre-sign framing swaps at frozen checkpoint `175`, showing a single heading choice (`token 72`) flips the fitted `91 J` calculation to `11 J`.
  - [development/2026-11-18-original-operation-witnesses/README.md](development/2026-11-18-original-operation-witnesses/README.md) — Independently audited operation-level annotations across `30` original BF16 physics generations.
  - [development/2026-11-19-streamed-working-state/README.md](development/2026-11-19-streamed-working-state/README.md) — Host-RAM streaming (`63.4 GB → 15.6 GB`) and changed-page-only GGUF installation engineering, verified bit-for-bit against prior updates.

---

## 4. Provenance, Storage & Maintenance Ledgers

- **[provenance/](provenance/ORGANIZATION.md)** — Repository consolidation maps ([INDEPENDENCE-PLAN.md](provenance/INDEPENDENCE-PLAN.md), [relocation.json](provenance/relocation.json), [versioned-files.json](provenance/versioned-files.json)) and top-level cross-reference tables ([studies.json](studies.json), [figures.json](figures.json), [helpers.json](helpers.json), [source-tables.json](source-tables.json)).
- **[storage/](storage/)** — Structured archival and local disk-release records:
  - [storage/2026-11-11-disk-release/README.md](storage/2026-11-11-disk-release/README.md) — Verified R2 upload and local release reducing root disk usage from `96%` (`69 GB` free) to `56%` (`792 GB` free).
  - [storage/2026-11-12-two-candidate-models/](storage/2026-11-12-two-candidate-models/manifest.json) — Candidate checkpoint archival manifest.
  - [storage/legacy-manifests/README.md](storage/legacy-manifests/README.md) — Uncited Nov 8–9 storage-release and archive JSON manifests consolidated from `records/` root (while manifests cited by active declarations, journals, or `CHECKPOINTS.md` remain at their original `records/` paths so every historical reference resolves cleanly).
- **[maintenance-runs/](maintenance-runs/)** — `70` audited storage, retention, and recovery operations, notably:
  - [maintenance-runs/2026-11-19-evening-studies-deletion/README.md](maintenance-runs/2026-11-19-evening-studies-deletion/README.md) & hash-locked [recorded-losses.json](maintenance-runs/2026-11-19-evening-studies-deletion/recorded-losses.json) — Complete forensic inventory of the November 19 unlogged `studies/` deletion and the `63` unarchived transient files excused by exact SHA-256 in `round-2-method.json`.
  - [maintenance-runs/2026-11-22-tree-and-panel-recovery/README.md](maintenance-runs/2026-11-22-tree-and-panel-recovery/README.md) — Hash-verified restoration of Round 2 inputs and the `8` blinded evaluation panel records from bucket and publication archives.
