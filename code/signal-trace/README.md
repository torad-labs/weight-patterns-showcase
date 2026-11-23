# signal-trace

Native residual-stream recording, token-free computational trace alignment, and installed-weight trajectory repair for Gemma 4 E2B.

- **Systems & Numerical Architecture:** [`ARCHITECTURE.md`](ARCHITECTURE.md) — end-to-end dataflow, `ggml`/CUDA vs. PyTorch precision boundaries, native K-cache anchoring, lossless BF16 + FP32 remainder installation, module map, and scientific guardrails.
- **Executable Training & Stopping Guide:** [`TRAINING.md`](TRAINING.md) — preparing, checking, and running the source-bound one-pair (`final_experiment`), native-key (`native_key_experiment`), and cohort contrast (`group_contrast_experiment`) learners.
- **Method & Experiment Records:** [`ROUND-2.md`](../../studies/04-round-2-overlap/ROUND-2.md), [`FINAL-EXPERIMENT.md`](../../FINAL-EXPERIMENT.md), and [`reports/descent-trace-repair/`](../../reports/descent-trace-repair/README.md).

Retired state-field evaluation commands and answer-grading scripts have been removed or reduced to provenance stubs so output text never enters training objectives (`records/development/2026-11-17-one-pair-code-cleanup/README.md`). Historical Round-2 solvers remain byte-for-byte intact because executed `declaration.json` records verify exact SHA-256 digests of their source closures (see [`ARCHITECTURE.md` §3](ARCHITECTURE.md#3-complete-module-map-active-core-vs-preserved-historical-solvers)).

```
signal-trace/
  pyproject.toml       PEP 621 package metadata, dependencies, and pytest/ruff/pyright config
  ARCHITECTURE.md      systems, numerical precision, memory streaming, and module guide
  TRAINING.md          CLI reference for installed-weight and group-contrast training
  recorder/            native C++/llama.cpp stream observers; see recorder/STREAM-RECORDER.md
  pipeline/
    paths.py           repository locations; the single place paths are derived
    native_records/    read native captures, excerpts, and weights; run_declaration.py preflight
    acquisition/       prepare cases, invoke native recorders, export traces, archive bulk data
    qualification/     verify bitwise equivalence between stream capture and plain inference
    decisions/         cohort population rules and recorder case manifests
    correspondence/    passage and KV-cache correspondence helpers (Round 1)
    observation/       observational progression analyses without weight edits (Round 1)
    overlap/           token-free 105-merge cosine DTW alignment and cohort overlap; see overlap/README.md
    evaluation/        legacy package marker; standalone answer graders removed
    observability/     zero-network JSONL span/metric recorder for out-of-process MLflow export
    weight_edit/       active 538-tensor trace learners + hash-bound historical Round-2 solvers
    research_guard/    verify_round2.py, R2 archive contracts, and recorded-loss accounting
    maintenance/       repository layout tools
  tests/               synthetic CPU unit and integration tests
```

## Measurement boundary

What is compared is the trace of the computation (`ROUND-2.md` Vocabulary): for every generated event, how the residual stream turns at each of the 105 merges ($T = \hat{y} - \hat{x}$), together with the operator inputs and outputs that produced each turn. Trajectories are aligned purely by those turns; no token ID, parsed numerical value, or lexical position determines correspondence or progress (`journal/2026-11-14-iterative-gap-closure.md`). The installed-weight baseline fits vector excess beyond fixed witness-calibrated tolerances across all aligned events and merges; group-contrast objectives fit signed failing-operation projections while preserving orthogonal correct-solution subspaces ([`TRAINING.md`](TRAINING.md)). Tokenization and decoding serve solely to run the native model on its prompts and identify recordings (`records/development/2026-11-15-tree-cleanup/README.md`).

## Quickstart (CPU Verification & CLI)

All verification tests use tiny synthetic native-format fixtures and run deterministically on CPU with `CUDA_VISIBLE_DEVICES=""` (requiring zero GPU memory or external GGUF weights).

Using [`uv`](https://docs.astral.sh/uv/) from `code/signal-trace/`:

```bash
# Install into a local virtual environment (with dev tools) and run CPU tests:
uv sync --extra dev
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --extra dev pytest -k "not repository and not test_current_tree_passes"

# Or run the core installed-weight learner test directly via stdlib unittest:
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python -m unittest tests.test_final_experiment
```

From the repository root, `make test-cpu`, `make test-guard`, and `make verify` run the standalone checksum, guard, protocol, figure, and synthetic 35-block CPU checks (`pipeline/research_guard/verify_round2.py` checks full-bench raw input and R2 restore invariants when run in the full lab archive).

## Building the C++ Stream Recorder (`recorder/`)

`recorder/observe_stream.cpp` and `recorder/observe_stream.late.cpp` compile against pinned `llama.cpp` commit `f089c9cb4bc285a11e510e1c4336b95aca449b08` (`sm_120`, `GGML_CUDA_DISABLE_FUSION=1` at capture time). The GGML scheduler patches and exact compile scripts are shipped in:
- [`records/development/2026-11-10-speed-review/recorder-kept/ggml-backend-eval-kept.patch`](../../records/development/2026-11-10-speed-review/recorder-kept/ggml-backend-eval-kept.patch) and [`make_kept.sh`](../../records/development/2026-11-10-speed-review/recorder-kept/make_kept.sh)
- [`records/development/2026-11-10-speed-review/recorder-deferred/ggml-backend-eval-deferred.patch`](../../records/development/2026-11-10-speed-review/recorder-deferred/ggml-backend-eval-deferred.patch) and [`make_deferred.sh`](../../records/development/2026-11-10-speed-review/recorder-deferred/make_deferred.sh)
- [`records/development/2026-11-10-speed-review/recorder-early-stop/reviewed/observe_stream-early-stop.consolidated.patch`](../../records/development/2026-11-10-speed-review/recorder-early-stop/reviewed/observe_stream-early-stop.consolidated.patch) and [`late-start/make_late.py`](../../records/development/2026-11-10-speed-review/late-start/make_late.py)

Modules named `*_a_*` were written for an earlier Round-1/initial cohort and are retained for provenance; consult [`ARCHITECTURE.md`](ARCHITECTURE.md) and `ROUND-2.md` before invoking historical modules.
