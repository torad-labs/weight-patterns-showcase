# Weight Patterns Showcase — Standalone CPU Verification & Paper Build Makefile
#
# Repository: https://github.com/torad-labs/weight-patterns-showcase
#
# All test and verification targets run strictly on CPU (CUDA_VISIBLE_DEVICES="")
# with bounded thread counts and require no GPU, GGUF weights, or network access.

SHELL := /usr/bin/env bash
.DEFAULT_GOAL := help

SHOWCASE_URL := https://github.com/torad-labs/weight-patterns-showcase
CPU_ENV := CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1

CORE_TEST_MODULES := \
	tests.test_final_experiment \
	tests.test_full_state \
	tests.test_full_graph \
	tests.test_full_graph_stream \
	tests.test_full_loss_maps \
	tests.test_native_graph \
	tests.test_native_key_anchor \
	tests.test_trace_response \
	tests.test_model_trace_response \
	tests.test_vector_chance \
	tests.test_turn_pair \
	tests.test_compare_occurrences \
	tests.test_complete_overlap \
	tests.test_common_prefix_training \
	tests.test_group_contrast \
	tests.test_group_contrast_training \
	tests.test_group_disk_state \
	tests.test_trace_diagnostic \
	tests.test_trace_stopping \
	tests.test_validation_stopping \
	tests.test_observability \
	tests.test_recorded_losses \
	tests.test_forbidden_scan \
	tests.test_forbidden_bytes \
	tests.test_parameter_sources \
	tests.test_research_layout \
	tests.test_storage_archives \
	tests.test_round2_verifier.Round2Verifier.test_complete_overlap_declaration_passes \
	tests.test_round2_verifier.Round2Verifier.test_the_runs_folder_itself_is_not_an_input \
	tests.test_round2_verifier.Round2Verifier.test_a_target_or_an_input_that_is_no_path_is_reported_not_raised \
	tests.test_round2_verifier.Round2Verifier.test_missing_reasoning_fields_fail \
	tests.test_round2_verifier.Round2Verifier.test_bare_or_underived_parameter_fails \
	tests.test_round2_verifier.Round2Verifier.test_edit_needs_an_overlap_target \
	tests.test_round2_verifier.Round2Verifier.test_first_attempt_reference_and_unlisted_run_fail \
	tests.test_round2_verifier.Round2Verifier.test_original_model_read_needs_declared_hash \
	tests.test_round2_verifier.Round2Verifier.test_uncommitted_declaration_and_result_are_valid_records \
	tests.test_round2_verifier.Round2Verifier.test_sweep_record_does_not_require_a_separate_step_5_run \
	tests.test_round2_verifier.Round2Verifier.test_native_record_can_be_updated_after_outputs_exist

MODULES ?= $(CORE_TEST_MODULES)

GUARD_TEST_MODULES := \
	tests.test_recorded_losses \
	tests.test_forbidden_scan \
	tests.test_forbidden_bytes \
	tests.test_parameter_sources \
	tests.test_research_layout \
	tests.test_storage_archives \
	tests.test_round2_verifier.Round2Verifier.test_complete_overlap_declaration_passes \
	tests.test_round2_verifier.Round2Verifier.test_missing_reasoning_fields_fail \
	tests.test_round2_verifier.Round2Verifier.test_bare_or_underived_parameter_fails \
	tests.test_round2_verifier.Round2Verifier.test_first_attempt_reference_and_unlisted_run_fail

.PHONY: help verify checksums test test-cpu test-guard score-protocol figures paper

help: ## Show available standalone verification targets
	@printf "Weight Patterns Showcase — Standalone Verification Targets\n"
	@printf "Repository: %s\n\n" "$(SHOWCASE_URL)"
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z0-9_-]+:.*## / {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@printf "\nVariables:\n"
	@printf "  MODULES=\"...\"        Override test modules for make test-cpu\n"

checksums: ## Verify SHA-256 digests of every exported file against MANIFEST.sha256
	@sha256sum -c MANIFEST.sha256 >/dev/null
	@echo "Verified all files in MANIFEST.sha256"

test: test-cpu ## Alias for test-cpu

test-cpu: ## Run deterministic synthetic 35-block CPU test suite (CUDA_VISIBLE_DEVICES="", no GGUF needed)
	@cd code/signal-trace && \
	if $(CPU_ENV) python3 -c "import torch, numpy, matplotlib" >/dev/null 2>&1; then \
		PY_CMD=(python3); \
	elif command -v uv >/dev/null 2>&1 && $(CPU_ENV) uv run --offline --no-project --python 3.12 --with torch==2.15.0 --with numpy==2.5.5 --with matplotlib==3.11.2 python -c "import torch" >/dev/null 2>&1; then \
		PY_CMD=(uv run --offline --no-project --python 3.12 --with torch==2.15.0 --with numpy==2.5.5 --with matplotlib==3.11.2 python); \
	elif command -v uv >/dev/null 2>&1; then \
		PY_CMD=(uv run --project . python); \
	else \
		echo "Error: torch/numpy/matplotlib not found in python3 and uv is unavailable." >&2; exit 1; \
	fi && \
	$(CPU_ENV) PYTHONPATH=. "$${PY_CMD[@]}" -c 'import ctypes, os, sys, unittest; \
libc = ctypes.CDLL("libc.so.6", use_errno=True) if sys.platform == "linux" else None; \
(not hasattr(os, "memfd_create") and libc is not None) and setattr(os, "memfd_create", lambda name, flags=1: (lambda fd: (_ for _ in ()).throw(OSError(ctypes.get_errno(), "memfd_create")) if fd < 0 else fd)(libc.syscall(319, name.encode() if isinstance(name, str) else name, flags))); \
unittest.main(module=None, argv=["unittest", *sys.argv[1:]])' $(MODULES)

test-guard: ## Run stdlib-only preregistration, parameter provenance, and archive guard tests (<1s)
	@cd code/signal-trace && $(CPU_ENV) PYTHONPATH=. python3 -m unittest $(GUARD_TEST_MODULES)

score-protocol: ## Verify prospective operation-forecast scorer (pass FORECAST=... OUTCOMES=... SHA256=... or run built-in check)
	@if [ -n "$(FORECAST)" ] && [ -n "$(OUTCOMES)" ] && [ -n "$(SHA256)" ]; then \
		$(CPU_ENV) python3 reports/descent-trace-repair/score_forecasts.py "$(FORECAST)" "$(OUTCOMES)" --forecast-sha256 "$(SHA256)"; \
	else \
		tmp="$$(mktemp -d "$${TMPDIR:-/tmp}/wp-protocol-check.XXXXXX")" && \
		trap 'rm -rf "$$tmp"' EXIT && \
		printf '%s\n' '{"study":"protocol-smoke","source_model":"gemma4-e2b-bf16","edit":"declared-edit-0","operations":[{"case":"held-A","operation":"descent-1","family":"family-A","before":"wrong","prediction":"repair","evidence":"pre-edit 105-merge turn alignment"},{"case":"held-A","operation":"friction-1","family":"family-A","before":"right","prediction":"preserve","evidence":"shared orthogonal subspace"}]}' > "$$tmp/forecast.json" && \
		printf '%s\n' '{"study":"protocol-smoke","source_model":"gemma4-e2b-bf16","edit":"declared-edit-0","operations":[{"case":"held-A","operation":"descent-1","after":"right"},{"case":"held-A","operation":"friction-1","after":"right"}]}' > "$$tmp/outcomes.json" && \
		digest="$$(sha256sum "$$tmp/forecast.json" | awk '{print $$1}')" && \
		$(CPU_ENV) python3 reports/descent-trace-repair/score_forecasts.py "$$tmp/forecast.json" "$$tmp/outcomes.json" --forecast-sha256 "$$digest"; \
	fi

figures: ## Rebuild all 6 manuscript SVG figures from committed measurement records (stdlib Python)
	@cd reports/descent-trace-repair && $(CPU_ENV) python3 make_figures.py
	@echo "Rebuilt 6 SVG figures in reports/descent-trace-repair/figures/"

paper: ## Typeset reports/descent-trace-repair/paper.pdf and paper.html via Typst
	@cd reports/descent-trace-repair && \
	if $(CPU_ENV) uv run --offline --no-project --with typst==0.15.2 python -c "import typst" >/dev/null 2>&1; then \
		$(CPU_ENV) uv run --offline --no-project --with typst==0.15.2 python render.py; \
	else \
		$(CPU_ENV) uv run --no-project --with typst==0.15.2 python render.py; \
	fi
	@echo "Rendered reports/descent-trace-repair/paper.pdf and paper.html"

verify: checksums test-guard score-protocol figures test-cpu ## Run complete standalone showcase verification suite
	@echo "All standalone showcase checks passed."
