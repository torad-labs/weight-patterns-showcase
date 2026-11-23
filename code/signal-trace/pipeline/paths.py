"""Repository locations shared by every pipeline stage.

Round two lives in studies/04-round-2-overlap: ROUND-2.md and round-2-method.json
describe it, inputs/ holds the read-only original-model data and runs/ holds one
folder per run. Older locations remain as links for historical records.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RECORDER = ROOT/'code/signal-trace/recorder'
ROUND_TWO = ROOT/'studies/04-round-2-overlap'
ROUND_TWO_INPUTS = ROUND_TWO/'inputs'
ROUND_TWO_RUNS = ROUND_TWO/'runs'
ACQUISITION = ROUND_TWO_INPUTS/'acquisition'
OBSERVATION = ROUND_TWO_INPUTS/'observation'
ORIGINAL_MODEL = ROOT/'models/gemma4/gemma4-e2b/gguf-bf16-bartowski/google_gemma-4-E2B-it-bf16.gguf'
# Historical modules resolve their recorded run root through this link.
SIGNAL_TRACE_RUN = ROOT/'runs/legs/gemma4-e2b/signal-trace-A-2026-11-07'
RUN = SIGNAL_TRACE_RUN
