"""What is left of the first round-two edit's module: the engine's BF16 rounding, its RMS-norm epsilon and the writer of a JSON record, which the solver and
the readers use.

The edit itself (`rows`, `build`, `inputs` and the module's commands) left this file on November 15, 2026. It was built on the positions of
full-overlap/pre-error, which end before an answer's first wrong token: a token decided which rows the construction took, so the construction was placed
by the output and not by the computation. The run's records stay as history (studies/04-round-2-overlap/runs/a2; its retired-programs.json names the
commit that holds the code). The commands that had the edited model answer the questions and judged the answers had left on November 14, 2026.
"""
import json

import numpy as np

NORM_EPSILON = 1e-6  # the engine's RMS-norm epsilon, as checked against the recorded contributions


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, ensure_ascii=False) + '\n')


def to_bf16(values):
    """Float32 to the engine's BF16: round to nearest even, subnormals to signed zero (ggml_fp32_to_bf16)."""
    bits = np.ascontiguousarray(values, dtype='<f4').view('<u4').astype('u8')
    out = ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16).astype('<u2')
    subnormal = (bits & 0x7f800000) == 0
    out[subnormal] = ((bits[subnormal] & 0x80000000) >> 16).astype('<u2')
    return out


def from_bf16(stored):
    return (stored.astype('<u4') << 16).view('<f4')
