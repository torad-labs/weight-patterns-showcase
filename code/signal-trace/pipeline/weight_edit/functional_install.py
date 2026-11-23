"""Publish a complete native joint update without inventing a legacy sweep."""
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import gap_state


def manifest(file: Path) -> Path:
    return Path(str(file) + '.joint.json')


def _save(file: Path, record) -> None:
    pending = Path(str(file) + '.next')
    with pending.open('w') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, file)


def install(start: gap_state.Working, expected_start_sha256: str, matrices: Sequence[str],
            weights: Mapping[str, np.ndarray], output: Path) -> dict:
    names, output = tuple(matrices), Path(output)
    if not names or len(set(names)) != len(names) or set(weights) != set(names):
        raise ValueError('every declared matrix must be supplied exactly once')
    if output.exists() or manifest(output).exists():
        raise ValueError('a joint native state is written once, never overwritten')
    kept = start.kept()
    if kept['model_sha256'] != expected_start_sha256:
        raise ValueError('the current native start differs from its declared state')
    # Model tensors are output x input; the native writer owns input x output.
    # Validate every matrix before creating a candidate file.
    for name in names:
        gap_state.encoded(weights[name].T, start.bindings[name])
    output.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = dict(format='functional-native-state/1', status='writing',
                  start_model_sha256=expected_start_sha256, matrices=list(names), written=[])
    _save(manifest(output), record)
    working = gap_state.Working.make(output, start.original, start.manifest['original_sha256'],
                                    start.bindings, start=dict(file=str(start.file), kept=kept))
    for name in names:
        working.put(name, weights[name].T)
        record['written'].append(name)
        _save(manifest(output), record)
    bound = working.kept()
    record.update(status='complete', model_sha256=bound['model_sha256'], kept=bound)
    _save(manifest(output), record)
    return record


def load(file: Path, expected_sha256: str, matrices: Sequence[str]) -> dict:
    file = Path(file)
    record = json.loads(manifest(file).read_text())
    names = list(matrices)
    if (record.get('format') != 'functional-native-state/1' or record.get('status') != 'complete'
            or record['matrices'] != names or record['written'] != names):
        raise ValueError('the complete declared joint update has not been published')
    if record['model_sha256'] != expected_sha256 or gap_state.digest(file) != expected_sha256:
        raise ValueError('the reached native file differs from its declared sha256')
    return record
