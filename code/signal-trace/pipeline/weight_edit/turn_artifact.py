"""Load an immutable original turn reference; never refit it from a current trace."""
from pathlib import Path

import numpy as np
import torch

from .functional_reference import checked
from .turn_pair import Reference
from . import vector_chance


def load(root: Path, binding, *, original_model_sha256: str, expected_merges: int) -> Reference | vector_chance.Reference:
    root = Path(root)
    record = checked(root, binding)
    if (record.get('format') not in ('single-original-turn-reference/1', 'single-original-vector-chance/1')
            or record.get('complete') is not True):
        raise ValueError('the complete original turn reference is required')
    definition = checked(root, record['source_definition'])
    if (record['source_model_sha256'] != original_model_sha256
            or definition['sources']['model']['sha256'] != original_model_sha256):
        raise ValueError('the reference belongs to another original model')
    names = [name for name, entry in definition['members'].items() if entry['group'] == 'correct']
    name = record['reference']
    if (record['original_successes'] != names or name not in names
            or set(record['original_turns']) != set(names) or set(record['scores']) != set(names)):
        raise ValueError('the reference must be selected from the complete original fitting-success inventory')
    if record['trace'] != record['original_turns'][name]:
        raise ValueError('the selected trace is not the recorded original member')

    def array(source):
        checked(root, source, parse=False)
        values = np.load(root / source['record'], mmap_mode='r', allow_pickle=False)
        if values.dtype != np.dtype('float64') or values.ndim != 3 or min(values.shape) < 1:
            raise ValueError('compiled turns and tolerance need full float64 event/merge/coordinate arrays')
        if values.shape[1] != expected_merges:
            raise ValueError('the compiled reference must hold every declared merge')
        if not np.isfinite(values).all():
            raise ValueError('compiled original data must be finite')
        return values

    original = array(record['trace'])
    if record['format'] == 'single-original-vector-chance/1':
        source = record['chance']
        checked(root, source, parse=False)
        chance = np.load(root / source['record'], mmap_mode='r', allow_pickle=False)
        if (chance.dtype != np.dtype('float64') or chance.shape != original.shape[:2]
                or list(chance.shape) != source['shape'] or source['dtype'] != 'float64'
                or not np.isfinite(chance).all() or bool((chance < 0).any())
                or list(original.shape) != record['trace']['shape']
                or record.get('normalization') != 'canonical-cpu-float64'):
            raise ValueError('original vector chance must match the complete canonical turn inventory')
        return vector_chance.Reference(
            [torch.from_numpy(np.array(original[:, index], copy=True)) for index in range(expected_merges)],
            [torch.from_numpy(np.array(chance[:, index], copy=True)) for index in range(expected_merges)],
            expected_merges=expected_merges)
    tolerance = array(record['tolerance'])
    if (list(original.shape) != record['trace']['shape'] or original.shape != tolerance.shape
            or record['trace']['dtype'] != 'float64'):
        raise ValueError('compiled original dimensions differ from their recorded shapes')
    return Reference.from_turns(
        [torch.from_numpy(np.array(original[:, index], copy=True)) for index in range(expected_merges)],
        [torch.from_numpy(np.array(tolerance[:, index], copy=True)) for index in range(expected_merges)],
        expected_merges=expected_merges)
