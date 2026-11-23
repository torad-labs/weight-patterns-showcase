"""Whole-prefix computational correspondence and a frozen turn/log-gain discrepancy."""
from functools import lru_cache
import importlib.util
from pathlib import Path

import numpy as np
import torch

from pipeline.overlap.complete_overlap import unit_steps
from pipeline.weight_edit import final_experiment as baseline
from pipeline.weight_edit.turn_pair import turns


def checked(root, record, sha256):
    path = (root / record).resolve()
    if not path.is_relative_to(root) or 'reserved-40b' in path.parts or baseline.binding(path)['sha256'] != sha256:
        raise ValueError(f'Invalid prefix diagnostic source: {record}')
    return path


@lru_cache(maxsize=1)
def reference_and_registrar(root, ref_record, ref_sha, method_record, method_sha):
    reference = np.load(checked(root, ref_record, ref_sha), mmap_mode='r', allow_pickle=False)
    if reference.ndim != 3:
        raise ValueError('Frozen reference must contain events, merges and coordinates')
    path = checked(root, method_record, method_sha)
    spec = importlib.util.spec_from_file_location('frozen_computation_prefix', path)
    method = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(method)
    return unit_steps(reference), reference.shape, method.endpoints


def locate(native, artifact, *, prompt_events):
    if artifact['format'] != 'whole-prefix-fixed-turn-log-gain/1':
        raise ValueError('Unknown whole-prefix score format')
    ref = artifact['reference_turns']
    method = artifact['registrar']
    reference, shape, endpoints = reference_and_registrar(
        Path(baseline.ROOT).resolve(), ref['record'], ref['sha256'], method['record'], method['sha256'])
    rows, merges, width = shape
    target = artifact['reference_event'] - artifact['first_reference_event']
    first = prompt_events - 1
    before, after = native['incoming'], native['merged']
    if not 0 <= target < rows or not 0 <= artifact['merge'] < merges or len(before) != len(after) or len(before) != merges:
        raise ValueError('Frozen site is outside the declared architecture or reference')
    length = len(before[0])
    if not 0 <= first < length:
        raise ValueError('Prompt boundary falls outside the observed execution')
    query = np.empty((length-first, merges, width), dtype=np.float32)
    # The original acquisition normalized each edge in CPU float64 then exported
    # its turn in float32. Build the full field one merge at a time to bound RAM.
    with torch.no_grad():
        for m, (a, y) in enumerate(zip(before, after, strict=True)):
            if tuple(a.shape) != (length, width) or a.shape != y.shape:
                raise ValueError('Incomplete merge or coordinate coverage')
            [turn] = turns({'incoming': [a[first:]], 'merged': [y[first:]]}, expected_merges=1)
            query[:, m] = turn.float().numpy()
    visits = endpoints(reference @ unit_steps(query).T)
    return [dict(key=artifact['key'], event=first+i, merge=artifact['merge'])
            for i, ties in enumerate(visits) if target in ties]


def score(forward, cell, artifact):
    if cell['key'] != artifact['key'] or cell['merge'] != artifact['merge']:
        raise ValueError('Observation belongs to another frozen site')
    m, i = cell['merge'], cell['event']
    before = forward['incoming'][m][i].to(device='cpu', dtype=torch.float64)
    after = forward['merged'][m][i].to(device='cpu', dtype=torch.float64)
    target = torch.as_tensor(artifact['reference_turn'], device='cpu', dtype=torch.float64)
    if before.ndim != 1 or before.shape != after.shape or target.shape != before.shape:
        raise ValueError('Frozen and observed turn widths differ')
    n0, n1 = torch.linalg.vector_norm(before), torch.linalg.vector_norm(after)
    if not bool(n0 > 0 and n1 > 0):
        raise ValueError('Undefined turn or log gain at a zero native endpoint')
    # The frozen scorer received turns stored in float32 and norms in float64.
    turn = (after / n1 - before / n0).float().double()
    log_discrepancy = torch.log(n1 / n0) - artifact['reference_log_gain']
    residual = torch.cat((turn - target, (log_discrepancy * artifact['sqrt_gain_weight']).reshape(1)))
    return torch.linalg.vector_norm(residual) - artifact['cutoff']
