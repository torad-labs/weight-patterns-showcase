"""Sparse native endpoints and per-merge moments from the two existing JVPs.

This stores no historical FP32 remainder and cannot resume an old optimizer.
"""
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import zipfile

import numpy as np
import torch


def binding(root, path):
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(record=str(path.relative_to(root)), sha256=digest)


def save(root, path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return binding(root, path)


def npz_bytes(specifications):
    """NPZ_STORED size, including large-archive offsets and ZIP64 records."""
    local = central = count_entries = 0
    for name, count, dtype in specifications:
        header = io.BytesIO()
        np.lib.format.write_array_header_1_0(header, dict(
            descr=np.dtype(dtype).str, fortran_order=False, shape=(count,)))
        payload = count * np.dtype(dtype).itemsize + len(header.getvalue())
        if payload > zipfile.ZIP64_LIMIT:
            raise ValueError('matrix entry exceeds the declared small-entry ZIP64 layout')
        length = len((name + '.npy').encode())
        central += 46 + length + (12 if local > zipfile.ZIP64_LIMIT else 0)
        local += 30 + length + 20 + payload  # NumPy forces the ZIP64 local extra.
        count_entries += 1
    zip64_end = (local > zipfile.ZIP64_LIMIT or central > zipfile.ZIP64_LIMIT
                 or count_entries > zipfile.ZIP_FILECOUNT_LIMIT)
    return local + central + 22 + (76 if zip64_end else 0)


def candidate_counts(before, after, seed):
    """Count exact transition bits and the seed-relative union for partial writes."""
    if (before.dtype != np.float32 or after.dtype != np.float32 or before.shape != after.shape
            or seed.dtype != np.dtype('<u2') or seed.shape != (before.size,)):
        raise ValueError('candidate storage inventory differs from native matrix layout')
    first, last = before.ravel().view(np.uint32), after.ravel().view(np.uint32)
    if np.any(first & 0xffff) or np.any(last & 0xffff):
        raise ValueError('storage inventory requires exact native BF16 endpoints')
    first, last = first >> 16, last >> 16
    return (int(np.count_nonzero(first != last)),
            int(np.count_nonzero((first != seed) | (last != seed))))


def storage_budget(rows, storage, exit_counts, delta_counts=None):
    names = [row['name'] for row in rows]
    for counts in (exit_counts,) if delta_counts is None else (exit_counts, delta_counts):
        if set(counts) != set(names) or any(type(counts[row['name']]) is not int
                or not 0 <= counts[row['name']] <= row['bytes'] // 4 for row in rows):
            raise ValueError('sparse storage counts omit or exceed a declared matrix')
    if delta_counts is not None and any(delta_counts[name] > exit_counts[name] for name in names):
        raise ValueError('the partial-install union cannot omit a changed coordinate')
    delta = 0 if delta_counts is None else npz_bytes([
        (f'{prefix}_{i}', delta_counts[name], dtype) for i, name in enumerate(names)
        for prefix, dtype in (('index', '<i8'), ('before', '<u2'), ('after', '<u2'))])
    overrides = npz_bytes([(f'{prefix}_{i}', exit_counts[name], dtype) for i, name in enumerate(names)
        for prefix, dtype in (('index', '<i8'), ('value', '<u2'))])
    remainder = sum(row['bytes'] for row in rows)
    captures = 1
    return dict(phase='entry' if delta_counts is None else 'rounded_candidate',
        capture_count=captures, capture_reserve_bytes=storage['capture_reserve_bytes'],
        delta_reserve_bytes=delta, exit_override_reserve_bytes=overrides,
        remainder_checkpoint_bytes=remainder, free_floor_bytes=storage['free_floor_bytes'],
        exit_counts=exit_counts, delta_counts=delta_counts,
        free_required_bytes=storage['free_floor_bytes'] + captures * storage['capture_reserve_bytes']
            + delta + overrides + remainder)


def reserve(root, run, path, budget):
    free = shutil.disk_usage(run).free
    receipt = save(root, path, dict(budget, observed_free_bytes=free,
                                    sufficient=free >= budget['free_required_bytes']))
    if free < budget['free_required_bytes']:
        raise OSError(f"{budget['phase']} requires {budget['free_required_bytes']} free bytes; available {free}")
    return receipt


class Observation:
    def __init__(self, folder, root, names, source_observation, source_identity,
                 target_identity, reference, correspondence, anchor, previous_commit, merges):
        self.folder, self.root, self.names = folder, root, tuple(names)
        folder.mkdir()
        self.metadata = dict(source_observation=source_observation, source_identity=source_identity,
            target_identity=target_identity, reference=reference, correspondence=correspondence,
            anchor_checkpoint=anchor, previous_commit=previous_commit)
        self.arrays, self.inventory, self.delta_counts, self.exit_counts, self.moments = {}, [], {}, {}, []
        self.continuous_moments = []
        self.merges = merges

    def add(self, name, before, after, seed):
        if len(self.inventory) >= len(self.names) or name != self.names[len(self.inventory)]:
            raise ValueError('sparse observation must follow the complete declared matrix order')
        count, union = candidate_counts(before, after, seed)
        index = len(self.inventory)
        old, new = before.ravel().view(np.uint32), after.ravel().view(np.uint32)
        selected = np.flatnonzero(old != new).astype('<i8')
        self.arrays[f'index_{index}'] = selected
        self.arrays[f'before_{index}'] = (old[selected] >> 16).astype('<u2')
        self.arrays[f'after_{index}'] = (new[selected] >> 16).astype('<u2')
        self.inventory.append(dict(name=name, shape=list(before.shape), changed_bit_patterns=count))
        self.delta_counts[name], self.exit_counts[name] = count, union

    def write_candidate(self, rows, storage, run):
        if tuple(row['name'] for row in self.inventory) != self.names:
            raise ValueError('prepared sparse edit omits matrices')
        budget = storage_budget(rows, storage, self.exit_counts, self.delta_counts)
        self.reservation = reserve(self.root, run, self.folder / 'storage.json', budget)
        path = self.folder / 'bf16-delta.npz'
        with path.open('xb') as stream:
            np.savez(stream, **self.arrays)
        if path.stat().st_size != budget['delta_reserve_bytes']:
            raise ValueError('sparse payload size differs from exact reservation')
        self.payload = binding(self.root, path)
        self.arrays.clear()
        return self.reservation

    def residuals(self, reference, pairs, forward, *, continuous=False):
        moments = self.continuous_moments if continuous else self.moments
        if moments:
            raise ValueError('each observation must use its existing JVP only once')
        for merge, part in enumerate(reference.measure(forward, pairs)):
            primal, tangent = torch.autograd.forward_ad.unpack_dual(part.residual)
            r = primal.detach().double()
            q = None if tangent is None else tangent.detach().double()
            baseline = float(r.square().sum())
            cross = 0. if q is None else 2 * float((r * q).sum())
            norm = 0. if q is None else float(q.square().sum())
            moments.append(dict(merge=merge, baseline=baseline, loss_cross=cross,
                response_norm_squared=norm, predicted_objective=baseline + cross + norm))
            yield part.residual  # The numerical reducer receives the original dual unchanged.

    def prepared(self, actual, continuous, applied_length):
        if len(self.moments) != self.merges or len(self.continuous_moments) != self.merges:
            raise ValueError('the per-merge table omits declared merges')
        totals = {key: math.fsum(row[key] for row in self.moments)
                  for key in ('baseline', 'loss_cross', 'response_norm_squared')}
        # The first JVP is along the freshly conditioned full-loss first-projection direction. Scale its scalar moments only;
        # the actual update and its original response-derived length are untouched.
        continuous_rows = []
        for row in self.continuous_moments:
            cross = applied_length * row['loss_cross']
            norm = applied_length * applied_length * row['response_norm_squared']
            continuous_rows.append(dict(merge=row['merge'], baseline=row['baseline'],
                loss_cross=cross, response_norm_squared=norm,
                predicted_objective=row['baseline'] + cross + norm))
        continuous_totals = {key: math.fsum(row[key] for row in continuous_rows)
                             for key in ('baseline', 'loss_cross', 'response_norm_squared')}
        raw_totals = {key: math.fsum(row[key] for row in self.continuous_moments)
                      for key in ('baseline', 'loss_cross', 'response_norm_squared')}
        return dict(format='sparse-native-transition/1', status='prepared', **self.metadata,
            payload=self.payload, inventory=self.inventory, storage_reservation=self.reservation,
            coordinates='Resident(output,input) C-order indices; exact little-endian u16 BF16 before/after codes.',
            continuous_applied_length=applied_length,
            continuous_per_merge_response=continuous_rows,
            continuous_per_merge_total_difference=dict(
                baseline=continuous_totals['baseline']-continuous.objective,
                loss_cross=continuous_totals['loss_cross']-applied_length * continuous.slope,
                response_norm_squared=continuous_totals['response_norm_squared']-applied_length * applied_length * continuous.curvature,
                predicted_objective=math.fsum(row['predicted_objective'] for row in continuous_rows)
                    -(continuous.objective + applied_length * continuous.slope + applied_length * applied_length * continuous.curvature)),
            continuous_raw_total_difference=dict(baseline=raw_totals['baseline']-continuous.objective,
                loss_cross=raw_totals['loss_cross']-continuous.slope,
                response_norm_squared=raw_totals['response_norm_squared']-continuous.curvature),
            continuous_scope='First existing JVP along the freshly conditioned full-loss first-attention direction, evaluated at prescribed applied_length; excludes carried remainder and BF16 rounding. Held-history STE prediction, not a component gradient or native result.',
            actual_displacement_coefficient=1, per_merge_response=self.moments,
            per_merge_total_difference=dict(baseline=totals['baseline']-actual.objective,
                loss_cross=totals['loss_cross']-actual.slope,
                response_norm_squared=totals['response_norm_squared']-actual.curvature),
            interpretation='Prepared endpoint edit only. Apply in a native replay only when a matching complete reached record commits it. Per-merge values are held-history STE moments, not native results.',
            limitation='No per-step FP32 remainder is saved. This supports native endpoint replay, not historical optimizer-state resume.')

    def committed(self, proposal, installation, reached_identity):
        if (installation['committed_matrices'] != list(self.names)
                or installation['in_progress'] is not None
                or reached_identity != self.metadata['target_identity']):
            raise ValueError('incomplete or differently identified installation cannot commit an edit')
        return dict(format='sparse-native-transition/1', status='committed', proposal=proposal,
            payload=self.payload, **self.metadata,
            committed_matrices=list(self.names), in_progress=None,
            interpretation='Complete native/resident/remainder installation; enclosing reached identity is the target. This is not a claim of fresh improvement.')
