"""Compile original-witness agreement bands for an explicitly supplied observable.

This module does not choose the computational observable or authorize a model
measurement. It consumes named ORIGINAL arrays and frozen correct-to-correct
paths, keeping the axes and exact leave-recipient-out extrema. No token enters.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from pipeline.weight_edit.agreement_loss import original_bounds

FORM = 'original-agreement-reference/2'


def digest(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def compile_arrays(folder, names, relations, paths, *, chunk, source, device='cpu'):
    """Compile a declared original bank, partitioning memory by original rows.

    `relations[name][domain]` is original-row x merge x observable-coordinate.
    `paths(axis, donor, domain)` gives [donor-row, axis-row]. Calibration never
    receives a candidate array. The caller binds where these inputs came from.
    """
    folder = Path(folder)
    if (folder/'reference.json').exists():
        raise ValueError('an original reference is compiled once')
    if len(names) < 3 or len(set(names)) != len(names) or set(names) != set(relations):
        raise ValueError('at least three distinct named original witnesses are required')
    if chunk < 1:
        raise ValueError('chunk must partition at least one original event')
    folder.mkdir(parents=True, exist_ok=True)
    domains = list(relations[names[0]])
    if any(set(relations[name]) != set(domains) for name in names):
        raise ValueError('every original witness must supply the same domains')
    axes: dict[str, dict] = {}
    record = dict(format=FORM, names=list(names), domains=domains, source=source, axes=axes)
    for axis in names:
        donors = [name for name in names if name != axis]
        axis_domains: dict[str, dict] = {}
        axis_record = dict(witnesses=donors, domains=axis_domains)
        for domain in domains:
            target = relations[axis][domain]
            if target.ndim != 3 or min(target.shape) < 1:
                raise ValueError('an observable must be original-row x merge x coordinate')
            count, merges, width = target.shape
            held = {}
            for donor in donors:
                values = relations[donor][domain]
                pairs = np.asarray(paths(axis, donor, domain))
                if (values.ndim != 3 or values.shape[1:] != (merges, width) or pairs.ndim != 2
                        or pairs.shape[1] != 2 or pairs.dtype.kind not in 'iu'
                        or len(pairs) == 0 or (pairs < 0).any()
                        or (pairs[:, 0] >= len(values)).any() or (pairs[:, 1] >= count).any()):
                    raise ValueError('an original-witness path has invalid events')
                if (len(np.unique(pairs[:, 0])) != len(values)
                        or len(np.unique(pairs[:, 1])) != count):
                    raise ValueError('an original-witness path must cover both complete traces')
                held[donor] = pairs.astype(np.int64)
            at = folder/'axes'/axis/domain
            at.mkdir(parents=True, exist_ok=True)
            shape = (2, count, merges, width)
            owners_dtype = np.min_scalar_type(len(donors) - 1)
            arrays = {kind: np.lib.format.open_memmap(at/f'{kind}.npy', mode='w+', dtype=np.float32, shape=shape)
                      for kind in ('lower', 'upper')}
            arrays.update({kind: np.lib.format.open_memmap(at/f'{kind}.npy', mode='w+', dtype=owners_dtype, shape=shape[1:])
                           for kind in ('lower_owner', 'upper_owner')})
            for start in range(0, count, chunk):
                stop = min(start + chunk, count)
                mapped_lower, mapped_upper = [], []
                for donor in donors:
                    path = held[donor]
                    chosen = path[(path[:, 1] >= start) & (path[:, 1] < stop)]
                    takes = torch.as_tensor(chosen[:, 1] - start, device=device)
                    values = torch.as_tensor(np.asarray(relations[donor][domain][chosen[:, 0]]), device=device)
                    index = takes[:, None, None].expand_as(values)
                    low = torch.full((stop - start, merges, width), float('inf'), dtype=values.dtype, device=device)
                    high = torch.full_like(low, -float('inf'))
                    low.scatter_reduce_(0, index, values, reduce='amin')
                    high.scatter_reduce_(0, index, values, reduce='amax')
                    mapped_lower.append(low)
                    mapped_upper.append(high)
                bounds = original_bounds(torch.stack(mapped_lower), torch.stack(mapped_upper))
                arrays['lower'][:, start:stop] = bounds.lower.cpu().numpy()
                arrays['upper'][:, start:stop] = bounds.upper.cpu().numpy()
                arrays['lower_owner'][start:stop] = bounds.lower_owner[0].cpu().numpy()
                arrays['upper_owner'][start:stop] = bounds.upper_owner[0].cpu().numpy()
            for array in arrays.values():
                array.flush()
            axis_domains[domain] = dict(shape=list(target.shape), files={
                kind: dict(file=str((at/f'{kind}.npy').relative_to(folder)), sha256=digest(at/f'{kind}.npy')) for kind in arrays})
            del arrays
        axes[axis] = axis_record
    with (folder/'reference.json').open('x') as file:
        json.dump(record, file, indent=1)
        file.write('\n')
    return record


class Bank:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.record = json.loads((self.folder/'reference.json').read_text())
        if self.record['format'] != FORM:
            raise ValueError('not an original-agreement reference')
        self.names = self.record['names']
        self._arrays = {}

    def _array(self, entry):
        path = (self.folder / entry['file']).resolve()
        if not path.is_relative_to(self.folder.resolve()):
            raise ValueError('an original reference file is outside its bank')
        def identity():
            stat = path.stat()
            return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        before = identity()
        if path in self._arrays:
            original, array = self._arrays[path]
            if before != original:
                raise ValueError(f'original reference changed after verification: {path.name}')
            return array
        if digest(path) != entry['sha256']:
            raise ValueError(f'original reference sha256 differs: {path.name}')
        array = np.load(path, mmap_mode='r')
        if identity() != before:
            raise ValueError(f'original reference changed while loading: {path.name}')
        self._arrays[path] = (before, array)
        return array

    def bounds(self, axis, recipient, domain):
        """Read a fixed axis, excluding the current recipient's original witness."""
        if recipient == axis:
            raise ValueError('a recipient cannot use its own original as a reference axis')
        entry = self.record['axes'][axis]
        files = entry['domains'][domain]['files']
        arrays = {key: self._array(value) for key, value in files.items()}
        excluded = entry['witnesses'].index(recipient) if recipient in entry['witnesses'] else None
        lower, upper = arrays['lower'][0], arrays['upper'][0]
        if excluded is not None:
            lower = np.where(arrays['lower_owner'] == excluded, arrays['lower'][1], lower)
            upper = np.where(arrays['upper_owner'] == excluded, arrays['upper'][1], upper)
        supported = np.full(lower.shape[:-1], len(entry['witnesses']) - int(excluded is not None) >= 2, dtype=bool)
        return lower, upper, supported
