"""Frozen repair-direction requests on computation-registered native prefix cells."""

from dataclasses import dataclass
import importlib.util
import math
from pathlib import Path

import numpy as np
import torch

from pipeline.overlap.complete_overlap import unit_steps
from pipeline.weight_edit import final_experiment as baseline
from pipeline.weight_edit.turn_pair import turns

FORMAT = 'common-prefix-turn-objective/1'
METRICS = {
    'objective': 'native.prefix_repair.objective',
    'supported_cells': 'native.prefix_repair.supported_cells',
    'represented_cells': 'native.prefix_repair.represented_cells',
    'matched_cells': 'native.prefix_repair.matched_cells',
    'active_cells': 'native.prefix_repair.active_cells',
}
COMBINED_METRIC = 'native.combined_objective'


def checked(row):
    path = (baseline.ROOT / row['record']).resolve()
    if not path.is_relative_to(baseline.ROOT) or 'reserved-40b' in path.parts or baseline.binding(path) != row:
        raise ValueError(f"Changed or inadmissible prefix objective source: {row['record']}")
    return path


def specification(binding):
    spec = baseline.read(checked(binding))
    if (spec.get('format') != FORMAT or type(spec.get('first_source_event')) is not int
            or spec['first_source_event'] < 0):
        raise ValueError('A frozen common-prefix objective with a source event origin is required')
    for row in [spec['arrays'], spec['registrar'], *spec.get('inputs', [])]:
        checked(row)
    return spec


@dataclass(frozen=True)
class Cells:
    events: torch.Tensor
    early: torch.Tensor
    axes: torch.Tensor
    targets: torch.Tensor
    sqrt_mass: torch.Tensor

    def residual(self, forward, merge):
        endpoints = {key: [forward[key][merge].index_select(0, self.events.to(forward[key][merge].device))]
                     for key in ('incoming', 'merged')}
        [current] = turns(endpoints, expected_merges=1)
        projection = ((current - self.early) * self.axes).sum(dim=1)
        # ReLU has zero derivative at equality. The ordinary vector constraint
        # continues to apply independently at the same computation.
        return torch.relu(self.targets - projection) * self.sqrt_mass


@dataclass(frozen=True)
class Registered:
    cells: tuple[Cells | None, ...]
    pairs: tuple[tuple[int, int], ...]
    supported: int
    represented: int

    def residuals(self, forward):
        for merge, cell in enumerate(self.cells):
            if cell is not None:
                yield merge, cell.residual(forward, merge)

    def measure(self, forward):
        losses, active, matches = [], 0, 0
        for merge, residual in self.residuals(forward):
            active += int(torch.count_nonzero(residual))
            matches += residual.numel()
            losses.append(dict(merge=merge, objective=float(residual.square().sum())))
        return dict(objective=math.fsum(row['objective'] for row in losses),
                    supported_cells=self.supported, represented_cells=self.represented,
                    matched_cells=matches, active_cells=active,
                    coverage=self.represented / self.supported,
                    status='registered' if matches else 'no_supported_correspondence', per_merge=losses)


class Objective:
    def __init__(self, spec):
        with np.load(baseline.ROOT / spec['arrays']['record'], allow_pickle=False) as payload:
            reference, early, axes, targets = (payload[k] for k in ('reference_turns', 'early_turns', 'axes', 'targets'))
        if (reference.ndim != 3 or min(reference.shape) < 1 or early.ndim != 3 or min(early.shape) < 1
                or early.shape != axes.shape or early.shape[1:] != reference.shape[1:]
                or early.shape[0] > reference.shape[0] or targets.shape != early.shape[:2]
                or not all(v.dtype.kind == 'f' for v in (reference, early, axes, targets))):
            raise ValueError('Prefix objective arrays do not cover one complete compatible architecture')
        if not np.array_equal(early, reference[:len(early)]):
            raise ValueError('The frozen early turns must be the measured prefix of the registration trace')
        if not np.isfinite(targets).all() or (targets < 0).any() or not (targets > 0).any():
            raise ValueError('Targets must be finite and nonnegative with positive support')
        self.support = targets > 0
        for merge in range(targets.shape[1]):
            valid = self.support[:, merge]
            direction = axes[valid, merge].astype(np.float64)
            if (not np.isfinite(direction).all()
                    or not np.allclose(np.linalg.norm(direction, axis=1), 1., rtol=8*np.finfo(np.float32).eps, atol=0.)):
                raise ValueError(f'Positive cells must have finite unit directions at merge {merge}')
        self.reference = unit_steps(reference)
        self.width = reference.shape[2]
        self.merges = reference.shape[1]
        self.early = early
        self.axes = axes
        self.targets = targets
        self.first_source_event = spec['first_source_event']
        self.supported = int(self.support.sum())
        source = baseline.ROOT / spec['registrar']['record']
        module_spec = importlib.util.spec_from_file_location('common_prefix_registrar', source)
        if module_spec is None or module_spec.loader is None:
            raise ValueError('Invalid prefix correspondence source')
        registrar = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(registrar)
        self.endpoints = registrar.endpoints

    @torch.no_grad()
    def register(self, native, *, prompt_events):
        before, after = native['incoming'], native['merged']
        first, length = prompt_events - 1, len(before[0])
        if not 0 <= first < length or len(before) != self.merges or len(after) != self.merges:
            raise ValueError('Native acquisition and frozen prefix architecture differ')
        query = np.empty((length-first, self.merges, self.width), dtype=np.float32)
        for merge, (incoming, merged) in enumerate(zip(before, after, strict=True)):
            if tuple(incoming.shape) != (length, self.width) or incoming.shape != merged.shape:
                raise ValueError('Native prefix registration requires every event, merge and coordinate')
            [turn] = turns({'incoming': [incoming[first:]], 'merged': [merged[first:]]}, expected_merges=1)
            query[:, merge] = turn.float().numpy()
        visits = self.endpoints(self.reference @ unit_steps(query).T)
        pairs = tuple((source, first + i) for i, ties in enumerate(visits) for source in ties
                      if source < len(self.targets) and self.support[source].any())
        source = np.asarray([i for i, _ in pairs], dtype=np.int64)
        found = np.asarray([i for _, i in pairs], dtype=np.int64)
        counts = np.bincount(source, minlength=len(self.targets))
        represented = int(self.support[counts > 0].sum())
        groups = []
        for merge in range(self.merges):
            supported = self.support[source, merge]
            rows = source[supported]
            groups.append(Cells(torch.from_numpy(found[supported]),
                torch.as_tensor(self.early[rows, merge], dtype=torch.float64),
                torch.as_tensor(self.axes[rows, merge], dtype=torch.float64),
                torch.as_tensor(self.targets[rows, merge], dtype=torch.float64),
                torch.from_numpy(1. / np.sqrt(self.supported * counts[rows]))) if len(rows) else None)
        return Registered(tuple(groups), pairs, self.supported, represented)
