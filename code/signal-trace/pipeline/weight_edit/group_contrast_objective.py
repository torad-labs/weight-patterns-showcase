"""Signed computational contrast on each freshly registered cohort member."""

from dataclasses import dataclass
import importlib.util
import math

import numpy as np
import torch

from pipeline.overlap.complete_overlap import unit_steps
from pipeline.weight_edit import common_prefix_objective as prefix, final_experiment as baseline
from pipeline.weight_edit.turn_pair import turns

FORMAT = 'signed-group-turn-objective/1'
ROLES = ('failing', 'preservation')
BASELINE_SUPPORT = 'baseline'
LIVE_SUPPORT = 'live_donor_support'
OBSERVED_STOP = 'observed_joint_outcome'


def support_policy(spec):
    policy = spec.get('support_policy', BASELINE_SUPPORT)
    if policy not in (BASELINE_SUPPORT, LIVE_SUPPORT):
        raise ValueError(f'Unsupported group support policy: {policy}')
    return policy


def specification(binding):
    spec = baseline.read(prefix.checked(binding))
    support_policy(spec)
    limit = spec.get('updates')
    if (spec.get('format') != FORMAT or not spec.get('members') or
            (limit != OBSERVED_STOP and (type(limit) is not int or limit < 2))):
        raise ValueError('Declare an observation-led stop or a cohort interval of at least two updates')
    info = [binding, spec['update_basis'], *spec.get('inputs', [])]
    prefix.checked(spec['registrar'])
    matchers = spec.get('operation_matchers', {})
    for entry in matchers.values():
        prefix.checked(entry['program'])
        data = baseline.read(prefix.checked(entry['policy']))
        info.append(entry['policy'])
        info.extend({key: donor['turns'][key] for key in ('record', 'sha256')} for donor in data['donors'])
    seen, roles = set(), set()
    for member in spec['members']:
        name = member['id']
        if (not name or name in seen or '/' in name or '\\' in name or name in ('.', '..')
                or type(member['first_source_event']) is not int or member['first_source_event'] < 0):
            raise ValueError('Each cohort member needs one safe unique ID and source event origin')
        seen.add(name)
        info.extend([member['case'], member['arrays']])
        [case] = baseline.read(prefix.checked(member['case']))
        if set(case) != {'id', 'tokens', 'max_tokens'} or case['id'] != name or not case['tokens']:
            raise ValueError('A cohort case must bind exactly one unconditional native prompt')
        fields = set()
        for op in member['operations']:
            if op['role'] not in ROLES or not op['name'] or op['name'] in fields or not op['field']:
                raise ValueError('Each member needs distinct supported operation names and declared roles')
            if 'matcher' in op and op['matcher'] not in matchers:
                raise ValueError(f"Unknown operation matcher for {op['name']}")
            if 'arrays' in op:
                info.append(op['arrays'])
            fields.add(op['name'])
            roles.add(op['role'])
        if not fields:
            raise ValueError('A cohort member must have at least one declared operation')
    if roles != set(ROLES):
        raise ValueError('Both failing and preservation computations are required')
    for row in info:
        prefix.checked(row)
    return spec, list({row['record']: row for row in info}.values())


@dataclass(frozen=True)
class Targets:
    cells: np.ndarray
    axes: np.ndarray | None
    offsets: np.ndarray | None
    origins: np.ndarray
    centres: np.ndarray | None
    variation: np.ndarray | None


@dataclass(frozen=True)
class Pieces:
    merge: int
    events: torch.Tensor
    source: torch.Tensor
    axes: torch.Tensor | None
    offsets: torch.Tensor | None
    origins: torch.Tensor
    sqrt_mass: torch.Tensor
    centres: torch.Tensor | None = None
    variation: torch.Tensor | None = None

    def values(self, forward):
        picked = {key: [forward[key][self.merge].index_select(0, self.events.to(forward[key][self.merge].device))]
                  for key in ('incoming', 'merged')}
        [current] = turns(picked, expected_merges=1)
        signed = (current*self.axes).sum(dim=1)-self.offsets if self.axes is not None else None
        if self.centres is None:
            return current, signed, signed
        change = current-self.centres
        projection = torch.bmm(self.variation, change.unsqueeze(2))
        nuisance = torch.bmm(self.variation.transpose(1,2), projection).squeeze(2)
        return current, signed, change-nuisance

    def weighted(self, residual):
        weight = self.sqrt_mass if residual.ndim == 1 else self.sqrt_mass[:,None]
        return residual*weight


@dataclass(frozen=True)
class Operation:
    name: str
    role: str
    supported: int
    represented: int
    groups: tuple
    required: frozenset
    observed_support: frozenset
    affine: bool
    donor_support: frozenset | None = None
    registration: dict | None = None

    def residuals(self, forward):
        for group in self.groups:
            _, _, residual = group.values(forward)
            yield group.weighted(residual)

    def measure(self, forward, first_source_event):
        contributions, cells = [], []
        for group in self.groups:
            current, signed, residual = group.values(forward)
            squared = group.weighted(residual).square()
            losses = squared.sum(1) if squared.ndim == 2 else squared
            contributions.append(float(squared.sum()))
            if group.centres is not None:
                outside = residual
            else:
                movement = current-group.origins
                outside = movement-(movement*group.axes).sum(1, keepdim=True)*group.axes
            magnitude = torch.linalg.vector_norm(residual, dim=1) if residual.ndim == 2 else residual.abs()
            signed_values = signed.tolist() if signed is not None else [None]*len(group.events)
            for row, event, error, off_axis, distance, loss in zip(group.source.tolist(), group.events.tolist(), signed_values,
                    torch.linalg.vector_norm(outside, dim=1).tolist(), magnitude.tolist(), losses.tolist(), strict=True):
                cells.append(dict(source_event=first_source_event+row, event=event, merge=group.merge,
                                  signed_error=error, orthogonal_turn_movement=off_axis, objective_residual_norm=distance,
                                  weighted_loss=loss))
        scope = self.required if self.donor_support is None else self.donor_support
        status = ('unmeasured' if not self.represented else 'complete' if scope <= self.observed_support else 'partial')
        identified = lambda cells: [[first_source_event+row, merge] for row, merge in sorted(cells)]
        result = dict(operation=self.name, role=self.role, status=status,
                    residual_geometry='affine_correct_variation' if self.affine else 'signed_projection',
                    objective=math.fsum(contributions) if cells else None, supported_cells=self.supported,
                    baseline_required_cells=len(self.required), baseline_support=identified(self.required),
                    represented_cells=self.represented, matched_cells=len(cells), cells=cells,
                    missing_baseline_support=identified(self.required-self.observed_support),
                    new_source_support=identified(self.observed_support-self.required))
        if self.donor_support is not None:
            result.update(support_policy=LIVE_SUPPORT, missing_donor_support=identified(scope-self.observed_support),
                          measured_contribution=result['objective'])
            if status != 'complete':
                result['objective'] = None
        if self.registration is not None:
            result['operation_registration'] = self.registration
        return result


def targets(payload, reference, field, role):
    cells = payload[field+'_cells']
    if (cells.ndim != 2 or cells.shape[1] != 2 or not len(cells) or cells.dtype.kind not in 'iu'
            or (cells < 0).any() or (cells >= np.asarray(reference.shape[:2])).any()
            or len(np.unique(cells, axis=0)) != len(cells)):
        raise ValueError(f'Invalid supported operation geometry: {field}')
    axes = offsets = centres = variation = None
    shape = (len(cells), reference.shape[2])
    if field+'_centres' in payload or field+'_variation' in payload:
        if role != 'preservation' or field+'_centres' not in payload or field+'_variation' not in payload:
            raise ValueError('Affine vector preservation needs both centres and nuisance variation')
        centres, variation = payload[field+'_centres'], payload[field+'_variation']
        if (centres.shape != shape or variation.ndim != 3 or variation.shape[0] != len(cells)
                or variation.shape[2] != reference.shape[2] or not np.isfinite(centres).all()
                or not np.isfinite(variation).all()):
            raise ValueError(f'Invalid affine preservation geometry: {field}')
        basis = variation.astype(np.float64)
        active = np.any(basis != 0., axis=2)
        expected = np.eye(basis.shape[1])[None]*active[:,:,None]
        if not np.allclose(basis @ basis.transpose(0,2,1), expected, rtol=0., atol=8*np.finfo(np.float32).eps):
            raise ValueError(f'Nuisance directions must be orthonormal and zero padded: {field}')
    else:
        axes, offsets = payload[field+'_axes'], payload[field+'_offsets']
        if (axes.shape != shape or offsets.shape != (len(cells),) or not np.isfinite(axes).all()
                or not np.isfinite(offsets).all()
                or not np.allclose(np.linalg.norm(axes.astype(np.float64), axis=1), 1.,
                                   rtol=8*np.finfo(np.float32).eps, atol=0.)):
            raise ValueError(f'Invalid signed direction geometry: {field}')
    return Targets(cells, axes, offsets, reference[cells[:,0], cells[:,1]], centres, variation)


class Objective:
    def __init__(self, spec):
        self.operation_matchers = {}
        for name, entry in spec.get('operation_matchers', {}).items():
            data = baseline.read(prefix.checked(entry['policy']))
            source = importlib.util.spec_from_file_location('frozen_operation_matcher', prefix.checked(entry['program']))
            mod = importlib.util.module_from_spec(source)
            source.loader.exec_module(mod)
            rows = entry['reference_rows']
            if (len(rows) != len(data['stage_order']) or len(set(rows)) != len(rows)
                    or any(type(row) is not int or row < 0 for row in rows)):
                raise ValueError('Every operation stage needs one distinct reference row')
            self.operation_matchers[name] = (mod.Matcher.from_spec(data, baseline.ROOT), data, rows)
        self.support_policy = support_policy(spec)
        path = prefix.checked(spec['registrar'])
        module_spec = importlib.util.spec_from_file_location('group_contrast_registration', path)
        if module_spec is None or module_spec.loader is None:
            raise ValueError('Unsupported group registration source')
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        self.endpoints = module.endpoints
        self.members = {}
        for record in {member['arrays']['record'] for member in spec['members']}:
            matching = [m for m in spec['members'] if m['arrays']['record'] == record]
            with np.load(baseline.ROOT/record, allow_pickle=False) as payload:
                reference = payload['reference_turns']
                if reference.ndim != 3 or min(reference.shape) < 1:
                    raise ValueError('Every registration source needs a complete native turn field')
                registered_reference = unit_steps(reference)
                loaded = {}
                for member in matching:
                    ops = []
                    for op in member['operations']:
                        field = op['field']
                        field_record = op.get('arrays', member['arrays'])['record']
                        key = field_record, field
                        if key not in loaded:
                            if field_record == record:
                                loaded[key] = targets(payload, reference, field, op['role'])
                            else:
                                with np.load(prefix.checked(op['arrays']), allow_pickle=False) as extra:
                                    loaded[key] = targets(extra, reference, field, op['role'])
                        target = loaded[key]
                        if 'matcher' in op:
                            support = self.operation_matchers[op['matcher']][2]
                            if max(support) >= len(reference) or not set(target.cells[:, 0]).issubset(support):
                                raise ValueError('Operation target and reference-stage registration disagree')
                        if target.centres is not None and op['role'] != 'preservation':
                            raise ValueError('A shared affine field is available for preservation operations')
                        ops.append((op, target))
                    self.members[member['id']] = (member, registered_reference, reference.shape[1:], ops)
        self.role_members = {role: sum(any(op['role'] == role for op in member['operations']) for member in spec['members'])
                             for role in ROLES}

    @torch.no_grad()
    def register(self, member, native, *, prompt_events, baseline_support=None, full_trace_pairs=None):
        member_spec, reference, (merges, width), targets_by_operation = self.members[member]
        incoming, merged = native['incoming'], native['merged']
        first = prompt_events - 1
        length = len(incoming[0])
        if not 0 <= first < length or len(incoming) != merges or len(merged) != merges:
            raise ValueError('Current native acquisition differs from the frozen trace geometry')
        query = np.empty((length-first, merges, width), dtype=np.float32)
        for m, (a, y) in enumerate(zip(incoming, merged, strict=True)):
            if a.shape != (length,width) or a.shape != y.shape:
                raise ValueError('Incomplete all-merge turn stream')
            [turn] = turns({'incoming':[a[first:]], 'merged':[y[first:]]}, expected_merges=1)
            query[:,m] = turn.float().numpy()
        prefix_visits = {}
        for event, ties in enumerate(self.endpoints(reference @ unit_steps(query).T), first):
            for source in ties:
                prefix_visits.setdefault(source, []).append(event)
        stage_visits = {}
        operations = []
        for op, target in targets_by_operation:
            visits, receipt = prefix_visits, None
            if op.get('confirm_full_trace') and full_trace_pairs is None:
                raise ValueError('This signed field requires the native whole-trace correspondence')
            if 'matcher' in op:
                name = op['matcher']
                if name not in stage_visits:
                    matcher, policy, reference_rows = self.operation_matchers[name]
                    measured = matcher.register(query, policy)
                    matched = {}
                    if measured['accepted']:
                        [pair] = measured['pairs']
                        phases = dict(zip(('symbolic', 'numeric'), pair, strict=True))
                        for row, stage in zip(reference_rows, policy['stage_order'], strict=True):
                            event = phases[stage['phase']] + stage['offset']
                            if not 0 <= event < len(query):
                                raise ValueError('Matched stage outside the native generation')
                            matched[row] = [first + event]
                    stage_visits[name] = matched, dict(matcher=name, first_query_event=first, **measured)
                visits, receipt = stage_visits[name]
            if op.get('confirm_full_trace'):
                [row] = np.unique(target.cells[:, 0]).tolist()
                source = member_spec['first_source_event'] + row
                mates = full_trace_pairs[full_trace_pairs[:, 0] == source, 1].tolist()
                recognized = visits.get(row, [])
                agrees = len(mates) == 1 and mates[0] in recognized
                receipt = dict(phase_read=receipt, reference_event=source, full_trace_events=mates,
                               phase_events=recognized, accepted=agrees)
                visits = {row: mates} if agrees else {}
                if len(target.cells) == 1 and target.offsets is not None:
                    receipt['target_projection'] = float(target.offsets[0])
            role = op['role']
            count = sum(other['role'] == role for other, _ in targets_by_operation)
            observed = frozenset((int(row), int(merge)) for row, merge in target.cells if row in visits)
            required = observed if baseline_support is None else baseline_support[op['name']]
            donor_support = None
            active = required & observed
            mass_cells = len(required)
            if self.support_policy == LIVE_SUPPORT:
                donor_support = frozenset((int(row), int(merge)) for row, merge in target.cells)
                active = observed
                mass_cells = len(donor_support)
            mass = 1. / (len(ROLES)*self.role_members[role]*count*mass_cells) if mass_cells else 0.
            represented = 0
            groups = []
            for merge in np.unique(target.cells[:,1]):
                indexed = [(i, visits[int(row)]) for i,(row,m) in enumerate(target.cells)
                           if m == merge and (row,m) in active]
                represented += len(indexed)
                if not indexed:
                    continue
                rows = np.asarray([i for i, events in indexed for _ in events], dtype=np.int64)
                groups.append(Pieces(int(merge), torch.tensor([e for _, events in indexed for e in events]),
                    torch.as_tensor(target.cells[rows,0].astype(np.int64)),
                    None if target.axes is None else torch.as_tensor(target.axes[rows], dtype=torch.float64),
                    None if target.offsets is None else torch.as_tensor(target.offsets[rows], dtype=torch.float64),
                    torch.as_tensor(target.origins[rows], dtype=torch.float64),
                    torch.tensor([math.sqrt(mass/len(events)) for _, events in indexed for _ in events], dtype=torch.float64),
                    None if target.centres is None else torch.as_tensor(target.centres[rows], dtype=torch.float64),
                    None if target.variation is None else torch.as_tensor(target.variation[rows], dtype=torch.float64)))
            operations.append(Operation(op['name'], role, len(target.cells), represented, tuple(groups),
                                        required, observed, target.centres is not None, donor_support, receipt))
        return tuple(operations)
