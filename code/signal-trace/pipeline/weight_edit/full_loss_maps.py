"""Local, computation-only loss records for FINAL-EXPERIMENT sections 5/6/11.

Inputs are the already selected turns, radius and correspondence. Nothing here
selects tolerance, changes a solver/graph, acquires a trace or contacts MLflow.
Only scalar-per-cell diagnostics are retained; hidden vectors and weights are not.
"""
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from pipeline.overlap.complete_overlap import validate_pairs


@dataclass(frozen=True)
class LossIdentity:
    loss_version: str
    reference_sha256: str
    tolerance_sha256: str
    tolerance_policy: str

    def __post_init__(self):
        if not self.loss_version or not self.tolerance_policy:
            raise ValueError('name the immutable loss version and tolerance policy')
        for value in (self.reference_sha256, self.tolerance_sha256):
            if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('reference and tolerance require exact SHA256 bindings')


@dataclass
class LossMaps:
    identity: LossIdentity
    observation: dict[str, Any]
    merge_labels: list[dict[str, Any]]
    arrays: dict[str, np.ndarray]
    summary: dict[str, Any]
    per_merge: list[dict[str, Any]]


def _number(value, *, nonnegative=False):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError('a numerical observation cannot be a Boolean')
    value = float(value)
    if not math.isfinite(value) or nonnegative and value < 0:
        raise ValueError('observations must be finite and in their declared domain')
    return value


def _tensor(value):
    result = torch.as_tensor(value).detach().to(device='cpu', dtype=torch.float64)
    if not bool(torch.isfinite(result).all()):
        raise ValueError('turns and radii must be finite')
    return result


def _labels(labels, merges):
    rows = [dict(row) for row in labels]
    if (len(rows) != merges or merges < 1 or merges % 3
            or any(type(r.get('block')) is not int or not isinstance(r.get('branch'), str)
                   or not r['branch'] for r in rows)
            or [r['block'] for r in rows] != [m // 3 for m in range(merges)]
            or len({(r['block'], r['branch']) for r in rows}) != merges):
        raise ValueError('supply all three distinct branch labels in every declared block')
    return [r | {'merge': m} for m, r in enumerate(rows)]


def build_loss_maps(reference_turns: Sequence, current_turns: Sequence,
                    radius: Sequence, pairs, *, expected_merges: int,
                    merge_labels: Sequence[Mapping], identity: LossIdentity,
                    observation: Mapping, declared_objective: float,
                    objective_reduction: str = 'fsum') -> LossMaps:
    """Read the complete radial residual without changing the differentiable loss.

    Turn/radius sequences are merge-major [position, coordinate]/[reference
    position]. The radius is consumed directly: no chance conversion or floor.
    ``fsum`` matches native measurement's fsum of per-merge residual-square
    reductions; ``torch_sum`` describes the existing gradient scalar reduction.
    The selected reduction must equal the supplied declared scalar exactly.
    Other regroupings expose their signed roundoff, never an invented tolerance.
    """
    if (type(expected_merges) is not int or expected_merges < 1
            or any(len(s) != expected_merges for s in (reference_turns, current_turns, radius))):
        raise ValueError('every declared merge is required')
    labels = _labels(merge_labels, expected_merges)
    observation = dict(observation)
    if (observation.get('kind') not in ('fresh_native', 'held_history',
                                      'continuous_prediction', 'installed_displacement_surrogate')
            or any(k not in observation or observation[k] is None for k in ('model_revision', 'acquisition_id', 'history_id'))
            or type(observation.get('complete')) is not bool):
        raise ValueError('label measurement kind, model revision, acquisition and history')
    original, current = _tensor(reference_turns[0]), _tensor(current_turns[0])
    if (original.ndim != 2 or current.ndim != 2 or min(original.shape) < 1
            or min(current.shape) < 1 or original.shape[1] != current.shape[1]):
        raise ValueError('complete reference/current position-by-coordinate turns are required')
    r, width = original.shape
    c = current.shape[0]
    index = np.asarray(pairs)
    validate_pairs(index, r, c)
    index = index.astype(np.int64, copy=True)
    ri, ci = torch.from_numpy(index[:, 0].copy()), torch.from_numpy(index[:, 1].copy())
    multiplicity = torch.bincount(ri, minlength=r)
    mass = 1. / (r * expected_merges * multiplicity[ri].double())
    shape = (len(index), expected_merges)
    arrays = {key: np.empty(shape, dtype=np.float64)
              for key in ('distance', 'radius', 'excess', 'weighted_loss', 'formula_weighted_loss')}
    arrays['outside'] = np.empty(shape, dtype=np.bool_)
    arrays['reference_radius'] = np.empty((r, expected_merges), dtype=np.float64)
    terms, per_merge = [], []
    with torch.no_grad():
        for m, (a, b, bound) in enumerate(zip(reference_turns, current_turns, radius)):
            a, b, bound = _tensor(a), _tensor(b), _tensor(bound)
            if a.shape != (r, width) or b.shape != (c, width) or bound.shape != (r,) or bool((bound < 0).any()):
                raise ValueError('turn/radius shape or nonnegative-radius inventory differs')
            delta = b[ci] - a[ri]
            distance = torch.linalg.vector_norm(delta, dim=-1)
            if not bool(torch.isfinite(distance).all()):
                raise ValueError('finite inputs overflowed the local distance calculation')
            tau = bound[ri]
            arrays['reference_radius'][:, m] = bound.numpy()
            outside = distance > tau
            excess = (distance - tau).clamp_min(0)
            denominator = torch.where(distance > 0, distance, torch.ones_like(distance))
            shrink = torch.where(outside, 1 - tau / denominator, torch.zeros_like(distance))
            shrink = torch.where(tau == 0, torch.ones_like(shrink), shrink)
            residual = delta * shrink[:, None] * mass.sqrt()[:, None]
            squared = residual.square()
            term = _number(squared.sum(), nonnegative=True)
            terms.append(term)
            values = dict(distance=distance, radius=tau, excess=excess, outside=outside,
                          weighted_loss=squared.sum(dim=-1), formula_weighted_loss=mass * excess.square())
            for key, value in values.items():
                arrays[key][:, m] = value.numpy()
            per_merge.append(labels[m] | dict(objective=term, outside=int(outside.sum()),
                cells=len(index), max_distance=float(distance.max()), max_excess=float(excess.max()),
                max_pair_weighted_loss=float(values['weighted_loss'].max())))
    if objective_reduction == 'fsum':
        total = math.fsum(terms)
    elif objective_reduction == 'torch_sum':
        total = float(torch.tensor(terms, dtype=torch.float64).sum())
    else:
        raise ValueError('name the actual declared scalar reduction: fsum or torch_sum')
    declared = _number(declared_objective, nonnegative=True)
    if total != declared:
        raise ValueError(f'local canonical scalar {total!r} differs from declared {declared!r}')
    reference_loss, current_loss = np.zeros((r, expected_merges)), np.zeros((c, expected_merges))
    np.add.at(reference_loss, index[:, 0], arrays['weighted_loss'])
    np.add.at(current_loss, index[:, 1], arrays['weighted_loss'])
    arrays.update(pairs=index, pair_mass=mass.numpy(), reference_multiplicity=multiplicity.numpy(),
        current_multiplicity=np.bincount(index[:, 1], minlength=c),
        reference_position_merge_loss=reference_loss, current_position_merge_loss=current_loss,
        reference_position_block_loss=reference_loss.reshape(r, -1, 3).sum(axis=2),
        current_position_block_loss=current_loss.reshape(c, -1, 3).sum(axis=2))
    regrouped = {key: float(arrays[key].sum()) for key in (
        'weighted_loss', 'formula_weighted_loss', 'reference_position_merge_loss',
        'current_position_merge_loss', 'reference_position_block_loss', 'current_position_block_loss')}
    summary = dict(format='complete-local-computational-loss/1', objective=declared,
        canonical_objective=total, canonical_matches_declared=True, objective_reduction=objective_reduction,
        reference_positions=r, current_positions=c, coordinates=width, pairs=len(index),
        merges=expected_merges, blocks=expected_merges // 3, pair_merge_cells=len(index) * expected_merges,
        outside=int(arrays['outside'].sum()), max_distance=float(arrays['distance'].max()),
        observation_complete=observation['complete'],
        tolerance_numeric_sha256=hashlib.sha256(arrays['reference_radius'].tobytes()).hexdigest(),
        max_excess=float(arrays['excess'].max()), max_pair_weighted_loss=float(arrays['weighted_loss'].max()),
        represented_mass=float(mass.sum()) * expected_merges,
        regrouped_sums=regrouped, regrouping_roundoff={k: v - declared for k, v in regrouped.items()},
        weighted_loss_definition='squared norm of actual weighted radial residual; formula_weighted_loss is w*excess^2 with its own recorded roundoff')
    return LossMaps(identity, observation, labels, arrays, summary, per_merge)


def correspondence_changes(before: LossMaps, after: LossMaps):
    """Save literal pair changes, multiplicities and local movement on common pairs.

    Equal numerical position IDs across changed histories are not a certificate
    of equal computation. Changed mass and raw distance/excess stay separate.
    """
    if (before.identity != after.identity or before.merge_labels != after.merge_labels
            or not np.array_equal(before.arrays['reference_radius'], after.arrays['reference_radius'])):
        raise ValueError('different loss/tolerance/reference versions cannot be relabeled as comparable')
    old = {tuple(pair): n for n, pair in enumerate(before.arrays['pairs'])}
    new = {tuple(pair): n for n, pair in enumerate(after.arrays['pairs'])}
    common = sorted(old.keys() & new.keys())
    oi, ni = [old[k] for k in common], [new[k] for k in common]
    arrays = dict(pairs_added=np.asarray(sorted(new.keys() - old.keys()), dtype=np.int64).reshape(-1, 2),
                  pairs_removed=np.asarray(sorted(old.keys() - new.keys()), dtype=np.int64).reshape(-1, 2),
                  common_pairs=np.asarray(common, dtype=np.int64).reshape(-1, 2),
                  mass_before=before.arrays['pair_mass'][oi], mass_after=after.arrays['pair_mass'][ni])
    for key in ('distance', 'radius', 'excess', 'weighted_loss', 'outside'):
        arrays[key + '_before'] = before.arrays[key][oi]
        arrays[key + '_after'] = after.arrays[key][ni]
    was, now = arrays['outside_before'], arrays['outside_after']
    summary = dict(pairs_added=len(arrays['pairs_added']), pairs_removed=len(arrays['pairs_removed']),
        common_pairs=len(common), source_history=before.observation['history_id'],
        target_history=after.observation['history_id'],
        same_history=before.observation['history_id'] == after.observation['history_id'],
        outside_to_inside=int((was & ~now).sum()), inside_to_outside=int((~was & now).sum()),
        inside_both_moved=int((~was & ~now & (arrays['distance_before'] != arrays['distance_after'])).sum()),
        common_pairs_changed_mass=int(np.count_nonzero(arrays['mass_before'] != arrays['mass_after'])),
        interpretation='literal saved pair coordinates, not token or semantic identities; inspect changed multiplicity before attributing a scalar decrease')
    return summary, arrays


def gradient_update_inventory(eligible: Sequence[Mapping], *, gradients: Mapping,
                              updates: Mapping | None = None):
    """Join explicit scalar receipts over the source inventory, never its subset.

    Gradients use status connected/disconnected/not_computed plus supplied finite,
    nonzero_entries and norm_squared when computed. Updates supply changed_entries
    and explicit complete=True before it can certify installation. When known,
    requested_nonzero_entries, displacement/remainder/row metadata are retained.
    Missing records stay absent; zero, disconnected and unobserved are distinct.
    Tensor reductions belong in the existing producer traversal, not a second pass.
    """
    eligible = [dict(r) for r in eligible]
    names = [r['name'] for r in eligible]
    updates = {} if updates is None else updates
    if not names or len(set(names)) != len(names) or (set(gradients) | set(updates)) - set(names):
        raise ValueError('use distinct complete eligible names and no foreign tensor receipts')
    rows = []
    for source in eligible:
        name = source['name']
        if not source.get('dtype') or not source.get('shape'):
            raise ValueError('eligible tensors require native dtype and complete shape')
        g, u = dict(gradients.get(name, {})), dict(updates.get(name, {}))
        status = g.get('status', 'absent')
        if status not in ('absent', 'not_computed', 'connected', 'disconnected'):
            raise ValueError('gradient connectivity must be explicit')
        gradient_state = status
        if status == 'connected':
            count = g.get('nonzero_entries')
            if type(count) is not int or count < 0 or type(g.get('finite')) is not bool:
                raise ValueError('computed gradients require finite status and nonzero count')
            if g['finite']:
                _number(g['norm_squared'], nonnegative=True)
                gradient_state = 'zero' if count == 0 else 'nonzero'
            else:
                gradient_state = 'nonfinite'
        changed = u.get('changed_entries')
        requested = u.get('requested_nonzero_entries')
        for count in (changed, requested):
            if count is not None and (type(count) is not int or count < 0):
                raise ValueError('update counts must be observed nonnegative integers')
        if 'complete' in u and type(u['complete']) is not bool:
            raise ValueError('installation completeness must be an observed Boolean')
        if not u:
            update_state = 'absent'
        elif u.get('complete') is False:
            update_state = 'incomplete'
        elif u.get('complete') is not True:
            update_state = 'completion_unobserved'
        elif changed is None:
            update_state = 'not_observed'
        elif changed:
            update_state = 'changed'
        elif u.get('requested_nonzero_entries', 0) > 0:
            update_state = 'rounded_away'
        elif u.get('requested_nonzero_entries') == 0:
            update_state = 'zero_requested'
        else:
            update_state = 'unchanged_request_unobserved'
        for key in ('displacement_squared', 'remainder_before_squared', 'remainder_after_squared'):
            if key in u:
                _number(u[key], nonnegative=True)
        rows.append(dict(tensor=source, gradient=g, gradient_state=gradient_state,
                         update=u, update_state=update_state))
    return dict(eligible_tensors=len(names), rows=rows,
        native_dtype_counts=dict(Counter(r['dtype'] for r in eligible)),
        gradient_states=dict(Counter(r['gradient_state'] for r in rows)),
        update_states=dict(Counter(r['update_state'] for r in rows)),
        missing_gradient_records=[n for n in names if n not in gradients],
        missing_update_records=[n for n in names if n not in updates],
        complete_gradient_inventory=all(n in gradients for n in names),
        complete_update_inventory=all(n in updates for n in names),
        all_gradients_observed=all(r['gradient_state'] not in ('absent', 'not_computed') for r in rows),
        all_updates_observed=all(r['update_state'] not in
            ('absent', 'not_observed', 'incomplete', 'completion_unobserved') for r in rows))


def response_comparison(source: LossMaps, fresh: LossMaps, *, continuous: Sequence[Mapping],
                        installed: Sequence[Mapping], applied_length: float):
    """Read existing JVP moments: continuous coefficient eta vs installed coefficient1.

    Moments are {merge, baseline, loss_cross=2<r,q>, response_norm_squared=||q||²}.
    A fitted-optimum predicted_objective, if present, is deliberately not read.
    No new JVP, response-based gate or choice of update length is performed.
    """
    if (source.identity != fresh.identity or source.merge_labels != fresh.merge_labels
            or not np.array_equal(source.arrays['reference_radius'], fresh.arrays['reference_radius'])):
        raise ValueError('response comparison needs one fixed loss identity and merge inventory')
    if fresh.observation['kind'] != 'fresh_native':
        raise ValueError('a prediction cannot be labeled fresh native loss')
    length = _number(applied_length)
    m = source.summary['merges']
    if any(len(x) != m or [r['merge'] for r in x] != list(range(m)) for x in (continuous, installed)):
        raise ValueError('both existing response tables must retain every merge in order')
    rows = []
    for a, b, observed in zip(continuous, installed, fresh.per_merge):
        a0, a1, a2 = (_number(a[k], nonnegative=k != 'loss_cross') for k in ('baseline', 'loss_cross', 'response_norm_squared'))
        b0, b1, b2 = (_number(b[k], nonnegative=k != 'loss_cross') for k in ('baseline', 'loss_cross', 'response_norm_squared'))
        rows.append(source.merge_labels[a['merge']] | dict(
            source_native_loss=source.per_merge[a['merge']]['objective'],
            continuous_prediction=_number(a0 + length * a1 + length * length * a2),
            continuous_baseline_difference=a0 - source.per_merge[a['merge']]['objective'],
            installed_displacement_surrogate=_number(b0 + b1 + b2),
            installed_baseline_difference=b0 - source.per_merge[a['merge']]['objective'],
            fresh_native_loss=observed['objective']))
    totals = {k: math.fsum(r[k] for r in rows) for k in
              ('source_native_loss', 'continuous_prediction', 'installed_displacement_surrogate', 'fresh_native_loss')}
    regrouping = dict(source_native_loss=totals['source_native_loss'] - source.summary['objective'],
                      fresh_native_loss=totals['fresh_native_loss'] - fresh.summary['objective'])
    totals.update(source_native_loss=source.summary['objective'], fresh_native_loss=fresh.summary['objective'])
    return dict(loss_identity=asdict(source.identity), source_observation=source.observation,
        fresh_observation=fresh.observation, applied_length=length, installed_displacement_coefficient=1,
        history_changed=source.observation['history_id'] != fresh.observation['history_id'],
        correspondence_changed=not np.array_equal(source.arrays['pairs'], fresh.arrays['pairs']),
        per_merge=rows, totals=totals, per_merge_native_aggregation_roundoff=regrouping,
        interpretation='two held-history, held-correspondence directional surrogates and one fresh native observation; fresh registration/history changes are explicit')


def recorder_attributes(maps: LossMaps, *, prefix='output.'):
    """Attach finite scalar summaries to the existing native.measure span.

    This emits nothing and does not duplicate the controller's native.objective.
    """
    values = {k: v for k, v in maps.summary.items() if isinstance(v, (str, int, float, bool))}
    values.update(asdict(maps.identity), observation_kind=maps.observation['kind'])
    return {prefix + k: v for k, v in values.items()}


def write_local(maps: LossMaps, folder, *, inventory=None, comparison=None, changes=None):
    """Write small local numeric diagnostics, never turns or weights; no overwrite."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    arrays = dict(maps.arrays)
    change_summary = None
    if changes is not None:
        change_summary, extra = changes
        arrays.update({'change_' + k: v for k, v in extra.items()})
    path = folder / 'loss-maps.npz'
    with path.open('xb') as file:
        np.savez_compressed(file, **arrays)
    metadata = dict(format='complete-local-computational-loss/1', identity=asdict(maps.identity),
        observation=maps.observation, summary=maps.summary, per_merge=maps.per_merge,
        index_contract=dict(pairs_columns=['reference_position', 'current_position'],
            cell_arrays_axes=['pair_row', 'merge'], radius_axes=['reference_position', 'merge'],
            position_merge_axes=['named_position', 'merge'], position_block_axes=['named_position', 'block'],
            merge_identity='per_merge[merge] supplies the exact block and branch'),
        arrays=dict(record=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    inventory={k: dict(shape=list(a.shape), dtype=str(a.dtype)) for k, a in arrays.items()}),
        inventory=inventory, response_comparison=comparison, correspondence_changes=change_summary,
        upload_scope='local scalar-per-cell maps; publish summaries/plots, not raw turns, hidden states or weights')
    with (folder / 'loss-maps.json').open('x') as file:
        json.dump(metadata, file, indent=2, allow_nan=False)
        file.write('\n')
    return metadata


def _headless_pyplot():
    """Files only: a long run must never open a connection to the desktop's display."""
    import matplotlib
    matplotlib.use('Agg', force=True)
    from matplotlib import pyplot
    return pyplot


def render_loss_views(maps: LossMaps, path):
    """Reference/current position×block maps and the exact saved DTW overlay."""
    plt = _headless_pyplot()
    a, b = (maps.arrays[k] for k in ('reference_position_block_loss', 'current_position_block_loss'))
    maximum = max(float(a.max()), float(b.max()))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for ax, values, title in zip(axes[:2], (a, b), ('Fixed reference positions', 'Fresh/current positions')):
        im = ax.imshow(values, aspect='auto', origin='lower', vmin=0, vmax=maximum)
        ax.set(xlabel='Block', ylabel='Acquisition position', title=title)
    fig.colorbar(im, ax=axes[:2], label='Sum of weighted local loss over three branches')
    pairs = maps.arrays['pairs']
    axes[2].plot(pairs[:, 0], pairs[:, 1], color='gray')
    dots = axes[2].scatter(pairs[:, 0], pairs[:, 1], c=maps.arrays['weighted_loss'].sum(axis=1))
    axes[2].set(xlabel='Reference position', ylabel='Current position', title='Saved computational correspondence')
    fig.colorbar(dots, ax=axes[2], label='Pair contribution across all merges')
    fig.suptitle(f"{maps.identity.loss_version} · {maps.observation['kind']} · L={maps.summary['objective']:.12g}")
    fig.savefig(path)
    plt.close(fig)


def render_cell_overlay(maps: LossMaps, reference_turns, current_turns, *, pair_index, merge, path):
    """Explicitly selected local turn overlay; does not persist whole trace arrays."""
    plt = _headless_pyplot()
    i, j = maps.arrays['pairs'][pair_index]
    a, b = _tensor(reference_turns[merge])[i], _tensor(current_turns[merge])[j]
    if a.shape != b.shape or float(torch.linalg.vector_norm(b - a)) != maps.arrays['distance'][pair_index, merge]:
        raise ValueError('selected turn vectors differ from the measured local cell')
    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    ax.plot(a.numpy(), label=f'Fixed reference position {i}')
    ax.plot(b.numpy(), label=f'Current position {j}')
    ax.set(xlabel='Turn coordinate', ylabel='unit(output) − unit(input)',
        title=f"Merge {merge}: distance={maps.arrays['distance'][pair_index, merge]:.8g}, radius={maps.arrays['radius'][pair_index, merge]:.8g}, excess={maps.arrays['excess'][pair_index, merge]:.8g}")
    ax.legend()
    fig.savefig(path)
    plt.close(fig)
