"""Connect the existing typed checkpoint restore to the one training loop.

Legacy317 bootstrap remains explicit. Full-parameter continuation retains every
native value and remainder through FullState.restore, not the legacy70 importer.
The caller creates its one working descriptor from the verified declared base.
"""
from pathlib import Path
from types import SimpleNamespace

from pipeline.weight_edit import full_state


def is_full(candidate):
    if candidate.get('format') == full_state.FORMAT:
        return True
    if 'format' not in candidate and 'weight_state' in candidate:
        return False
    raise ValueError('unsupported continuation checkpoint format')


def validate(candidate, initial, expected_start, expected_loss_identity=None):
    full = is_full(candidate)
    ready = (candidate.get('status') == 'complete' and bool(candidate.get('checkpoint'))
             if full else candidate.get('weight_state') == 'complete'
             and candidate.get('pending_install') is None
             and candidate.get('pending_transition') is None
             and candidate.get('transition_state') == 'complete'
             and candidate.get('accumulator', {}).get('complete') is True)
    if (not ready or candidate.get('last_measurement') != expected_start
            or candidate.get('model_identity') != initial.get('model_identity')
            or initial.get('complete') is not True or initial.get('ended') is not True):
        raise ValueError('continuation needs a complete checkpoint and its own complete fresh measurement')
    if full and expected_loss_identity is not None:
        if candidate.get('loss_identity') != expected_loss_identity:
            raise ValueError('full continuation loss identity differs')
    return full


def checkpoint_record(root, candidate, checked):
    """Bind the outer receipt to the exact existing typed checkpoint metadata."""
    if Path(candidate['checkpoint']['record']).name != 'state.json':
        raise ValueError('typed restore requires the bound state.json manifest')
    record = checked(root, candidate['checkpoint'])
    keys = ('format', 'status', 'iteration', 'model_identity', 'base_identity',
            'inventory_identity', 'transition')
    if not record.get('complete') or any(record.get(key) != candidate.get(key) for key in keys):
        raise ValueError('full checkpoint metadata differs from its final-state receipt')
    return record


def seed(candidate):
    return candidate['base_identity'] if is_full(candidate) else candidate['base_seed']


def iteration(candidate):
    return candidate['iteration'] if is_full(candidate) else candidate['update']


def historical_step(declaration, candidate):
    counters = declaration['counter_semantics']
    previous = (counters['historical_predecessor'] if is_full(candidate)
                else counters['historical_ffn_update'])
    return previous + iteration(candidate)


def remainder_inventory(root, candidate, checked):
    """Expose retained row counts to the existing entry storage calculation."""
    if not is_full(candidate):
        return None
    record = checkpoint_record(root, candidate, checked)
    return SimpleNamespace(remainders={row['name']: row['remainder']['rows']
        for row in record['layout'] if row['remainder']['kind'] == 'rows'})


def payloads(root, candidate, source_folder, checked):
    if is_full(candidate):
        record = checkpoint_record(root, candidate, checked)
        metadata = root / candidate['checkpoint']['record']
        folder = metadata.parent
        infos = (record['native'], record['remainder'])
        paths = [metadata]
    else:
        folder, infos, paths = source_folder, (candidate['overrides'], candidate['accumulator']), []
    for info in infos:
        name = Path(info['file'])
        if name.is_absolute() or len(name.parts) != 1:
            raise ValueError('checkpoint payload must be a local file basename')
        paths.append(folder / name)
    return paths


def restore(root, fd, candidate, source_binding, specs, *, chunk_elements, journal, checked):
    base = seed(candidate)
    if is_full(candidate):
        checkpoint_record(root, candidate, checked)
        return full_state.FullState.restore(fd, (root / candidate['checkpoint']['record']).parent,
            specs=specs, expected_base_identity=base,
            expected_model_identity=candidate['model_identity'], chunk_elements=chunk_elements,
            sync=lambda name, start, values: None, journal=journal)
    state = full_state.FullState(fd, specs, model_identity=candidate['model_identity'],
        base_identity=base, iteration=candidate['update'], complete=False,
        chunk_elements=chunk_elements, journal=journal)
    state.restore_legacy(candidate, (root / source_binding['record']).parent,
        expected_base_seed=base, sync=lambda name, start, values: None)
    return state
