"""Release one consumed capture through the existing verified archive operator.

The controller calls this after measuring the next complete native capture, outside
the card lock, and waits for return before producing another capture. Only the two
old bulk files leave; the current handoff and every small research record stay.
"""
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / 'code/signal-trace'))
sys.path.insert(0, str(ROOT / 'tools/box-image/bin'))
from pipeline.research_guard import archive_inputs as archive
import wp_overlap as overlap

_spec = importlib.util.spec_from_file_location('consumed_trace_cache_hash', ROOT / 'tools/torad-research/src/cache-hash.py')
cache = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cache)
# Existing finite archive helper's buffer, inherited from the R2 range reader.
HASH_CHUNK_BYTES = 16 * 1024 * 1024
STREAMS = ('stream.f32', 'stream.jsonl')


def source_files():
    """Exact helper/operator/projection/transport sources for the declaration."""
    modules = (archive, archive.archives, archive.sources, archive.storage, cache,
               overlap, sys.modules['wp_training_models'])
    paths = {Path(__file__).resolve(), *(Path(module.__file__).resolve() for module in modules),
             ROOT / 'code/signal-trace/pipeline/research_guard/verify_round2.py'}
    paths.update(ROOT / 'tools/torad-research/src' / name for name in
                 ('cli.ts', 'bucket.ts', 'store.ts', 'extract.ts'))
    return sorted(paths)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def observation(root, run, number, *, local):
    """Check exact producer records without rereading a current bulk payload."""
    step = run / 'loop' / str(number)
    measured_path = step / 'measurement.json'
    measured = read(measured_path)
    capture_path = step / 'capture/capture.json'
    identity = measured.get('model_identity', {})
    if (measured.get('complete') is not True or measured.get('ended') is not True
            or measured.get('step') != number or measured.get('generation') != number
            or identity.get('kind') != 'native-working-revision/1'
            or identity.get('run') != run.relative_to(root).as_posix()
            or identity.get('generation') != number):
        raise ValueError('archive boundary requires the exact complete native observation')
    if measured.get('capture') != dict(record=capture_path.relative_to(root).as_posix(), sha256=digest(capture_path)):
        raise ValueError('measurement does not bind its exact local capture')
    captured = read(capture_path)
    if captured.get('model_identity') != identity:
        raise ValueError('capture and measurement identify different native revisions')
    for name in STREAMS:
        binding = captured['raw'][name]
        expected = step / 'capture/raw' / measured['member'] / name
        if root / binding['record'] != expected or expected.resolve() != expected:
            raise ValueError('only this step\'s exact plain stream paths may be archived')
        if local and (not expected.is_file() or expected.is_symlink()):
            raise ValueError('the retained current handoff must have both local stream files')
    return measured, captured


def published(root, run, number, measured, state_path):
    """The observer's persisted receipt is written only after both MLflow uploads."""
    try:
        state = read(state_path)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    receipt = state.get(run.name, {}).get('training_view', {}).get('overlap', {}).get(str(number), {})
    if not receipt.get('published'):
        return None
    measured_sha = digest(run / 'loop' / str(number) / 'measurement.json')
    if receipt.get('measurement_sha256') != measured_sha:
        raise ValueError('published overlap identifies a different measurement')
    # A changed renderer must first finish its own publication; never fake its receipt.
    if receipt.get('renderer_sha256') != overlap.SOURCE_SHA256:
        return None
    key = hashlib.sha256((measured_sha + overlap.SOURCE_SHA256).encode()).hexdigest()
    if receipt['published'] != key:
        raise ValueError('published overlap key is not bound to this measurement and renderer')
    target = Path(receipt['cache']) / 'overlap.json'
    if digest(target) != receipt.get('scalar_sha256'):
        raise ValueError('published overlap cache changed')
    projected = read(target)
    if (projected.get('format') != overlap.SCHEMA or not overlap.projection_verified(projected)
            or projected.get('measurement', {}).get('sha256') != measured_sha
            or any(projected.get(k) != measured[k] for k in ('objective', 'outside', 'pair_merge_cells'))
            or projected.get('producer') != receipt.get('projection_source')
            or any(projected.get('producer', {}).get(k) != v for k, v in overlap.raw_projection_sources(root).items())):
        raise ValueError('published overlap does not reproduce the bound native measurement')
    return receipt


@contextmanager
def archive_environment(allowlist=None, name=None):
    keys = ('TORAD_RESEARCH_CACHE_READ_ALLOWLIST', 'TORAD_RESEARCH_RESTORED_ARCHIVE')
    old = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ.pop(key, None)
        if allowlist is not None:
            os.environ[keys[0]] = str(allowlist)
            os.environ[keys[1]] = name
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def inventoried(root, binding, membership, output):
    """Reuse an exact retry inventory; otherwise use the existing guarded maker."""
    target = root / binding['record']
    name = hashlib.sha256(binding['record'].encode()).hexdigest()[:12] + '-' + target.name + '.json'
    allowlist = output / (target.name + '-allowlist.json')
    if target.exists():
        allowed = {str(target): cache.identity(os.lstat(target))}
        save(allowlist, allowed)
    else:
        allowed = {}
    if not (root / archive.archives.REGISTRY / name).exists():
        original = archive.archives.read_held

        def advised(path):
            if Path(path) != target:
                return original(path)
            value = cache.digest(str(target), allowed, HASH_CHUNK_BYTES)
            if value != binding['sha256']:
                return None, 'differs from the original capture byte hash'
            return value, allowed[str(target)]

        archive.archives.read_held = advised
        try:
            made, problems = archive.make(root, binding['record'], membership)
        finally:
            archive.archives.read_held = original
        if problems or made != name:
            raise RuntimeError(f'archive inventory refused: {problems}')
    record, problems = archive.archives.one(root, name)
    if (problems or record is None or record['input'] != binding['record']
            or set(record['files']) != {'.'} or record['files']['.']['sha256'] != binding['sha256']):
        raise ValueError(f'archive inventory is not the exact captured file: {problems}')
    if target.exists() and archive.archives.unchanged(root, record):
        raise ValueError('local stream changed since its archive inventory')
    return name, record, allowlist


def archive_previous(run: Path, previous_step: int, current_step: int, declaration: dict, *, stop_requested=None) -> dict:
    """Synchronously wait for publication, verify both old files remotely, then release.

    Required declaration.consumed_trace_archive: active_runs, observer_state,
    observer_interval_seconds (the existing observer's configured interval), and
    retained_steps. No capture may start while this call waits or after it raises.
    """
    run = Path(run).absolute()
    root = run.parents[3]
    if run != root / 'studies/04-round-2-overlap/runs' / run.name or run.resolve() != run:
        raise ValueError('expected a canonical round-two run folder')
    if (type(previous_step) is not int or type(current_step) is not int
            or previous_step < 0 or current_step != previous_step + 1):
        raise ValueError('only the immediately consumed predecessor may be released')
    if read(run / 'declaration.json') != declaration:
        raise ValueError('archive configuration is not the on-disk declaration')
    config = declaration['consumed_trace_archive']
    active = config['active_runs']
    if run.name not in active:
        raise ValueError('the current run must remain in the archive active-input guard')
    interval = config['observer_interval_seconds']
    if not isinstance(interval, (int, float)) or not math.isfinite(interval) or interval <= 0:
        raise ValueError('publication wait uses the declared existing observer interval')
    previous, capture = observation(root, run, previous_step, local=False)
    current, _ = observation(root, run, current_step, local=True)
    if (previous['member'] != current['member']
            or previous['model_identity']['original_sha256'] != current['model_identity']['original_sha256']):
        raise ValueError('old and current observations must belong to the same native continuation')
    prior_binding = dict(record=(run / 'loop' / str(previous_step) / 'measurement.json').relative_to(root).as_posix(),
                         sha256=digest(run / 'loop' / str(previous_step) / 'measurement.json'))
    if read(run / 'loop' / str(current_step) / 'proposal.json').get('source_observation') != prior_binding:
        raise ValueError('the current derivative did not consume this previous observation')
    if previous_step in config['retained_steps']:
        return dict(previous_step=previous_step, current_step=current_step, retained=True, archives=[])
    targets = [root / capture['raw'][name]['record'] for name in STREAMS]
    needs, problems = archive.needed(root, active)
    if problems or any(target == root / path or target.is_relative_to(root / path) for target in targets for path in needs):
        raise ValueError(f'an active declared input cannot be released: {problems}')
    output = run / 'trace-archives' / str(previous_step)
    receipt = dict(previous_step=previous_step, current_step=current_step, archives=[], phase='waiting_for_publication')
    save(output / 'release.json', receipt)
    while True:
        if stop_requested is not None and stop_requested():
            raise InterruptedError('stopped while waiting for the consumed trace publication; local files retained')
        publication = published(root, run, previous_step, previous, Path(config['observer_state']))
        if publication is not None:
            break
        time.sleep(interval)
    receipt.update(phase='archiving', publication=publication)
    save(output / 'release.json', receipt)
    membership = dict(sources=[(run / 'loop' / str(previous_step) / 'capture/capture.json').relative_to(root).as_posix(),
                               prior_binding['record'], (run / 'declaration.json').relative_to(root).as_posix()],
                      answers=[previous['member']], pooled={'.': [previous['member']]})
    save(output / 'membership.json', membership)
    ready = []
    for stream in STREAMS:
        binding = capture['raw'][stream]
        name, record, allowlist = inventoried(root, binding, membership, output)
        if (root / binding['record']).exists():
            with archive_environment():
                said, failed = archive.bucket(root, 'upload', record, sorted(record['files']))
            (output / (stream + '-upload.log')).write_text(said)
            if failed:
                raise RuntimeError('archive upload failed; no local stream released')
            with archive_environment(allowlist, name):
                problems = archive.verify(root, name)
            if problems:
                raise RuntimeError(f'archive readback failed; no local stream released: {problems}')
        elif record.get('remote_bytes_verified') is None:
            raise ValueError('missing local stream has no verified archive; refusing to proceed')
        ready.append(name)
    # All transfers succeeded before either removal. Recheck the retained handoff.
    observation(root, run, current_step, local=True)
    for name in ready:
        done, problems = archive.release(root, name, active)
        if problems or done is None or done['files_released'] + done['already_gone'] != 1:
            raise RuntimeError(f'guarded archive release did not finish: {problems}; {done}')
        receipt['archives'].append(dict(record=name, release=done))
        save(output / 'release.json', receipt)
    receipt['phase'] = 'complete'
    save(output / 'release.json', receipt)
    return receipt
