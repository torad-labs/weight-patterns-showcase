"""Two prescribed native states on one saved acquisition, in one anonymous model.

The sole controller launches this after assessing the same-card reconstruction.
Only this diagnostic's outputs are written. Original/current models are opened
read-only. Full-compute teacher mode emits incidental vocabulary metadata;
this program leaves it unread and compares only complete computational traces.
"""
import ctypes
import fcntl
import hashlib
import importlib.util
import io
import json
import mmap
import os
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
held_path = HERE.parent / 'native-held/replay.py'
spec = importlib.util.spec_from_file_location('held_native_diagnostic', held_path)
held = importlib.util.module_from_spec(spec)
spec.loader.exec_module(held)
from pipeline.weight_edit import gap_state

# The local uv Python was built without these Linux wrappers/constants.
# Values are the installed /usr/include/linux/memfd.h and
# /usr/include/x86_64-linux-gnu/bits/fcntl-linux.h ABI, not research parameters.
MFD_CLOEXEC, MFD_ALLOW_SEALING = 0x0001, 0x0002
F_ADD_SEALS, F_GET_SEALS = 1033, 1034
F_SEAL_SEAL, F_SEAL_SHRINK, F_SEAL_GROW, F_SEAL_WRITE = 0x0001, 0x0002, 0x0004, 0x0008


def new_memfd():
    libc = ctypes.CDLL(None, use_errno=True)
    libc.memfd_create.argtypes = (ctypes.c_char_p, ctypes.c_uint)
    libc.memfd_create.restype = ctypes.c_int
    fd = libc.memfd_create(b'wp-first-bad-diagnostic', MFD_CLOEXEC | MFD_ALLOW_SEALING)
    if fd < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return fd


def save(path, value):
    with Path(path).open('x') as file:
        json.dump(value, file, indent=2)
        file.write('\n')


def memory_model(seed, expected_sha256):
    """One seed read, hash, and RAM copy; no disk model copy or second hash pass."""
    before = held.working_identity.fingerprint(seed)
    fd = new_memfd()
    try:
        os.ftruncate(fd, before[2])
        digest = hashlib.sha256()
        with Path(seed).open('rb') as source, os.fdopen(os.dup(fd), 'wb') as destination:
            os.posix_fadvise(source.fileno(), 0, 0, os.POSIX_FADV_NOREUSE)
            offset = 0
            while block := source.read(gap_state.BLOCK):
                digest.update(block)
                destination.write(block)
                os.posix_fadvise(source.fileno(), offset, len(block), os.POSIX_FADV_DONTNEED)
                offset += len(block)
        if (digest.hexdigest() != expected_sha256 or held.working_identity.fingerprint(seed) != before
                or os.fstat(fd).st_size != before[2] or offset != before[2]):
            raise ValueError('the retained seed changed or differs from its declared bytes')
        fcntl.fcntl(fd, F_ADD_SEALS, F_SEAL_GROW | F_SEAL_SHRINK)
        return fd
    except BaseException:
        os.close(fd)
        raise


def load_changes(candidate, candidate_binding, tensors, names, expected_update, original):
    if (candidate['update'] != expected_update or candidate['base_seed'] != original['working_model']['start']
            or [row['name'] for row in candidate['inventory']] != list(names)):
        raise ValueError('the candidate is not this complete prescribed reconstruction')
    source = Path(candidate_binding['record']).parent / candidate['overrides']['file']
    if Path(candidate['overrides']['file']).name != candidate['overrides']['file']:
        raise ValueError('the sparse payload must be beside its bound candidate record')
    data = held.replay_updates.payload(ROOT / source, candidate['overrides']['sha256'])
    changes = []
    with np.load(io.BytesIO(data), allow_pickle=False) as arrays:
        if set(arrays.files) != {f'{prefix}_{i}' for i in range(len(names)) for prefix in ('index', 'value')}:
            raise ValueError('the sparse payload omits or adds a matrix')
        for i, name in enumerate(names):
            index, value = arrays[f'index_{i}'], arrays[f'value_{i}']
            if (index.dtype != np.dtype('<i8') or value.dtype != np.dtype('<u2')
                    or index.ndim != 1 or value.shape != index.shape
                    or len(index) != candidate['inventory'][i]['entries']
                    or (len(index) and (index[0] < 0 or index[-1] >= tensors[name]['bytes'] // 2))
                    or bool((np.diff(index) <= 0).any()) or bool(((value & 0x7f80) == 0x7f80).any())):
                raise ValueError('sparse BF16 locations/values are incomplete, duplicate, out of range, or nonfinite')
            changes.append((name, index.copy(), value.copy()))
    return changes, dict(record=str(source), sha256=candidate['overrides']['sha256'])


def apply_changes(fd, tensors, changes, restore=()):
    """Reset only prior sparse locations, then install the next seed-relative set."""
    undo = []
    with mmap.mmap(fd, 0, access=mmap.ACCESS_WRITE) as memory:
        def apply(name, index, values, remember):
            tensor = tensors[name]
            array = np.ndarray((tensor['bytes'] // 2,), dtype='<u2', buffer=memory, offset=tensor['offset_bytes'])
            if remember:
                undo.append((name, index.copy(), array[index].copy()))
            array[index] = values
            if not np.array_equal(array[index], values):
                raise ValueError('a sparse BF16 write did not reach the anonymous model')
            del array
        for name, index, values in restore:
            apply(name, index, values, False)
        for name, index, values in changes:
            apply(name, index, values, True)
    return undo


def capture(fd, folder, identity, cases_file, engine, inputs):
    """Only a read-only model descriptor reaches the native worker."""
    folder.mkdir()
    read_fd = os.open(f'/proc/self/fd/{fd}', os.O_RDONLY | os.O_CLOEXEC)
    try:
        before = held.working_identity.fingerprint(f'/proc/self/fd/{fd}')
        command = [str(held.generation_engine.RECORDER), f'/proc/self/fd/{read_fd}', str(folder / 'raw'),
                   str(cases_file), str(ROOT / 'dependencies/llama-cpp-build'), 'capture', 'residual', 'stream']
        with Path(f'/run/user/{os.getuid()}/gpu0.lock').open('r+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            held.frozen(inputs)
            with (folder / 'stdout.jsonl').open('x') as stdout, (folder / 'stderr.log').open('x') as stderr:
                process = subprocess.Popen(command, env=engine.env, stdout=stdout, stderr=stderr,
                                           pass_fds=(read_fd, lock.fileno()))
                watch = held.complete_watch(engine, process)
                code = process.wait()
            if code:
                raise subprocess.CalledProcessError(code, command)
            account = engine.ended(watch, folder / 'stderr.log', cases=[inputs['member']])
        if held.working_identity.fingerprint(f'/proc/self/fd/{fd}') != before:
            raise ValueError('the anonymous model changed during native computation')
        held.frozen(inputs)
        record = dict(model_identity=identity, command=command, engine=engine.described(passes=[account]),
            model_storage='one anonymous memfd, child inherits read-only descriptor',
            seals=fcntl.fcntl(fd, F_GET_SEALS), file_identity_during_pass=before,
            immutability='sole diagnostic writer performs no writes until the child exits; stat checked unchanged; final state additionally sealed against writes',
            cases=held.binding(cases_file),
            raw={name: held.binding(folder / 'raw' / inputs['member'] / name) for name in ('stream.f32', 'stream.jsonl')})
        save(folder / 'capture.json', record)
        return record
    finally:
        os.close(read_fd)


def trace_comparison(actual, expected):
    rows = []
    for i in range(len(expected['merged'])):
        record = dict(merge=i)
        for field in ('incoming', 'merged'):
            a, b = actual[field][i].numpy(), expected[field][i].numpy()
            if a.shape != b.shape:
                raise ValueError('the baseline trace has another complete event inventory')
            record[field + '_different_bits'] = int(np.count_nonzero(a.view('<u4') != b.view('<u4')))
            record[field + '_max_abs'] = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
        rows.append(record)
    return rows


def main():
    declaration = held.check_native_run(HERE, HERE / 'declaration.json')
    for key, path in (('program', Path(__file__).resolve()), ('native_held_program', held_path),
                      ('native_held_inputs', held.HERE / 'inputs.json')):
        if declaration[key] != held.binding(path):
            raise ValueError(f'the actual diagnostic source/input differs: {key}')
    inputs, original, _, _ = held.load_inputs()
    if os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID' or os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise ValueError('use the declared original physical GPU0')
    replay_report = held.checked(ROOT, declaration['same_card_report'])
    candidate_bindings = declaration['candidates']
    if len(candidate_bindings) != 2 or replay_report['seed_verified_sha256'] != original['working_model']['start']['model_sha256']:
        raise ValueError('both prescribed candidates and their verified reconstruction are required')
    tensors = held.checked(ROOT, original['tensors'])
    architecture = held.checked(ROOT, original['architecture'])
    definition = held.checked(ROOT, original['source_definition'])
    names = tuple(original['functional_loop']['matrices'])
    seed = original['working_model']['start']
    size = (ROOT / seed['file']).stat().st_size
    spans = sorted((tensors[name]['offset_bytes'], tensors[name]['offset_bytes'] + tensors[name]['bytes']) for name in names)
    if (len(names) != 70 or len(set(names)) != 70 or any(a < 0 or b > size or a >= b for a, b in spans)
            or any(a[1] > b[0] for a, b in zip(spans, spans[1:]))
            or any(tensors[name]['native_type'] != 'bf16' or tensors[name]['bytes'] != 2 * int(np.prod(tensors[name]['shape'])) for name in names)):
        raise ValueError('the seventy BF16 spans do not partition the declared editable matrix inventory')
    candidates = [held.checked(ROOT, source) for source in candidate_bindings]
    prepared = [load_changes(candidate, source, tensors, names, update, original)
                for candidate, source, update in zip(candidates, candidate_bindings, (5, 6))]
    if (ROOT / declaration['baseline_observation']['record'] != held.SOURCE / 'loop/6/measurement.json'
            or ROOT / declaration['recorded_proposal']['record'] != held.SOURCE / 'loop/6/proposal.json'):
        raise ValueError('the diagnostic must bind the prescribed first-bad transition')
    measured = held.checked(ROOT, declaration['baseline_observation'])
    recorded_proposal = held.checked(ROOT, declaration['recorded_proposal'])
    reading = held.checked(ROOT, measured['reading'])
    if measured['generation'] != 5 or measured['member'] != inputs['member'] or len(reading['merges']) != 105:
        raise ValueError('the baseline must be saved observation6 at generation5 with every merge')
    folder = (ROOT / measured['reading']['record']).parent
    table = next(row for row in reading['table'] if row['id'] == inputs['member'])
    acquisition = held.checked(ROOT, dict(record=str((folder / 'answers' / inputs['member'] / 'answer.json').relative_to(ROOT)),
                                         sha256=table['answer_json_sha256']))
    labels = sorted({node[key] for node in reading['merges'] for key in ('incoming', 'branch', 'connection')})
    cases_file = HERE / 'cases.json'
    save(cases_file, [dict(id=inputs['member'], tokens=definition['prompts'][inputs['member']],
        teacher_tokens=acquisition['tokens'], stream_labels='^(' + '|'.join(map(re.escape, labels)) + ')$', stop_after_last_label=False)])
    engine = held.generation_engine.Generation()
    if engine.record['sha256'] != reading['generation_sha256'] or engine.record['engine_sha256'] != reading['libraries_sha256']:
        raise ValueError('the replay must use the saved native engine')
    reference = held.turn_artifact.load(ROOT, original['reference'], original_model_sha256=original['model_sha256'], expected_merges=105)
    forward_spec = importlib.util.spec_from_file_location('diagnostic_storage', ROOT / original['forward']['record'])
    forward = importlib.util.module_from_spec(forward_spec)
    forward_spec.loader.exec_module(forward)
    fd = memory_model(ROOT / seed['file'], seed['model_sha256'])
    try:
        model = SimpleNamespace(file=Path(f'/proc/self/fd/{fd}'), tensors=tensors, device=torch.device('cpu'),
                                width=architecture['width'], layers=architecture['layers'])
        model.w = {f'blk.{i}.layer_output_scale.weight': forward.stored(model.file, tensors[f'blk.{i}.layer_output_scale.weight'])
                   for i in range(model.layers)}
        model.rows = lambda name, ids: forward.Model.rows(model, name, ids)
        _, _, expected, pairs = held.replay_updates.saved_case(held.SOURCE, 6, model, definition)
        baseline = held.turn_correction.measurement(reference, expected, pairs)
        if baseline != {key: measured[key] for key in baseline}:
            raise ValueError('the complete saved computational loss does not reproduce')
        results, restore = [], []
        for update, source, (changes, payload) in zip((5, 6), candidate_bindings, prepared):
            restore = apply_changes(fd, tensors, changes, restore)
            if update == 6:
                fcntl.fcntl(fd, F_ADD_SEALS, F_SEAL_WRITE | F_SEAL_SEAL)
            identity = dict(kind='diagnostic-seed-sparse-bf16/1', reconstruction_update=update,
                seed=dict(file=seed['file'], sha256=seed['model_sha256']), candidate=source, overrides=payload)
            output = HERE / f'candidate-{update}'
            capture(fd, output, identity, cases_file, engine, inputs)
            with torch.no_grad():
                native = held.raw_forward(output / 'raw' / inputs['member'], reading['merges'], measured['events'], model.width)
                fixed = held.turn_correction.measurement(reference, native, pairs)
                row = dict(update=update, model_identity=identity, fixed_correspondence=fixed,
                    capture=held.binding(output / 'capture.json'), reference=original['reference'], correspondence=measured['correspondence'])
                if update == 5:
                    compared = trace_comparison(native, expected)
                    exact = fixed == baseline and all(r['incoming_different_bits'] == r['merged_different_bits'] == 0 for r in compared)
                    row.update(baseline_exact=exact, saved_baseline=baseline, native_trace_comparison=compared)
                else:
                    remapped = held.turn_correction.correspondence(reference, native)
                    row['computational_remapping'] = held.turn_correction.measurement(reference, native, remapped)
                    row['recorded_update_prediction'] = recorded_proposal['response']
                    replayed = next(item for item in replay_report['observations'] if item['update'] == 6)
                    row['same_card_calculation'] = {key: replayed[key] for key in ('replay', 'finite_unrounded_held_graph',
                        'finite_bf16_held_graph', 'actual_bf16_displacement_response') if key in replayed}
                    row['native_fixed_loss_change'] = fixed['objective'] - results[0]['fixed_correspondence']['objective']
                    row['native_fixed_loss_change_from_saved_baseline'] = fixed['objective'] - baseline['objective']
                save(output / 'measurement.json', row)
                results.append(row)
                print(json.dumps({k: row[k] for k in ('update', 'fixed_correspondence', 'baseline_exact',
                    'native_fixed_loss_change', 'native_fixed_loss_change_from_saved_baseline') if k in row}), flush=True)
                del native
        held.frozen(inputs)
        save(HERE / 'result.json', dict(format='prescribed-first-bad-native-diagnostic/1', results=results,
            baseline_reproduced=results[0]['baseline_exact'], same_card_report=declaration['same_card_report'],
            frozen_training_revision=inputs['state']['model_identity'], seed_verified_during_ram_copy=True,
            model_disk_copies=0, current_or_original_writes=0, model_storage='one memfd reused for both prescribed states',
            interpretation='Held-acquisition computation at prescribed diagnostic reconstructions; not training or a historical checkpoint claim. Primary change compares native6 with native5; the saved5 comparison remains separate even when reconstructed5 differs.'))
    finally:
        os.close(fd)


if __name__ == '__main__':
    main()
