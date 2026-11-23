"""One read-only native replay of revision 9 on observation 9's saved inputs.

prepare: bind small input records; no model load, GPU work, or array scan.
run: root's declared diagnostic, under the ordinary GPU lock, then measure.
measure: measure an already completed replay without another native pass.

The saved input IDs drive computation only. No logits, decoded answers, greedy
agreement, new correspondence, or generated-answer success judgment is read.
The unchanged recorder emits vocabulary metadata in this full-compute mode;
this diagnostic leaves that incidental metadata unread and measures tensors.
"""
import argparse
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
SOURCE = ROOT / 'studies/04-round-2-overlap/runs/vector-chance-working-from-update-3'
INSTRUMENTS = ROOT / 'records/development/2026-11-12-path-forward'
CGROUP = Path('/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/app.slice/weight-patterns-vector-working.service')
sys.path[:0] = [str(ROOT / 'code/signal-trace'), str(INSTRUMENTS), str(HERE.parent)]

from pipeline.native_records.run_declaration import check_native_run
from pipeline.weight_edit import turn_artifact, turn_correction, working_identity
from pipeline.weight_edit.functional_reading import digest, tokens_hash
from pipeline.weight_edit.functional_reference import checked
import generation_engine
import replay_updates
import states_reading


def binding(path):
    return dict(record=str(Path(path).relative_to(ROOT)), sha256=digest(Path(path)))


def write_once(name, value):
    with (HERE / name).open('x') as file:
        json.dump(value, file, indent=2)
        file.write('\n')


def frozen(inputs):
    events = dict(line.split() for line in (CGROUP / 'cgroup.events').read_text().splitlines())
    if events.get('frozen') != '1':
        raise ValueError('the sole training controller must remain frozen during this read-only diagnostic')
    state = working_identity.load_state(SOURCE / 'working/state.json',
        model=SOURCE / 'working/model.gguf', expected_identity=inputs['state']['model_identity'])
    if state != inputs['state']:
        raise ValueError('the prepared frozen working state changed')
    return state


def prepare():
    original = json.loads((SOURCE / 'declaration.json').read_text())
    measured = json.loads((SOURCE / 'loop/9/measurement.json').read_text())
    reading = checked(ROOT, measured['reading'])
    folder = (ROOT / measured['reading']['record']).parent
    rows = checked(ROOT, dict(record=str((folder / 'row-map.json').relative_to(ROOT)),
                             sha256=reading['row_map']['sha256']))
    member = measured['member']
    if reading['answers_read'] != [member] or len(rows['answers']) != 1 or len(reading['table']) != 1:
        raise ValueError('the diagnostic needs exactly the saved fitting member')
    entry, table = rows['answers'][0], reading['table'][0]
    acquisition_binding = dict(record=str((folder / 'answers' / member / 'answer.json').relative_to(ROOT)),
                               sha256=table['answer_json_sha256'])
    acquisition = checked(ROOT, acquisition_binding)
    definition = checked(ROOT, original['source_definition'])
    prompt, supplied = definition['prompts'][member], acquisition['tokens']
    if (entry['id'] != member or table['id'] != member or len(prompt) != entry['question'] + 1
            or tokens_hash(prompt) != entry['prompt_sha256'] or tokens_hash(supplied) != entry['text_sha256']
            or len(supplied) != entry['steps'] or len(prompt) + len(supplied) - 1 != measured['events']
            or reading['model_identity'] != measured['model_identity'] or measured['generation'] != 8
            or [m['merge'] for m in reading['merges']] != list(range(105))):
        raise ValueError('the saved acquisition or complete computational inventory differs')
    state = working_identity.load_state(SOURCE / 'working/state.json')
    expected = measured['model_identity'] | dict(generation=9)
    if state['model_identity'] != expected:
        raise ValueError('the replay must read frozen revision 9')
    labels = sorted({m[k] for m in reading['merges'] for k in ('incoming', 'branch', 'connection')})
    cases = [dict(id=member, tokens=prompt, teacher_tokens=supplied,
                  stream_labels='^(' + '|'.join(map(re.escape, labels)) + ')$', stop_after_last_label=False)]
    # A forced replay is a separate diagnostic. It cannot be passed off as the
    # native-events-only own-generation capture required by the training loop.
    write_once('cases.json', cases)
    inputs = dict(format='held-input-computational-diagnostic/1', observation=9, member=member,
        state=state, source_declaration=binding(SOURCE / 'declaration.json'),
        measurement=binding(SOURCE / 'loop/9/measurement.json'), acquisition=acquisition_binding,
        cases=binding(HERE / 'cases.json'), program=binding(Path(__file__).resolve()),
        saved_reader=binding(HERE.parent / 'replay_updates.py'), cgroup=str(CGROUP),
        physical_gpu='0', labels=labels, interpretation='conditional replay at fixed saved acquisition and correspondence, not fresh-generation progress',
        stop_semantics='Full compute, no early stop: the qualified kept scheduler rejects early-stop callbacks.',
        incidental_metadata='The existing recorder computes and emits vocabulary/greedy metadata in ordinary teacher_tokens mode. This helper never opens result.json or reads, scores, or compares that metadata; only complete stream tensors enter its loss.',
        watch_semantics='Accept one actual live process map only after every engine.files path appears; preserve engine.ended stamp, scheduler inode/digest, and execution-account checks.',
        previous_failed_attempts=[str((HERE / name / 'archive.json').relative_to(ROOT))
                                  for name in ('failed-stop-mode', 'failed-library-watch')])
    frozen(inputs)
    write_once('inputs.json', inputs)
    print(json.dumps(dict(inputs=binding(HERE / 'inputs.json'), cases=1, events=measured['events'], merges=105,
                          generation=9, gpu_work=False, model_bytes_read=0)), flush=True)


def load_inputs():
    declaration = check_native_run(HERE, HERE.parent / 'declaration.json')
    source = declaration['native_held_inputs']
    if ROOT / source['record'] != HERE / 'inputs.json':
        raise ValueError('the parent declaration must bind this prepared diagnostic')
    inputs = checked(ROOT, source)
    for key in ('program', 'saved_reader'):
        checked(ROOT, inputs[key], parse=False)
    cases = checked(ROOT, inputs['cases'])
    original = checked(ROOT, inputs['source_declaration'])
    for source in original['implementation']:
        checked(ROOT, source, parse=False)
    checked(ROOT, inputs['acquisition'])
    measured = checked(ROOT, inputs['measurement'])
    reading = checked(ROOT, measured['reading'])
    if (cases[0]['id'] != inputs['member'] or len(cases) != 1
            or reading['generation_sha256'] != reading['engine']['sha256']):
        raise ValueError('prepared case or saved engine binding differs')
    frozen(inputs)
    return inputs, original, measured, reading


def complete_watch(engine, process):
    """The early BASE snapshot can precede a lazy backend's dlopen."""
    watch = engine.watch(process)
    required = {str(path) for path in engine.files.values()}
    while process.poll() is None:
        seen = generation_engine.recorder_engine.mapped(process.pid)
        if seen is not None and required.issubset(seen):
            # Stop the original reader before replacing its early snapshot.
            # This is one actual map, never a union or an inferred mapping.
            watch.thread.join()
            watch.seen, watch.pid = seen, process.pid
            return watch
        time.sleep(watch.every)
    raise ValueError('recorder exited before all bound libraries were observed in its live process map')


def capture(inputs, reading):
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != inputs['physical_gpu']
            or os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID'):
        raise ValueError('this comparison uses CUDA_DEVICE_ORDER=PCI_BUS_ID and CUDA_VISIBLE_DEVICES=0')
    engine = generation_engine.Generation()
    if (engine.record['sha256'] != reading['generation_sha256']
            or engine.record['engine_sha256'] != reading['libraries_sha256']):
        raise ValueError('the replay must use the saved capture engine and libraries')
    command = [str(generation_engine.RECORDER), inputs['state']['model_file'], str(HERE / 'raw'),
               str(HERE / 'cases.json'), str(ROOT / 'dependencies/llama-cpp-build'), 'capture', 'residual', 'stream']
    # No owner.lock impersonation: this is a read-only diagnostic under an
    # externally frozen sole controller. GPU ownership still uses its real lock.
    with Path(f'/run/user/{os.getuid()}/gpu0.lock').open('r+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('GPU0 is still owned, including by a frozen recorder; root must resolve its ownership before launch') from error
        frozen(inputs)
        with (HERE / 'recorder.stdout.jsonl').open('x') as stdout, (HERE / 'recorder.stderr.log').open('x') as stderr:
            process = subprocess.Popen(command, env=engine.env, stdout=stdout, stderr=stderr, pass_fds=(lock.fileno(),))
            watch = complete_watch(engine, process)
            code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)
        engine.ended(watch, HERE / 'recorder.stderr.log', cases=[inputs['member']])
    frozen(inputs)
    write_once('capture.json', dict(command=command, engine=engine.described(), model_identity=inputs['state']['model_identity'],
        inputs=binding(HERE / 'inputs.json'), stream=binding(HERE / 'raw' / inputs['member'] / 'stream.f32'),
        index=binding(HERE / 'raw' / inputs['member'] / 'stream.jsonl')))


def raw_forward(raw, nodes, count, width):
    found = states_reading.rows_in(raw, count)
    values = np.memmap(raw / 'stream.f32', dtype='<f4', mode='r')
    positions = np.arange(count)
    native = dict(incoming=[], merged=[])
    for index, node in enumerate(nodes):
        if node['merge'] != index:
            raise ValueError('native merge order differs')
        output = found[node['connection']]
        if output['operation'] != 'ADD' or output['sources'] != sorted([node['incoming'], node['branch']]):
            raise ValueError('the native recorded merge is not its declared ADD')
        before, branch, after = [states_reading.taken(values, found[node[key]], positions)
                                 for key in ('incoming', 'branch', 'connection')]
        if (any(value.shape != (count, width) or not bool(np.isfinite(value).all()) for value in (before, branch, after))
                or not np.array_equal(np.add(before, branch).view('u4'), after.view('u4'))):
            raise ValueError('native ADD, complete event inventory, or finite width differs')
        native['incoming'].append(torch.from_numpy(before))
        native['merged'].append(torch.from_numpy(after))
    return native


def measure(inputs, original, measured, reading):
    captured = json.loads((HERE / 'capture.json').read_text())
    if captured['inputs'] != binding(HERE / 'inputs.json') or captured['model_identity'] != inputs['state']['model_identity']:
        raise ValueError('the completed replay belongs to other prepared inputs or revision')
    for key in ('stream', 'index'):
        checked(ROOT, captured[key], parse=False)
    tensors = checked(ROOT, original['tensors'])
    architecture = checked(ROOT, original['architecture'])
    definition = checked(ROOT, original['source_definition'])
    spec = importlib.util.spec_from_file_location('held_forward_storage', ROOT / original['forward']['record'])
    forward = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(forward)
    # Only the fixed layer scales and the acquired embedding rows are needed to
    # reconstruct saved incoming values. Do not instantiate or scan a full model.
    model = SimpleNamespace(file=Path(inputs['state']['model_file']), tensors=tensors, device=torch.device('cpu'),
        width=architecture['width'], layers=architecture['layers'])
    model.w = {f'blk.{i}.layer_output_scale.weight': forward.stored(model.file, tensors[f'blk.{i}.layer_output_scale.weight'])
               for i in range(model.layers)}
    model.rows = lambda name, ids: forward.Model.rows(model, name, ids)
    reference = turn_artifact.load(ROOT, original['reference'], original_model_sha256=original['model_sha256'],
                                   expected_merges=3 * model.layers)
    _, _, previous, pairs = replay_updates.saved_case(SOURCE, inputs['observation'], model, definition)
    with torch.no_grad():
        before = turn_correction.measurement(reference, previous, pairs)
        if before != {key: measured[key] for key in before}:
            raise ValueError(f'saved-array whole loss differs from the original measurement: {before}')
        del previous
        native = raw_forward(HERE / 'raw' / inputs['member'], reading['merges'], measured['events'], model.width)
        after = turn_correction.measurement(reference, native, pairs)
        per_merge = [dict(merge=i, objective=float(value.residual.double().square().sum()), outside=int(value.outside.sum()))
                     for i, value in enumerate(reference.measure(native, pairs))]
        if math.fsum(row['objective'] for row in per_merge) != after['objective'] or sum(row['outside'] for row in per_merge) != after['outside']:
            raise ValueError('per-merge loss does not equal the complete native loss')
    frozen(inputs)
    report = dict(format='held-input-computational-result/1', interpretation=inputs['interpretation'],
        generation=inputs['state']['generation'], history_generation=measured['generation'], model_identity=inputs['state']['model_identity'],
        inputs=binding(HERE / 'inputs.json'), capture=binding(HERE / 'capture.json'), correspondence=measured['correspondence'],
        reference=original['reference'], saved_baseline_reproduced_exactly=True, baseline=before, native_held=after,
        objective_change=after['objective'] - before['objective'], per_merge=per_merge, fresh_generation=False,
        model_writes=0, model_full_scans=0, rows=measured['events'], merges=len(native['merged']))
    write_once('result.json', report)
    print(json.dumps({key: report[key] for key in ('generation', 'saved_baseline_reproduced_exactly', 'baseline', 'native_held', 'objective_change')}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'measure'))
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
        return
    inputs, original, measured, reading = load_inputs()
    if args.action == 'run':
        capture(inputs, reading)
    measure(inputs, original, measured, reading)


if __name__ == '__main__':
    main()
