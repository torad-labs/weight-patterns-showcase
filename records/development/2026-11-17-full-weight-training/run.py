"""Full-path correction through every declared trainable weight block.

Each update differentiates all105 original weighted computational residual terms
through the current complete graph into all70 trainable matrices. One measured
response scales the joint negative gradient, complete FP32 remainders survive
BF16 installation, and ordinary fresh generation supplies the next observation.
"""
import argparse
from contextlib import contextmanager, redirect_stdout
from dataclasses import asdict
from datetime import datetime
import fcntl
import gc
import importlib.util
import io
import json
import math
import mmap
import os
from pathlib import Path
import signal
import shutil
import sys
import time

import numpy as np
import torch

SELF = Path(__file__).resolve().parent
HERE = SELF.parents[2] / 'studies/04-round-2-overlap/runs/vector-chance-loss-rise-diagnosis/empirical-eta-continuation'


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evaluation = load_module('continued_native_evaluation', HERE.parent / 'response-step-experiment/evaluate.py')
factory, held, ROOT = evaluation.factory, evaluation.held, evaluation.ROOT
peer = load_module('continued_staging_source', HERE.parent / 'per-merge-response/measure.py')
precision = load_module('prospective_ffn_precision', HERE.parent / 'per-merge-response/ffn-native-precision-adapter/adapter.py')
from pipeline.weight_edit import bf16_direction, gap_state, model_trace_response, native_graph, trace_gradient
from pipeline.observability.recorder import Recorder
import pipeline.observability.recorder as event_source
archive = load_module('consumed_trace_archive', HERE / 'consumed_trace_archive.py')
accumulation = load_module('bf16_accumulation', HERE / 'bf16_accumulation.py')
native_encoding = load_module('native_bf16_encoding', HERE / 'native_bf16_encoding.py')
reclaim = load_module('verified_deleted_handles', ROOT / 'records/development/2026-11-10-speed-review/declared-inputs/held_deleted.py')
import native_events
observation_source = load_module('sparse_native_observation', ROOT / 'records/development/2026-11-16-full-request-continuation/observation.py')
projection_adapter = load_module('explicit_projection_derivative', ROOT / 'records/development/2026-11-16-projection-gradient-adapter/adapter.py')
projection_geometry = load_module('native_projection_geometry', ROOT / 'records/development/2026-11-16-attention-projection-precision/evaluate.py')
capture_source = load_module('continuation_capture', SELF / 'capture.py')

# Preserve source-bound archive worker context without changing the central helper.
archive.archive = load_module('full_weight_archive_operator', ROOT / 'records/development/2026-11-17-capture-context/archive_inputs.py')


def source_files():
    """Additional sources, beside the original declaration's complete inventory."""
    modules = (evaluation, factory, held, held.replay_updates, peer, native_events,
               bf16_direction, gap_state, model_trace_response, native_graph, trace_gradient, precision, archive, accumulation, native_encoding, event_source, reclaim, observation_source, projection_adapter, projection_geometry, projection_geometry.projection, capture_source)
    return sorted({Path(__file__).resolve(), *(Path(m.__file__).resolve() for m in modules), *archive.source_files()})


def retired_predecessor(inputs):
    """The prior writer is gone; preserve its existing ready-file identity checks."""
    events_file = held.CGROUP / 'cgroup.events'
    try:
        events = dict(line.split() for line in events_file.read_text().splitlines())
    except FileNotFoundError:
        events = {'populated': '0'}
    if events.get('populated') != '0':
        raise ValueError('the retired predecessor still has a live process')
    state = held.working_identity.load_state(held.SOURCE / 'working/state.json',
        model=held.SOURCE / 'working/model.gguf', expected_identity=inputs['state']['model_identity'])
    if state != inputs['state']:
        raise ValueError('the retained predecessor working state changed')
    return state


# Process-local replacement of the old diagnostic's live-frozen requirement.
# Original helper files and every numerical function remain unchanged.
held.frozen = retired_predecessor


def now():
    return datetime.now().astimezone().isoformat()


@contextmanager
def card():
    with Path(f'/run/user/{os.getuid()}/gpu0.lock').open('r+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def resident(module, fd, tensors, architecture, names):
    """The peer's existing CPU staging; every actual operand uses GPU0."""
    model = module.Model(Path(f'/proc/self/fd/{fd}'), tensors, architecture,
                         device='cpu', leaves=lambda name: name in names, cache='float16')
    if tuple(model.leaves) != tuple(names):
        raise ValueError('the resident model must expose every declared matrix in order')
    model.device = torch.device('cuda:0')
    staged_parts = {'attn_q', 'attn_k', 'attn_v', 'ffn_up', 'ffn_gate', 'inp_gate', 'proj'}
    with torch.no_grad():
        for name in tuple(model.w):
            staged = name not in names and (name == 'per_layer_model_proj.weight' or
                (name.startswith('blk.') and name.split('.')[2] in staged_parts))
            if not staged:
                model.w[name] = model.w[name].detach().to(model.device)
        model.leaves = {name: model.w[name] for name in names}
        weight = model.weight
        model.weight = lambda name: weight(name).to(model.device)
    return model


def native_bits(fd, tensors, names):
    """Read only the declared editable spans in RAM, in native storage order."""
    with mmap.mmap(fd, 0, access=mmap.ACCESS_READ) as memory:
        return {name: np.ndarray((tensors[name]['bytes'] // 2,), dtype='<u2',
                buffer=memory, offset=tensors[name]['offset_bytes']).copy() for name in names}


def install(fd, model, tensors, names, weights, remainders, next_remainders, rows, installation):
    """Write complete native matrices, verify the RAM write, then refresh resident leaves."""
    if set(weights) != set(names):
        raise ValueError('installation requires every declared matrix')
    accumulation.validate_arrays(next_remainders, rows)
    # Validate all payloads before the first write; encoding is the existing
    # (input,output) adapter for resident (output,input) weights.
    encoded = {name: native_encoding.encoded_resident(weights[name], tensors[name]) for name in names}
    with mmap.mmap(fd, 0, access=mmap.ACCESS_WRITE) as memory, torch.no_grad():
        for name in names:
            installation['in_progress'] = name
            binding, raw = tensors[name], encoded.pop(name)
            start, size = binding['offset_bytes'], binding['bytes']
            memory[start:start + size] = raw
            if memory[start:start + size] != raw:
                raise ValueError(f'the complete RAM write differs for {name}')
            model.w[name].copy_(torch.from_numpy(weights[name]).to(model.w[name]))
            remainders[name] = next_remainders.pop(name)
            installation['committed_matrices'].append(name)
            installation['in_progress'] = None
        model.leaves = {name: model.w[name] for name in names}


def checkpoint(run, fd, tensors, names, seed_bits, seed, generation, **details):
    """One exit checkpoint, reflecting actual RAM bytes even after a failed write."""
    arrays, inventory = {}, []
    with mmap.mmap(fd, 0, access=mmap.ACCESS_READ) as memory:
        for index, name in enumerate(names):
            current = np.ndarray((tensors[name]['bytes'] // 2,), dtype='<u2',
                                 buffer=memory, offset=tensors[name]['offset_bytes'])
            selected = np.flatnonzero(current != seed_bits[name]).astype('<i8')
            arrays[f'index_{index}'], arrays[f'value_{index}'] = selected, current[selected].copy()
            inventory.append(dict(name=name, entries=len(selected)))
            del current
    payload = run / 'final-overrides.npz'
    with payload.open('xb') as file:
        np.savez(file, **arrays)
    result = dict(update=generation, base_seed=seed, inventory=inventory,
        overrides=dict(file=payload.name, sha256=held.binding(payload)['sha256']),
        saved_at=now(), interpretation='Exit-only RAM continuation checkpoint; local update count, not historical revision identity', **details)
    factory.save(run / 'final-state.json', result)
    return held.binding(run / 'final-state.json')


def checkpoint_with_remainder(run, fd, tensors, names, seed_bits, seed, generation,
                              remainders, rows, installation, **details):
    """Keep the native exit checkpoint even when the additional payload cannot save."""
    remainder_error = None
    try:
        saved = accumulation.save(run, remainders, rows,
            model_identity=details['model_identity'], iteration=generation,
            complete=details['pending_install'] is None, installation=installation)
    except Exception as error:
        remainder_error = error
        saved = dict(format=accumulation.FORMAT, complete=False,
                     error=f'{type(error).__name__}: {error}', installation=installation)
    binding = checkpoint(run, fd, tensors, names, seed_bits, seed, generation,
                         accumulator=saved, **details)
    return binding, remainder_error


def propose(model, history, native, reference, pairs, measured, eta, remainders, progress, observation, seed_bits, rows, storage, run, recorder):
    graph = native_graph.Graph(model, history, native)
    original_forward = graph.forward
    origin_checks = []

    def checked_forward(*args, **kwargs):
        forward = original_forward(*args, **kwargs)
        for field, captured in (('incoming', graph.native_in), ('merged', graph.native_out)):
            if len(forward[field]) != len(captured):
                raise ValueError('the differentiated origin omits captured merges')
            for value, expected in zip(forward[field], captured):
                primal = torch.autograd.forward_ad.unpack_dual(value).primal.detach()
                if (primal.shape != expected.shape or primal.dtype != expected.dtype
                        or not torch.equal(primal, expected)):
                    raise ValueError('the differentiated origin differs from the captured native tensors')
        origin_checks.append(True)
        return forward  # Keep both reverse gradients and forward tangents untouched.

    graph.forward = checked_forward
    progress('full_computational_derivative_started', trainable_matrices=len(graph.leaves), merges=reference.merges)
    gradient = trace_gradient.weight_gradient(graph, history, reference, pairs=pairs)
    if set(gradient.direction) != set(model.leaves):
        raise ValueError('the joint gradient omits a declared trainable matrix')
    if any(gradient.before[key] != measured[key] for key in ('outside', 'pair_merge_cells')):
        raise ValueError('the joint derivative origin has another complete residual inventory')
    record = dict(before=gradient.before, eta=eta,
        gradient_norm_squared=gradient.norm_squared,
        full_request_objective=gradient.before['objective'],
        full_request_reduction_difference=gradient.before['objective']-measured['objective'],
        direction_matrices=len(gradient.direction), request_merges=reference.merges,
        gradient_scope='All original weighted computational losses differentiated through the complete current graph into every declared trainable matrix.',
        direction_scope='Fresh joint negative weight gradient in all70 declared trainable blocks; no selected projection or manually zeroed other blocks.',
        timings=dict(weight_derivative_seconds=gradient.seconds),
        gradient_scalar_reduction_difference=gradient.before['objective']-measured['objective'])
    progress('full_weight_direction_complete', **record)
    if gradient.norm_squared == 0:
        return dict(record, status='zero_weight_gradient_not_full_loss_closure'), None, None
    started = time.monotonic()
    with recorder.context(response_kind='continuous_direction', solver_product_index=1):
        response = model_trace_response.measure_model(graph, tuple(model.leaves), history,
            gradient.direction, lambda forward: observation.residuals(reference, pairs, forward, continuous=True))
    record.update(response=asdict(response),
        response_scalar_reduction_difference=response.objective-measured['objective'],
        captured_origin_checks=len(origin_checks))
    record['timings']['directional_response_seconds'] = time.monotonic() - started
    progress('directional_response_complete', **record)
    if response.status != 'proposal' or response.length is None:
        return dict(record, status=response.status), None, None
    length, weights, next_remainders, displacement = eta * response.length, {}, {}, {}
    changed = remainder_changed = 0
    remainder_before, remainder_after, displacement_norms = [], [], []
    started = time.monotonic()
    for name in model.leaves:
        with recorder.context(matrix=name):
            base = graph.w[name].detach().cpu().numpy()
            direction = gradient.direction.pop(name).cpu().numpy()
            stored, following = accumulation.step(base, remainders[name], direction, length)
            delta = accumulation.actual_delta(base, stored)
            changed += int(np.count_nonzero(stored != base))
            remainder_changed += int(np.count_nonzero(following != remainders[name]))
            remainder_before.append(accumulation.squared(remainders[name]))
            remainder_after.append(accumulation.squared(following))
            displacement_norms.append(accumulation.squared(delta))
            observation.add(name, base, stored, seed_bits[name])
            weights[name], next_remainders[name] = stored, following
            displacement[name] = torch.from_numpy(delta).to(graph.w[name])
    record['timings']['native_rounding_seconds'] = time.monotonic() - started
    record['sparse_storage_reservation'] = observation.write_candidate(rows, storage, run)
    started = time.monotonic()
    # The coefficient here is exactly one: displacement already includes the
    # new scaled gradient, all carried remainder, and the actual BF16 rounding.
    # measure_model's fitted length/status are diagnostics, never an update gate.
    with recorder.context(response_kind='actual_bf16_displacement', solver_product_index=2):
        actual = model_trace_response.measure_model(graph, tuple(model.leaves), history,
            displacement, lambda forward: observation.residuals(reference, pairs, forward))
    record['timings']['actual_displacement_response_seconds'] = time.monotonic() - started
    record.update(status='ready', changed_values=changed, native_weights_unchanged=changed == 0,
        applied_length=length,
        linear_prediction_at_applied_length=response.objective + length * response.slope + length * length * response.curvature,
        new_gradient_prediction_scope='fresh scaled full105-loss joint negative gradient in all70 trainable matrices, before carried remainder and BF16 rounding',
        actual_bf16_response=asdict(actual),
        actual_bf16_prediction=actual.objective + actual.slope + actual.curvature,
        actual_bf16_response_coefficient=1,
        actual_bf16_response_scope='J_r times the entire installed BF16 displacement; fitted length and status are unused',
        actual_displacement_squared=math.fsum(displacement_norms),
        remainder_before_squared=math.fsum(remainder_before),
        remainder_after_squared=math.fsum(remainder_after), remainder_changed_values=remainder_changed,
        captured_origin_checks=len(origin_checks),
        acceptance='prescribed accumulated native update followed by fresh measurement, including zero changed weights; no response gate',
        value_is='local held-history directional predictions; not fresh native progress')
    record['native_transition'] = observation.prepared(actual, response, length)
    record['native_transition']['continuous_scope'] = ('First existing JVP along the fresh full105-loss joint negative weight gradient in all70 trainable matrices, evaluated at prescribed applied_length; excludes carried remainder and BF16 rounding. Held-history STE prediction, not a native result.')
    return record, weights, next_remainders


def retained_weights(candidate, previous, previous_binding, reference):
    """The last measured trace is provenance only; it never seeds this derivative."""
    if (candidate['pending_install'] is not None or candidate['weight_state'] != 'complete'
            or candidate['last_measurement'] != previous_binding
            or candidate['model_identity']['generation'] != candidate['update']
            or candidate['model_identity']['run'] != previous['model_identity']['run']
            or candidate['update'] < previous['generation']):
        raise ValueError('resume requires a complete, consistently identified installed checkpoint')
    if previous['reference'] != reference or not previous['complete'] or not previous['ended']:
        raise ValueError('the predecessor provenance must retain the original complete reference')
    return candidate['update']


def acquire(run, number, fd, model, reference, case, cases_binding, engine,
            inputs, reading, original, identity, previous, previous_loss, recorder,
            checkpoint_binding, parent_update, storage):
    """Acquire at current weights, then measure the complete new computation."""
    free = shutil.disk_usage(run).free
    required = (storage['free_floor_bytes'] + storage['capture_reserve_bytes']
                + storage['remainder_checkpoint_bytes'])
    if free < required:
        raise OSError(f'capture requires {required} free bytes including the existing reserve; available {free}')
    step = run / 'loop' / str(number)
    step.mkdir(parents=True, exist_ok=True)
    with recorder.context(training_step=number, model_generation=number), recorder.span('native.capture', detached=True,
            attributes={'input.training_member': inputs['member'], 'input.max_acquisition_steps': case['max_tokens']}) as capture_outcome:
        capture = factory.capture(fd, step / 'capture', identity(number),
                                  ROOT / cases_binding['record'], engine, inputs)
        capture_outcome.update({'output.model_generation': number, 'output.capture_returned': True})
    capture['immutability'] = 'size-sealed RAM model; sole writer waits for native child exit before any next write; only read-only descriptor inherited; stat continuity checked'
    (step / 'capture/capture.json').write_text(json.dumps(capture, indent=2) + '\n')
    raw = step / 'capture/raw' / inputs['member']
    member = dict(id=case['id'], tokens=case['tokens'], question=len(case['tokens']) - 1)
    answer = native_events.answer_of(member, json.loads((raw / 'result.json').read_text()), case['max_tokens'])
    factory.save(step / 'answer.json', answer)
    history = tuple(case['tokens'] + answer['tokens'][:-1])
    if len(history) != member['question'] + answer['steps']:
        raise ValueError('the new fresh computation has an inconsistent event inventory')
    with torch.no_grad(), recorder.context(training_step=number, model_generation=number), recorder.span('native.measure', detached=True,
            attributes={'input.events': len(history), 'input.merges': len(reading['merges'])}) as measure_outcome:
        native = held.raw_forward(raw, reading['merges'], len(history), model.width)
        pairs = held.turn_correction.correspondence(reference, native)
        measured = held.turn_correction.measurement(reference, native, pairs)
        mapped = evaluation.loss_map(reference, native, pairs, step / 'residual-squared.npy')
        measure_outcome.update(capture_source.scalars(measured, 'output.'))
        measure_outcome.update({'output.events': len(history), 'output.merges': len(reading['merges']),
            'output.pairs': len(pairs), 'output.complete': bool(answer['ended_by_itself'])})
    with (step / 'pairs.npy').open('xb') as file:
        np.save(file, pairs, allow_pickle=False)
    ended = answer['ended_by_itself']
    observation = dict(measured, step=number, generation=number, events=len(history), merges=105,
        complete=bool(ended), ended=ended, measured_at=now(), member=inputs['member'],
        model_identity=identity(number), model_sha256=None,
        reference=original['reference'], reference_sha256=original['reference']['sha256'],
        correspondence=held.binding(step / 'pairs.npy'), capture=held.binding(step / 'capture/capture.json'),
        acquisition=held.binding(step / 'answer.json'), residual_squared=mapped,
        map_aggregation_roundoff=mapped['sum'] - measured['objective'],
        previous_measurement=previous, resumed_checkpoint=checkpoint_binding,
        predecessor_counter=parent_update + number,
        measure='fresh native input-output turns; no output-token judge',
        scope='complete naturally ended fresh path' if ended else 'partial fresh path: original acquisition limit reached')
    if previous_loss is not None:
        observation['loss_change'] = measured['objective'] - previous_loss
    factory.save(step / 'measurement.json', observation)
    with recorder.context(training_step=number, model_generation=number, training_member=inputs['member']):
        recorder.metric('native.objective', measured['objective'], step=number)
    return native, pairs, history, measured, raw, held.binding(step / 'measurement.json'), ended


def execute(run, declaration, recorder, capture, completion):
    with recorder.span('run.bind', detached=True) as binding_outcome:
        if declaration['program'] != held.binding(Path(__file__).resolve()):
            raise ValueError('the continuation source differs from its declaration')
        if (declaration['projection_gradient_adapter'] != held.binding(Path(projection_adapter.__file__))
                or declaration['gradient_helper'] != held.binding(Path(trace_gradient.__file__))):
            raise ValueError('the projection precision adapter or full weight-gradient helper differs from its declaration')
        sources = {ROOT / row['record']: row for row in declaration['implementation']}
        for path in source_files():
            if path not in sources:
                raise ValueError(f'the actual helper is not source-bound: {path}')
        for source in sources.values():
            held.checked(ROOT, source, parse=False)
        pilot = held.checked(ROOT, declaration['start_experiment'])
        initial = held.checked(ROOT, declaration['start_fresh'])
        eta = declaration['eta']
        if not math.isfinite(eta) or eta <= 0 or eta != pilot['candidate_rule']['eta']:
            raise ValueError('the fixed empirical eta must be exactly the completed pilot value')
        if os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID' or os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
            raise ValueError('use the original physical GPU0')
        inputs, original, _, reading = held.load_inputs()
        candidate = held.checked(ROOT, pilot['candidate'])
        if (declaration['resume_checkpoint'] != pilot['candidate']
                or declaration['resume_predecessor_generation'] != candidate['update']):
            raise ValueError('the restart must use its exact declared installed checkpoint')
        parent_update = retained_weights(candidate, initial, declaration['start_fresh'], original['reference'])
        initial_capture = held.checked(ROOT, initial['capture'])
        answer = held.checked(ROOT, initial['acquisition'])
        if initial_capture['model_identity'] != initial['model_identity']:
            raise ValueError('the saved fresh capture belongs to another candidate')
        cases = held.checked(ROOT, initial_capture['cases'])
        if len(cases) != 1 or set(cases[0]) != {'id', 'tokens', 'max_tokens'} or cases[0]['id'] != inputs['member']:
            raise ValueError('reuse the exact original one-member fresh acquisition settings')
        case = cases[0]
        tensors, architecture = held.checked(ROOT, original['tensors']), held.checked(ROOT, original['architecture'])
        names = tuple(original['functional_loop']['matrices'])
        rows = accumulation.layout(tensors, names)
        if declaration['accumulation']['initialization'] != 'resume':
            raise ValueError('the continuation must retain every incoming FP32 remainder')
        if [row['name'] for row in candidate['inventory']] != list(names):
            raise ValueError('the current exit inventory omits matrices')
        budget = observation_source.storage_budget(rows, declaration['storage'],
            {row['name']: row['entries'] for row in candidate['inventory']})
        if budget != declaration['entry_storage']:
            raise ValueError('the exact current-state entry storage budget differs')
        observation_source.reserve(ROOT, run, run / 'entry-storage.json', budget)
        if declaration['storage']['remainder_checkpoint_bytes'] != sum(row['bytes'] for row in rows):
            raise ValueError('the full FP32 remainder checkpoint must have its exact additional disk reserve')
        remainders = accumulation.load(ROOT, pilot['candidate'], candidate,
            declaration['accumulation']['initialization'], rows)
        changes, _ = factory.load_changes(candidate, pilot['candidate'], tensors, names, candidate['update'], original)
        if len(names) != 70 or len(reading['merges']) != 105:
            raise ValueError('the complete seventy-matrix and105-merge inventory is required')
        module = peer.load_module('continued_torch_model', ROOT / original['forward']['record'])
        reference = held.turn_artifact.load(ROOT, original['reference'], original_model_sha256=original['model_sha256'], expected_merges=105)
        comparison = held.checked(ROOT, pilot['factory_declaration'])
        replay_report = held.checked(ROOT, comparison['same_card_report'])
        # The pilot producer used one CPU thread. Reporting reductions are not a
        # weight identity test; tensor/cell identities below remain exact.
        torch.set_num_threads(declaration['cpu_threads'])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        # Match the operand-precision qualification's FP32 accumulation policy.
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        engine = held.generation_engine.Generation()
        if engine.record['sha256'] != reading['generation_sha256'] or engine.record['engine_sha256'] != reading['libraries_sha256']:
            raise ValueError('the continuation must use the unchanged qualified native engine')
        binding_outcome.update({'output.editable_matrices': len(names), 'output.merges': len(reading['merges']),
            'output.eta': eta, 'output.predecessor_generation': parent_update})
    stop = []
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(signum, lambda number, frame: stop.append(number))
    generation, pending, reason, last_measurement = 0, None, 'stop_requested', None
    last_transition, pending_transition = None, None
    failed, checkpoint_error = False, None
    installation = dict(target_iteration=0, committed_matrices=[], in_progress=None)
    fd, seed_bits, model = None, None, None
    seed = candidate['base_seed']
    owner = (run / 'owner.lock').open('a')
    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    controller = Path(f'/run/user/{os.getuid()}/weight-patterns-controller.lock').open('a')
    fcntl.flock(controller, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def progress(phase, **details):
        row = dict(at=now(), phase=phase, generation=generation, **details)
        with (run / 'progress.jsonl').open('a') as file:
            file.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)

    def identity(number):
        return dict(kind=held.working_identity.KIND, run=str(run.relative_to(ROOT)),
                    generation=number, original_sha256=original['model_sha256'])

    try:
        fd = factory.memory_model(ROOT / seed['file'], seed['model_sha256'])
        seed_bits = native_bits(fd, tensors, names)
        pending = 0
        factory.apply_changes(fd, tensors, changes)
        pending = None
        with card(), recorder.span('model.load') as model_outcome:
            model = resident(module, fd, tensors, architecture, names)
            capture.outcome(model_outcome, 'model', model)
        with recorder.context(training_step=0, model_generation=0, training_member=inputs['member']), recorder.span('training.baseline', detached=True):
            native, pairs, history, measured, raw, last_measurement, ended = acquire(
                run, 0, fd, model, reference, case, initial_capture['cases'], engine,
                inputs, reading, original, identity, None, None, recorder,
                pilot['candidate'], parent_update, declaration['storage'])
        progress('baseline_acquired', **measured, ended=ended, predecessor_counter=parent_update)
        if not ended:
            reason = 'incomplete_recovery_acquisition'
            return
        while not stop:
            number = generation + 1
            with recorder.context(training_step=number, source_model_generation=generation, model_generation=generation,
                    target_model_generation=number, training_member=inputs['member'],
                    source_observation_record=last_measurement['record'], source_observation_sha256=last_measurement['sha256'],
                    model_identity_kind=identity(generation)['kind'], model_identity_run=identity(generation)['run']), recorder.span('training.step', detached=True):
                step = run / 'loop' / str(number)
                step.mkdir()
                held.frozen(inputs)
                source = held.checked(ROOT, last_measurement)
                if source['model_identity'] != identity(generation):
                    raise ValueError('sparse edit source is not the current measured native state')
                observation = observation_source.Observation(step / 'native-transition', ROOT, names,
                    last_measurement, identity(generation), identity(number), original['reference'],
                    source['correspondence'], pilot['candidate'], last_transition, len(reading['merges']))
                with card():
                    with recorder.span('proposal.prepare', attributes={'input.events': len(history), 'input.merges': len(reading['merges'])}):
                        calls = precision.calls_from_index(raw / 'stream.jsonl', events=len(history),
                            layers=model.layers, width=model.width)
                        wrapped = precision.Model(model, calls, rms=module.rms, gelu=module.gelu,
                            batch_single_column=declaration['ffn_precision_policy']['batch_single_column'])
                        attention_calls = projection_geometry.calls_from_index(raw / 'stream.jsonl',
                            events=len(history), layers=model.layers, width=model.width)
                        third_calls = projection_geometry.projection.calls_from_index(raw / 'stream.jsonl',
                            events=len(history), layers=model.layers, width=model.width)
                        wrapped = projection_adapter.Model(wrapped, attention_calls, third_calls)
                    with recorder.span('proposal.compute', detached=True, attributes={
                            'input.events': len(history), 'input.pairs': len(pairs), 'input.merges': len(reading['merges']),
                            'input.editable_matrices': len(names), 'input.objective': measured['objective'], 'input.eta': eta}) as proposal_outcome:
                        proposal, weights, next_remainders = propose(wrapped, history, native, reference, pairs, measured, eta, remainders, progress, observation, seed_bits, rows, declaration['storage'], run, recorder)
                        capture.outcome(proposal_outcome, 'proposal', (proposal, weights, next_remainders))
                    del wrapped
                    proposal['ffn_operand_precision'] = dict(
                        adapter=declaration['ffn_precision'],
                        geometry='Every actual native FFN call from the source observation; no text-position selection',
                        derivative='Explicit identity-through-operand-rounding surrogate at rounded operands; not the derivative of discrete BF16 rounding',
                        attention_and_third_merge='separate declared projection adapter; other arithmetic unchanged')
                    proposal['projection_operand_precision'] = dict(
                        adapter=declaration['projection_gradient_adapter'],
                        gradient_helper=declaration['gradient_helper'],
                        geometry='Every actual attention/third native call from the fresh source observation',
                        derivative='Explicit F32 input-and-weight straight-through derivatives at rounded operands; ordinary F32 singles',
                        weight_gradient='Full chain-rule derivative through every declared trainable matrix; no first-projection request or factor map')
                    factory.save(step / 'proposal.json', dict(proposal, step=number, at=now(), source_observation=last_measurement))
                    for key in ('full_request_objective', 'gradient_norm_squared', 'direction_matrices'):
                        recorder.metric('proposal.' + key, proposal[key], step=number)
                    if weights is None:
                        reason = proposal['status']
                        break
                    pending_transition = held.binding(step / 'proposal.json')
                    pending = number
                    installation = dict(target_iteration=number, committed_matrices=[], in_progress=None)
                    with recorder.span('weights.install', detached=True, attributes={'input.editable_matrices': len(names),
                            'input.changed_values': proposal['changed_values'], 'input.target_generation': number}) as install_outcome:
                        try:
                            install(fd, model, tensors, names, weights, remainders, next_remainders, rows, installation)
                        finally:
                            install_outcome.update({'output.committed_matrices': len(installation['committed_matrices']),
                                'output.in_progress': installation['in_progress'] or '',
                                'output.complete': installation['in_progress'] is None and len(installation['committed_matrices']) == len(names)})
                    generation, pending = number, None
                    recorder.annotate_context(model_generation=generation)
                    del weights, next_remainders
                    gc.collect()
                    torch.cuda.empty_cache()
                commit = observation.committed(pending_transition, installation, identity(generation))
                factory.save(step / 'reached.json', dict(native_transition=commit, model_identity=identity(generation), model_sha256=None,
                    generation=generation, matrices=list(names), installed_at=now(), storage='same anonymous RAM GGUF and resident leaves',
                    accumulator=dict(committed_matrices=installation['committed_matrices'],
                        remainder_squared=proposal['remainder_after_squared'],
                        remainder_changed_values=proposal['remainder_changed_values'])))
                last_transition = held.binding(step / 'reached.json')
                pending_transition = None
                del observation
                progress('native_weights_installed', changed_values=proposal['changed_values'])
                del native
                previous_loss = measured['objective']
                native, pairs, history, measured, raw, last_measurement, ended = acquire(
                    run, number, fd, model, reference, case, initial_capture['cases'], engine,
                    inputs, reading, original, identity, last_measurement, previous_loss, recorder,
                    pilot['candidate'], parent_update, declaration['storage'])
                progress('fresh_measurement_complete', **measured, ended=ended,
                         loss_change=measured['objective'] - previous_loss)
                if not ended:
                    reason = 'incomplete_fresh_acquisition'
                    break
                with recorder.span('storage.archive_consumed', detached=True, attributes={
                        'input.previous_generation': number - 1, 'input.current_generation': number}) as archive_outcome:
                    receipt = archive.archive_previous(run, number - 1, number, declaration, stop_requested=lambda: bool(stop))
                    capture.outcome(archive_outcome, 'archive', receipt)
                if receipt.get('archives'):
                    with (run / 'trace-archives' / str(number - 1) / 'physical-release.log').open('a') as output, redirect_stdout(output):
                        reclaim.free([item['record'] for item in receipt['archives']])
                progress('consumed_trace_archived', previous_generation=number - 1, archive_receipt=receipt)
                capture.coverage(run / 'observability/coverage.json')
        held.frozen(inputs)
    except InterruptedError as error:
        if not stop:
            failed = True
            reason = f'{type(error).__name__}: {error}'
            raise
        reason = 'stop_requested'
    except BaseException as error:
        failed = True
        reason = f'{type(error).__name__}: {error}'
        raise
    finally:
        if fd is not None:
            try:
                if seed_bits is not None:
                    with recorder.context(training_step=generation, model_generation=generation), recorder.span('state.checkpoint', detached=True,
                            attributes={'input.generation': generation, 'input.pending_install': pending if pending is not None else -1,
                                'input.committed_matrices': len(installation['committed_matrices'])}) as checkpoint_outcome:
                        saved, checkpoint_error = checkpoint_with_remainder(
                            run, fd, tensors, names, seed_bits, seed, generation, remainders, rows, installation,
                            start_candidate=pilot['candidate'], start_fresh=declaration['start_fresh'], eta=eta,
                            model_identity=identity(generation), pending_install=pending,
                            weight_state='complete' if pending is None else 'partial installation; not a completed update',
                            last_measurement=last_measurement, stop_signals=stop, reason=reason,
                            last_committed_transition=last_transition, pending_transition=pending_transition,
                            transition_state=('partial_install' if pending is not None else
                                'installed_commit_record_missing' if pending_transition is not None else 'complete'))
                        checkpoint_outcome.update({'output.record': saved['record'], 'output.sha256': saved['sha256'],
                            'output.remainder_save_error': checkpoint_error is not None})
                    if checkpoint_error is not None:
                        failed = True
                        reason = f'remainder checkpoint failed: {type(checkpoint_error).__name__}: {checkpoint_error}'
                    progress('exit_checkpoint_saved', checkpoint=saved, reason=reason, pending_install=pending)
            except BaseException as error:
                failed = True
                checkpoint_error = error
                reason = f'exit checkpoint failed: {type(error).__name__}: {error}'
            finally:
                os.close(fd)
        completion.update(status='FAILED' if failed else 'FINISHED', reason=reason, generation=generation)
        factory.save(run / 'launcher-exit.json', dict(exit_code=1 if failed else 0, ended_at=now(),
            status='failed' if failed else 'stopped', installed_updates=generation, reason=reason,
            final_state=held.binding(run / 'final-state.json') if (run / 'final-state.json').exists() else None))
        controller.close()
        owner.close()
        if checkpoint_error is not None:
            raise checkpoint_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    run = parser.parse_args().run.resolve()
    # Keep the original admission guard before any newly created event file.
    declaration = held.check_native_run(run, run / 'declaration.json')
    recorder = Recorder(run / 'observability/events.jsonl', dict(
        project='weight-patterns', folder=run.name, operation=declaration['operation'],
        entrypoint=declaration['program']['record'], run=str(run.relative_to(ROOT)),
        declaration_sha256=held.binding(run / 'declaration.json')['sha256'],
        original_model_sha256=declaration['original_model']['sha256'],
        reference_sha256=declaration['reference']['sha256'],
        resumed_checkpoint_sha256=declaration['resume_checkpoint']['sha256'],
        model_identity_kind=held.working_identity.KIND, model_identity_run=str(run.relative_to(ROOT)),
        capture_schema=capture_source.FORMAT))
    capture = capture_source.Capture(recorder)
    completion = dict(status='FAILED', reason='before execution')
    try:
        with recorder.span('research.invocation', detached=True) as outcome:
            with capture.instrument(globals()):
                execute(run, declaration, recorder, capture, completion)
            outcome.update(capture_source.scalars(completion, 'output.'))
    except BaseException as error:
        completion.update(status='FAILED', reason=type(error).__name__)
        raise
    finally:
        capture.coverage(run / 'observability/coverage.json')
        status = completion.pop('status')
        recorder.close(status, **completion)


if __name__ == '__main__':
    main()
