"""One-pair computational learning over every eligible learned parameter.

The declaration supplies the reviewed immutable tolerance and actual source
bindings. No fallback tolerance, extra answer target or perfect-step search lives
here. Native capture/bootstrap are reused from the preceding qualified runner.
"""
import argparse
from contextlib import contextmanager, redirect_stdout
from dataclasses import asdict
from datetime import datetime
from functools import partial
import fcntl
import gc
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
SELF = Path(__file__).resolve().parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


legacy = load_module('final_experiment_native_helpers',
    ROOT / 'records/development/2026-11-17-full-weight-training/run.py')
factory, held = legacy.factory, legacy.held
from pipeline.observability.recorder import Recorder
from pipeline.weight_edit import full_graph, full_parameters, full_state, full_loss_maps, turn_pair, vector_chance

continuation = load_module('final_experiment_continuation', SELF / 'continuation.py')
local_diagnostic = load_module('final_experiment_local_gradient_diagnostic',
    SELF / 'local_gradient_diagnostic.py')


def now():
    return datetime.now().astimezone().isoformat()


def save(path, value):
    path = Path(path)
    with path.open('x') as out:
        json.dump(value, out, indent=2, allow_nan=False)
        out.write('\n')


def binding(path):
    path = Path(path).resolve()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(record=str(path.relative_to(ROOT)), sha256=digest)


def source_files():
    modules = (full_graph, full_parameters, full_state, full_loss_maps, turn_pair, vector_chance)
    paths = set(legacy.source_files()) | {Path(__file__).resolve()}
    paths.update(Path(m.__file__).resolve() for m in modules)
    # The executed extraction reader, omitted from the predecessor's direct list.
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        # torch.ops/classes expose pseudo filenames such as '_ops.py'. They
        # are not project modules and must not become invented cwd-relative
        # bindings. Real loaded project modules have absolute __file__ paths;
        # a missing absolute project source still fails normal hash validation.
        if filename and Path(filename).is_absolute():
            path = Path(filename).resolve()
            if path.is_relative_to(ROOT) and path.suffix == '.py':
                paths.add(path)
    return sorted(paths)


def load_reference(declaration):
    """Bind the explicitly declared calibration around the fixed original trace."""
    original = held.checked(ROOT, declaration['reference'])
    if declaration.get('tolerance_policy') == 'retained-original-chance':
        if (original.get('format') != 'single-original-vector-chance/1'
                or declaration['tolerance'] != original['chance']):
            raise ValueError('retained tolerance must be the exact original chance binding')
        reference = held.turn_artifact.load(ROOT, declaration['reference'],
            original_model_sha256=declaration['model_sha256'], expected_merges=105)
        return reference, full_loss_maps.LossIdentity(original['format'],
            declaration['reference']['sha256'], original['chance']['sha256'],
            'retained-original-chance')
    tolerance = held.checked(ROOT, declaration['tolerance'])
    if (tolerance['format'] != 'final-experiment-tolerance/1'
            or tolerance['reference'] != declaration['reference']
            or tolerance['original_model_sha256'] != original['source_model_sha256']
            or tolerance.get('status') != 'reviewed'
            or not tolerance.get('derivation') or not tolerance.get('measured_from')):
        raise ValueError('the reviewed original-derived tolerance is not fully bound')
    held.checked(ROOT, original['trace'], parse=False)
    held.checked(ROOT, tolerance['chance'], parse=False)
    turns = np.load(ROOT / original['trace']['record'], mmap_mode='r', allow_pickle=False)
    chance = np.load(ROOT / tolerance['chance']['record'], allow_pickle=False)
    if (chance.shape != turns.shape[:2] or not np.isfinite(chance).all()
            or np.any(chance <= 0)):
        raise ValueError('the final experiment requires a supported positive movement allowance at every cell')
    reference = vector_chance.Reference(
        [torch.from_numpy(np.array(turns[:, m], copy=True)) for m in range(turns.shape[1])],
        [torch.from_numpy(chance[:, m].copy()) for m in range(chance.shape[1])],
        expected_merges=turns.shape[1])
    identity = full_loss_maps.LossIdentity(tolerance['loss_version'],
        declaration['reference']['sha256'], declaration['tolerance']['sha256'], tolerance['policy'])
    return reference, identity


def storage_budget(specs, state, context, inherited):
    """Reserve the exact maximum next-step state plus existing capture/free floor."""
    native = sum(row.nbytes for row in specs)
    remainder = 0
    for row in specs:
        if row.sparse_rows:
            prior = len(state.remainders.get(row.name, {})) if state else 0
            remainder += min(row.shape[0], prior + context) * row.shape[1] * 4
        else:
            remainder += row.size * 4
    result = dict(native_checkpoint_bytes=native, remainder_checkpoint_bytes=remainder,
        capture_reserve_bytes=inherited['capture_reserve_bytes'], free_floor_bytes=inherited['free_floor_bytes'])
    result['free_required_bytes'] = sum(result.values())
    return result


def reserve(run, budget):
    available = shutil.disk_usage(run).free
    if available < budget['free_required_bytes']:
        raise OSError(f"complete capture/checkpoint reserve needs {budget['free_required_bytes']} bytes; {available} available")
    return dict(budget, observed_free_bytes=available)


def residuals(reference, pairs, moments, selected_moments=None):
    """Record both existing directional calculations without changing a dual."""
    def read(forward):
        if moments:
            raise ValueError('each response table belongs to one directional calculation')
        for merge, observed in enumerate(reference.measure(forward, pairs)):
            if merge == 0 and selected_moments is not None:
                selected_moments.append(local_diagnostic.moments(
                    observed.residual, local_diagnostic.selection(pairs)))
            primal, tangent = torch.autograd.forward_ad.unpack_dual(observed.residual)
            r = primal.detach().double()
            q = None if tangent is None else tangent.detach().double()
            moments.append(dict(merge=merge, baseline=float(r.square().sum()),
                loss_cross=0. if q is None else 2 * float((r*q).sum()),
                response_norm_squared=0. if q is None else float(q.square().sum())))
            yield observed.residual
    return read


def sync_parameter(model, state, name, start, values):
    """Copy one physical span into the same persistent CPU derivative weights."""
    spec = state.specs[name]
    with torch.no_grad():
        if spec.sparse_rows:
            row, column = divmod(start, spec.shape[1])
            index = model.parameter_rows[name].index(row)
            target = model.weights[name][index, column:column+len(values)]
        else:
            target = model.weights[name].view(-1)[start:start+len(values)]
        target.copy_(torch.from_numpy(values))
        if not np.array_equal(target.numpy(), values):
            raise ValueError('derivative state differs from installed native values')


def preview(state, graph, direction, length):
    """Keep one CPU displacement, scalar receipts and chunk hashes, not endpoints."""
    displacement, updates, hashes = {}, {}, {}
    for name in graph.leaves:
        delta = torch.empty_like(graph.w[name], device='cpu')
        target = delta.numpy().reshape(-1)
        cursor, changed, norm, remainder_norm, cast_error = 0, 0, [], [], []
        entries = []
        for chunk in state.preview(name, direction[name].numpy(), length,
                                   rows=graph.parameter_rows.get(name)):
            count = len(chunk.stored)
            target[cursor:cursor+count] = chunk.displacement.astype(np.float32)
            cast_error.append(0. if chunk.exact_in_f32 else float(
                np.square(chunk.displacement-target[cursor:cursor+count].astype(np.float64)).sum()))
            changed += int(np.count_nonzero(chunk.displacement))
            norm.append(float(np.dot(chunk.displacement, chunk.displacement)))
            remainder_norm.append(float(np.square(chunk.remainder.astype(np.float64)).sum()))
            entries.append(dict(start=chunk.start, count=count, sha256=hashlib.sha256(chunk.raw).hexdigest()))
            cursor += count
        if cursor != target.size:
            raise ValueError('preview omitted part of a declared parameter')
        displacement[name] = delta
        updates[name] = dict(changed_entries=changed, displacement_squared=math.fsum(norm),
            remainder_after_squared=math.fsum(remainder_norm), jvp_displacement_cast_error_squared=math.fsum(cast_error))
        hashes[name] = entries
    return displacement, updates, hashes


def install(state, parameter_model, graph, direction, length, hashes, next_identity):
    selections = {name: graph.parameter_rows.get(name) for name in graph.leaves}
    state.begin(selections, expected_identity=state.model_identity, next_identity=next_identity,
        sync=lambda name, start, values: sync_parameter(parameter_model, state, name, start, values))
    for name in graph.leaves:
        seen = 0
        for chunk, expected in zip(state.preview(name, direction[name].numpy(), length,
                rows=selections[name]), hashes[name], strict=True):
            if (chunk.start != expected['start'] or len(chunk.stored) != expected['count']
                    or hashlib.sha256(chunk.raw).hexdigest() != expected['sha256']):
                raise ValueError('installed candidate differs from the displacement prediction')
            state.install(chunk)
            seen += 1
        if seen != len(hashes[name]):
            raise ValueError('installation omitted a previewed span')
    return state.finish()


def propose(graph, history, reference, pairs, measured, state, eta, recorder):
    checked = []
    forward = graph.forward
    def anchored(*args, **kwargs):
        value = forward(*args, **kwargs)
        for field, expected in [('incoming', graph.native_in), ('merged', graph.native_out)]:
            if len(value[field]) != reference.merges:
                raise ValueError('origin lost a merge')
            for actual, target in zip(value[field], expected):
                actual = torch.autograd.forward_ad.unpack_dual(actual).primal.detach()
                if not torch.equal(actual, target):
                    raise ValueError('differentiated origin differs from native capture')
        checked.append(True)
        return value
    graph.forward = anchored
    with recorder.span('proposal.weight_derivative', detached=True):
        gradient = full_graph.weight_gradient(graph, history, reference, pairs=pairs)
    if set(gradient.direction) != set(state.specs):
        raise ValueError('gradient omits an eligible physical learned tensor')
    if any(row['gradient_was_none'] for row in gradient.diagnostics.values()):
        raise ValueError('an eligible tensor is disconnected/detached; do not install a silently zeroed direction')
    if any(gradient.before[k] != measured[k] for k in ('outside', 'pair_merge_cells')):
        raise ValueError('gradient and fresh measurement have different residual inventories')
    result = dict(before=gradient.before, gradient_norm_squared=gradient.norm_squared,
        gradient_scalar_reduction_difference=gradient.before['objective']-measured['objective'],
        eligible_tensors=len(graph.leaves), request_merges=reference.merges, eta=eta,
        gradient_diagnostics=gradient.diagnostics, gradient_seconds=gradient.seconds)
    with recorder.span('proposal.local_derivative_diagnostic', detached=True) as outcome:
        try:
            diagnostic = local_diagnostic.attribute(
                graph, history, reference, pairs, gradient.direction)
        except Exception as error:
            diagnostic = dict(status='diagnostic_error', error=repr(error))
        result['local_gradient_diagnostic'] = diagnostic
        outcome.update({key: value for key, value in diagnostic.get('totals', {}).items()
                        if value is not None})
        outcome['diagnostic_status'] = diagnostic['status']
    if gradient.norm_squared == 0:
        return dict(result, status='zero_weight_gradient_not_full_loss_closure'), None
    continuous, installed, selected_continuous, selected_installed = [], [], [], []
    with recorder.context(response_kind='continuous_direction', solver_product_index=1), recorder.span('solver.jvp', detached=True):
        response = full_graph.directional_response(graph, history, gradient.direction,
            residuals(reference, pairs, continuous, selected_continuous))
    result.update(continuous_response=asdict(response), continuous_moments=continuous,
                  selected_cell_continuous_moments=selected_continuous)
    if response.status != 'proposal' or response.length is None:
        return dict(result, status=response.status), None
    length = eta * response.length
    with recorder.span('proposal.native_rounding', detached=True):
        displacement, updates, hashes = preview(state, graph, gradient.direction, length)
    with recorder.context(response_kind='actual_native_displacement', solver_product_index=2), recorder.span('solver.jvp', detached=True):
        actual = full_graph.directional_response(graph, history, displacement,
            residuals(reference, pairs, installed, selected_installed))
    del displacement
    result.update(status='ready', applied_length=length,
        continuous_prediction=response.objective+length*response.slope+length*length*response.curvature,
        installed_displacement_response=asdict(actual), installed_moments=installed,
        selected_cell_installed_moments=selected_installed,
        installed_displacement_prediction=actual.objective+actual.slope+actual.curvature,
        installed_displacement_coefficient=1, update_inventory=updates,
        candidate_chunk_hashes=hashes, captured_origin_checks=len(checked),
        changed_values=sum(row['changed_entries'] for row in updates.values()),
        scope='All eligible continuous learned-parameter derivatives; held-history STE predictions, fresh native measurement judges progress')
    return result, gradient.direction


def acquire(run, number, fd, reference, loss_identity, case, cases_binding,
            engine, inputs, reading, model_identity, previous, recorder, budget):
    reserve(run, budget)
    step = run / 'loop' / str(number)
    step.mkdir(parents=True, exist_ok=True)
    with recorder.context(training_step=number, model_generation=number):
        with recorder.span('native.capture', detached=True):
            capture = factory.capture(fd, step/'capture', model_identity,
                ROOT/cases_binding['record'], engine, inputs)
        # The native producer already wrote capture.json. Retain that exact
        # receipt instead of attempting to create the same file a second time.
        raw = step/'capture/raw'/inputs['member']
        member = dict(id=case['id'], tokens=case['tokens'], question=len(case['tokens'])-1)
        answer = legacy.native_events.answer_of(member, json.loads((raw/'result.json').read_text()), case['max_tokens'])
        save(step/'answer.json', answer)
        history = tuple(case['tokens']+answer['tokens'][:-1])
        if len(history) != member['question']+answer['steps']:
            raise ValueError('inconsistent native acquisition event inventory')
        with torch.no_grad(), recorder.span('native.measure', detached=True) as outcome:
            native = held.raw_forward(raw, reading['merges'], len(history), reference.width)
            pairs = held.turn_correction.correspondence(reference, native)
            measured = held.turn_correction.measurement(reference, native, pairs)
            current_turns = turn_pair.turns(native, expected_merges=reference.merges)
            maps = full_loss_maps.build_loss_maps(reference._reference, current_turns,
                [torch.sqrt(2*c) for c in reference._chance], pairs,
                expected_merges=reference.merges,
                merge_labels=[dict(block=m//3, branch=('attention','feed_forward','per_layer_input')[m%3]) for m in range(reference.merges)],
                identity=loss_identity, observation=dict(kind='fresh_native', model_revision=model_identity,
                    complete=bool(answer['ended_by_itself']),
                    acquisition_id=binding(step/'answer.json')['sha256'],
                    history_id=hashlib.sha256(np.asarray(history,dtype='<i8').tobytes()).hexdigest()),
                declared_objective=measured['objective'])
            outcome.update(full_loss_maps.recorder_attributes(maps))
        with (step/'pairs.npy').open('xb') as stream:
            np.save(stream, pairs, allow_pickle=False)
        complete = bool(answer['ended_by_itself'])
        record = dict(measured, step=number, generation=number, events=len(history), merges=reference.merges,
            complete=complete, ended=complete, measured_at=now(), member=inputs['member'],
            model_identity=model_identity, reference_sha256=loss_identity.reference_sha256,
            loss_identity=asdict(loss_identity), correspondence=binding(step/'pairs.npy'),
            capture=binding(step/'capture/capture.json'), acquisition=binding(step/'answer.json'),
            previous_measurement=previous, measure='fresh native input-output turns; no output-token judge')
        save(step/'measurement.json', record)
        recorder.metric('native.objective', measured['objective'], step=number)
    return native, pairs, history, measured, raw, binding(step/'measurement.json'), complete, maps


def state_specs(tensors, architecture):
    """Inventory starts at every stored tensor, not the old seventy-name list."""
    inventory = full_parameters.inventory(tensors, architecture)
    specs = tuple(full_state.TensorSpec.from_binding(tensors[row.name],
        sparse_rows=row.name in full_parameters.TABLES, aliases=row.aliases)
        for row in inventory if row.eligible)
    return inventory, specs


def load_inputs(declaration):
    """Read immutable acquisition metadata under THIS program's source bindings.

    The historical helper also re-admits its old executable sources. Those hashes
    belong to the historical run; a new source version is bound prospectively in
    this declaration and must not rewrite or pretend to execute the older one.
    """
    inputs = held.checked(ROOT, declaration['native_inputs'])
    original = held.checked(ROOT, inputs['source_declaration'])
    reading = held.checked(ROOT, declaration['native_reading'])
    if (original['model_sha256'] != declaration['original_model']['sha256']
            or original['reference'] != declaration['reference']
            or reading['generation_sha256'] != reading['engine']['sha256']):
        raise ValueError('original reference/model or qualified acquisition engine differs')
    held.frozen(inputs)
    return inputs, original, reading


def graph_for(model, state, history, native, raw, declaration):
    rows = {name: state.required_rows(name, sorted(set(history)))
            for name in full_parameters.TABLES}
    groups = full_parameters.shared_calls_from_index(raw/'stream.jsonl',
        events=len(history), width=model.width)
    parameters = full_parameters.ParameterModel(model, history, device='cuda:0',
        row_ids=rows, shared_linear=partial(full_parameters.cpu_shared_linear, groups=groups))
    calls = legacy.precision.calls_from_index(raw/'stream.jsonl', events=len(history),
        layers=model.layers, width=model.width)
    wrapped = legacy.precision.Model(parameters, calls,
        **forward_functions(model),
        batch_single_column=declaration['ffn_precision_policy']['batch_single_column'])
    attention = legacy.projection_geometry.calls_from_index(raw/'stream.jsonl',
        events=len(history), layers=model.layers, width=model.width)
    third = legacy.projection_geometry.projection.calls_from_index(raw/'stream.jsonl',
        events=len(history), layers=model.layers, width=model.width)
    wrapped = legacy.projection_adapter.Model(wrapped, attention, third)
    return full_graph.Graph(wrapped, history, native)


def forward_functions(model):
    """Use numerical helpers from the original functions under instrumentation."""
    return dict(rms=inspect.unwrap(type(model).per_layer_input).__globals__['rms'],
        gelu=inspect.unwrap(type(model).feed_forward).__globals__['gelu'])


def parameter_report(specs, diagnostics=None, updates=None):
    gradients = {}
    for name, row in (diagnostics or {}).items():
        gradients[name] = dict(row, status=('disconnected' if row['gradient_was_none'] else 'connected'))
    return full_loss_maps.gradient_update_inventory(
        [dict(name=s.name, dtype=s.native_type, shape=list(s.shape)) for s in specs],
        gradients=gradients, updates=updates)


def publish_maps(run, number, maps, recorder, *, reference, native, inventory, comparison=None, changes=None):
    folder = run/'loop'/str(number)/'diagnostics'
    with recorder.span('computation.loss_maps', detached=True) as outcome:
        full_loss_maps.write_local(maps, folder, inventory=inventory,
            comparison=comparison, changes=changes)
        full_loss_maps.render_loss_views(maps, folder/'computation-overlap.png')
        # One explicitly labeled illustrative vector, beside the COMPLETE maps.
        # This choice affects only the plot, never loss, gradients or update scope.
        pair_index, merge = np.unravel_index(np.argmax(maps.arrays['weighted_loss']),
            maps.arrays['weighted_loss'].shape)
        full_loss_maps.render_cell_overlay(maps, reference._reference,
            turn_pair.turns(native, expected_merges=reference.merges),
            pair_index=int(pair_index), merge=int(merge), path=folder/'local-turn-overlay.png')
        save(folder/'overlay-selection.json', dict(pair_index=int(pair_index), merge=int(merge),
            rule='largest weighted local loss for illustration; full maps and gradient retain every cell'))
        # The observer publishes these existing local files; there is no network
        # client, backend query or remote training dependency here.
        outcome.update({'output.metadata_record': str((folder/'loss-maps.json').relative_to(ROOT)),
            'output.overlap_plot_record': str((folder/'computation-overlap.png').relative_to(ROOT)),
            'output.local_turn_overlay_record': str((folder/'local-turn-overlay.png').relative_to(ROOT))})


def execute(run, declaration, recorder, capture, completion):
    with recorder.span('run.bind', detached=True):
        if declaration['program'] != binding(Path(__file__)):
            raise ValueError('controller differs from its declared source')
        sources = {ROOT/row['record']: row for row in declaration['implementation']}
        for path in source_files():
            if path not in sources:
                raise ValueError(f'actual executed helper is not source-bound: {path}')
        for row in sources.values():
            held.checked(ROOT, row, parse=False)
        held.checked(ROOT, declaration['whole_build_review'], parse=False)
        held.checked(ROOT, declaration['plan'], parse=False)
        if os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID' or os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
            raise ValueError('the declared native/derivative device is physical GPU0')
        inputs, original, reading = load_inputs(declaration)
        candidate = held.checked(ROOT, declaration['resume_checkpoint'])
        initial = held.checked(ROOT, declaration['start_fresh'])
        continuation.validate(candidate, initial, declaration['start_fresh'], declaration['loss_identity'])
        eta = declaration['eta']
        if not math.isfinite(eta) or eta <= 0 or eta != candidate['eta']:
            raise ValueError('use the explicitly retained empirical eta')
        initial_capture = held.checked(ROOT, initial['capture'])
        cases_binding = initial_capture['cases']
        cases = held.checked(ROOT, cases_binding)
        if len(cases) != 1 or set(cases[0]) != {'id','tokens','max_tokens'} or cases[0]['id'] != inputs['member']:
            raise ValueError('only the selected wrong response is regenerated')
        tensors = held.checked(ROOT, original['tensors'])
        architecture = held.checked(ROOT, original['architecture'])
        inventory, specs = state_specs(tensors, architecture)
        if json.loads(json.dumps([asdict(row) for row in inventory])) != declaration['parameter_inventory']:
            raise ValueError('eligible and excluded tensor inventory differs from the declaration')
        reference, loss_identity = load_reference(declaration)
        if reference.merges != len(reading['merges']):
            raise ValueError('all native merges must enter the reference')
        module = load_module('final_experiment_cpu_model', ROOT/original['forward']['record'])
        torch.set_num_threads(declaration['cpu_threads'])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        engine = held.generation_engine.Generation()
        if engine.record['sha256'] != reading['generation_sha256'] or engine.record['engine_sha256'] != reading['libraries_sha256']:
            raise ValueError('native generation engine changed')
    stop = []
    old_signals = {n: signal.signal(n, lambda number, frame: stop.append(number))
                   for n in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1)}
    generation, fd, state, last_measurement = 0, None, None, None
    reason, failed, checkpoint_error = 'stop_requested', False, None
    owner = (run/'owner.lock').open('a')
    controller = Path(f'/run/user/{os.getuid()}/weight-patterns-controller.lock').open('a')
    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    fcntl.flock(controller, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def identity(number):
        return dict(kind=held.working_identity.KIND, run=str(run.relative_to(ROOT)),
            generation=number, original_sha256=original['model_sha256'])

    def progress(phase, **details):
        row = dict(at=now(), phase=phase, generation=generation, **details)
        with (run/'progress.jsonl').open('a') as out:
            out.write(json.dumps(row, allow_nan=False)+'\n')
        print(json.dumps(row, allow_nan=False), flush=True)

    try:
        seed = continuation.seed(candidate)
        retained = continuation.remainder_inventory(ROOT, candidate, held.checked)
        reserve(run, storage_budget(specs, retained, declaration['native_context'], declaration['storage']))
        with recorder.span('state.restore', detached=True):
            fd = factory.memory_model(ROOT/seed['file'], seed['model_sha256'])
            state = continuation.restore(ROOT, fd, candidate, declaration['resume_checkpoint'], specs,
                chunk_elements=max(s.size for s in specs if not s.sparse_rows),
                journal=run/'working-state.json', checked=held.checked)
            state.rebind_complete_identity(identity(0), provenance=declaration['resume_checkpoint'])
        with recorder.span('model.load', detached=True):
            # Construct AFTER native restore. Dense tensors subsequently share the
            # ParameterModel storage, and table rows always read this same fd.
            model = module.Model(f'/proc/{os.getpid()}/fd/{fd}', tensors, architecture,
                device='cpu', leaves=lambda name: False, cache='float16')
        budget = storage_budget(specs, state, declaration['native_context'], declaration['storage'])
        with recorder.span('training.baseline', detached=True):
            native, pairs, history, measured, raw, last_measurement, ended, maps = acquire(
                run, 0, fd, reference, loss_identity, cases[0], cases_binding, engine, inputs, reading,
                identity(0), None, recorder, budget)
            publish_maps(run, 0, maps, recorder, reference=reference, native=native, inventory=parameter_report(specs))
        save(run/'inventory.json', dict(stored=[asdict(r) for r in inventory],
            eligible=[asdict(s) for s in specs], memory=full_parameters.memory_plan(
                full_parameters.inventory(tensors, architecture, history),
                events=len(history), width=model.width, layers=model.layers)))
        progress('baseline_acquired', **measured, ended=ended,
            loss_version=loss_identity.loss_version, resumed_checkpoint=declaration['resume_checkpoint'])
        if not ended:
            reason = 'incomplete_baseline_acquisition'
            return
        while not stop:
            number = generation+1
            with recorder.context(training_step=number, source_model_generation=generation,
                    model_generation=generation, target_model_generation=number,
                    training_member=inputs['member'], source_observation_record=last_measurement['record'],
                    source_observation_sha256=last_measurement['sha256']), recorder.span('training.step', detached=True):
                step = run/'loop'/str(number)
                step.mkdir()
                held.frozen(inputs)
                source = held.checked(ROOT, last_measurement)
                if source['model_identity'] != state.model_identity or state.model_identity != identity(generation):
                    raise ValueError('gradient source is not the currently installed and measured state')
                with legacy.card():
                    with recorder.span('proposal.prepare', detached=True):
                        graph = graph_for(model, state, history, native, raw, declaration)
                    with recorder.span('proposal.compute', detached=True):
                        proposal, direction = propose(graph, history, reference, pairs, measured, state, eta, recorder)
                    save(step/'proposal.json', dict(proposal, at=now(),
                        source_observation=last_measurement, source_model_identity=identity(generation),
                        target_model_identity=identity(number), loss_identity=asdict(loss_identity)))
                    recorder.metric('proposal.gradient_norm_squared', proposal['gradient_norm_squared'], step=number)
                    if direction is None:
                        reason = proposal['status']
                        progress('proposal_not_installed', reason=reason)
                        break
                    with recorder.span('weights.install', detached=True):
                        receipt = install(state, graph.parameter_model, graph, direction,
                            proposal['applied_length'], proposal['candidate_chunk_hashes'], identity(number))
                    generation = number
                    save(step/'reached.json', dict(receipt, at=now(), source_observation=last_measurement))
                    update_inventory = parameter_report(specs, proposal['gradient_diagnostics'],
                        receipt['transition']['committed'])
                    del direction, graph
                    gc.collect()
                    torch.cuda.empty_cache()
                progress('native_weights_installed', changed_values=proposal['changed_values'],
                    eligible_tensors=len(specs))
                previous, old_maps, previous_loss = last_measurement, maps, measured['objective']
                del native
                budget = storage_budget(specs, state, declaration['native_context'], declaration['storage'])
                native, pairs, history, measured, raw, last_measurement, ended, maps = acquire(
                    run, number, fd, reference, loss_identity, cases[0], cases_binding, engine, inputs, reading,
                    identity(number), previous, recorder, budget)
                comparison = full_loss_maps.response_comparison(old_maps, maps,
                    continuous=proposal['continuous_moments'], installed=proposal['installed_moments'],
                    applied_length=proposal['applied_length'])
                publish_maps(run, number, maps, recorder, reference=reference, native=native,
                    inventory=update_inventory, comparison=comparison,
                    changes=full_loss_maps.correspondence_changes(old_maps, maps))
                progress('fresh_measurement_complete', **measured, ended=ended,
                    loss_change=measured['objective']-previous_loss)
                del old_maps
                if not ended:
                    reason = 'incomplete_fresh_acquisition'
                    break
                with recorder.span('storage.archive_consumed', detached=True):
                    receipt = archive_source.archive_previous(run, number-1, number, declaration,
                        stop_requested=lambda: bool(stop))
                if receipt.get('archives'):
                    with (run/'trace-archives'/str(number-1)/'physical-release.log').open('a') as output, redirect_stdout(output):
                        legacy.reclaim.free([item['record'] for item in receipt['archives']])
                progress('consumed_trace_archived', previous_generation=number-1, archive_receipt=receipt)
                capture.coverage(run/'observability/coverage.json')
    except BaseException as error:
        failed, reason = True, f'{type(error).__name__}: {error}'
        raise
    finally:
        if state is not None:
            try:
                with recorder.span('state.checkpoint', detached=True):
                    state.checkpoint(run/'final-checkpoint')
                    save(run/'final-state.json', dict(state.receipt(), at=now(), reason=reason,
                        checkpoint=binding(run/'final-checkpoint/state.json'),
                        last_measurement=last_measurement, loss_identity=asdict(loss_identity),
                        eta=eta, stop_signals=stop))
                progress('exit_checkpoint_saved', status=state.status,
                    checkpoint=binding(run/'final-state.json'), reason=reason)
            except BaseException as error:
                checkpoint_error, failed = error, True
                reason = f'checkpoint failed: {type(error).__name__}: {error}'
        if fd is not None:
            os.close(fd)
        owner.close()
        controller.close()
        for number, handler in old_signals.items():
            signal.signal(number, handler)
        completion.update(status='FAILED' if failed else 'FINISHED', reason=reason, generation=generation)
        save(run/'launcher-exit.json', dict(at=now(), exit_code=int(failed), reason=reason,
            installed_updates=generation, state_status=None if state is None else state.status,
            last_measurement=last_measurement))
        if checkpoint_error is not None:
            raise checkpoint_error


def main():
    global archive_source
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    run = parser.parse_args().run.resolve()
    declaration = held.check_native_run(run, run/'declaration.json')
    capture_source = load_module('final_experiment_capture', SELF/'capture.py')
    archive_source = load_module('final_experiment_archive', SELF/'archive.py')
    recorder = Recorder(run/'observability/events.jsonl', dict(project='weight-patterns',
        folder=run.name, run=str(run.relative_to(ROOT)), operation=declaration['operation'],
        entrypoint=declaration['program']['record'], declaration_sha256=binding(run/'declaration.json')['sha256'],
        original_model_sha256=declaration['original_model']['sha256'],
        reference_sha256=declaration['reference']['sha256'], tolerance_sha256=declaration['tolerance']['sha256'],
        resumed_checkpoint_sha256=declaration['resume_checkpoint']['sha256'],
        model_identity_kind=held.working_identity.KIND, model_identity_run=str(run.relative_to(ROOT)),
        capture_schema=capture_source.FORMAT))
    capture = capture_source.Capture(recorder)
    completion = dict(status='FAILED', reason='before execution')
    try:
        with recorder.span('research.invocation', detached=True) as outcome:
            with capture.instrument(globals()):
                execute(run, declaration, recorder, capture, completion)
            outcome.update(legacy.capture_source.scalars(completion, 'output.'))
    finally:
        capture.coverage(run/'observability/coverage.json')
        recorder.close(completion.pop('status'), **completion)


if __name__ == '__main__':
    main()
