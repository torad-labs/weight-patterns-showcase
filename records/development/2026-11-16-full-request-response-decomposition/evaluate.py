"""Read-only global/local/native response decomposition of committed236-to237.

No derivative, JVP, candidate choice, native acquisition or optimizer state change.
"""
import argparse
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
FINITE = ROOT/'records/development/2026-11-16-full-request-finite-endpoint/evaluate.py'
spec = importlib.util.spec_from_file_location('decomposition_finite_endpoint', FINITE)
finite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(finite)
helper, Recorder = finite.helper, finite.Recorder


def endpoint_raw(dec, baseline, endpoint, reading):
    """Require the actual restored endpoint, with the same recorded call geometry."""
    history, capture = finite.history(endpoint)
    _, baseline_capture = finite.history(baseline)
    if capture['raw'] != dec['endpoint_raw']:
        raise ValueError('restored endpoint raw binding differs from its acquisition')
    if (capture['raw']['stream.jsonl']['sha256'] != baseline_capture['raw']['stream.jsonl']['sha256']
            or capture['engine']['sha256'] != reading['generation_sha256']):
        raise ValueError('endpoint native engine or complete call geometry differs')
    paths = {name: helper.checked(row) for name, row in dec['endpoint_raw'].items()}
    if set(paths) != {'stream.f32', 'stream.jsonl'} or paths['stream.f32'].parent != paths['stream.jsonl'].parent:
        raise ValueError('restored native arrays and index must be one complete capture')
    return paths['stream.f32'].parent, history


def finite_paths(graph, history, native_endpoint, weights):
    """Freeze236; compare global propagation with native-input local operations.

    The second Graph is used ONLY for its .computed local operations. Its native
    anchors are never used as a zero-change endpoint prediction. Its initializer
    reconstructs shared cache/auxiliaries from each native producer's own input.
    """
    original = graph.w
    try:
        with torch.no_grad():
            baseline = graph.forward(history, again=False)
            helper.assert_anchor(graph, baseline)
            graph.w = weights
            global_endpoint = graph.forward(history, again=False)
            local_graph = helper.native_graph.Graph(graph.model, history, native_endpoint)
            local_endpoint = dict(incoming=local_graph.native_in, merged=tuple(
                origin + (current - frozen)
                for origin, current, frozen in zip(graph.native_out, local_graph.computed, graph.computed)))
            # Native1 shares the validated device copies, without another model.
            native = dict(incoming=local_graph.native_in, merged=local_graph.native_out)
            return dict(baseline=baseline, global_endpoint=global_endpoint,
                        local_endpoint=local_endpoint, native_endpoint=native)
    finally:
        graph.w = original


SUM_KEYS = ('baseline_squared', 'native_squared', 'global_squared', 'local_squared',
            'global_dot_native', 'local_dot_native', 'global_dot_local',
            'total_error_squared', 'propagation_error_squared', 'local_error_squared',
            'twice_error_cross', 'vector_identity_error_squared')


def derived_moments(row):
    """Null angles for zero movement; retain signed cross terms and roundoff."""
    for label in ('native', 'global', 'local'):
        row[label+'_norm'] = math.sqrt(row[label+'_squared'])
    for a, b in (('global', 'native'), ('local', 'native'), ('global', 'local')):
        denominator = row[a+'_norm'] * row[b+'_norm']
        row[a+'_'+b+'_cosine'] = row[a+'_dot_'+b]/denominator if denominator else None
    row['error_energy_sum_with_cross'] = math.fsum((row['propagation_error_squared'],
        row['local_error_squared'], row['twice_error_cross']))
    row['error_energy_identity_difference'] = row['total_error_squared']-row['error_energy_sum_with_cross']
    return row


def moments(baseline, global_endpoint, local_endpoint, native_endpoint):
    """Coordinate-space errors; global-native = (global-local)+(local-native)."""
    values = [x.detach().to(device='cpu', dtype=torch.float64)
              for x in (baseline, global_endpoint, local_endpoint, native_endpoint)]
    if any(x.shape != values[0].shape or not bool(torch.isfinite(x).all()) for x in values):
        raise ValueError('response vectors must have equal complete finite shapes')
    base, glob, local, native = values
    g, l, n = glob-base, local-base, native-base
    total, propagated, local_error = g-n, g-l, l-n
    identity = total-(propagated+local_error)
    squared = lambda x: float(x.square().sum())
    dot = lambda a, b: float((a*b).sum())
    row = dict(coordinates=base.numel(), baseline_squared=squared(base),
        native_squared=squared(n), global_squared=squared(g), local_squared=squared(l),
        global_dot_native=dot(g, n), local_dot_native=dot(l, n), global_dot_local=dot(g, l),
        total_error_squared=squared(total), propagation_error_squared=squared(propagated),
        local_error_squared=squared(local_error), twice_error_cross=2*dot(propagated, local_error),
        vector_identity_error_squared=squared(identity),
        vector_identity_max_abs=float(identity.abs().max()))
    return derived_moments(row)


def aggregate(rows):
    row = {key: math.fsum(part[key] for part in rows) for key in SUM_KEYS}
    row['coordinates'] = sum(part['coordinates'] for part in rows)
    row['vector_identity_max_abs'] = max(part['vector_identity_max_abs'] for part in rows)
    return derived_moments(row)


def compare_paths(paths, reference, pairs):
    """Use the actual full Reference.measure; never locally renormalize a merge."""
    labels = ('baseline', 'global_endpoint', 'local_endpoint', 'native_endpoint')
    if set(paths) != set(labels):
        raise ValueError('all four paths are required')
    for path in paths.values():
        if any(len(path[key]) != reference.merges for key in ('incoming', 'merged')):
            raise ValueError('a response path omitted a declared merge')
    maps = {label: np.zeros((reference.rows, reference.merges), dtype='<f8') for label in labels}
    generators = [reference.measure(paths[label], pairs) for label in labels]
    rows = []
    with torch.no_grad():
        for i in range(reference.merges):
            parts = [next(iterator).residual for iterator in generators]
            losses = {}
            for label, residual in zip(labels, parts):
                squared = residual.double().square()
                np.add.at(maps[label][:, i], pairs[:, 0], squared.sum(dim=1).cpu().numpy())
                losses[label] = float(squared.sum())
            row = dict(merge=i, losses=losses,
                incoming=moments(*(paths[label]['incoming'][i] for label in labels)),
                outgoing=moments(*(paths[label]['merged'][i] for label in labels)),
                weighted_residual=moments(*parts))
            rows.append(row)
        if any(next(iterator, None) is not None for iterator in generators):
            raise ValueError('reference supplied undeclared merge terms')
    return dict(per_merge=rows,
        objectives={label: math.fsum(row['losses'][label] for row in rows) for label in labels},
        response={space: aggregate([row[space] for row in rows])
                  for space in ('incoming', 'outgoing', 'weighted_residual')}), maps


def saved_map_check(maps, baseline, endpoint, prior):
    checks = {}
    bindings = dict(baseline=baseline['residual_squared'], native_endpoint=endpoint['residual_squared'],
                    global_endpoint=prior['endpoint']['residual_map'])
    for label, binding in bindings.items():
        saved = np.load(helper.checked(binding), allow_pickle=False)
        if saved.dtype != np.dtype('<f8') or saved.shape != maps[label].shape:
            raise ValueError('saved loss map has different shape or representation')
        checks[label] = dict(binding=binding, exact=bool(np.array_equal(saved, maps[label])),
                            max_abs_error=float(np.max(np.abs(saved-maps[label]))))
        if label != 'global_endpoint' and not checks[label]['exact']:
            raise ValueError('restored native loss map differs from the recorded observation: '+label)
    return checks


def worker(run, dec, lock_fd):
    lock = Path(f'/run/user/{os.getuid()}/gpu1.lock')
    if os.fstat(lock_fd).st_ino != lock.stat().st_ino or os.environ.get('CUDA_VISIBLE_DEVICES') != dec['gpu_uuid']:
        raise ValueError('worker lacks the declared GPU1 reservation')
    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source, precision = helper.configure(dec)
    names = tuple(source['functional_loop']['matrices'])
    baseline, endpoint, proposal, transition = finite.evidence(dec, names)
    prior = helper.record(dec['finite_result'])
    if (prior['status'] != 'complete' or prior['program'] != dec['finite_evaluator']
            or any(prior[key] != dec[key] for key in ('checkpoint', 'baseline_measurement',
                'endpoint_measurement', 'endpoint_proposal', 'endpoint_reached'))
            or prior['payload'] != transition['payload']):
        raise ValueError('prior finite comparison belongs to another exact transition')
    rec = Recorder(run/'observability/events.jsonl', dict(project='weight-patterns', folder=run.name,
        operation=dec['operation'], run=str(run.relative_to(ROOT)), entrypoint=dec['program']['record'],
        declaration_sha256=helper.sha(run/'declaration.json'),
        parent_run=str(Path(dec['endpoint_proposal']['record']).parents[2]),
        original_model_sha256=source['model_sha256'], reference_sha256=source['reference']['sha256'],
        checkpoint_sha256=dec['checkpoint']['sha256'], endpoint_payload_sha256=transition['payload']['sha256'],
        physical_gpu=1, gpu_uuid=dec['gpu_uuid']))
    status = 'FAILED'
    try:
        with rec.span('diagnostic.prepare'):
            reading = helper.record(dec['reading'])
            raw, endpoint_history = endpoint_raw(dec, baseline, endpoint, reading)
            graph, history, reference, pairs, _, _, _ = helper.prepare(dec, source)
            if endpoint_history != history:
                raise ValueError('native endpoint has another execution history')
            factory = helper.load(dec['reconstruction_helper'], 'decomposition_native_reader')
            native = factory.held.raw_forward(raw, reading['merges'], len(history), graph.width)
            with np.load(helper.checked(transition['payload']), allow_pickle=False) as payload:
                weights, replay = finite.endpoint_weights(graph.w, names, transition['inventory'], payload)
            if (replay['changed_coordinates'] != proposal['changed_values']
                    or replay['actual_displacement_squared'] != proposal['actual_displacement_squared']):
                raise ValueError('reconstructed endpoint differs from the installed displacement')
        with rec.context(training_step=endpoint['step'], proposal_evaluation_index=0,
                proposal_evaluation_kind='fixed_committed_endpoint',
                model_identity=json.dumps(transition['target_identity'], sort_keys=True)):
            with rec.span('proposal.evaluate', attributes={'evaluation_scope': 'global_and_native_input_local_response'}):
                paths = finite_paths(graph, history, native, weights)
            with rec.span('diagnostic.compare'):
                result, maps = compare_paths(paths, reference, pairs)
                reproduction = saved_map_check(maps, baseline, endpoint, prior)
                for label in ('global_endpoint', 'local_endpoint'):
                    path = run/(label+'-residual-squared.npy')
                    with path.open('xb') as stream:
                        np.save(stream, maps[label], allow_pickle=False)
                    result[label+'_map'] = helper.binding(path)
                result.update(status='complete', program=dec['program'], checkpoint=dec['checkpoint'],
                    baseline_measurement=dec['baseline_measurement'], endpoint_measurement=dec['endpoint_measurement'],
                    endpoint_proposal=dec['endpoint_proposal'], endpoint_reached=dec['endpoint_reached'],
                    endpoint_raw=dec['endpoint_raw'], payload=transition['payload'], replay=replay,
                    finite_result=dec['finite_result'], saved_map_reproduction=reproduction,
                    counts=dict(events=len(history), reference_events=reference.rows, pairs=len(pairs), merges=reference.merges),
                    precision=dict(effective=precision, cpu_threads=torch.get_num_threads()),
                    hardware=dict(declared=dec['hardware'], torch_device=torch.cuda.get_device_name(0),
                                  peak_gpu_bytes=torch.cuda.max_memory_allocated()),
                    decomposition='global-native = (global-local)+(local-native); squared errors retain twice the cross term',
                    local_scope='Native inputs at both endpoints; auxiliary/cache reconstructed at native producer inputs, not captured native cache bytes. Diagnostic intervention only.',
                    prediction_vs_result=dict(prediction=dec['prediction'], interpretation='Complete measured response moments retained; independent interpretation pending.'),
                    failures_only='Every original reference term, all current events and all105 merges retained; no error-selected filtering.',
                    what_this_changes='Read-only localization of an already committed response; no learner, derivative, reference or acquisition change.',
                    limitation='GPU1 surrogate versus saved GPU0 native. Error legs are counterfactual vector differences, not causal percentages or a replacement gradient.')
                for label, objective in result['objectives'].items():
                    rec.metric('proposal.decomposition_'+label+'_objective', objective)
                rec.metric('proposal.global_native_residual_error_squared', result['response']['weighted_residual']['total_error_squared'])
                helper.save(run/'result.json', result)
        status = 'FINISHED'
    finally:
        rec.close(status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--worker-lock-fd', type=int)
    args = parser.parse_args()
    run = args.run.resolve()
    if not run.is_relative_to(ROOT/'studies/04-round-2-overlap/runs'):
        raise ValueError('outputs must remain inside the declared run')
    dec = helper.check_native_run(run, run/'declaration.json')
    if dec['program'] != helper.binding(__file__) or dec['finite_evaluator'] != helper.binding(FINITE) or dec['physical_gpu'] != 1:
        raise ValueError('wrong program, inherited finite evaluator or physical GPU')
    for row in dec['implementation']:
        helper.checked(row)
    if dec['observability']['events'] != 'observability/events.jsonl':
        raise ValueError('missing bound Recorder output')
    if args.worker_lock_fd is not None:
        worker(run, dec, args.worker_lock_fd)
        return
    with Path(f'/run/user/{os.getuid()}/gpu1.lock').open('r+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        hardware = subprocess.check_output(['nvidia-smi', '--id=1', '--query-gpu=index,name,uuid,compute_cap', '--format=csv,noheader'], text=True).strip()
        if hardware != dec['hardware'] or hardware.split(', ')[2] != dec['gpu_uuid']:
            raise ValueError('physical GPU differs from declaration')
        env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=dec['gpu_uuid'])
        with (run/'worker.stdout.log').open('x') as stdout, (run/'worker.stderr.log').open('x') as stderr:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), str(run), '--worker-lock-fd', str(lock.fileno())],
                env=env, stdout=stdout, stderr=stderr, pass_fds=(lock.fileno(),), check=True)


if __name__ == '__main__':
    main()
