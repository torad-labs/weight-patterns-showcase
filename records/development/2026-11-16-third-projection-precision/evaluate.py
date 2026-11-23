"""Finite third-projection policy comparison at the already committed236→237.

No gradient, JVP, candidate, native acquisition or optimizer state is produced.
"""
import argparse
from collections import Counter
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
DECOMPOSITION = ROOT/'records/development/2026-11-16-full-request-response-decomposition/evaluate.py'
spec = importlib.util.spec_from_file_location('third_precision_decomposition', DECOMPOSITION)
decomposition = importlib.util.module_from_spec(spec)
spec.loader.exec_module(decomposition)
finite, helper, Recorder = decomposition.finite, decomposition.helper, decomposition.Recorder


def calls_from_index(path, *, events, layers, width):
    """Native operator labels identify geometry, never computational correctness."""
    labels = {f'per_layer_embd_out-{layer}': layer for layer in range(layers)}
    calls = {layer: [] for layer in range(layers)}
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get('label') not in labels:
                continue
            positions, columns = tuple(row['positions']), row['microbatch_count']
            if (type(columns) is not int or columns < 1 or len(positions) != columns
                    or any(type(i) is not int for i in positions)
                    or row['name'] != row['label'] or row['operation'] != 'MUL'
                    or row['native_type'] != 'f32' or row['token_axis'] != 1
                    or row['shape'] != [width, columns, 1, 1]
                    or row['strides'] != [4, 4*width, 4*width*columns, 4*width*columns]):
                raise ValueError('third boundary metadata does not match its native positions')
            calls[labels[row['label']]].append(dict(call=row['call'],
                microbatch=row['microbatch'], positions=positions))
    for layer, rows in calls.items():
        if (sorted(i for row in rows for i in row['positions']) != list(range(events))
                or len({(row['call'], row['microbatch']) for row in rows}) != len(rows)):
            raise ValueError(f'layer {layer} has missing or repeated native calls/events')
    return calls


def geometry_inventory(calls):
    return [dict(layer=layer, calls=len(rows), events=sum(len(row['positions']) for row in rows),
        columns={str(n): count for n, count in sorted(Counter(len(row['positions']) for row in rows).items())})
        for layer, rows in sorted(calls.items())]


def validate_policy(model, calls):
    """Only the already inspected dense BF16 geometry, without vector fallback."""
    inventory = []
    for layer, rows in calls.items():
        for part in ('inp_gate', 'proj'):
            name = f'blk.{layer}.{part}.weight'
            tensor = model.tensors[name]
            inner, outer, dim2, dim3 = tensor['shape']
            if (tensor['native_type'] != 'bf16' or dim2 != 1 or dim3 != 1
                    or inner % 64 or outer % 32
                    or any(1 < len(row['positions']) <= 8 and outer*len(row['positions']) <= 1024
                           for row in rows)):
                raise ValueError('third projection is outside the reviewed native operand policy')
            inventory.append(dict(name=name, native_type=tensor['native_type'], shape=tensor['shape']))
    return inventory


def bf16_product(value, weight):
    # Finite primal only. The CUDA output is F32; no STE or reverse expression.
    return torch.mm(value.to(torch.bfloat16), weight.to(torch.bfloat16).T,
                    out_dtype=torch.float32)


def grouped_linear(value, weight, rows, linear):
    if (value.ndim != 2 or weight.ndim != 2 or value.shape[1] != weight.shape[1]
            or value.dtype != torch.float32 or weight.dtype != torch.float32
            or value.device != weight.device
            or not bool(torch.isfinite(value).all()) or not bool(torch.isfinite(weight).all())
            or not torch.equal(weight, weight.to(torch.bfloat16).float())):
        raise ValueError('finite F32 activations and exact native BF16 weights are required')
    order = [i for row in rows for i in row['positions']]
    if sorted(order) != list(range(len(value))):
        raise ValueError('projection calls do not cover each event exactly once')
    values = []
    for row in rows:
        part = value[list(row['positions'])]
        values.append(linear(part, weight) if len(part) == 1 else bf16_product(part, weight))
    return torch.cat(values)[torch.argsort(torch.tensor(order, device=value.device))]


def third(model, layer, incoming, auxiliary, rows, *, policy):
    """Invoke the unchanged third function; intercept only its two products."""
    if not policy:
        return model.third(layer, incoming, auxiliary)
    linear, seen = F.linear, []
    shapes = [tuple(reversed(model.tensors[f'blk.{layer}.{part}.weight']['shape'][:2]))
              for part in ('inp_gate', 'proj')]

    def product(value, weight, bias=None):
        if bias is not None or len(seen) >= 2 or tuple(weight.shape) != shapes[len(seen)]:
            raise ValueError('third function changed its two declared projection calls')
        seen.append(tuple(weight.shape))
        return grouped_linear(value, weight, rows, linear)

    with patch.object(F, 'linear', product):
        result = model.third(layer, incoming, auxiliary)
    if len(seen) != 2:
        raise ValueError('third function did not call both declared projections')
    return result


def third_paths(graph, history, native_endpoint, weights, calls):
    """Two separately anchored finite policies, each at both native endpoints.

    Other merges stay at native237 only to give Reference.measure its complete
    shape. Their losses are never attributed to a third-branch intervention.
    """
    original = graph.w
    try:
        with torch.no_grad():
            native = {key: tuple(value.detach().to(graph.device) for value in native_endpoint[key])
                      for key in ('incoming', 'merged')}
            if any(len(native[key]) != 3*graph.layers for key in native):
                raise ValueError('native endpoint omitted merges')
            for values in native.values():
                if any(x.shape != (len(history), graph.width) or x.dtype != torch.float32
                       or not bool(torch.isfinite(x).all()) for x in values):
                    raise ValueError('native endpoint requires complete finite F32 arrays')
            baseline = dict(incoming=graph.native_in, merged=graph.native_out)
            computed, origin_exact = {}, True
            for endpoint, state, incoming in [('before', original, graph.native_in),
                                               ('after', weights, native['incoming'])]:
                graph.w = state
                per = graph.model.per_layer_input(incoming[0], history)
                for layer, rows in calls.items():
                    i = 3*layer+2
                    x = incoming[i]
                    existing = x + third(graph.model, layer, x, per[:, layer], rows, policy=False)
                    if endpoint == 'before':
                        exact = torch.equal(existing, graph.computed[i])
                        origin_exact &= exact
                        if not exact:
                            raise ValueError('existing third reconstruction differs from frozen236 computation')
                    changed = x + third(graph.model, layer, x, per[:, layer], rows, policy=True)
                    computed[endpoint, layer] = existing, changed
            paths = dict(baseline=baseline, native=native)
            for policy, index in [('existing', 0), ('policy', 1)]:
                merged = list(native['merged'])
                for layer in calls:
                    i = 3*layer+2
                    merged[i] = graph.native_out[i] + (computed['after', layer][index] - computed['before', layer][index])
                paths[policy] = dict(incoming=native['incoming'], merged=merged)
            return paths, origin_exact
    finally:
        graph.w = original


MOMENT_KEYS = ('baseline_squared', 'native_squared', 'existing_squared', 'policy_squared',
    'existing_dot_native', 'policy_dot_native', 'existing_dot_policy',
    'existing_native_error_squared', 'policy_native_error_squared', 'policy_change_squared')


def derived(row):
    for a, b in [('existing', 'native'), ('policy', 'native'), ('existing', 'policy')]:
        denominator = math.sqrt(row[a+'_squared']*row[b+'_squared'])
        row[a+'_'+b+'_cosine'] = row[a+'_dot_'+b]/denominator if denominator else None
    return row


def moments(base, existing, policy, native):
    row = decomposition.moments(base, existing, policy, native)
    rename = dict(global_squared='existing_squared', local_squared='policy_squared',
        global_dot_native='existing_dot_native', local_dot_native='policy_dot_native',
        global_dot_local='existing_dot_policy', total_error_squared='existing_native_error_squared',
        local_error_squared='policy_native_error_squared', propagation_error_squared='policy_change_squared')
    result = {key: row[key] for key in ('coordinates', 'baseline_squared', 'native_squared')}
    result.update({target: row[source] for source, target in rename.items()})
    return derived(result)


def aggregate(rows):
    return derived(dict(coordinates=sum(row['coordinates'] for row in rows),
        **{key: math.fsum(row[key] for row in rows) for key in MOMENT_KEYS}))


def compare(paths, reference, pairs, calls):
    """Partition already weighted rows by current event; never renormalize."""
    labels = ('baseline', 'existing', 'policy', 'native')
    maps = {label: np.zeros((reference.rows, reference.merges), dtype='<f8') for label in labels}
    generators = [reference.measure(paths[label], pairs) for label in labels]
    rows = []
    with torch.no_grad():
        for merge in range(reference.merges):
            parts = [next(iterator).residual for iterator in generators]
            for label, part in zip(labels, parts):
                np.add.at(maps[label][:, merge], pairs[:, 0], part.double().square().sum(1).cpu().numpy())
            if merge % 3 != 2:
                continue
            layer = merge//3
            outputs = [paths[label]['merged'][merge] for label in labels]
            sizes = np.empty(len(outputs[0]), dtype=np.int64)
            for call in calls[layer]:
                sizes[list(call['positions'])] = len(call['positions'])
            groups = []
            for columns in sorted(set(sizes.tolist())):
                current = np.flatnonzero(sizes == columns)
                paired = np.flatnonzero(sizes[pairs[:, 1]] == columns)
                groups.append(dict(columns=columns, events=len(current), pairs=len(paired),
                    losses={label: float(part[paired].double().square().sum()) for label, part in zip(labels, parts)},
                    outgoing=moments(*(x[current] for x in outputs)),
                    weighted_residual=moments(*(x[paired] for x in parts))))
            row = dict(merge=merge, layer=layer, events=len(outputs[0]), pairs=len(parts[0]),
                losses={label: float(part.double().square().sum()) for label, part in zip(labels, parts)},
                outgoing=moments(*outputs), weighted_residual=moments(*parts), by_columns=groups)
            row['group_sum_difference'] = {space: {key: row[space][key]-math.fsum(g[space][key] for g in groups)
                for key in MOMENT_KEYS} for space in ('outgoing', 'weighted_residual')}
            row['group_loss_sum_difference'] = {label: row['losses'][label]-math.fsum(g['losses'][label] for g in groups)
                                                for label in labels}
            rows.append(row)
        if any(next(iterator, None) is not None for iterator in generators) or len(rows) != len(calls):
            raise ValueError('reference or third inventory differs from complete declared merges')
    def total(selected):
        return dict(event_merge_observations=sum(row['events'] for row in selected),
            pair_merge_observations=sum(row['pairs'] for row in selected),
            losses={label: math.fsum(row['losses'][label] for row in selected) for label in labels},
            **{space: aggregate([row[space] for row in selected]) for space in ('outgoing', 'weighted_residual')})
    return dict(per_merge=rows, all_events=total(rows), by_columns={str(n): total(
        [g for row in rows for g in row['by_columns'] if g['columns'] == n])
        for n in sorted({g['columns'] for row in rows for g in row['by_columns']})}), maps


def reproduction(maps, baseline, endpoint, prior):
    result = {}
    for label, binding in [('baseline', baseline['residual_squared']), ('native', endpoint['residual_squared']),
                           ('existing', prior['local_endpoint_map'])]:
        saved = np.load(helper.checked(binding), allow_pickle=False)
        if saved.dtype != np.dtype('<f8') or saved.shape != maps[label].shape:
            raise ValueError('saved reproduction map has different shape/type')
        wanted, actual = (saved[:, 2::3], maps[label][:, 2::3]) if label == 'existing' else (saved, maps[label])
        result[label] = dict(binding=binding, scope='all third merges' if label == 'existing' else 'all merges',
            exact=bool(np.array_equal(wanted, actual)), max_abs_error=float(np.max(np.abs(wanted-actual))))
        if label != 'existing' and not result[label]['exact']:
            raise ValueError('saved native loss map was not reproduced: '+label)
    return result


def worker(run, dec, lock_fd):
    lock = Path(f'/run/user/{os.getuid()}/gpu1.lock')
    if os.fstat(lock_fd).st_ino != lock.stat().st_ino or os.environ.get('CUDA_VISIBLE_DEVICES') != dec['gpu_uuid']:
        raise ValueError('worker lacks the declared GPU1 reservation')
    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source, precision = helper.configure(dec)
    names = tuple(source['functional_loop']['matrices'])
    baseline, endpoint, proposal, transition = finite.evidence(dec, names)
    prior = helper.record(dec['decomposition_result'])
    if (prior['status'] != 'complete' or prior['program'] != dec['decomposition_evaluator']
            or any(prior[key] != dec[key] for key in ('checkpoint', 'baseline_measurement',
                'endpoint_measurement', 'endpoint_proposal', 'endpoint_reached', 'endpoint_raw'))
            or prior['payload'] != transition['payload']):
        raise ValueError('prior local comparison belongs to another exact transition')
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
            raw, endpoint_history = decomposition.endpoint_raw(dec, baseline, endpoint, reading)
            graph, history, reference, pairs, _, _, _ = helper.prepare(dec, source)
            if endpoint_history != history:
                raise ValueError('native endpoint has another execution history')
            calls = calls_from_index(helper.checked(dec['raw']['stream.jsonl']),
                events=len(history), layers=graph.layers, width=graph.width)
            geometry = geometry_inventory(calls)
            if geometry != dec['third_geometry']:
                raise ValueError('recorded third geometry differs from declaration')
            matrices = validate_policy(graph.model, calls)
            factory = helper.load(dec['reconstruction_helper'], 'third_precision_native_reader')
            native = factory.held.raw_forward(raw, reading['merges'], len(history), graph.width)
            with np.load(helper.checked(transition['payload']), allow_pickle=False) as payload:
                weights, replay = finite.endpoint_weights(graph.w, names, transition['inventory'], payload)
            if (replay['changed_coordinates'] != proposal['changed_values']
                    or replay['actual_displacement_squared'] != proposal['actual_displacement_squared']):
                raise ValueError('reconstructed endpoint differs from installed displacement')
        with rec.context(training_step=endpoint['step'], proposal_evaluation_index=0,
                proposal_evaluation_kind='fixed_committed_endpoint',
                model_identity=json.dumps(transition['target_identity'], sort_keys=True)):
            with rec.span('proposal.evaluate', attributes={'evaluation_scope': 'native_input_third_projection_policy'}):
                paths, origin_exact = third_paths(graph, history, native, weights, calls)
            with rec.span('diagnostic.compare'):
                result, maps = compare(paths, reference, pairs, calls)
                checks = reproduction(maps, baseline, endpoint, prior)
                for label in ('existing', 'policy'):
                    path = run/(label+'-third-residual-squared.npy')
                    with path.open('xb') as stream:
                        np.save(stream, maps[label][:, 2::3], allow_pickle=False)
                    result[label+'_third_map'] = helper.binding(path)
                result.update(status='complete', program=dec['program'], checkpoint=dec['checkpoint'],
                    baseline_measurement=dec['baseline_measurement'], endpoint_measurement=dec['endpoint_measurement'],
                    endpoint_proposal=dec['endpoint_proposal'], endpoint_reached=dec['endpoint_reached'],
                    endpoint_raw=dec['endpoint_raw'], payload=transition['payload'], replay=replay,
                    decomposition_result=dec['decomposition_result'], saved_map_reproduction=checks,
                    baseline_existing_computation_exact=origin_exact, geometry=geometry, projection_matrices=matrices,
                    counts=dict(events=len(history), reference_events=reference.rows, pairs=len(pairs),
                        reference_merges=reference.merges, compared_third_merges=len(calls)),
                    precision=dict(effective=precision, cpu_threads=torch.get_num_threads()),
                    hardware=dict(declared=dec['hardware'], torch_device=torch.cuda.get_device_name(0),
                        peak_gpu_bytes=torch.cuda.max_memory_allocated()),
                    native_full_objectives={label: float(maps[label].sum()) for label in ('baseline', 'native')},
                    prediction_vs_result=dict(prediction=dec['prediction'], interpretation='Measured finite responses; independent interpretation pending.'),
                    failures_only='Every current event and mapped pair at all35 third merges; original105-merge mass, no error filtering or group renormalization.',
                    what_this_changes='Tests only the third projection operand/call policy at an existing transition; no learner, reference or derivative change.',
                    limitation='GPU1 Torch versus saved GPU0 native. Call geometry and operand policy do not reproduce native kernels; auxiliary values remain reconstructed. Finite response does not validate a derivative.')
                for label, loss in result['all_events']['losses'].items():
                    rec.metric('proposal.third_'+label+'_loss', loss)
                rec.metric('proposal.third_policy_native_residual_error_squared', result['all_events']['weighted_residual']['policy_native_error_squared'])
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
    if dec['program'] != helper.binding(__file__) or dec['decomposition_evaluator'] != helper.binding(DECOMPOSITION) or dec['physical_gpu'] != 1:
        raise ValueError('wrong program, inherited evaluator or physical GPU')
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
