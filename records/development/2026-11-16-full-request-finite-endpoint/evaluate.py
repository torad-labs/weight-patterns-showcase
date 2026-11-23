"""Read-only finite replay of one already committed native weight transition.

No derivative, directional response, candidate search or native acquisition.
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
HELPER = ROOT / 'records/development/2026-11-16-component-gradient-diagnosis/diagnose.py'
spec = importlib.util.spec_from_file_location('finite_endpoint_reconstruction', HELPER)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
from pipeline.observability.recorder import Recorder


def history(measurement):
    capture = helper.record(measurement['capture'])
    cases = helper.record(capture['cases'])
    answer = helper.record(measurement['acquisition'])
    if (len(cases) != 1 or cases[0]['id'] != measurement['member']
            or answer['id'] != measurement['member'] or not answer['ended_by_itself']
            or capture['model_identity'] != measurement['model_identity']):
        raise ValueError('recorded acquisition identity is inconsistent')
    tokens = tuple(cases[0]['tokens'] + answer['tokens'][:-1])
    if len(tokens) != measurement['events']:
        raise ValueError('recorded acquisition has another event inventory')
    return tokens, capture


def evidence(dec, names):
    """Bridge retained236 to its identical new baseline and committed237 endpoint."""
    retained = helper.record(dec['measurement'])
    baseline = helper.record(dec['baseline_measurement'])
    endpoint = helper.record(dec['endpoint_measurement'])
    proposal = helper.record(dec['endpoint_proposal'])
    reached = helper.record(dec['endpoint_reached'])
    transition, committed = proposal['native_transition'], reached['native_transition']
    keys = ('source_observation', 'source_identity', 'target_identity', 'reference',
            'correspondence', 'anchor_checkpoint', 'payload')
    if (transition['status'] != 'prepared' or committed['status'] != 'committed'
            or committed['in_progress'] is not None
            or committed['proposal'] != dec['endpoint_proposal']
            or any(transition[key] != committed[key] for key in keys)
            or transition['source_observation'] != dec['baseline_measurement']
            or transition['source_identity'] != baseline['model_identity']
            or transition['target_identity'] != endpoint['model_identity']
            or reached['model_identity'] != endpoint['model_identity']
            or reached['generation'] != endpoint['generation']
            or transition['anchor_checkpoint'] != dec['checkpoint']
            or tuple(reached['matrices']) != names
            or tuple(committed['committed_matrices']) != names
            or tuple(row['name'] for row in transition['inventory']) != names):
        raise ValueError('endpoint is not the complete declared committed transition')
    if (transition['reference'] != baseline['reference']
            or transition['correspondence'] != baseline['correspondence']):
        raise ValueError('transition comparison bindings differ from its baseline')
    for key in ('objective', 'outside', 'events', 'merges', 'pair_merge_cells',
                'complete', 'ended', 'reference', 'member'):
        if retained[key] != baseline[key]:
            raise ValueError('retained state and new baseline differ: ' + key)
    for item in (retained, baseline, endpoint):
        if not item['complete'] or not item['ended']:
            raise ValueError('finite comparison requires complete recorded paths')
        if item['reference'] != baseline['reference']:
            raise ValueError('finite comparison cannot change the original reference')
        helper.checked(item['correspondence'])
        helper.checked(item['residual_squared'])
    if (retained['correspondence']['sha256'] != baseline['correspondence']['sha256']
            or retained['residual_squared']['sha256'] != baseline['residual_squared']['sha256']
            or endpoint['correspondence']['sha256'] != baseline['correspondence']['sha256']
            or endpoint['merges'] != baseline['merges']):
        raise ValueError('held comparison or retained baseline map differs')
    old_history, old_capture = history(retained)
    base_history, base_capture = history(baseline)
    end_history, _ = history(endpoint)
    if old_history != base_history or base_history != end_history:
        raise ValueError('endpoint is not the recorded same-history transition')
    if ({key: value['sha256'] for key, value in old_capture['raw'].items()}
            != {key: value['sha256'] for key, value in base_capture['raw'].items()}):
        raise ValueError('retained anchors differ from the actual transition baseline')
    helper.checked(transition['payload'])
    return baseline, endpoint, proposal, transition


def endpoint_weights(original, names, inventory, payload):
    """Decode exact native codes into private changed tensors; never mutate inputs."""
    if tuple(row['name'] for row in inventory) != tuple(names):
        raise ValueError('sparse payload inventory does not cover the editable weights')
    expected = {f'{kind}_{i}' for i in range(len(names)) for kind in ('index', 'before', 'after')}
    if set(payload) != expected:
        raise ValueError('sparse payload array inventory differs')
    weights, changed, squared = dict(original), 0, []
    for i, row in enumerate(inventory):
        name = row['name']
        old = original[name]
        index, before, after = (payload[f'{kind}_{i}'] for kind in ('index', 'before', 'after'))
        if (old.dtype != torch.float32 or not old.is_contiguous() or list(old.shape) != row['shape']
                or index.dtype != np.dtype('<i8') or before.dtype != np.dtype('<u2')
                or after.dtype != np.dtype('<u2') or index.ndim != 1
                or before.shape != index.shape or after.shape != index.shape
                or len(index) != row['changed_bit_patterns']
                or (len(index) and (index[0] < 0 or index[-1] >= old.numel()
                                   or bool((index[1:] <= index[:-1]).any())))
                or bool((before == after).any())):
            raise ValueError('invalid exact sparse endpoint arrays: ' + name)
        if not len(index):
            continue
        selected = old.detach().reshape(-1)[torch.from_numpy(index).to(old.device)].cpu().numpy()
        actual = selected.view(np.uint32)
        if bool((actual & 0xffff).any()) or not np.array_equal(actual >> 16, before):
            raise ValueError('sparse before-codes do not match retained native weights: ' + name)
        values = (after.astype(np.uint32) << 16).view(np.float32)
        if not np.isfinite(selected).all() or not np.isfinite(values).all():
            raise ValueError('sparse endpoint contains nonfinite native values')
        replacement = old.detach().clone()
        replacement.reshape(-1)[torch.from_numpy(index).to(old.device)] = torch.from_numpy(values).to(old)
        weights[name] = replacement
        squared.append(float(np.square(values.astype(np.float64)-selected.astype(np.float64)).sum()))
        changed += len(index)
    return weights, dict(changed_coordinates=changed, actual_displacement_squared=math.fsum(squared))


def finite(graph, history_tokens, reference, pairs, weights, *, origin=False):
    """Evaluate all original residuals once, with frozen anchors and held pairs."""
    original = graph.w
    try:
        graph.w = weights
        with torch.no_grad():
            forward = graph.forward(history_tokens, again=False)
            if origin:
                helper.assert_anchor(graph, forward)
            contributions = np.zeros((reference.rows, reference.merges), dtype='<f8')
            rows = []
            for i, part in enumerate(reference.measure(forward, pairs)):
                squared = part.residual.double().square()
                values = squared.sum(dim=1).cpu().numpy()
                np.add.at(contributions[:, i], pairs[:, 0], values)
                rows.append(dict(merge=i, loss=float(squared.sum())))
            if len(rows) != reference.merges:
                raise ValueError('finite endpoint omitted original merge residuals')
        return dict(objective=math.fsum(row['loss'] for row in rows), per_merge=rows), contributions
    finally:
        graph.w = original


def worker(run, dec, lock_fd):
    lock = Path(f'/run/user/{os.getuid()}/gpu1.lock')
    if os.fstat(lock_fd).st_ino != lock.stat().st_ino or os.environ.get('CUDA_VISIBLE_DEVICES') != dec['gpu_uuid']:
        raise ValueError('worker lacks the declared GPU1 reservation')
    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source, precision = helper.configure(dec)
    names = tuple(source['functional_loop']['matrices'])
    baseline_native, endpoint_native, proposal, transition = evidence(dec, names)
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
            graph, history_tokens, reference, pairs, _, _, _ = helper.prepare(dec, source)
            with np.load(helper.checked(transition['payload']), allow_pickle=False) as payload:
                weights, replay = endpoint_weights(graph.w, names, transition['inventory'], payload)
            if (replay['changed_coordinates'] != proposal['changed_values']
                    or replay['actual_displacement_squared'] != proposal['actual_displacement_squared']):
                raise ValueError('reconstructed endpoint differs from the installed displacement')
        result, maps = {}, {}
        for i, (label, given) in enumerate((('baseline', graph.w), ('endpoint', weights))):
            with rec.context(training_step=baseline_native['step'] if i == 0 else endpoint_native['step'],
                    proposal_evaluation_index=i, proposal_evaluation_kind='baseline' if i == 0 else 'candidate',
                    model_identity=json.dumps(transition['source_identity'] if i == 0 else transition['target_identity'], sort_keys=True)):
                with rec.span('proposal.evaluate', attributes={'endpoint_role': label, 'evaluation_scope': 'fixed_recorded_endpoint'}):
                    result[label], maps[label] = finite(graph, history_tokens, reference, pairs, given, origin=i == 0)
                rec.metric('proposal.finite_objective', result[label]['objective'], step=i)
        with rec.span('diagnostic.compare'):
            base_map = np.load(helper.checked(baseline_native['residual_squared']), allow_pickle=False)
            end_map = np.load(helper.checked(endpoint_native['residual_squared']), allow_pickle=False)
            predicted = proposal['native_transition']['per_merge_response']
            rows = []
            for i in range(reference.merges):
                rows.append(dict(merge=i, native_before=float(base_map[:, i].sum()), native_after=float(end_map[:, i].sum()),
                    finite_before=result['baseline']['per_merge'][i]['loss'], finite_after=result['endpoint']['per_merge'][i]['loss'],
                    recorded_jvp_prediction=predicted[i]['predicted_objective']))
            for label, values in maps.items():
                path = run/(label+'-residual-squared.npy')
                with path.open('xb') as stream:
                    np.save(stream, values, allow_pickle=False)
                result[label]['residual_map'] = helper.binding(path)
            result.update(status='complete', program=dec['program'], checkpoint=dec['checkpoint'],
                baseline_measurement=dec['baseline_measurement'], endpoint_measurement=dec['endpoint_measurement'],
                endpoint_proposal=dec['endpoint_proposal'], endpoint_reached=dec['endpoint_reached'],
                payload=transition['payload'], replay=replay, per_merge=rows,
                finite_change=result['endpoint']['objective']-result['baseline']['objective'],
                native_change=endpoint_native['objective']-baseline_native['objective'],
                recorded_jvp_change=proposal['actual_bf16_prediction']-baseline_native['objective'],
                baseline_minus_native=result['baseline']['objective']-baseline_native['objective'],
                baseline_map_max_abs_error=float(np.max(np.abs(maps['baseline']-base_map))),
                finite_minus_native=result['endpoint']['objective']-endpoint_native['objective'],
                precision=dict(effective=precision, cpu_threads=torch.get_num_threads()),
                hardware=dict(declared=dec['hardware'], torch_device=torch.cuda.get_device_name(0), peak_gpu_bytes=torch.cuda.max_memory_allocated()),
                prediction_vs_result=dict(prediction=dec['prediction'],
                    finite_change=result['endpoint']['objective']-result['baseline']['objective'],
                    native_change=endpoint_native['objective']-baseline_native['objective'],
                    interpretation='Measured comparison recorded; independent interpretation pending.'),
                failures_only='All original reference rows and all105 merges remain in both finite evaluations and the native comparison; no result-based filtering.',
                what_this_changes='Read-only evidence about the already committed endpoint. No training update, reference change, native acquisition or new gradient.',
                scope='One already-installed endpoint under fixed original reference, native anchors and computational pairs; no new derivative, JVP, search, native capture, or optimizer/remainder change.',
                limitation='Finite anchored surrogate on GPU1, not a new native observation. Recorded native measurements were acquired on GPU0; device/finite reconstruction mismatch remains explicit.')
            rec.metric('proposal.finite_change', result['finite_change'])
            rec.metric('proposal.recorded_jvp_change', result['recorded_jvp_change'])
            rec.metric('native.recorded_change', result['native_change'])
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
    if dec['program'] != helper.binding(__file__) or dec['physical_gpu'] != 1:
        raise ValueError('wrong program or physical GPU')
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
