"""One prescribed response comparison and one empirical gain-compensated candidate.

No native execution, training process control, model-file writes or step search.
All vectors use the existing whole-trace residual and fixed saved correspondence.
"""
import gc
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.autograd import forward_ad

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT / 'code/signal-trace'))
from pipeline.native_records.run_declaration import check_native_run
from pipeline.weight_edit import bf16_direction, gap_state, model_trace_response, native_graph, trace_gradient, turn_artifact
from pipeline.weight_edit.functional_reference import checked


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def binding(path):
    path = Path(path)
    return dict(record=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def save(name, value):
    with (HERE / name).open('x') as file:
        json.dump(value, file, indent=2)
        file.write('\n')


def residuals(reference, forward, pairs):
    return [item.residual.detach().double().cpu() for item in reference.measure(forward, pairs)]


def moments(graph, history, reference, pairs, direction, *, native_before=None, native_after=None, finite=None):
    rows = []
    def measured(forward):
        for index, item in enumerate(reference.measure(forward, pairs)):
            primal, tangent = forward_ad.unpack_dual(item.residual)
            r = primal.detach().double().cpu()
            q = torch.zeros_like(r) if tangent is None else tangent.detach().double().cpu()
            row = dict(merge=index, objective=float(r.square().sum()),
                       slope=float(2 * (r * q).sum()), curvature=float(q.square().sum()))
            if native_before is not None:
                row['native_anchor_residual_error_squared'] = float((r - native_before[index]).square().sum())
            if finite is not None:
                dt = finite[index] - r
                dn = native_after[index] - native_before[index]
                row.update(finite_torch_loss=float(finite[index].square().sum()),
                    finite_torch_cross=float(2 * (r * dt).sum()), finite_torch_square=float(dt.square().sum()),
                    native_loss_before=float(native_before[index].square().sum()),
                    native_loss_after=float(native_after[index].square().sum()),
                    native_cross=float(2 * (native_before[index] * dn).sum()), native_square=float(dn.square().sum()),
                    graph_finite_minus_jvp_squared=float((dt-q).square().sum()),
                    native_minus_graph_finite_squared=float((dn-dt).square().sum()),
                    native_minus_jvp_squared=float((dn-q).square().sum()),
                    native_dot_jvp=float((dn*q).sum()), graph_finite_dot_jvp=float((dt*q).sum()))
            rows.append(row)
            yield item.residual
    response = model_trace_response.measure_model(graph, tuple(graph.leaves), history, direction, measured)
    if len(rows) != reference.merges:
        raise ValueError('incomplete per-merge response')
    totals = {key: math.fsum(row[key] for row in rows) for key in rows[0] if key != 'merge'}
    # CPU reductions and the existing GPU reductions can differ at rounding precision;
    # retain both instead of inventing a tolerance or calling them exact identities.
    return dict(merges=rows, totals=totals, response=vars(response),
                reduction_difference={k: totals[k]-getattr(response,k) for k in ('objective','slope','curvature')})


def main():
    started = time.monotonic()
    declaration = check_native_run(HERE, HERE / 'declaration.json')
    if declaration['program'] != binding(Path(__file__)):
        raise ValueError('diagnostic source differs from declaration')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != declaration['gpu_uuid']:
        raise ValueError('use the declared diagnostic GPU')
    original = checked(ROOT, declaration['source_declaration'])
    for source in original['implementation']:
        checked(ROOT, source, parse=False)
    comparison = checked(ROOT, declaration['native_comparison'])
    native_result = checked(ROOT, declaration['native_result'])
    numerical = checked(ROOT, declaration['gain_measurement'])
    replay_configuration = checked(ROOT, comparison['same_card_report'])
    torch.set_num_threads(replay_configuration['cpu_threads'])
    if [row['update'] for row in native_result['results']] != [5,6]:
        raise ValueError('native comparison must contain exactly prescribed5 and6')
    eta = math.sqrt(numerical['anchored_actual_bf16_jvp']['curvature'] /
                    numerical['total']['finite_squared_residual_change'])
    checked(ROOT, comparison['program'], parse=False)
    checked(ROOT, original['forward'], parse=False)
    native = load_module('prescribed_native_for_response', ROOT / comparison['program']['record'])
    forward = load_module('prescribed_torch_for_response', ROOT / original['forward']['record'])
    tensors, architecture = checked(ROOT, original['tensors']), checked(ROOT, original['architecture'])
    names = tuple(original['functional_loop']['matrices'])
    candidates = [checked(ROOT, source) for source in comparison['candidates']]
    changes = [native.load_changes(candidate, source, tensors, names, step, original)[0]
               for candidate, source, step in zip(candidates, comparison['candidates'], (5, 6))]
    seed = ROOT / original['working_model']['start']['file']
    seed_stat = native.held.working_identity.fingerprint(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # CPU construction keeps immutable matrices off the display GPU. Editable
    # operands and every actual calculation use the declared GPU, unchanged dtype.
    model = forward.Model(seed, tensors, architecture, device='cpu',
                          leaves=lambda name: name in names, cache='float16')
    if tuple(model.leaves) != names:
        raise ValueError('editable matrix inventory differs')
    seed_bits = {name: model.w[name].detach().to(torch.bfloat16).view(torch.uint16).numpy().copy().reshape(-1)
                 for name in names}
    transition = []
    with torch.no_grad():
        for (name, i5, v5), (name6, i6, v6) in zip(*changes):
            if name != name6:
                raise ValueError('candidate matrices differ')
            union = np.union1d(i5, i6)
            before, after = seed_bits[name][union].copy(), seed_bits[name][union].copy()
            before[np.searchsorted(union, i5)] = v5
            after[np.searchsorted(union, i6)] = v6
            f5 = torch.from_numpy(before.copy()).view(torch.bfloat16).float()
            f6 = torch.from_numpy(after.copy()).view(torch.bfloat16).float()
            model.w[name].view(-1)[torch.from_numpy(union)] = f5
            transition.append((name, union, f5, f6))
        model.device = torch.device('cuda:0')
        staged_parts = {'attn_q','attn_k','attn_v','ffn_up','ffn_gate','inp_gate','proj'}
        for name in tuple(model.w):
            staged = name not in names and (name == 'per_layer_model_proj.weight' or
                (name.startswith('blk.') and name.split('.')[2] in staged_parts))
            if not staged:
                model.w[name] = model.w[name].detach().to(model.device)
        model.leaves = {name: model.w[name] for name in names}
        weight = model.weight
        model.weight = lambda name: weight(name).to(model.device)
    print('Loaded prescribed reconstructed5; immutable matrices remain on CPU', flush=True)
    measured = checked(ROOT, comparison['baseline_observation'])
    reading = checked(ROOT, measured['reading'])
    definition = checked(ROOT, original['source_definition'])
    table = next(row for row in reading['table'] if row['id'] == measured['member'])
    folder = (ROOT / measured['reading']['record']).parent
    acquisition = checked(ROOT, dict(record=str((folder/'answers'/measured['member']/'answer.json').relative_to(ROOT)),
                                     sha256=table['answer_json_sha256']))
    history = tuple(definition['prompts'][measured['member']] + acquisition['tokens'][:-1])
    checked(ROOT, measured['correspondence'], parse=False)
    pairs = np.load(ROOT / measured['correspondence']['record'], allow_pickle=False)
    if len(history) != measured['events']:
        raise ValueError('acquisition event count differs')
    reference = turn_artifact.load(ROOT, original['reference'],
        original_model_sha256=original['model_sha256'], expected_merges=3*model.layers)
    captured = []
    for step, row in zip((5,6), native_result['results']):
        capture = checked(ROOT, row['capture'])
        if capture['model_identity']['candidate'] != comparison['candidates'][step-5]:
            raise ValueError('native capture is not the prescribed candidate')
        path = (ROOT / row['capture']['record']).parent / 'raw' / measured['member']
        trace = native.held.raw_forward(path, reading['merges'], measured['events'], model.width)
        captured.append(residuals(reference, trace, pairs))
        loss = math.fsum(float(r.square().sum()) for r in captured[-1])
        if loss != row['fixed_correspondence']['objective']:
            raise ValueError('saved native computational loss does not reproduce')
        if step == 5:
            graph = native_graph.Graph(model, history, trace)
        del trace
    print('Both saved native losses reproduced; graph anchored at actual native5', flush=True)
    gradient = trace_gradient.weight_gradient(graph, history, reference, pairs=pairs)
    raw = moments(graph, history, reference, pairs, gradient.direction, native_before=captured[0])
    alpha = raw['response']['length']
    if raw['response']['status'] != 'proposal' or alpha is None:
        raise ValueError('raw negative gradient did not yield a finite response scale')
    raw.update(gradient_norm_squared=gradient.norm_squared, gradient_seconds=gradient.seconds,
               eta=eta, candidate_scale=eta*alpha,
               slope_identity_difference=raw['response']['slope']+gradient.norm_squared)
    save('raw-gradient-response.json', raw)
    # Export only the ONE requested candidate's seed-relative BF16 overrides.
    # Consume the raw gradient matrix by matrix; no full direction is saved.
    arrays, inventory, stats = {}, [], []
    for index, name in enumerate(names):
        base = graph.w[name].detach().cpu().numpy()
        direction = gradient.direction.pop(name).cpu().numpy()
        wanted = bf16_direction.rounded({name:base}, {name:direction}, eta*alpha)[name]
        bits = np.frombuffer(gap_state.encoded(wanted.T, tensors[name]), dtype='<u2')
        selected = np.flatnonzero(bits != seed_bits[name]).astype('<i8')
        arrays[f'index_{index}'], arrays[f'value_{index}'] = selected, bits[selected].copy()
        inventory.append(dict(name=name, entries=int(len(selected))))
        delta = wanted.astype(np.float64)-base.astype(np.float64)
        stats.append(dict(name=name, changed_values=int(np.count_nonzero(delta)),
            displacement_norm_squared=float(np.square(delta).sum()),
            gradient_dot_displacement=float((-direction.astype(np.float64)*delta).sum())))
        del base, direction, wanted, bits, delta
    payload = HERE / 'candidate-gain-compensated-overrides.npz'
    with payload.open('xb') as file:
        np.savez(file, **arrays)
    candidate = dict(update=6, base_seed=original['working_model']['start'], inventory=inventory,
        overrides=dict(file=payload.name, sha256=binding(payload)['sha256']),
        interpretation='ONE empirical response-gain candidate from prescribed reconstructed5; not historical6 or a safe-step theorem',
        preupdate_candidate=comparison['candidates'][0], anchor=native_result['results'][0]['capture'],
        calculation=binding(HERE/'raw-gradient-response.json'), eta=eta, alpha=alpha, applied_scale=eta*alpha,
        formula='BF16(W5 + eta * alpha * S), S=-grad L at native5-anchored W5',
        gain_measurement=declaration['gain_measurement'],
        arithmetic=dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
            limitation='GPU1 and actual native5 anchor differ from historical/same-card calculation; comparison to old6 is not strict scale-only'),
        displacement=dict(matrices=stats, **{key:math.fsum(row[key] for row in stats)
            for key in ('changed_values','displacement_norm_squared','gradient_dot_displacement')}))
    save('candidate-gain-compensated.json', candidate)
    print('CANDIDATE_READY ' + str(HERE/'candidate-gain-compensated.json'), flush=True)
    del arrays, seed_bits, gradient
    gc.collect(); torch.cuda.empty_cache()
    # Finite Torch evaluation at the EXISTING prescribed6, then restore W5.
    with torch.no_grad():
        for name, index, before, after in transition:
            graph.w[name].view(-1)[torch.from_numpy(index).to(model.device)] = after.to(model.device)
        finite = residuals(reference, graph.forward(history), pairs)
        for name, index, before, after in transition:
            graph.w[name].view(-1)[torch.from_numpy(index).to(model.device)] = before.to(model.device)
        delta = {name:torch.zeros_like(graph.w[name]) for name in names}
        for name, index, before, after in transition:
            delta[name].view(-1)[torch.from_numpy(index).to(model.device)] = (after-before).to(model.device)
    rounded = moments(graph, history, reference, pairs, delta,
        native_before=captured[0], native_after=captured[1], finite=finite)
    save('prescribed-bf16-response.json', rounded)
    if native.held.working_identity.fingerprint(seed) != seed_stat:
        raise ValueError('the retained seed changed during diagnostic reading')
    save('result.json', dict(status='complete', raw_gradient=binding(HERE/'raw-gradient-response.json'),
        actual_bf16_displacement=binding(HERE/'prescribed-bf16-response.json'),
        candidate=binding(HERE/'candidate-gain-compensated.json'),
        elapsed_seconds=time.monotonic()-started, peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
        seed_identity='Retained immutable seed SHA inherited from prior verified reconstruction; stat unchanged; no repeated full-file hash',
        seed_stat_before=seed_stat, native_source=declaration['native_result'],
        limitations=['Prescribed reconstructions are not historical weight identity.',
            'Raw S is recomputed on GPU1 with actual native5 anchors.',
            'Finite response differences do not alone qualify or refute the native Jacobian.',
            'Candidate prediction is empirical; only owner native held/fresh captures measure its effect.']))
    print('COMPLETE ' + str(HERE/'result.json'), flush=True)


if __name__ == '__main__':
    main()
