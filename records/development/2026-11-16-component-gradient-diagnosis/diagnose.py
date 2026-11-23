"""Read-only full/component gradient comparison at one retained native state.

Two reverse passes through the existing anchored STE graph. No optimizer,
JVP, native acquisition, model-file write or gradient-array output.
"""
import argparse
import fcntl
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'code/signal-trace'))
from pipeline.native_records.run_declaration import check_native_run
from pipeline.observability.recorder import Recorder
from pipeline.weight_edit import native_graph, turn_artifact

CUDA_PRECISION = dict(matmul_allow_tf32=False, cudnn_allow_tf32=False,
                      matmul_allow_bf16_reduced_precision_reduction=False)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def binding(path):
    return dict(record=str(Path(path).resolve().relative_to(ROOT)), sha256=sha(path))


def checked(row):
    path = (ROOT / row['record']).resolve()
    if not path.is_relative_to(ROOT) or sha(path) != row['sha256']:
        raise ValueError('missing or changed binding: ' + row['record'])
    return path


def record(row):
    return json.loads(checked(row).read_text())


def load(row, name):
    spec = importlib.util.spec_from_file_location(name, checked(row))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def select_merge(loss_map, merges):
    if (loss_map.dtype != np.dtype('<f8') or loss_map.ndim != 2
            or loss_map.shape[1] != merges or not np.isfinite(loss_map).all()
            or bool((loss_map < 0).any())):
        raise ValueError('selection requires the complete finite saved loss map')
    totals = [math.fsum(float(x) for x in loss_map[:, i]) for i in range(merges)]
    selected = np.flatnonzero(np.asarray(totals) == max(totals))
    if len(selected) != 1:
        raise ValueError('saved largest merge is ambiguous')
    return int(selected[0]), totals


def assert_anchor(graph, forward):
    for key, native in (('incoming', graph.native_in), ('merged', graph.native_out)):
        if len(forward[key]) != len(native):
            raise ValueError('incomplete differentiated native origin')
        for value, expected in zip(forward[key], native):
            if value.dtype != expected.dtype or not torch.equal(value.detach(), expected):
                raise ValueError('differentiated origin differs from captured native arrays')


def differentiate(graph, history, reference, pairs, consume, selected_merge=None):
    """Same tensor reduction/leaves as trace_gradient; retain unused before zero-fill.

    The full pass uses stack(terms).sum() exactly as weight_gradient. The
    component pass selects that same weighted term, without renormalization.
    consume receives one private autograd output at a time, or None if unused.
    """
    started = time.monotonic()
    names, original = tuple(graph.leaves), graph.w
    if (not names or len(names) != len(set(names))
            or any(name not in original or original[name].dtype != torch.float32 for name in names)):
        raise ValueError('complete distinct native float32 leaves are required')
    if selected_merge is not None and not 0 <= selected_merge < reference.merges:
        raise ValueError('selected merge outside full inventory')
    with torch.enable_grad():
        weights = tuple(original[name].detach().requires_grad_(True) for name in names)
        graph.w = original | dict(zip(names, weights))
        try:
            forward = graph.forward(history)
            assert_anchor(graph, forward)
            terms, values, outside = [], [], []
            for observed in reference.measure(forward, pairs):
                term = observed.residual.double().square().sum()
                terms.append(term)
                values.append(float(term.detach()))
                outside.append(int(observed.outside.sum()))
            if len(terms) != reference.merges or not all(math.isfinite(x) for x in values):
                raise ValueError('every finite weighted residual is required')
            objective = torch.stack(terms).sum() if selected_merge is None else terms[selected_merge]
            gradients = list(torch.autograd.grad(objective, weights, allow_unused=True)) if objective.requires_grad else [None] * len(weights)
        finally:
            graph.w = original
    # Release all forward graphs before CPU transfers and comparison reductions.
    del forward, terms, objective, weights, observed, term
    for i, name in enumerate(names):
        gradient = gradients[i]
        gradients[i] = None
        if gradient is not None and not bool(torch.isfinite(gradient).all()):
            raise ValueError('nonfinite weight derivative')
        consume(name, None if gradient is None else gradient.detach())
        del gradient
    return dict(selected_merge=selected_merge, per_merge_objective=values,
                full_objective=math.fsum(values), outside=sum(outside),
                selected_objective=math.fsum(values) if selected_merge is None else values[selected_merge],
                pair_merge_cells=len(pairs) * reference.merges,
                anchor_primal_exact=True, seconds=time.monotonic()-started)


def vector_moments(full, component):
    """None is unused; a connected zero tensor keeps its separate flag."""
    f = None if full is None else full.double()
    c = None if component is None else component.double()
    ff = 0. if f is None else float(f.square().sum())
    cc = 0. if c is None else float(c.square().sum())
    dot = 0. if f is None or c is None else float((f * c).sum())
    other = -c if f is None and c is not None else f if c is None else f-c
    oo = 0. if other is None else float(other.square().sum())
    co = 0. if c is None or other is None else float((c * other).sum())
    return dict(full_norm_squared=ff, component_norm_squared=cc, other_norm_squared=oo,
                component_dot_full=dot, component_dot_other=co,
                dot_decomposition_difference=dot-(cc+co),
                full_unused=full is None, component_unused=component is None,
                full_nonzero=full is not None and bool(torch.count_nonzero(full)),
                component_nonzero=component is not None and bool(torch.count_nonzero(component)))


def summarize(rows):
    keys = ('full_norm_squared', 'component_norm_squared', 'other_norm_squared',
            'component_dot_full', 'component_dot_other')
    total = {key: math.fsum(row[key] for row in rows) for key in keys}
    ff, cc, oo = (total[key] for key in keys[:3])
    total.update(full_norm=math.sqrt(ff), component_norm=math.sqrt(cc), other_norm=math.sqrt(oo),
        component_full_cosine=None if ff == 0 or cc == 0 else total['component_dot_full']/math.sqrt(ff*cc),
        component_other_cosine=None if oo == 0 or cc == 0 else total['component_dot_other']/math.sqrt(oo*cc),
        component_descent_along_negative_full=-total['component_dot_full'],
        component_descent_along_negative_component=-cc,
        other_to_self_projection=None if cc == 0 else total['component_dot_other']/cc,
        full_norm_identity_difference=ff-(cc+oo+2*total['component_dot_other']))
    return total


def configure(dec):
    source = record(dec['source_declaration'])
    if source['program'] != dec['precision_source'] or dec['cuda_precision'] != CUDA_PRECISION or dec['cpu_threads'] != source['cpu_threads']:
        raise ValueError('precision provenance differs from retained learning source')
    checked(dec['precision_source'])
    torch.set_num_threads(dec['cpu_threads'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    effective = dict(matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        matmul_allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
    if effective != CUDA_PRECISION:
        raise ValueError('precision flags did not take effect')
    return source, effective


def restore_model(dec, source, checkpoint, forward, factory):
    tensors, architecture = record(source['tensors']), record(source['architecture'])
    names = tuple(source['functional_loop']['matrices'])
    original = record(dec['reconstruction_original'])
    if (any(original[key] != source[key] for key in ('forward', 'tensors', 'architecture', 'reference'))
            or any(original['functional_loop'][key] != source['functional_loop'][key]
                   for key in ('matrices', 'training_members'))):
        raise ValueError('reconstruction source differs from retained numerical definition')
    changes, payload = factory.load_changes(checkpoint, dec['checkpoint'], tensors, names, checkpoint['update'], original)
    if payload != dec['overrides'] or checkpoint['base_seed'] != original['working_model']['start']:
        raise ValueError('checkpoint seed or sparse payload differs')
    seed = checked(dec['seed'])
    if dec['seed'] != dict(record=checkpoint['base_seed']['file'], sha256=checkpoint['base_seed']['model_sha256']):
        raise ValueError('wrong seed binding')
    model = forward.Model(seed, tensors, architecture, device='cpu', leaves=lambda name: name in names, cache='float16')
    if tuple(model.leaves) != names:
        raise ValueError('editable inventory differs')
    with torch.no_grad():
        for name, index, bits in changes:
            values = (bits.astype(np.uint32) << 16).view(np.float32)
            model.w[name].view(-1)[torch.from_numpy(index)] = torch.from_numpy(values)
            actual = model.w[name].detach().numpy().ravel().view(np.uint32)[index] >> 16
            if not np.array_equal(actual, bits):
                raise ValueError('sparse native reconstruction differs')
        # Same CPU-backed immutable matrices as the retained GPU1 response path.
        model.device = torch.device('cuda:0')
        staged_parts = {'attn_q', 'attn_k', 'attn_v', 'ffn_up', 'ffn_gate', 'inp_gate', 'proj'}
        for name in tuple(model.w):
            staged = name not in names and (name == 'per_layer_model_proj.weight' or
                (name.startswith('blk.') and name.split('.')[2] in staged_parts))
            if not staged:
                model.w[name] = model.w[name].detach().to(model.device)
        model.leaves = {name: model.w[name] for name in names}
        weight = model.weight
        model.weight = lambda name: weight(name).to(model.device)
    return model


def prepare(dec, source):
    checkpoint, measurement = record(dec['checkpoint']), record(dec['measurement'])
    if (checkpoint['pending_install'] is not None or checkpoint['weight_state'] != 'complete'
            or checkpoint['last_measurement'] != dec['measurement']
            or checkpoint['model_identity'] != measurement['model_identity']
            or checkpoint['update'] != measurement['generation']
            or not measurement['complete'] or not measurement['ended']
            or measurement['reference'] != source['reference']):
        raise ValueError('requires exact complete checkpoint and its complete measured history')
    capture = record(measurement['capture'])
    if capture['model_identity'] != checkpoint['model_identity'] or capture['raw'] != dec['raw']:
        raise ValueError('native capture identity or retained raw binding differs')
    for row in dec['raw'].values(): checked(row)
    raw = checked(dec['raw']['stream.jsonl']).parent
    if checked(dec['raw']['stream.f32']).parent != raw:
        raise ValueError('raw arrays and index are not one capture')
    cases, answer = record(capture['cases']), record(measurement['acquisition'])
    if len(cases) != 1 or cases[0]['id'] != measurement['member'] or answer['id'] != measurement['member'] or not answer['ended_by_itself']:
        raise ValueError('acquisition does not match retained member/history')
    history = tuple(cases[0]['tokens'] + answer['tokens'][:-1])
    if len(history) != measurement['events']:
        raise ValueError('retained history inventory differs')
    pairs = np.load(checked(measurement['correspondence']), allow_pickle=False)
    loss_map = np.load(checked(measurement['residual_squared']), allow_pickle=False)
    selected, totals = select_merge(loss_map, measurement['merges'])
    if selected != dec['selected_merge'] or list(loss_map.shape) != measurement['residual_squared']['shape']:
        raise ValueError('declared argmax or map shape differs')
    forward = load(source['forward'], 'component_forward')
    factory = load(dec['reconstruction_helper'], 'component_checkpoint_reader')
    reading = record(dec['reading'])
    if len(reading['merges']) != measurement['merges'] or reading['generation_sha256'] != capture['engine']['sha256']:
        raise ValueError('merge layout or capture engine differs')
    model = restore_model(dec, source, checkpoint, forward, factory)
    native = factory.held.raw_forward(raw, reading['merges'], len(history), model.width)
    reference = turn_artifact.load(ROOT, source['reference'], original_model_sha256=source['model_sha256'], expected_merges=measurement['merges'])
    if (loss_map.shape[0] != reference.rows or pairs.shape != (measurement['pair_merge_cells']//reference.merges, 2)):
        raise ValueError('saved reference/pair inventory differs')
    adapter = load(source['ffn_precision'], 'component_precision')
    calls = adapter.calls_from_index(raw/'stream.jsonl', events=len(history), layers=model.layers, width=model.width)
    wrapped = adapter.Model(model, calls, rms=forward.rms, gelu=forward.gelu,
                            batch_single_column=source['ffn_precision_policy']['batch_single_column'])
    graph = native_graph.Graph(wrapped, history, native)
    return graph, history, reference, pairs, selected, totals, measurement


def worker(run, dec, lock_fd):
    lock = Path(f'/run/user/{os.getuid()}/gpu1.lock')
    if (os.fstat(lock_fd).st_ino != lock.stat().st_ino
            or os.environ.get('CUDA_VISIBLE_DEVICES') != dec['gpu_uuid']):
        raise ValueError('child lacks the bound GPU1 reservation')
    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source, precision = configure(dec)
    rec = Recorder(run/'observability/events.jsonl', dict(project='weight-patterns', folder=run.name,
        operation=dec['operation'], run=str(run.relative_to(ROOT)), entrypoint=dec['program']['record'],
        declaration_sha256=sha(run/'declaration.json'), parent_run=str(Path(dec['checkpoint']['record']).parent),
        original_model_sha256=source['model_sha256'], reference_sha256=source['reference']['sha256'],
        checkpoint_sha256=dec['checkpoint']['sha256'], physical_gpu=1, gpu_uuid=dec['gpu_uuid']))
    status = 'FAILED'
    try:
        with rec.span('diagnostic.prepare'):
            graph, history, reference, pairs, selected, losses, measured = prepare(dec, source)
        component = {}
        def keep(name, gradient):
            component[name] = None if gradient is None else gradient.cpu()
        with rec.span('proposal.compute', attributes={'gradient_scope': 'selected_component', 'merge': selected}):
            component_pass = differentiate(graph, history, reference, pairs, keep, selected)
        gc.collect(); torch.cuda.empty_cache()
        rows = []
        def compare(name, gradient):
            value = None if gradient is None else gradient.cpu()
            row = vector_moments(value, component.pop(name))
            rows.append(dict(name=name, shape=list(graph.w[name].shape), **row))
        with rec.span('proposal.compute', attributes={'gradient_scope': 'full'}):
            full_pass = differentiate(graph, history, reference, pairs, compare)
        if component or [row['name'] for row in rows] != list(graph.leaves):
            raise ValueError('gradient comparison omitted a declared matrix')
        with rec.span('diagnostic.compare'):
            result = dict(program=dec['program'], checkpoint=dec['checkpoint'], measurement=dec['measurement'],
                selected_merge=selected, selection='unique argmax of all saved per-merge weighted loss totals',
                saved_per_merge_loss=losses, full_pass=full_pass, component_pass=component_pass,
                full_loss_minus_saved=full_pass['full_objective']-measured['objective'],
                per_merge_loss_minus_saved=[a-b for a,b in zip(full_pass['per_merge_objective'],losses)],
                repeated_primal_loss_difference=[a-b for a,b in zip(component_pass['per_merge_objective'],full_pass['per_merge_objective'])],
                matrices=rows, totals=summarize(rows),
                precision=dict(effective=precision, source=dec['precision_source'], cpu_threads=torch.get_num_threads()),
                hardware=dict(declared=dec['hardware'], torch_device=torch.cuda.get_device_name(0), torch_version=torch.__version__, peak_gpu_bytes=torch.cuda.max_memory_allocated()),
                prediction_vs_result=dict(prediction=dec['prediction'], measured='Full and selected gradients, norms and projections are reported; independent interpretation pending.'),
                failures_only='No event filtering. All original weighted residuals and all declared matrices remain in the full pass; the diagnostic component is selected only by saved argmax.',
                what_this_changes='Read-only gradient attribution at retained206. No update, optimizer, current learner source or native capture changed.',
                limitations=['Anchored Torch/FFN STE derivative, not a native discrete derivative.',
                    'GPU1 differs from the learning GPU0; identical native anchors do not establish derivative equality between devices.',
                    'Fixed recorded histories/correspondence; no generated-path effect, fresh-loss reduction, repair or transfer is measured.',
                    'Gradient magnitude is conditional on this parameterization and editable scope; no arbitrary weak-gradient threshold is applied.'])
            save(run/'result.json', result)
        status = 'FINISHED'
    finally:
        rec.close(status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--worker-lock-fd', type=int)
    args = parser.parse_args(); run = args.run.resolve()
    if not run.is_relative_to(ROOT/'studies/04-round-2-overlap/runs'):
        raise ValueError('outputs must remain inside an admitted run')
    dec = check_native_run(run, run/'declaration.json')
    if dec['program'] != binding(__file__) or dec['physical_gpu'] != 1:
        raise ValueError('wrong program or physical GPU')
    for row in dec['implementation']: checked(row)
    if dec['observability']['events'] != 'observability/events.jsonl':
        raise ValueError('missing bound Recorder output')
    if args.worker_lock_fd is not None:
        worker(run, dec, args.worker_lock_fd); return
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
