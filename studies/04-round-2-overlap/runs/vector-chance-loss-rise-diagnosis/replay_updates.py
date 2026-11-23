"""Read-only replay of the actual saved updates; never install or generate."""
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / 'code/signal-trace'))
from pipeline.weight_edit import bf16_direction, gap_state, native_graph, trace_gradient, turn_artifact, turn_correction
from pipeline.weight_edit.functional_reference import checked


def save(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2) + '\n')


def payload(path, digest):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f'changed input: {path}')
    return data


def saved_case(run, number, model, definition):
    step = run / 'loop' / str(number)
    measured = json.loads((step / 'measurement.json').read_text())
    reading = checked(ROOT, measured['reading'])
    if reading['model_identity'] != measured['model_identity']:
        raise ValueError('saved native reading and observation identities differ')
    if not measured['complete'] or not measured['ended']:
        raise ValueError('only complete saved observations enter the replay')
    folder = (ROOT / measured['reading']['record']).parent
    row_map = json.loads(payload(folder / 'row-map.json', reading['row_map']['sha256']))
    member = measured['member']
    entry = next(x for x in row_map['answers'] if x['id'] == member)
    table = next(x for x in reading['table'] if x['id'] == member)
    acquisition = json.loads(payload(folder / 'answers' / member / 'answer.json', table['answer_json_sha256']))
    # Acquisition IDs are numerical inputs only: never decoded or compared as outcomes.
    history = tuple(definition['prompts'][member] + acquisition['tokens'][:-1])
    order = np.r_[np.arange(entry['question_rows_from'], entry['question_rows_from'] + entry['question']),
                  np.arange(entry['answer_rows_from'], entry['answer_rows_from'] + entry['steps'])]
    if len(history) != measured['events'] or len(order) != len(history):
        raise ValueError('saved causal inventory differs')
    merged = []
    for index, node in enumerate(reading['merges']):
        if node['merge'] != index:
            raise ValueError('merge inventory changed')
        array = np.load(io.BytesIO(payload(folder / node['file'], node['sha256'])), allow_pickle=False)
        if array.dtype != np.float32 or list(array.shape) != node['shape']:
            raise ValueError('saved merge array format changed')
        merged.append(torch.from_numpy(array[order].copy()))
    if len(merged) != 3 * model.layers:
        raise ValueError('incomplete merge inventory')
    incoming = [model.rows('token_embd.weight', history).cpu() * math.sqrt(model.width)]
    for index in range(1, len(merged)):
        scale = model.w[f'blk.{index // 3 - 1}.layer_output_scale.weight'].detach().cpu() if index % 3 == 0 else 1
        incoming.append(merged[index - 1] * scale)
    pair_binding = measured['correspondence']
    pairs = np.load(io.BytesIO(payload(ROOT / pair_binding['record'], pair_binding['sha256'])), allow_pickle=False)
    return measured, history, dict(incoming=incoming, merged=merged), pairs


def main():
    declaration = json.loads((HERE / 'declaration.json').read_text())
    run = ROOT / declaration['source_run']
    original = checked(ROOT, declaration['source_declaration'])
    state_path = run / 'working/state.json'
    frozen_state = state_path.read_bytes()
    state = json.loads(frozen_state)
    if state['status'] != 'ready' or state['generation'] != declaration['replay_updates']:
        raise ValueError('the frozen completed generation changed')
    if (Path(declaration['cgroup']) / 'cgroup.freeze').read_text().strip() != '1':
        raise ValueError('the training controller must stay frozen')
    for source in original['implementation']:
        checked(ROOT, source, parse=False)
    tensors = checked(ROOT, original['tensors'])
    architecture = checked(ROOT, original['architecture'])
    definition = checked(ROOT, original['source_definition'])
    reference = turn_artifact.load(ROOT, original['reference'], original_model_sha256=original['model_sha256'],
                                   expected_merges=3 * architecture['layers'])
    spec = importlib.util.spec_from_file_location('diagnostic_forward', ROOT / original['forward']['record'])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    names = tuple(original['functional_loop']['matrices'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if os.environ.get('CUDA_VISIBLE_DEVICES') != declaration['hardware']['replay_uuid']:
        raise ValueError('the replay must use its declared physical GPU')
    seed = ROOT / original['working_model']['start']['file']
    seed_hash = hashlib.sha256()
    with seed.open('rb') as source:
        os.posix_fadvise(source.fileno(), 0, 0, os.POSIX_FADV_NOREUSE)
        offset = 0
        while block := source.read(gap_state.BLOCK):
            seed_hash.update(block)
            os.posix_fadvise(source.fileno(), offset, len(block), os.POSIX_FADV_DONTNEED)
            offset += len(block)
    if seed_hash.hexdigest() != original['working_model']['start']['model_sha256']:
        raise ValueError('the retained starting model does not match its declared bytes')
    print(json.dumps(dict(seed_verified_sha256=seed_hash.hexdigest())), flush=True)
    model = module.Model(seed, tensors, architecture,
                         device='cuda:0', leaves=lambda name: name in names, cache='float16')
    if tuple(model.leaves) != names:
        raise ValueError('the declared editable matrix inventory changed')
    report = dict(device=torch.cuda.get_device_name(), visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  torch=torch.__version__, cpu_threads=torch.get_num_threads(), observations=[],
                  seed_verified_sha256=seed_hash.hexdigest(),
                  limitation='Replay and endpoint checks test reproducibility, not native derivative fidelity.')
    save('replay-progress.json', report)
    for number in range(1, declaration['replay_updates'] + 1):
        started = time.monotonic()
        measured, history, native, pairs = saved_case(run, number, model, definition)
        direct = turn_correction.measurement(reference, native, pairs)
        if direct != {k: measured[k] for k in direct}:
            raise ValueError('independent saved-array loss does not reproduce the observation')
        graph = native_graph.Graph(model, history, native)
        unrounded = {}
        rounding = bf16_direction.rounded

        def capture_round(base, direction, length):
            if number == declaration['first_bad_update']:
                for name in base:
                    unrounded[name] = (base[name].astype(np.float64) + length * direction[name].astype(np.float64)).astype(np.float32)
            return rounding(base, direction, length)

        bf16_direction.rounded = capture_round
        try:
            proposal = trace_gradient.propose(graph, history, reference, pairs=pairs)
        finally:
            bf16_direction.rounded = rounding
        historical = json.loads((run / 'loop' / str(number) / 'proposal.json').read_text())
        actual = dict(gradient_norm_squared=proposal.gradient_norm_squared,
                      changed_values=proposal.changed_values,
                      **{k: getattr(proposal.response, k) for k in ('slope', 'curvature', 'length', 'predicted_objective')})
        expected = dict(gradient_norm_squared=historical['gradient_norm_squared'],
                        changed_values=historical['changed_values'],
                        **{k: historical['response'][k] for k in ('slope', 'curvature', 'length', 'predicted_objective')})
        observation = dict(update=number, native_before=direct, graph_before=proposal.before,
                           response_origin_objective=proposal.response.objective,
                           graph_origin_exact=(proposal.before == direct and proposal.response.objective == direct['objective']),
                           replay=actual, recorded=expected,
                           exact_equal={k: actual[k] == expected[k] for k in actual},
                           differences={k: actual[k] - expected[k] for k in actual})
        if proposal.weights is None:
            raise ValueError('the historical accepted update did not replay as a proposal')
        if unrounded:
            base_weights = graph.w
            for label, candidate in (('finite_unrounded_held_graph', unrounded), ('finite_bf16_held_graph', proposal.weights)):
                graph.w = base_weights | {name: torch.from_numpy(candidate[name]).to(base_weights[name]) for name in names}
                try:
                    with torch.no_grad():
                        observation[label] = turn_correction.measurement(reference, graph.forward(history), pairs)
                finally:
                    graph.w = base_weights
                gc.collect()
                torch.cuda.empty_cache()
        with torch.no_grad():
            for name in names:
                model.w[name].copy_(torch.from_numpy(proposal.weights[name]).to(model.w[name]))
        observation['elapsed_seconds'] = time.monotonic() - started
        report['observations'].append(observation)
        save('replay-progress.json', report)
        print(json.dumps(observation), flush=True)
        del graph, proposal, native, history, pairs, unrounded
        gc.collect()
        torch.cuda.empty_cache()
    report['endpoint'] = []
    file = run / 'working/model.gguf'
    with file.open('rb') as source:
        for name in names:
            binding = tensors[name]
            expected = gap_state.encoded(model.w[name].detach().cpu().numpy().T, binding)
            actual = os.pread(source.fileno(), binding['bytes'], binding['offset_bytes'])
            report['endpoint'].append(dict(name=name, bytes=len(actual), exact_equal=actual == expected,
                different_bf16_entries=int(np.count_nonzero(np.frombuffer(actual, dtype='<u2') != np.frombuffer(expected, dtype='<u2')))))
    if state_path.read_bytes() != frozen_state:
        raise ValueError('the training state changed during the read-only replay')
    report['endpoint_all_exact'] = all(x['exact_equal'] for x in report['endpoint'])
    save('replay-report.json', report)
    print(json.dumps(dict(endpoint_all_exact=report['endpoint_all_exact'],
                         different_entries=sum(x['different_bf16_entries'] for x in report['endpoint']))), flush=True)


if __name__ == '__main__':
    main()
