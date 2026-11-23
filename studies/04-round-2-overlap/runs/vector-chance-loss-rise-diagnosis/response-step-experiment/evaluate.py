"""Evaluate one already-produced BF16 candidate: held computation, then fresh.

No update rule, search, output judge, controller, or model disk copy. Both native
passes read the same sealed memfd. The original reference and every merge remain
unchanged; zero loss is not an acceptance condition for this experiment.
"""
import fcntl
import importlib.util
import io
import json
import os
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
FACTORY = HERE.parent / 'native-first-bad/replay.py'
spec = importlib.util.spec_from_file_location('response_native_factory', FACTORY)
factory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(factory)
held = factory.held


def loss_map(reference, native, pairs, out):
    """Existing weighted residual squared, summed by reference event and merge."""
    contribution = np.zeros((reference.rows, reference.merges), dtype='<f8')
    for index, observed in enumerate(reference.measure(native, pairs)):
        row = observed.residual.double().square().sum(dim=1).cpu().numpy()
        np.add.at(contribution[:, index], pairs[:, 0], row)
    with out.open('xb') as file:
        np.save(file, contribution, allow_pickle=False)
    return dict(**held.binding(out), shape=list(contribution.shape), dtype='<f8',
        sum=float(contribution.sum()), definition='existing reference.measure weighted residual squared; aggregate mapped pairs by original reference event and merge')


def main():
    declaration = held.check_native_run(HERE, HERE / 'declaration.json')
    if (declaration['program'] != held.binding(Path(__file__).resolve())
            or declaration['factory_program'] != held.binding(FACTORY)):
        raise ValueError('the evaluator or native factory differs from its declared source')
    if not isinstance(declaration.get('candidate'), dict):
        raise ValueError('the prospectively specified candidate payload has not been bound yet')
    parent = held.checked(ROOT, declaration['factory_declaration'])
    for key, path in (('program', FACTORY), ('native_held_program', factory.held_path),
                      ('native_held_inputs', held.HERE / 'inputs.json')):
        if parent[key] != held.binding(path):
            raise ValueError(f'the admitted native helper/input changed: {key}')
    inputs, original, _, _ = held.load_inputs()
    held.checked(ROOT, declaration['scale_record'])
    if os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID' or os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise ValueError('this experiment uses the original physical GPU0')
    controls = {key: held.checked(ROOT, source) for key, source in declaration['controls'].items()}
    if set(controls) != {'held5', 'held6', 'fresh5', 'fresh6'}:
        raise ValueError('the paired held and fresh controls must all remain bound')
    baseline = held.checked(ROOT, parent['baseline_observation'])
    reading = held.checked(ROOT, baseline['reading'])
    if baseline['generation'] != 5 or baseline['member'] != inputs['member'] or len(reading['merges']) != 105:
        raise ValueError('the held acquisition must be the complete saved5 computation')
    fixed_capture = held.checked(ROOT, controls['held5']['capture'])
    fresh_capture = held.checked(ROOT, controls['fresh5']['capture'])
    fixed_cases = held.checked(ROOT, fixed_capture['cases'])
    fresh_cases = held.checked(ROOT, fresh_capture['cases'])
    if (len(fixed_cases) != 1 or len(fresh_cases) != 1
            or fixed_cases[0]['id'] != inputs['member'] or fresh_cases[0]['id'] != inputs['member']
            or set(fresh_cases[0]) != {'id', 'tokens', 'max_tokens'}
            or len(fixed_cases[0]['tokens']) + len(fixed_cases[0]['teacher_tokens']) - 1 != baseline['events']):
        raise ValueError('the original held/fresh acquisition inventories differ')
    tensors = held.checked(ROOT, original['tensors'])
    names = tuple(original['functional_loop']['matrices'])
    candidate = held.checked(ROOT, declaration['candidate'])
    changes, payload = factory.load_changes(candidate, declaration['candidate'], tensors, names, candidate['update'], original)
    if len(names) != 70 or len(changes) != 70:
        raise ValueError('the candidate must include the complete seventy-matrix BF16 inventory')
    reference = held.turn_artifact.load(ROOT, original['reference'], original_model_sha256=original['model_sha256'], expected_merges=105)
    pairs = np.load(io.BytesIO(held.replay_updates.payload(ROOT / baseline['correspondence']['record'],
                    baseline['correspondence']['sha256'])), allow_pickle=False)
    identity = dict(kind='diagnostic-seed-sparse-bf16/1', experiment=str(HERE.relative_to(ROOT)),
        base_candidate=parent['candidates'][0], candidate=declaration['candidate'], overrides=payload,
        seed=dict(file=candidate['base_seed']['file'], sha256=candidate['base_seed']['model_sha256']),
        rule=declaration['candidate_rule'], scale_record=declaration['scale_record'], historical_revision_claim=False)
    engine = held.generation_engine.Generation()
    if engine.record['sha256'] != reading['generation_sha256'] or engine.record['engine_sha256'] != reading['libraries_sha256']:
        raise ValueError('the experiment must use the same qualified native engine')
    import native_events
    fd = factory.memory_model(ROOT / identity['seed']['file'], identity['seed']['sha256'])
    try:
        factory.apply_changes(fd, tensors, changes)
        fcntl.fcntl(fd, factory.F_ADD_SEALS, factory.F_SEAL_WRITE | factory.F_SEAL_SEAL)
        results = {}
        for mode, cases_binding in (('held', fixed_capture['cases']), ('fresh', fresh_capture['cases'])):
            folder = HERE / mode
            factory.capture(fd, folder, identity, ROOT / cases_binding['record'], engine, inputs)
            raw = folder / 'raw' / inputs['member']
            count = baseline['events']
            answer_binding = None
            ended = None
            if mode == 'fresh':
                case = fresh_cases[0]
                member = dict(id=case['id'], tokens=case['tokens'], question=len(case['tokens']) - 1)
                answer = native_events.answer_of(member, json.loads((raw / 'result.json').read_text()), case['max_tokens'])
                factory.save(folder / 'answer.json', answer)
                answer_binding = held.binding(folder / 'answer.json')
                count = member['question'] + answer['steps']
                ended = answer['ended_by_itself']
            with torch.no_grad():
                native = held.raw_forward(raw, reading['merges'], count, reading['width'])
                correspondence = pairs if mode == 'held' else held.turn_correction.correspondence(reference, native)
                measured = held.turn_correction.measurement(reference, native, correspondence)
                mapped = loss_map(reference, native, correspondence, folder / 'residual-squared.npy')
            with (folder / 'pairs.npy').open('xb') as file:
                np.save(file, correspondence, allow_pickle=False)
            control_metrics = {key: (value['measurement'] if 'measurement' in value else value['fixed_correspondence'])
                               for key, value in controls.items() if key.startswith(mode)}
            result = dict(model_identity=identity, measurement=measured, computational_events=count, merges=len(native['merged']),
                reference=original['reference'], correspondence=held.binding(folder / 'pairs.npy'), capture=held.binding(folder / 'capture.json'),
                acquisition=answer_binding, acquisition_ended_by_itself=ended,
                scope=('complete saved computational acquisition' if mode == 'held' else
                       'complete naturally ended fresh path' if ended else 'partial fresh path: original acquisition limit reached'),
                residual_squared=mapped, map_aggregation_roundoff=mapped['sum'] - measured['objective'],
                controls=control_metrics, loss_change_from_controls={key: measured['objective'] - value['objective'] for key, value in control_metrics.items()})
            factory.save(folder / 'measurement.json', result)
            results[mode] = result
            print(json.dumps(dict(mode=mode, measurement=measured, computational_events=count,
                                  scope=result['scope'], acquisition_ended_by_itself=ended,
                                  loss_change_from_controls=result['loss_change_from_controls'])), flush=True)
            del native
        held.frozen(inputs)
        factory.save(HERE / 'result.json', dict(format='one-candidate-native-held-and-fresh/1', results=results,
            candidate=declaration['candidate'], rule=declaration['candidate_rule'], scale_record=declaration['scale_record'],
            controls=declaration['controls'], model_disk_copies=0, current_or_original_writes=0,
            fresh_scope_complete=results['fresh']['acquisition_ended_by_itself'],
            interpretation='One prospectively specified BF16 candidate; fresh computational improvement is primary. Held and fresh losses stay separate; no token/output judge or zero-loss requirement.'))
    finally:
        os.close(fd)


if __name__ == '__main__':
    main()
