"""Declaration-bound full native comparison; local solver scores accept nothing."""
import json
import sys
from pathlib import Path
from typing import Any

from pipeline.paths import ROOT
from pipeline.native_records.run_declaration import check_native_run
from pipeline.native_records.native_trace import checkpoint_values
from pipeline.weight_edit import context_targets as c, gap_targets
from pipeline.weight_edit.whole_context_score import NativeReading, score

CAPTURE_EXECUTION = ('recorder_sha256', 'libraries_sha256', 'card', 'driver', 'visible')
REGISTRATION = ('cohort', 'reference_sha256', 'registration', 'tube_sha256', 'the_text_before')


def verify_correspondence_files(record, capture, reading):
    """Hold the consumed inline path to the actual producer-closed files too."""
    import hashlib
    from io import BytesIO
    import numpy as np
    own = Path(capture)/'own'
    raw = (own/'text.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != record['capture']['text_json_sha256']:
        raise ValueError('Capture record changed after correspondence publication')
    text = json.loads(raw)
    if text['model_sha256'] != record['model_sha256']:
        raise ValueError('Correspondence and producer model differ')
    if any(text[k] != record['capture'][k] for k in REGISTRATION) or any(text[k] != record['execution'][k] for k in CAPTURE_EXECUTION):
        raise ValueError('Inline correspondence provenance differs from its producer')
    for answer in reading.answers:
        name = answer['id']
        bound = text['files'][name]
        pair_bytes = (own/name/'pairs-original.npy').read_bytes()
        answer_bytes = (own/name/'answer.json').read_bytes()
        if hashlib.sha256(pair_bytes).hexdigest() != bound['pairs-original.npy'] or hashlib.sha256(answer_bytes).hexdigest() != bound['answer.json']:
            raise ValueError('Correspondence files differ from the producer-closed bytes')
        if gap_targets.of_tokens(json.loads(answer_bytes)['tokens']) != answer['text_sha256']:
            raise ValueError('Producer answer is not the native reading history')
        pairs = np.load(BytesIO(pair_bytes), allow_pickle=False)
        if name in record['answers']:
            supplied = record['answers'][name]
            if (supplied['pairs_sha256'] != bound['pairs-original.npy']
                    or supplied['answer_json_sha256'] != bound['answer.json']
                    or not np.array_equal(pairs, supplied['pairs'])
                    or supplied['original_computation'] != text['original_computation'][name]):
                raise ValueError('Inline path or original computation differs from its producer')
        elif not np.array_equal(pairs, np.column_stack([np.arange(answer['steps'])]*2)):
            raise ValueError('Unchanged history requires producer identity pairs')


def _engine_accounts(reading, engine, validator_sha256):
    import importlib.util
    path = Path(__file__).resolve().parents[4]/'records/development/2026-11-12-path-forward/recorder_engine.py'
    if c.sha256(path) != validator_sha256:
        raise ValueError('Engine account validator differs from the frozen protocol')
    spec = importlib.util.spec_from_file_location('comparison_recorder_engine', path)
    if spec is None or spec.loader is None:
        raise ValueError('Missing acquisition engine validator')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.held(reading.record.get('engine'), engine, 'reading')


def _capture_execution(source, candidate):
    original = source['execution']
    if source['format'] == 'whole-gap-reader-protocol/1':
        if 'capture_execution' in source or 'the_two_executions' in source:
            raise ValueError('Legacy protocol cannot declare a second execution')
        return original
    try:
        declared = source['capture_execution']
        capture = {key: declared[key] for key in original}
        engine, bridge = declared['engine'], source['the_two_executions']
        difference = {key: dict(control=original[key], capture=capture[key])
                      for key in original if original[key] != capture[key]}
        same = {key for key in original if original[key] == capture[key]}
        if (set(difference) != {'libraries_sha256'} or bridge['differ'] != difference
                or set(bridge['the_same']) != same
                or engine['sha256'] != capture['libraries_sha256']
                or engine['installed']['libraries_sha256'] != original['libraries_sha256']
                or bridge['libraries_that_are_other_files'] != engine['apart_from_the_installed_engine']):
            raise ValueError('Protocol does not bind the qualified library-only execution transition')
        evidence = bridge['rests_on']
        folder = (ROOT/evidence['run']).resolve()
        if not folder.is_relative_to(ROOT.resolve()):
            raise ValueError('Qualification is outside the repository')
        records = {}
        for name, key in (('declaration.json', 'declaration_sha256'), ('result.json', 'result_sha256'),
                          ('compare/kept.json', 'comparison_sha256')):
            path = folder/name
            if c.sha256(path) != evidence[key]:
                raise ValueError('Qualification differs from the frozen protocol')
            records[name] = json.loads(path.read_text())
        declaration, result, paired = (records[name] for name in ('declaration.json', 'result.json', 'compare/kept.json'))
        made = declaration['implementation']
        if (declaration.get('admission', {}).get('by') != 'weight-research-astra'
                or not str(result.get('status', '')).startswith('finished')
                or not paired['answers_read'] or paired['same_bytes'] != paired['answers_read'] or paired['apart']
                or evidence['answers_read'] != paired['answers_read']
                or evidence['answers_of_the_same_bytes'] != paired['same_bytes']
                or declaration['model_sha256'] != evidence['model_sha256']
                or declaration['model_sha256'] != source['original_control']['model_sha256']
                or made['recorder_sha256'] != capture['recorder_sha256']
                or made['kept_library_sha256'] != engine['overlay']['sha256']
                or made['kept_library_sha256'] != engine['libraries']['libggml-base.so.0']['sha256']
                or made['installed_library_sha256'] != engine['apart_from_the_installed_engine']['libggml-base.so.0']['installed']['sha256']):
            raise ValueError('Qualification does not cover these executions and original model')
    except (KeyError, TypeError, OSError) as error:
        raise ValueError('Incomplete qualified execution transition') from error
    _engine_accounts(candidate, engine, source.get('instruments', {}).get(
        'records/development/2026-11-12-path-forward/recorder_engine.py'))
    return capture


def comparison_protocol(source, sha256, control, candidate, target_sha256, candidate_sha256, history_before):
    """Use the acquisition owner's protocol, not a second competing protocol file."""
    if source.get('format') not in ('whole-gap-reader-protocol/1', 'whole-gap-reader-protocol/2'):
        raise ValueError('Not the admitted reader protocol format')
    execution = _capture_execution(source, candidate)
    if control.execution != source['execution'] or candidate.execution != execution:
        raise ValueError('Reading differs from its declared control or capture execution')
    original = source['original_control']
    expected = dict(reading_sha256=control.sha256, model_sha256=control.record['model_sha256'],
                    rows=control.record['rows'], merges=len(control.record['merges']),
                    answers=[a['id'] for a in control.answers])
    if any(original.get(k) != v for k,v in expected.items()) or original.get('made_with') != control.record.get('made_with'):
        raise ValueError('Protocol names another original control or its production metadata')
    files = source['instruments']
    for name in ('states_reading.py','whole_reading.py','reading_identity.py'):
        matches = [value for path,value in files.items() if path.endswith('/'+name)]
        if len(matches) != 1 or candidate.record.get('made_with',{}).get(name) != matches[0]:
            raise ValueError('Candidate was made by another reader than the protocol binds')
    tube = [value for path,value in files.items() if path.endswith('/pipeline/weight_edit/tube.py')]
    if len(tube) != 1:
        raise ValueError('Protocol must bind one registration implementation')
    return dict(original_model_sha256=original['model_sha256'], candidate_model_sha256=candidate_sha256,
                target_manifest_sha256=target_sha256, execution=execution,
                **(dict(original_execution=source['execution']) if source['format'] == 'whole-gap-reader-protocol/2' else {}),
                reader_protocol_sha256=sha256, original_control_reading_sha256=control.sha256,
                registration={**{k:source[k] for k in ('cohort','reference_sha256','registration')},
                              'tube_sha256':tube[0], 'the_text_before':history_before},
                original_computation=source['original_computation'])


def compare(control, candidate, targets, normalization, protocol, control_score, correspondence=None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compare one frozen proposal with an already-scored, hash-bound control.

    The caller validates the protocol/cache files' hashes. Cached control arrays
    need not be restored or scored again; their metadata identity must agree.
    """
    for reading, model, execution in (
            (control, protocol['original_model_sha256'], protocol.get('original_execution', protocol['execution'])),
            (candidate, protocol['candidate_model_sha256'], protocol['execution'])):
        if reading.record['model_sha256'] != model:
            raise ValueError('Reading is not the declared original or frozen candidate')
        if reading.execution != execution:
            raise ValueError('Reading is not the admitted canonical execution')
    if targets.sha256 != protocol['target_manifest_sha256']:
        raise ValueError('Comparison uses another target')
    if str(protocol['execution']['visible']) != '0':
        raise ValueError('This comparison requires canonical card 0')
    expected_cache = dict(format='whole-gap-score/1', reading_sha256=control.sha256,
                          target_manifest_sha256=targets.sha256, model_sha256=control.record['model_sha256'],
                          execution=control.execution, complete=True, merges=len(targets.nodes),
                          reference_rows=targets.rows, comparisons_per_channel=targets.rows*len(targets.nodes))
    if any(control_score.get(key) != value for key, value in expected_cache.items()):
        raise ValueError('Cached control is not the complete score of this original reading and target')
    ids = [a['id'] for a in targets.answers]
    if set(control_score['members']) != set(ids) or any(not m['complete'] for m in control_score['members'].values()):
        raise ValueError('Cached control omits a complete fitting member')
    if [a['id'] for a in control.answers] != ids or any(
            any(a[key] != b[key] for key in ('text_sha256','prompt_sha256','steps','question'))
            for a,b in zip(control.answers,targets.answers)):
        raise ValueError('Control is not the original target histories')
    if [a['id'] for a in candidate.answers] != ids:
        raise ValueError('Candidate changed fitting membership or order')
    changed = [a['id'] for a,b in zip(candidate.answers,targets.answers) if a['text_sha256'] != b['text_sha256']]
    if changed or correspondence is not None:
        if correspondence is None or correspondence.get('format') != 'whole-gap-correspondence/2':
            raise ValueError('Changed-history comparison needs producer-bound correspondence/2')
        expected = dict(basis='original-computation', reading_sha256=candidate.sha256,
                        target_manifest_sha256=targets.sha256, model_sha256=candidate.record['model_sha256'],
                        execution={key:candidate.record[key] for key in CAPTURE_EXECUTION})
        if any(correspondence.get(key) != value for key,value in expected.items()):
            raise ValueError('Correspondence is not of the candidate, target and admitted capture execution')
        held_protocol = correspondence.get('protocol', {})
        if (held_protocol.get('sha256') != protocol['reader_protocol_sha256']
                or held_protocol.get('original_control_reading_sha256') != control.sha256):
            raise ValueError('Correspondence was not held to this declared reader protocol and control')
        capture = correspondence.get('capture', {})
        if not capture.get('text_json_sha256') or any(capture.get(k) != protocol['registration'][k] for k in REGISTRATION):
            raise ValueError('Registration is not bound to the declared original reference and producer')
        if set(correspondence.get('answers', {})) != set(changed) or correspondence.get('unchanged') != [i for i in ids if i not in changed]:
            raise ValueError('Correspondence changed/unchanged inventory differs')
        for name in changed:
            actual = correspondence['answers'][name].get('original_computation', {})
            expected_original = protocol['original_computation'].get(name)
            if not expected_original or any(actual.get(k) != expected_original[k] for k in ('record','sha256')):
                raise ValueError(f'{name}: correspondence used another or undeclared original computation')
    native = score(candidate, targets, normalization, correspondence)
    members, groups = {}, {}
    for original in targets.answers:
        name = original['id']
        before, after = control_score['members'][name], native['members'][name]
        members[name] = dict(correct=original['correct'], complete=after['complete'],
                             control={k:before[k] for k in ('direction','size')},
                             candidate={k:after[k] for k in ('direction','size')},
                             change={k:None if after[k] is None else after[k]-before[k] for k in ('direction','size')})
        group = groups.setdefault('correct' if original['correct'] else 'wrong', dict(control=0., candidate=0., complete=True))
        group['control'] += before['direction']+before['size']
        group['complete'] &= after['complete']
        if after['complete']:
            group['candidate'] += after['direction']+after['size']
    for group in groups.values():
        group['candidate'] = group['candidate'] if group['complete'] else None
        group['change'] = None if group['candidate'] is None else group['candidate']-group['control']
    report = dict(format='whole-gap-comparison/1',
        prediction_vs_result=dict(complete=native['complete'], control_total=control_score['total'], candidate_total=native['total'],
                                  total_change=None if native['total'] is None else native['total']-control_score['total'],
                                  members=members, groups=groups, changed_histories=changed,
                                  reference_rows=targets.rows, merges=len(targets.nodes), comparisons_per_channel=native['comparisons_per_channel']),
        failures_only=dict(incomplete_members=[k for k,v in members.items() if not v['complete']],
                           limits=['A complete monotonic correspondence is not proof of semantic equivalence.',
                                   'This is a fitting-cohort comparison under a constructed field, not no-fit transfer.',
                                   'Signed changes are reported without a numerical significance threshold.']),
        what_this_changes='The frozen proposal has a complete native comparison where resolved; completeness or lower target loss alone is not a repair verdict.')
    return report, native


def _source(name, declaration):
    path = (ROOT/name).resolve()
    if not any(path.is_relative_to((ROOT/entry).resolve()) for entry in declaration['inputs']):
        raise ValueError(f'Comparison source was not declared: {name}')
    return path


def _bound(record, declaration):
    path = _source(record['record'], declaration)
    if c.binding(path, ROOT) != record:
        raise ValueError(f'Comparison source binding changed: {record["record"]}')
    return json.loads(path.read_text())


def _output(run, name):
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Generated comparison paths must be relative to their own run')
    path = (run/relative).resolve()
    if not path.is_relative_to(run):
        raise ValueError('Generated comparison path leaves its run')
    return path


def main(folder):
    run = Path(folder).resolve()
    if not run.is_relative_to(ROOT/'studies/04-round-2-overlap/runs'):
        raise ValueError('Native comparison must belong to a declared run')
    declaration = check_native_run(run, run/'declaration.json')
    if declaration['method_step'] != 5:
        raise ValueError('Native comparison requires a new step-5 construction declaration')
    asked = declaration['native_comparison']
    for binding in asked['implementation']:
        if c.binding(ROOT/binding['record'], ROOT) != binding:
            raise ValueError('Comparison implementation changed since declaration')
    output = _output(run, asked['output'])
    if output.exists():
        raise FileExistsError('Preserve an existing or partial native comparison')
    targets = gap_targets.read(_source(asked['target'], declaration), ROOT)
    if targets.sha256 != asked['target_sha256']:
        raise ValueError('Target differs from the native declaration')
    freeze_path = _output(run, asked['sweep_record'])
    frozen = json.loads(freeze_path.read_text())
    if frozen['sweep']['original_sha256'] != declaration['model_sha256'] or frozen['sweep']['target_sha256'] != targets.sha256:
        raise ValueError('Frozen sweep used another original or target')
    names = frozen['sweep']['matrices']
    if not names or [m['matrix'] for m in frozen['matrices']] != names:
        raise ValueError('A partial sweep is not the declared completed proposal')
    last = frozen['last_state']
    if last['every_other_byte_is_the_originals'] is not True or frozen['matrices'][-1]['state_after'] != last['identity']:
        raise ValueError('Frozen final state is not the sweep records final state')
    source_protocol = _bound(asked['protocol'], declaration)
    if source_protocol['original_control']['model_sha256'] != declaration['model_sha256']:
        raise ValueError('Reader protocol is not bound to the declared original')
    for name, digest in source_protocol['instruments'].items():
        if c.sha256(ROOT/name) != digest:
            raise ValueError(f'Acquisition implementation changed since protocol: {name}')
    control = NativeReading(_source(source_protocol['original_control']['reading'], declaration))
    cached = _bound(asked['control_score'], declaration)
    candidate = NativeReading(_output(run, asked['candidate_reading']))
    protocol = comparison_protocol(source_protocol,asked['protocol']['sha256'],control,candidate,targets.sha256,
                                   last['model_sha256'],asked['history_before'])
    mapping_path = _output(run, asked['correspondence']) if asked.get('correspondence') else None
    correspondence = json.loads(mapping_path.read_text()) if mapping_path is not None else None
    if correspondence is not None:
        verify_correspondence_files(correspondence, _output(run, asked['capture']), candidate)
    normal = asked['normalization']
    tensors, reference = _bound(normal['tensors'], declaration), _bound(normal['reference'], declaration)
    from pipeline.weight_edit.gap_run import path_order
    if names != path_order(tensors) or frozen['sweep']['fitting'] != [m['id'] for m in targets.answers]:
        raise ValueError('Frozen proposal is not the full path and fitting inventory')
    if declaration['sweep']['rule'] != frozen['sweep']['rule']:
        raise ValueError('Frozen proposal used another construction rule')
    if declaration['sweep']['matrices'] != 'path order' and declaration['sweep']['matrices'] != names:
        raise ValueError('Frozen proposal differs from its declared matrix order')
    terminal = reference['terminal']
    model = _source(normal['model'], declaration)
    weight = checkpoint_values(model, tensors[terminal['output_norm']]).astype('<f8').ravel()
    import hashlib
    if hashlib.sha256(weight.tobytes()).hexdigest() != terminal['output_norm_sha256']:
        raise ValueError('Original output normalization changed')
    name, scale = next(iter(terminal['layer_output_scale'].items()))
    if checkpoint_values(model, tensors[name]).ravel().tolist() != [scale]:
        raise ValueError('Original terminal scale changed')
    report, native = compare(control,candidate,targets,dict(weight=weight,scale=scale,eps=terminal['eps']),protocol,cached,correspondence)
    c.write_json(output/'candidate-score.json', native)
    report['bindings'] = dict(declaration=c.binding(run/'declaration.json',ROOT), sweep=c.binding(freeze_path,ROOT),
                             protocol=asked['protocol'], control_reading_sha256=control.sha256, control_score=asked['control_score'],
                             candidate_reading_sha256=candidate.sha256, target_manifest_sha256=targets.sha256,
                             correspondence=None if mapping_path is None else c.binding(mapping_path,ROOT),
                             candidate_score=c.binding(output/'candidate-score.json',ROOT))
    c.write_json(output/'result.json', report)
    print(json.dumps(dict(complete=report['prediction_vs_result']['complete'], groups=report['prediction_vs_result']['groups'],
                          changed_histories=len(report['prediction_vs_result']['changed_histories']), result=str((output/'result.json').relative_to(ROOT)))),flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
