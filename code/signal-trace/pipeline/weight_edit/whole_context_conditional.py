"""Score original weights on a frozen candidate's supplied histories.

The primary NativeReading and comparison contracts are unchanged. No alignment,
model, capture or target is made here; cached primary scores remain primary.
"""
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from pipeline.paths import ROOT
from pipeline.native_records.run_declaration import check_native_run
from pipeline.native_records.native_trace import checkpoint_values
from pipeline.weight_edit import context_targets as c, gap_targets
from pipeline.weight_edit.whole_context_score import NativeReading, _json, _local, _layout, score
from pipeline.weight_edit.whole_context_compare import (
    CAPTURE_EXECUTION, REGISTRATION, comparison_protocol, verify_correspondence_files,
    _bound, _source,
)

FORM = 'whole-gap-conditional-correspondence/1'
SAME = ('current_text_sha256', 'original_text_sha256', 'steps', 'original_steps',
        'pairs_sha256', 'answer_json_sha256', 'original_computation', 'pairs')


def require(condition, message):
    if not condition:
        raise ValueError(message)


class ConditionalReading:
    """Complete supplied histories, explicitly not certified as own generation."""

    array = NativeReading.array

    def __init__(self, folder):
        self.folder = Path(folder).resolve()
        self.record, self.sha256 = _json(self.folder/'reading.json')
        r = self.record
        require(r.get('format') == 'whole-gap-reading/1', 'Not a native state reading')
        identity = ('model_sha256', *CAPTURE_EXECUTION, 'fusion_disabled', 'recorder_arguments')
        require(all(k in r and r[k] not in (None, '') for k in identity), 'Missing conditional execution identity')
        self.execution = {k:r[k] for k in identity if k != 'model_sha256'}
        require(r['row_map']['file'] == 'row-map.json', 'Unexpected conditional row map')
        self.row_map, _ = _json(_local(self.folder,'row-map.json'), r['row_map']['sha256'])
        self.answer_rows = _layout(self.row_map)
        self.answers = self.row_map['answers']
        require(r['rows'] == self.row_map['rows'], 'Conditional row count differs')
        ids = [a['id'] for a in self.answers]
        require(r['answers_read'] == ids == [a['id'] for a in r['table']], 'Incomplete conditional history inventory')
        self.histories, self.table = {}, {}
        conditional = 0
        for row, table in zip(self.answers,r['table']):
            name = row['id']
            answer, _ = _json(_local(self.folder,f'answers/{name}/answer.json'),table['answer_json_sha256'])
            require(all(answer[k] == row[k] for k in ('id','steps','question','prompt_sha256','text_sha256')),
                    'Conditional answer and row identity differ')
            require(len(answer['tokens']) == row['steps'] and gap_targets.of_tokens(answer['tokens']) == row['text_sha256'],
                    'Conditional token identity differs')
            first, count = answer['first_step_not_its_own'], answer['steps_not_its_own']
            require(isinstance(count,int) and not isinstance(count,bool) and 0 <= count <= row['steps'],
                    'Invalid conditional departure count')
            require((first is None and count == 0) or
                    (isinstance(first,int) and not isinstance(first,bool) and 0 <= first < row['steps'] and 0 < count <= row['steps']-first),
                    'Invalid conditional first departure')
            require(table['first_step_not_its_own'] == first and table['steps_not_its_own'] == count
                    and table['the_whole_answer_is_this_models_own'] is (first is None),
                    'Conditional departure evidence differs')
            conditional += first is not None
            self.histories[name], self.table[name] = answer, table
        require(r['answers_whose_text_is_not_this_models_own'] == conditional, 'Conditional aggregate differs')
        # Endings belong to the candidate that generated these histories. The
        # original's choice after a supplied suffix is not that candidate's ending.


def shared_check(conditional, original, checked):
    """Bind the owner's completed byte comparison, without repeating its pass."""
    expected = dict(format='whole-gap-conditional-check/1', control_reading_sha256=conditional.sha256,
                    original_control_reading_sha256=original.sha256,
                    one_model_on_one_execution=dict(model_sha256=original.record['model_sha256'],**conditional.execution),
                    rows_of_the_control=conditional.record['rows'], merges_compared=len(original.record['merges']),
                    answers=len(original.answers), the_same_bits_everywhere=True,
                    merges_with_a_row_not_the_same_bits=[])
    require(all(checked.get(k) == v for k,v in expected.items()), 'Shared-input check is failed, incomplete or unbound')
    if conditional.execution != original.execution:
        engines = checked.get('the_two_engines', {})
        require(engines.get('original_control_libraries_sha256') == original.execution['libraries_sha256']
                and engines.get('control_engine_sha256') == conditional.execution['libraries_sha256'],
                'Shared-input check does not bind both engine identities')
    table, answer_count, question_count = [], 0, 0
    for now, before in zip(conditional.answers,original.answers):
        name = now['id']
        require((now['id'],now['prompt_sha256'],now['question']) ==
                (before['id'],before['prompt_sha256'],before['question']), 'Shared-input prompts differ')
        given, old = conditional.histories[name]['tokens'], original.histories[name]['tokens']
        least = min(len(given),len(old))
        leaves = next((t for t in range(least) if given[t] != old[t]), None if len(given)==len(old) else least)
        count = least if leaves is None or leaves >= least else leaves+1
        first = conditional.histories[name]['first_step_not_its_own']
        table.append(dict(id=name,first_step_not_the_recordings=leaves,first_step_not_its_own=first,
                          the_same_step=first==leaves,answer_rows_compared=count,
                          steps=len(given),recorded_steps=len(old)))
        answer_count += count
        question_count += now['question']
    require(checked['table'] == table and all(t['the_same_step'] for t in table),
            'Shared-input first-departure coverage differs')
    counts = dict(answer_rows_compared=answer_count, question_rows_compared=question_count,
                  rows_compared=answer_count+question_count,
                  answers_whose_first_step_not_its_own_is_where_the_text_leaves_the_recording=len(table))
    require(all(checked.get(k) == v for k,v in counts.items()), 'Shared-input row coverage differs')
    require(set(checked['terminal']) == {'l_out','result_norm'}, 'Shared-input terminal inventory differs')
    for name,n in (('l_out',answer_count+question_count),('result_norm',answer_count)):
        require(checked['terminal'][name] == dict(rows_compared=n,rows_not_the_same_bits=0,largest_difference=0.),
                'Shared-input terminal check failed')


def cached_score(held, reading, targets):
    expected = dict(format='whole-gap-score/1', reading_sha256=reading.sha256,
                    model_sha256=reading.record['model_sha256'], target_manifest_sha256=targets.sha256,
                    execution=reading.execution, complete=True, merges=len(targets.nodes),
                    reference_rows=targets.rows, comparisons_per_channel=targets.rows*len(targets.nodes))
    require(all(held.get(k) == v for k,v in expected.items()), 'Cached primary score identity or coverage differs')
    require(set(held['members']) == {a['id'] for a in targets.answers}, 'Cached primary member inventory differs')
    for a in targets.answers:
        member = held['members'][a['id']]
        require(member['complete'] and member['correct'] == a['correct']
                and member['reference_comparisons'] == (a['steps']+a['question'])*len(targets.nodes),
                'Cached primary member coverage differs')


def components(before, conditional, after):
    def channels(x):
        return dict(direction=x['direction'],size=x['size'],total=x['direction']+x['size'])
    b,h,a = map(channels,(before,conditional,after))
    return dict(original=b,conditional=h,candidate=a,
                history_change={k:h[k]-b[k] for k in b},
                weight_change={k:a[k]-h[k] for k in b},
                total_change={k:a[k]-b[k] for k in b})


def compare_conditional(*, original, candidate, conditional, targets, normalization,
                        mapping, primary, primary_sha256, protocol, protocol_sha256,
                        check, capture, original_score, candidate_score):
    require(mapping.get('format') == FORM, 'Not a conditional correspondence')
    held = comparison_protocol(protocol,protocol_sha256,original,conditional,targets.sha256,
                               candidate.record['model_sha256'],'recordings')
    require(original.execution == held.get('original_execution', held['execution'])
            and conditional.execution == candidate.execution == held['execution']
            and str(held['execution']['visible']) == '0', 'Conditional comparison execution differs')
    require(conditional.record['model_sha256'] == original.record['model_sha256'] != candidate.record['model_sha256'],
            'Conditional comparison requires original weights on another model history')
    require(conditional.answers == candidate.answers, 'Conditional rows are not the complete candidate histories')
    require([a['id'] for a in original.answers] == [a['id'] for a in targets.answers]
            and all(all(a[k]==b[k] for k in ('text_sha256','prompt_sha256','steps','question'))
                    for a,b in zip(original.answers,targets.answers)), 'Original reference histories differ')
    require(all(conditional.table[a['id']]['limit'] == table['limit']
                for a,table in zip(candidate.answers,candidate.record['table'])), 'Conditional generation bounds differ')
    expected = dict(basis='original-computation', reading_sha256=conditional.sha256,
                    reading_model_sha256=original.record['model_sha256'],target_manifest_sha256=targets.sha256,
                    execution={k:conditional.record[k] for k in CAPTURE_EXECUTION},
                    answers_whose_text_is_not_the_originals_own=conditional.record['answers_whose_text_is_not_this_models_own'])
    require(all(mapping.get(k) == v for k,v in expected.items()), 'Conditional correspondence identity differs')
    require(primary.get('format') == 'whole-gap-correspondence/2'
            and primary.get('basis') == 'original-computation'
            and primary['reading_sha256'] == candidate.sha256
            and primary['model_sha256'] == candidate.record['model_sha256']
            and primary['target_manifest_sha256'] == targets.sha256
            and primary['execution'] == expected['execution'], 'Primary correspondence identity differs')
    require(mapping['primary_correspondence'] == dict(format=primary['format'],sha256=primary_sha256,
                reading_sha256=candidate.sha256,protocol_sha256=primary['protocol']['sha256']),
            'Conditional mapping is not bound to the primary correspondence')
    require(mapping['history']['model_sha256'] == candidate.record['model_sha256']
            and mapping['history']['capture_text_json_sha256'] == primary['capture']['text_json_sha256']
            and mapping['capture'] == primary['capture'], 'Conditional capture provenance differs')
    require(mapping['protocol']['sha256'] == protocol_sha256
            and mapping['protocol']['original_control_reading_sha256'] == original.sha256
            and mapping['protocol']['execution'] == held['execution']
            and primary['protocol']['original_control_reading_sha256'] == original.sha256,
            'Conditional protocol or original control differs')
    require(all(primary['capture'][k] == held['registration'][k] for k in REGISTRATION),
            'Original registration provenance differs')
    changed = [a['id'] for a,b in zip(candidate.answers,targets.answers) if a['text_sha256'] != b['text_sha256']]
    unchanged = [a['id'] for a in targets.answers if a['id'] not in changed]
    require(set(mapping['answers']) == set(primary['answers']) == set(changed)
            and mapping['unchanged'] == primary['unchanged'] == unchanged, 'Changed/unchanged inventory differs')
    for name in changed:
        a,b = mapping['answers'][name],primary['answers'][name]
        require(all(a[k] == b[k] for k in SAME), 'Conditional correspondence changed the primary mapping')
        expected_original = held['original_computation'].get(name)
        require(expected_original and all(b['original_computation'].get(k) == expected_original[k] for k in ('record','sha256')),
                'Conditional mapping uses another original computation')
        answer = conditional.histories[name]
        require(a['first_step_not_the_originals_own'] == answer['first_step_not_its_own']
                and a['steps_not_the_originals_own'] == answer['steps_not_its_own']
                and a['the_whole_text_is_the_originals_own'] is (answer['first_step_not_its_own'] is None),
                'Conditional mapping departure evidence differs')
    verify_correspondence_files(primary,capture,candidate)
    shared_check(conditional,original,check)
    cached_score(original_score,original,targets)
    cached_score(candidate_score,candidate,targets)
    native = score(conditional,targets,normalization,mapping)
    native.update(format='whole-gap-conditional-score/1', condition=mapping.get('condition','original weights on supplied candidate histories'),
                  interpretation='Conditional field error, not an own-answer score or a replacement primary result')
    require(native['complete'], 'Conditional field is unresolved; no complete decomposition')
    members = {}
    groups = {}
    for a in targets.answers:
        name = a['id']
        members[name] = dict(correct=a['correct'],**components(original_score['members'][name],
                              native['members'][name],candidate_score['members'][name]))
        group = groups.setdefault('correct' if a['correct'] else 'wrong',{})
        for term,values in members[name].items():
            if term == 'correct':
                continue
            total = group.setdefault(term,dict(direction=0.,size=0.,total=0.))
            for channel,value in values.items():
                total[channel] += value
    report: dict[str, Any] = dict(format='whole-gap-conditional-comparison/1',
        prediction_vs_result=dict(complete=True,reference_rows=targets.rows,conditional_rows=conditional.record['rows'],
            merges=len(targets.nodes),comparisons_per_channel=native['comparisons_per_channel'],
            original_total=original_score['total'],conditional_total=native['total'],candidate_total=candidate_score['total'],
            members=members,groups=groups),
        failures_only=dict(limits=[
            'Ordered conditional contrast, not a unique causal share or history-independent repair number.',
            'Correspondence is fixed to the primary capture; complete coverage does not prove semantic equivalence.',
            'The fields are experimental targets; lower field error alone is not capability repair.',
            'Fitting members only; no no-fit or sealed data. Neither primary result is regraded.']),
        what_this_changes='Separates the observed field-loss change into history under original weights and weights on that fixed history.')
    return report,native


def main(folder):
    run = Path(folder).resolve()
    require(run.is_relative_to(ROOT/'studies/04-round-2-overlap/runs'), 'Conditional scoring needs its own run')
    d = check_native_run(run,run/'declaration.json')
    require(d.get('operation') == 'whole-context-conditional-score' and d['method_step'] == 5, 'Wrong conditional operation')
    for bound in d['implementation']:
        require(c.binding(ROOT/bound['record'],ROOT) == bound, 'Conditional scoring implementation changed')
    require(not (run/'conditional-score.json').exists() and not (run/'result.json').exists(), 'Preserve existing scoring results')
    asked = d['conditional_score']
    records = {k:_bound(asked[k],d) for k in ('mapping','primary','protocol','check','original_score','candidate_score','primary_result')}
    targets = gap_targets.read(_source(asked['target'],d),ROOT)
    require(targets.sha256 == asked['target_sha256'], 'Declared conditional target changed')
    readings = []
    for key, loader in (('original_reading',NativeReading),('candidate_reading',NativeReading),('conditional_reading',ConditionalReading)):
        folder = _source(asked[key]+'/reading.json',d).parent
        reading = loader(folder)
        _source(asked[key]+'/row-map.json',d)
        for a in reading.answers:
            _source(asked[key]+'/answers/'+a['id']+'/answer.json',d)
        readings.append(reading)
    original,candidate,conditional = readings
    capture = _source(asked['capture']+'/own/text.json',d).parent.parent
    for a in candidate.answers:
        for name in ('answer.json','pairs-original.npy'):
            _source(asked['capture']+'/own/'+a['id']+'/'+name,d)
    require(original.record['model_sha256'] == d['model_sha256'], 'Declared original identity differs')
    bindings = records.pop('primary_result')['bindings']
    for key,entry in (('control_score','original_score'),('candidate_score','candidate_score'),('correspondence','primary')):
        require(bindings[key] == asked[entry], 'Cached scores no longer bind the primary result')
    require(bindings['candidate_reading_sha256'] == candidate.sha256
            and bindings['control_reading_sha256'] == original.sha256
            and bindings['target_manifest_sha256'] == targets.sha256
            and bindings['protocol']['sha256'] == records['primary']['protocol']['sha256'],
            'Primary result provenance differs')
    normal = asked['normalization']
    tensors,reference = _bound(normal['tensors'],d),_bound(normal['reference'],d)
    terminal = reference['terminal']
    model = _source(normal['model'],d)
    weight = checkpoint_values(model,tensors[terminal['output_norm']]).astype('<f8').ravel()
    require(hashlib.sha256(weight.tobytes()).hexdigest() == terminal['output_norm_sha256'], 'Original output normalization changed')
    name,scale = next(iter(terminal['layer_output_scale'].items()))
    require(checkpoint_values(model,tensors[name]).ravel().tolist() == [scale], 'Original terminal scale changed')
    report,native = compare_conditional(original=original,candidate=candidate,conditional=conditional,targets=targets,
        normalization=dict(weight=weight,scale=scale,eps=terminal['eps']),capture=capture,
        primary_sha256=asked['primary']['sha256'],protocol_sha256=asked['protocol']['sha256'],**records)
    c.write_json(run/'conditional-score.json',native)
    report['bindings'] = dict(declaration=c.binding(run/'declaration.json',ROOT),**asked,
                             conditional_score=c.binding(run/'conditional-score.json',ROOT))
    c.write_json(run/'result.json',report)
    print(json.dumps(dict(run=run.name,**{k:v for k,v in report['prediction_vs_result'].items() if k!='members'})),flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
