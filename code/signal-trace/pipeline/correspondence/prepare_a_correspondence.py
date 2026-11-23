"""Materialize A's reviewed construction response index from saved executions.

This reads no fresh evaluation answer and fits no edit. Native capture execution
IDs are attached later from each completed capture's audit, not assumed from a
token index or the pilot's microbatch count.
"""
import hashlib
import json
from pathlib import Path

from pipeline.acquisition.run_a_baseline import ROOT, RUN
from pipeline.correspondence.solution_correspondence import annotate, correspond


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def prepare(output):
    cohort = json.loads((ROOT/'research/experiments/signal-trace-repair/A-cohort.json').read_text())
    output.mkdir(exist_ok=False)
    (output/'cases').mkdir()
    (output/'pairs').mkdir()
    cases, annotations = {}, {}
    for case in cohort['cases']:
        cid = case['id']
        source = ROOT/case['baseline']/'result.json'
        prompt_source = ROOT/case['input']
        assert hashlib.sha256(prompt_source.read_bytes()).hexdigest() == case['input_sha256']
        prompt = json.loads(prompt_source.read_text())[0]
        result = json.loads(source.read_text())
        item = annotate(result, prompt_tokens=prompt['tokens'])
        item.update(id=cid, role=case['role'], source=str(source.relative_to(ROOT)),
                    gpu=case.get('gpu', 0), baseline_execution=case.get('baseline_execution'),
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    prompt_source=str(prompt_source.relative_to(ROOT)),
                    prompt_source_sha256=case['input_sha256'])
        write(output/'cases'/(cid+'.json'), item)
        cases[cid] = dict(role=case['role'], baseline=case['baseline'], capture=case['capture'],
                          gpu=case.get('gpu', 0), baseline_execution=case.get('baseline_execution'),
                          input=case['input'], decisions=len(item['tokens']),
                          annotation='cases/'+cid+'.json', unknown_sections=[s for s in item['sections'] if not s['automatic']])
        annotations[cid] = item
    specs = []
    references = {}
    eligible_roles = ('failure', 'natural_numerical_success')
    for case in cohort['cases']:
        if case['role'] in eligible_roles:
            references.setdefault((case.get('gpu', 0), case['role']), case['id'])
    for cid in ('block-changed', 'block-base'):
        case = cases[cid]
        references[case['gpu'], case['role']] = cid
    for case in cohort['cases']:
        if case['role'] not in eligible_roles:
            continue
        reference = references[case.get('gpu', 0), case['role']]
        specs.append((reference, case['id'], case['role']+'_overlay'))
    for pair in cohort['natural_comparisons']:
        specs.append((pair['failure'], pair['natural_success'], 'natural_failure_success'))
    for left, right in cohort['successful_context_controls']:
        specs.append((left, right, 'successful_context_contrast'))
    summaries = []
    for index, (left, right, relation) in enumerate(specs):
        pair = correspond(annotations[left], annotations[right])
        pair.update(left=left, right=right, relation=relation,
                    left_gpu=cases[left]['gpu'], right_gpu=cases[right]['gpu'],
                    same_hardware=cases[left]['gpu'] == cases[right]['gpu'])
        name=f'{index:03}-{left}--{right}.json'
        write(output/'pairs'/name, pair)
        summaries.append(dict(left=left, right=right, relation=relation, file='pairs/'+name,
                              left_gpu=pair['left_gpu'], right_gpu=pair['right_gpu'],
                              same_hardware=pair['same_hardware'],
                              matched=len(pair['predictions']), unmatched_left=len(pair['unmatched_left']),
                              unmatched_right=len(pair['unmatched_right']), same_prompt=pair['same_prompt']))
    report = dict(cases=cases, pairs=summaries,
                  method='research/experiments/signal-trace-repair/A-GROUP-CORRESPONDENCE.md',
                  scope='Response correspondence only; no state recurrence or edit fit',
                  fresh_evaluation_used=False)
    write(output/'index.json', report)
    print(json.dumps(dict(cases=len(cases), pairs=len(summaries),
                          unknown_sections=sum(len(c['unknown_sections']) for c in cases.values()),
                          output=str(output)), indent=2))


if __name__ == '__main__':
    prepare(RUN/'response-correspondence')
