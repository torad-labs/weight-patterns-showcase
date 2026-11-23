"""Freeze the revised pooled-acquisition queue from retained question metadata."""
import argparse
import json
import math
from pathlib import Path

from pipeline.acquisition.single_card_plan import build_plan
from pipeline.native_records.run_declaration import check_native_run, digest
from pipeline.paths import ROOT, ACQUISITION, OBSERVATION, ROUND_TWO_RUNS


def prepare(run):
    run = Path(run).resolve()
    if run.parent != ROUND_TWO_RUNS.resolve():
        raise ValueError('Acquisition belongs in its own round-two run folder')
    output = run/'matched-design'
    check_native_run(output, run/'declaration.json')
    if output.exists():
        raise ValueError('Never overwrite a frozen acquisition design')
    cases = json.loads((run/'original-questions.json').read_text())
    source = json.loads((OBSERVATION/'response-correspondence/index.json').read_text())
    roles = {c['id']: source['cases'][c['id']]['role'] for c in cases}
    sources = [ACQUISITION/name for name in
               ['historical-cases.json', 'discovery-cases.json', 'expanded-cases.json', 'fresh-cases.json']]
    sources.append(ACQUISITION/'reserved-40b/cases.json')
    excluded = set()
    provenance = []
    for path in sources:
        questions = json.loads(path.read_text())
        excluded.update(c['prompt'] for c in questions)
        provenance.append(dict(path=str(path.relative_to(ROOT)), sha256=digest(path), questions=len(questions)))
    plan = build_plan(cases, roles, excluded)
    candidates = plan.pop('candidates')
    plan['physical_queue_count'] = len(candidates)
    plan['historical_conditional_assignment_count'] = math.prod(
        math.comb(2*r['matched_quota'], r['matched_quota']) for r in plan['cells'])
    plan['null_scope'] = ('This is the assignment count if every historical matched quota is filled, '
        'not a computed statistic null. The actual pooled held-out statistic and its within-cell '
        'label calibration must be computed on the frozen, common-card cohort before interpretation. '
        'The large joint assignment space requires a separately reviewed computational plan.')
    plan['roles_source'] = str((OBSERVATION/'response-correspondence/index.json').relative_to(ROOT))
    plan['source_roles'] = roles
    plan['source_case_order'] = [c['id'] for c in cases]
    plan['exclusions'] = provenance
    plan['generation'] = 'python3 -m pipeline.acquisition.prepare_single_card_plan --run '+str(run)
    output.mkdir()
    for name, value in [('plan.json',plan), ('candidate-questions.json',candidates)]:
        (output/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    summary = {key:plan[key] for key in ['physical_queue_count','initial_screening_ceiling',
        'initial_match_deficit','conditional_matched_per_class','historical_conditional_assignment_count']}
    print(json.dumps(summary))
    return plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROUND_TWO_RUNS/'acquisition-single-card')
    prepare(parser.parse_args().run)
