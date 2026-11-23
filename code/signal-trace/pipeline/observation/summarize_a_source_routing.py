"""Describe context-relative source routing along the recorded attention path.

Uses diagnostic probabilities reconstructed from native parents. The report
retains all heads, questions, controls and exclusions; selected regularities
are exploratory candidates for intervention, not causal findings.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from pipeline.native_records.group_observation import publish
from pipeline.acquisition.prepare_a_native_excerpts import OUTPUT


def main():
    cohort = json.loads(Path('research/experiments/signal-trace-repair/A-cohort.json').read_text())
    cases = {c['id']: c for c in cohort['cases']}
    observed = {p.parent.parent.name: json.loads(p.read_text())
                for p in OUTPUT.glob('*/attention-messages/index.json')}
    rows = {(cid, r['anchor'], r['layer']): r for cid, case in observed.items() for r in case['rows']}
    pairs = []
    controls = []
    for q in cohort['natural_comparisons']:
        a, b = q['failure'], q['natural_success']
        if cases[a]['role'] == 'failure' and cases[b]['role'] == 'natural_numerical_success' and cases[a]['gpu'] == cases[b]['gpu']:
            pairs.append((a, b))
    factors = json.loads((OUTPUT / 'factor-contrasts/index.json').read_text())['pairs']
    for q in factors:
        pair = (q['left'], q['right'])
        if q['relation'] == 'failure_success' and pair not in pairs:
            pairs.append(pair)
        elif q['relation'] == 'success_success':
            controls.append(pair)
    for a, b in cohort['successful_context_controls']:
        if cases[a]['gpu'] == cases[b]['gpu'] and (a, b) not in controls:
            controls.append((a, b))
    anchor = 'climb_reference_body'
    eligible = [(a, b) for a, b in pairs if (a, anchor, 0) in rows and (b, anchor, 0) in rows]
    eligible_controls = [(a, b) for a, b in controls if (a, anchor, 0) in rows and (b, anchor, 0) in rows]

    def observation(cid, layer, head):
        r = rows[cid, anchor, layer]
        groups = {g['group']: g for g in r['groups']}
        descent = groups['generated_descent']['probability_by_head'][head]
        climb = groups['prompt_climb']['probability_by_head'][head]
        if min(descent, climb) <= 0:
            raise ValueError('A finite observed source ratio is required; do not hide underflow with clipping')
        return dict(case=cid, descent=descent, climb=climb, log_ratio=float(np.log(descent / climb)),
                    event=r['event'], projection_event=r['projection_event'], projection_weight=r['projection_weight']['name'])

    heads = []
    for layer in range(35):
        for head in range(8):
            data = {c: observation(c, layer, head) for c in sorted({c for pair in eligible + eligible_controls for c in pair})}
            per_case = defaultdict(list)
            for a, b in eligible:
                per_case[a].append(dict(success=b, gap=data[a]['log_ratio'] - data[b]['log_ratio']))
            gap = np.array([np.mean([x['gap'] for x in v]) for v in per_case.values()])
            control = [dict(left=a, right=b, gap=data[a]['log_ratio'] - data[b]['log_ratio']) for a, b in eligible_controls]
            control75 = float(np.quantile([abs(x['gap']) for x in control], .75))
            median = float(np.median(gap))
            sign_count = int(np.sum(gap * np.sign(median) > 0))
            heads.append(dict(layer=layer, head=head, observations=list(data.values()),
                              failures=[dict(case=c, comparisons=v, mean_gap=float(np.mean([x['gap'] for x in v]))) for c, v in per_case.items()],
                              controls=control, median_gap=median, same_sign_cases=sign_count, cases=len(gap),
                              control_absolute_q75=control75,
                              exploratory_recurrence=sign_count == len(gap) and abs(median) > control75))
    out = OUTPUT / 'source-routing-summary'
    out.mkdir(exist_ok=False)
    publish(out / 'index.json', dict(anchor=anchor, eligible_pairs=eligible, eligible_controls=eligible_controls,
                                    missing_pairs=[p for p in pairs if p not in eligible], heads=heads,
                                    cases={cid: dict(role=c['role'], gpu=c['gpu'], parameters=c['parameters']) for cid, c in cases.items()},
                                    selected=[dict(layer=h['layer'], head=h['head']) for h in heads if h['exploratory_recurrence']],
                                    definition='log(probability assigned to the naturally generated descent passage / probability assigned to the prompt climb clause), at the prediction of climb in the literal body phrase. Failure minus naturally successful comparison; average mates within each failing question.',
                                    selection='Exploratory: all 14 available failing questions share the sign; median absolute gap exceeds the 75th percentile of absolute correct/correct context gaps. All 280 heads remain reported. This threshold is not a significance test.',
                                    limits=['A source probability is not causal importance or a capability label. Inspect the full value message, output projection and downstream path before attributing a repair.',
                                            'The source probabilities are derived from recorded Q/K/masks, not directly recorded fused-attention internals.',
                                            'Some comparisons change final height; same-height physical-factor comparisons and correct/correct height controls remain separately identifiable.',
                                            'Pair and question observations are dependent, and all selection is exploratory on this construction cohort.',
                                            'No new weight edits, generation or held-out evaluation.']))
    print(json.dumps(dict(output=str(out), pairs=len(eligible), failures=len({a for a, b in eligible}), controls=len(eligible_controls), selected=[(h['layer'], h['head']) for h in heads if h['exploratory_recurrence']])))


if __name__ == '__main__':
    main()
