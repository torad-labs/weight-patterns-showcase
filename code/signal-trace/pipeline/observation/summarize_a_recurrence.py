"""Summarize an observed connected fragment, with its counterexamples intact.

The rows are changes of direction at actual consecutive residual merges. This
does not identify a semantic coordinate or select a stored-weight intervention.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from pipeline.native_records.group_observation import publish
from pipeline.acquisition.prepare_a_native_excerpts import OUTPUT


def unit(values):
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-30)


def summarize():
    fields = json.loads((OUTPUT / 'turn-fields/index.json').read_text())
    values = np.load(OUTPUT / 'turn-fields/changes.npy', mmap_mode='r')
    indices = {(r['case'], r['anchor'], r['offset']): i
               for i, r in enumerate(fields['records'])}
    pairs = json.loads((OUTPUT / 'factor-contrasts/index.json').read_text())['pairs']
    out = OUTPUT / 'recurrence-summary'
    out.mkdir(exist_ok=False)
    contexts = []
    for anchor in ['after_loss_heading_end', 'destination_heading_end', 'climb_reference_body']:
        for offset in range(-4, 5):
            contrasts = []
            controls = []
            missing = []
            per_question = defaultdict(list)
            for pair_id, pair in enumerate(pairs):
                ia = indices.get((pair['left'], anchor, offset))
                ib = indices.get((pair['right'], anchor, offset))
                if ia is None or ib is None:
                    missing.append(pair_id)
                    continue
                difference = values[ia].astype('f8') - values[ib]
                if pair['relation'] == 'failure_success':
                    contrasts.append((pair_id, pair, difference))
                    per_question[pair['left']].append(difference)
                elif pair['relation'] == 'success_success':
                    controls.append((pair_id, difference))
            differences = np.stack([np.mean(v, axis=0) for v in per_question.values()])
            whole = unit(differences.reshape(len(differences), -1))
            direction = unit(whole.mean(axis=0))
            control = np.stack([v for _, v in controls])
            control_projection = np.abs(control.reshape(len(control), -1) @ direction)
            cross = []
            for pair_id, pair, delta in contrasts:
                used = {pair['left'], pair['right']}
                available = [(qid, other, d) for qid, other, d in contrasts
                             if not used.intersection([other['left'], other['right']])]
                row = dict(pair=pair_id, left=pair['left'], right=pair['right'], gpu=pair['gpu'], factor=pair['factor'])
                for name, candidates in [('disjoint', available), ('other_factor', [x for x in available if x[1]['factor'] != pair['factor']])]:
                    # Every distinct failing question has equal influence in
                    # the reference, even when it has several successful mates.
                    grouped = defaultdict(list)
                    for qid, other, d in candidates:
                        grouped[other['left']].append(d)
                    if grouped:
                        reference = unit(np.stack([unit(np.mean(v, axis=0).ravel()) for v in grouped.values()]).mean(axis=0))
                        row[name] = dict(cosine=float(unit(delta.ravel()) @ reference),
                                         pairs=[x[0] for x in candidates], failures=list(grouped))
                    else:
                        row[name] = None
                cross.append(row)
            branch_unit = unit(differences)
            branch_direction = unit(branch_unit.mean(axis=0))
            branch_projection = np.einsum('ncd,cd->nc', differences, branch_direction)
            branch_control = np.abs(np.einsum('ncd,cd->nc', control, branch_direction))
            loo = []
            for i, case in enumerate(per_question):
                ref = unit(np.delete(whole, i, axis=0).mean(axis=0))
                loo.append(dict(case=case, cosine=float(whole[i] @ ref)))
            stem = anchor + '_' + str(offset)
            np.save(out / (stem + '-differences.npy'), differences.astype('<f4'))
            np.save(out / (stem + '-direction.npy'), direction.reshape(105, 1536).astype('<f4'))
            contexts.append(dict(anchor=anchor, offset=offset, failures=list(per_question),
                                 contrast_pairs=[x[0] for x in contrasts], control_pairs=[x[0] for x in controls],
                                 missing_pairs=missing, whole_trace_coherence=float(np.linalg.norm(whole.mean(axis=0))),
                                 leading_gram_eigenvalue_fraction=float(np.linalg.eigvalsh(whole @ whole.T)[-1] / len(whole)),
                                 per_case_projection=(differences.reshape(len(differences), -1) @ direction).tolist(),
                                 control_absolute_projection=control_projection.tolist(), leave_one_failure_out=loo,
                                 disjoint_question_checks=cross, branch_coherence=np.linalg.norm(branch_unit.mean(axis=0), axis=1).tolist(),
                                 branch_projection=branch_projection.tolist(), branch_control_absolute_projection=branch_control.tolist(),
                                 differences=str(out / (stem + '-differences.npy')), direction=str(out / (stem + '-direction.npy'))))
    all_attention = [json.loads(p.read_text()) for p in OUTPUT.glob('*/attention-messages/index.json')]
    rows = [r for c in all_attention for r in c['rows']]
    reconstruction = {key: dict(median=float(np.median([r[key] for r in rows])),
                                maximum=float(max(r[key] for r in rows)))
                      for key in ['relative_attention_reconstruction_error', 'relative_projection_reconstruction_error']}
    publish(out / 'index.json', dict(contexts=contexts, pairs=pairs, connections=fields['connections'],
                                    source=str(OUTPUT / 'turn-fields/index.json'), attention_cases=len(all_attention),
                                    attention_observations=len(rows), reconstruction=reconstruction,
                                    definition='Each observation retains the ordered 105 full 1536-dimensional direction changes. Each contrast uses the same GPU, mass, descent and climb; speed or dissipation differs. Repeated mates are averaged within each failing question before the descriptive overlap.',
                                    limits=['These are conditional observational differences, not a discovered exclusive capability or a weight edit.',
                                            'Headings and literal phrases supply context, not the onset of internal calculation.',
                                            'Shared questions make pair counts dependent. Disjoint checks remove every reused question from the reference.',
                                            'The nine neighboring predictions are separate observations. Their local wording is not necessarily identical, especially after the anchor.',
                                            'Fused attention probabilities and source messages are diagnostic reconstructions, not recorded internal arithmetic.']))
    return out


if __name__ == '__main__':
    print(summarize())
