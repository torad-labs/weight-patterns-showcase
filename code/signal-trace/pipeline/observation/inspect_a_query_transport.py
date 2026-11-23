"""Follow a context-dependent routing change into its native value message.

This is a local diagnostic on recorded parents, not an inference intervention,
edited trajectory, or checkpoint. Its input-dependent query targets are saved
so that a later persistent, coupled weight construction can be audited.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from pipeline.native_records.group_observation import frames, publish
from pipeline.native_records.native_excerpts import ExcerptReader
from pipeline.native_records.native_trace import checkpoint_values
from pipeline.observation.observe_a_weight_transport import MODEL
from pipeline.acquisition.prepare_a_native_excerpts import OUTPUT
from pipeline.observation.query_transport import query, source_ratio, solve_query
from pipeline.acquisition.run_a_baseline import RUN


def epsilon(event):
    return float(np.array(event['output']['op_params_i32'], dtype='<i4').view('<f4')[0])


def normalize(x):
    return x / max(float(np.linalg.norm(x)), 1e-30)


def rms(x, gain, eps):
    return x * gain / np.sqrt(np.mean(x * x) + eps)


def main():
    summary = json.loads((OUTPUT / 'source-routing-summary/index.json').read_text())
    selected = [h for h in summary['heads'] if h['exploratory_recurrence']]
    failures = sorted({r['case'] for h in selected for r in h['failures']})
    needed = sorted(set(failures) | {b for a, b in summary['eligible_pairs']})
    out = OUTPUT / 'query-transport'
    out.mkdir(exist_ok=False)
    weight_cache = {}
    contexts = {}
    all_rows = []
    for cid in needed:
        root = OUTPUT / cid
        attention = json.loads((root / 'attention-messages/index.json').read_text())
        body = {r['layer']: r for r in attention['rows'] if r['anchor'] == 'climb_reference_body'}
        execution = next(iter(body.values()))['execution']
        reader = ExcerptReader(root / 'native')
        ann = json.loads((RUN / 'response-correspondence/cases' / (cid + '.json')).read_text())
        supplied = json.loads((RUN / 'cohort-inputs' / (cid + '.json')).read_text())
        supplied = supplied[0] if isinstance(supplied, list) else supplied
        roles = {s['index']: s['role'] for s in ann['sections']}
        descent_positions = {len(supplied['tokens']) + t['prediction'] for t in ann['tokens'] if roles.get(t['section']) == 'descent'}
        climb_positions = set(attention['prompt_clauses']['prompt_climb']['positions'])
        case_rows = []
        payload = {}
        try:
            for observed_execution, events in frames(root / 'events.jsonl.zst'):
                if observed_execution != execution:
                    continue
                by_event = {e['event']: e for e in events}
                by_name = {e['name']: e for e in events}
                for layer in sorted({h['layer'] for h in selected}):
                    row = body[layer]
                    event = by_event[row['event']]
                    qevent = by_name[f'Qcur-{layer}']
                    rotation_event = by_name[f'Qcur_pos-{layer}']
                    normed_event = by_name[f'Qcur_normed-{layer}']
                    norm_event = by_event[normed_event['inputs'][0]['producer_event']]
                    params = rotation_event['output']['op_params_i32']
                    dimension = rotation_event['output']['shape'][0]
                    if params[1] != dimension or params[2] != 2:
                        raise ValueError('This diagnostic requires the observed full-width NeoX rotation')
                    raw = reader.values(qevent['output']).reshape(8, dimension).astype('f8')
                    before = reader.values(normed_event['output']).reshape(8, dimension).astype('f8')
                    after = reader.values(rotation_event['output']).reshape(8, dimension).astype('f8')
                    half = dimension // 2
                    x, y = before[:, :half], before[:, half:]
                    u, v = after[:, :half], after[:, half:]
                    denominator = np.sum(x*x + y*y, axis=0)
                    if np.any(denominator <= 1e-20):
                        raise ValueError('Rotation is not determined by the recorded head vectors')
                    cosine = np.sum(x*u + y*v, axis=0) / denominator
                    sine = np.sum(x*v - y*u, axis=0) / denominator
                    rotation = (cosine, sine)
                    gain = checkpoint_values(MODEL, normed_event['inputs'][1]['weight']).astype('f8')
                    eps = epsilon(norm_event)
                    reconstructed = np.stack([query(x, gain, eps, rotation) for x in raw])
                    rotation_error = float(np.linalg.norm(reconstructed - after) / np.linalg.norm(after))
                    if rotation_error > 1e-6:
                        raise ValueError(f'Query transform reconstruction differs: {cid}, {layer}, {rotation_error}')
                    mask = reader.values(event['inputs'][3]['snapshot']).astype('f8')
                    valid = np.flatnonzero(np.isfinite(mask))
                    if np.any(mask[valid] != 0):
                        raise ValueError('A nonzero finite attention bias needs explicit inclusion in the source-ratio solver')
                    keys = reader.values(event['inputs'][1]['snapshot']).reshape(-1, dimension)[valid].astype('f8')
                    values = reader.values(event['inputs'][2]['snapshot']).reshape(-1, dimension)[valid].astype('f8')
                    positions = row['cache_positions']
                    descent = np.array([s in descent_positions for s in positions])
                    climb = np.array([s in climb_positions for s in positions])
                    scale = float(np.array(event['output']['op_params_i32'], dtype='<i4').view('<f4')[0])
                    logits = reconstructed @ keys.T * scale + mask[valid][None, :]
                    probability = np.exp(logits - logits.max(axis=1, keepdims=True))
                    probability /= probability.sum(axis=1, keepdims=True)
                    projection = by_event[row['projection_event']]
                    binding = row['projection_weight']
                    if binding['name'] not in weight_cache:
                        weight_cache[binding['name']] = checkpoint_values(MODEL, binding).reshape(binding['shape'][1], binding['shape'][0]).astype('f8')
                    weight = weight_cache[binding['name']]
                    actual_projection = reader.values(projection['output']).astype('f8')
                    post = by_name[f'attn_post_norm-{layer}']
                    post_norm = by_event[post['inputs'][0]['producer_event']]
                    post_gain = checkpoint_values(MODEL, post['inputs'][1]['weight']).astype('f8')
                    post_eps = epsilon(post_norm)
                    merge = by_name[f'attn_out-{layer}']
                    incoming = reader.values(merge['inputs'][1]['snapshot']).astype('f8')
                    branch = reader.values(post['output']).astype('f8')
                    merged = reader.values(merge['output']).astype('f8')
                    local = dict(raw=raw, gain=gain, eps=eps, rotation=rotation, keys=keys, values=values,
                                 descent=descent, climb=climb, scale=scale, probability=probability, weight=weight,
                                 projection=actual_projection, post_gain=post_gain, post_eps=post_eps,
                                 incoming=incoming, branch=branch, merged=merged)
                    contexts[cid, layer] = local
                    case_rows.append(dict(layer=layer, execution=execution, prediction=row['prediction'],
                                          query_event=qevent['event'], attention_event=row['event'], projection_event=row['projection_event'],
                                          query_binding=qevent['inputs'][0]['weight'], query_rotation_reconstruction_error=rotation_error,
                                          source_events=str(root / 'events.jsonl.zst'), source_native=str(root / 'native/excerpt.json')))
                    payload[f'l{layer}_input'] = reader.values(qevent['inputs'][1]['snapshot'])
                    payload[f'l{layer}_raw'] = raw
                break
        finally:
            reader.close()
        if cid in failures:
            for layer in sorted({h['layer'] for h in selected}):
                local = contexts[cid, layer]
                changed = local['raw'].copy()
                details = []
                for head_record in [h for h in selected if h['layer'] == layer]:
                    head = head_record['head']
                    comparisons = next(f['comparisons'] for f in head_record['failures'] if f['case'] == cid)
                    intended_gap = float(np.mean([m['gap'] for m in comparisons]))
                    current, _ = source_ratio(query(changed[head], local['gain'], local['eps'], local['rotation']), local['keys'], local['descent'], local['climb'], local['scale'])
                    new, detail = solve_query(changed[head], local['gain'], local['eps'], local['rotation'], local['keys'], local['descent'], local['climb'], local['scale'], current - intended_gap)
                    changed[head] = new
                    details.append(dict(head=head, comparisons=comparisons, **detail))
                qnew = np.stack([query(q, local['gain'], local['eps'], local['rotation']) for q in changed])
                logits = qnew @ local['keys'].T * local['scale']
                new_probability = np.exp(logits - logits.max(axis=1, keepdims=True))
                new_probability /= new_probability.sum(axis=1, keepdims=True)
                message = (new_probability - local['probability']) @ local['values']
                projection_change = local['weight'] @ message.ravel()
                branch_change = rms(local['projection'] + projection_change, local['post_gain'], local['post_eps']) - rms(local['projection'], local['post_gain'], local['post_eps'])
                direction_change = normalize(local['merged'] + branch_change) - normalize(local['merged'])
                local['proposed_direction_change'] = direction_change
                payload[f'l{layer}_target_raw'] = changed
                payload[f'l{layer}_direction_change'] = direction_change
                case_rows_layer = next(r for r in case_rows if r['layer'] == layer)
                case_rows_layer.update(heads=details,
                                      relative_raw_query_change=float(np.linalg.norm(changed - local['raw']) / np.linalg.norm(local['raw'])),
                                      relative_projection_change=float(np.linalg.norm(projection_change) / np.linalg.norm(local['projection'])),
                                      relative_branch_change=float(np.linalg.norm(branch_change) / np.linalg.norm(local['branch'])),
                                      residual_direction_change_norm=float(np.linalg.norm(direction_change)))
        np.savez(out / (cid + '.npz'), **payload)
        all_rows.append(dict(case=cid, failure=cid in failures, rows=case_rows, values=str(out / (cid + '.npz'))))
        print(cid, flush=True)
    for case in all_rows:
        if not case['failure']:
            continue
        for row in case['rows']:
            layer = row['layer']
            local = contexts[case['case'], layer]
            mates = sorted({b for a, b in summary['eligible_pairs'] if a == case['case']})
            target_differences = []
            for mate in mates:
                correct = contexts[mate, layer]
                target_differences.append(normalize(correct['merged']) - normalize(correct['incoming']) - (normalize(local['merged']) - normalize(local['incoming'])))
            desired = np.mean(target_differences, axis=0)
            actual = local['proposed_direction_change']
            row['local_direction_cosine_with_natural_correct_comparisons'] = float(normalize(actual) @ normalize(desired))
            row['comparison_cases'] = mates
    publish(out / 'index.json', dict(cases=all_rows, source=str(OUTPUT / 'source-routing-summary/index.json'),
                                    interpretation='A local, nonlinear recorded-parent diagnostic. Each query rotation uses that case\'s own key/value history and matches its observed successful-comparison source ratio. Actual output weights and post-normalization carry the changed message into the residual.',
                                    limits=['No checkpoint changed. These are not edited native trajectories.',
                                            'Each layer is inspected separately on its original inputs. A persistent candidate still needs a coupled construction and native tests of upstream/downstream effects.',
                                            'Source ratios and local cosine agreement do not establish a capability mechanism.',
                                            'Only the seven exploratory heads are changed in this diagnostic; the unmodified branch inputs, own historical value vectors and all intervening operations remain explicit.']))
    print(out, flush=True)


if __name__ == '__main__':
    main()
