"""Map every recorded passage and merge before selecting a joint weight edit.

This is an observation, not a fitting target. References retain the full ordered
vectors and graph bindings. Scalar distances describe their differences; they
do not label every context difference as an error or assign an edit location.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from pipeline.acquisition.run_a_baseline import RUN
from pipeline.native_records.group_observation import publish


METRICS = ['direction_squared_distance', 'right_minus_left_log_gain',
           'log_gain_squared_distance', 'right_minus_left_log_incoming_norm',
           'right_minus_left_log_outgoing_norm', 'left_direction_squared_norm',
           'right_direction_squared_norm', 'direction_dot']
ELIGIBLE = {'failure', 'natural_numerical_success'}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite_list(values):
    return [float(x) if np.isfinite(x) else None for x in values]


def finite_mean(values, axis=0):
    array = np.asarray(values)
    count = np.isfinite(array).sum(axis=axis)
    return np.divide(np.nansum(array, axis=axis), count,
                     out=np.full(np.shape(count), np.nan), where=count > 0)


def metrics(left, right, left_norms, right_norms):
    difference = right - left
    norms = np.stack([left_norms, right_norms])
    valid = np.all(np.isfinite(norms) & (norms > 0), axis=(0, 3))
    with np.errstate(divide='ignore', invalid='ignore'):
        log_norm_difference = np.log(right_norms) - np.log(left_norms)
    log_norm_difference[~valid] = np.nan
    gain = log_norm_difference[..., 1] - log_norm_difference[..., 0]
    return np.stack([np.sum(difference ** 2, axis=-1), gain, gain ** 2,
                     log_norm_difference[..., 0], log_norm_difference[..., 1],
                     np.sum(left ** 2, axis=-1), np.sum(right ** 2, axis=-1),
                     np.sum(left * right, axis=-1)], axis=-1)


def map_population(correspondence_index, original_root, destination):
    correspondence_index, original_root, destination = map(
        Path, (correspondence_index, original_root, destination))
    if destination.exists():
        raise ValueError(f'Destination already exists: {destination}')
    source = read(correspondence_index)
    if not read(original_root / 'index.json')['complete']:
        raise ValueError('Complete original vector population is required')
    destination.mkdir(parents=True)
    arrays, metadata, roles, connections, cases = {}, {}, {}, {}, {}
    counts = Counter()
    for cid, meta in source['cases'].items():
        if meta['role'] not in ELIGIBLE:
            continue
        local = original_root / cid
        index = read(local / 'index.json')
        turns = np.load(local / 'turns.npy', mmap_mode='r')
        norms = np.load(local / 'norms.npy', mmap_mode='r')
        if turns.ndim != 3 or turns.shape[1] != 105 or norms.shape != (*turns.shape[:2], 2):
            raise ValueError(f'{cid}: all 105 ordered merges and native norms are required')
        if index['predictions'] != len(turns) or index['gpu'] != meta['gpu']:
            raise ValueError(f'{cid}: source recording identity mismatch')
        coordinates = [(c['layer'], c['kind']) for c in index['connections']]
        if len(coordinates) != 105 or {x[0] for x in coordinates} != set(range(35)):
            raise ValueError(f'{cid}: incomplete graph-bound layer coverage')
        if connections and coordinates != next(iter(connections.values())):
            raise ValueError(f'{cid}: merge coordinates disagree across recordings')
        connections[cid] = coordinates
        annotation = read(correspondence_index.parent / 'cases' / f'{cid}.json')
        section = {s['index']: s['role'] for s in annotation['sections']}
        token_roles = {t['prediction']: section.get(t['section'], 'unresolved')
                       for t in annotation['tokens']}
        roles[cid] = {p: token_roles.get(p, 'unresolved') for p in range(len(turns))}
        arrays[cid] = (turns, norms)
        metadata[cid] = dict(index=str((local / 'index.json').resolve()),
                            index_sha256=digest(local / 'index.json'),
                            vectors=str((local / 'turns.npy').resolve()),
                            norms=str((local / 'norms.npy').resolve()),
                            vector_sha256=index.get('vector_sha256'),
                            native_archive=index.get('full_original_archive_record'))
        cases[cid] = dict(role=meta['role'], gpu=meta['gpu'], predictions=len(turns),
                          passage_positions=dict(Counter(roles[cid].values())),
                          source=metadata[cid])
        counts[meta['role']] += 1
    # Each matched coordinate is retained. Aggregation is for the report only.
    pair_records, excluded_pairs, per_question = [], [], defaultdict(list)
    covered = defaultdict(set)
    self_count = 0
    total_matched = 0
    for number, pair in enumerate(source['pairs']):
        left, right = pair['left'], pair['right']
        reason = None
        if left not in arrays or right not in arrays:
            reason = 'Outside eligible error/success population'
        elif not pair['same_hardware'] or cases[left]['gpu'] != cases[right]['gpu']:
            reason = 'Different card baselines; excluded from the overlap'
        if reason:
            excluded_pairs.append(dict(number=number, pair=pair, reason=reason))
            continue
        mapping_path = correspondence_index.parent / pair['file']
        mapping = read(mapping_path)
        rows, excluded_rows = [], []
        seen_left, seen_right = set(), set()
        for row in mapping['predictions']:
            a, b = row['left_prediction'], row['right_prediction']
            if a not in roles[left] or b not in roles[right]:
                raise ValueError(f'{pair["file"]}: prediction outside recording')
            if a in seen_left or b in seen_right:
                raise ValueError(f'{pair["file"]}: ambiguous repeated correspondence')
            seen_left.add(a); seen_right.add(b)
            role = row['section'] or 'unresolved'
            if roles[left][a] != role or roles[right][b] != role:
                excluded_rows.append(dict(row=row, reason='Passage role mismatch'))
                continue
            rows.append(dict(row, role=role))
        for side, seen in [('left', seen_left), ('right', seen_right)]:
            unmatched = mapping.get(f'unmatched_{side}', [])
            n = cases[pair[side]]['predictions']
            if set(unmatched) & seen or set(unmatched) | seen != set(range(n)):
                raise ValueError(f'{pair["file"]}: missing or duplicate {side} position accounting')
        local = destination / 'pairs' / f'{number:03}-{left}--{right}'
        local.mkdir(parents=True)
        values = np.empty((len(rows), 105, len(METRICS)), dtype='<f8')
        for start in range(0, len(rows), 16):
            chunk = rows[start:start + 16]
            a = [r['left_prediction'] for r in chunk]
            b = [r['right_prediction'] for r in chunk]
            values[start:start + len(chunk)] = metrics(
                arrays[left][0][a].astype('f8'), arrays[right][0][b].astype('f8'),
                arrays[left][1][a].astype('f8'), arrays[right][1][b].astype('f8'))
        if not np.isfinite(values[..., 0]).all():
            raise ValueError(f'{left}/{right}: nonfinite directional vectors')
        np.save(local / 'metrics.npy', values)
        relation, is_self = pair['relation'], left == right
        self_count += is_self
        total_matched += len(rows)
        summaries = []
        member = left if relation == 'natural_failure_success' else right
        for role in sorted({r['role'] for r in rows}):
            selection = [i for i, row in enumerate(rows) if row['role'] == role]
            mean = finite_mean(values[selection], axis=0)
            summaries.append(dict(role=role, positions=len(selection),
                                  direction_squared_by_merge=finite_list(mean[:, 0]),
                                  gain_squared_by_merge=finite_list(mean[:, 2])))
            if not is_self:
                per_question[(relation, member, role)].append(mean)
        if relation == 'natural_failure_success':
            if cases[left]['role'] != 'failure' or cases[right]['role'] != 'natural_numerical_success':
                raise ValueError('Failure/success relation has incorrect source roles')
            covered[left].update(r['left_prediction'] for r in rows)
        record = dict(pair=pair, rows=rows, excluded_rows=excluded_rows,
                      unmatched_left=mapping['unmatched_left'], unmatched_right=mapping['unmatched_right'],
                      correspondence=str(mapping_path.resolve()), correspondence_sha256=digest(mapping_path),
                      left_source=metadata[left], right_source=metadata[right],
                      left_graph=metadata[left]['index'], right_graph=metadata[right]['index'],
                      metrics=METRICS, metrics_file=str((local / 'metrics.npy').resolve()),
                      shape=list(values.shape), invalid_gain_entries=int(np.isnan(values[..., 1]).sum()),
                      self_comparison=is_self, summaries=summaries)
        publish(local / 'index.json', record)
        pair_records.append(dict(number=number, left=left, right=right, relation=relation,
                                 gpu=cases[left]['gpu'], self_comparison=is_self,
                                 matched=len(rows), record=str((local / 'index.json').resolve())))
        print(json.dumps(dict(pair=number, left=left, right=right, positions=len(rows))), flush=True)
    questions, grouped_values = [], defaultdict(list)
    for (relation, cid, role), entries in per_question.items():
        mean = finite_mean(entries, axis=0)
        questions.append(dict(relation=relation, case=cid, gpu=cases[cid]['gpu'], role=role,
                              mates=len(entries), direction_squared_by_merge=finite_list(mean[:, 0]),
                              gain_squared_by_merge=finite_list(mean[:, 2])))
        grouped_values[(relation, role)].append(mean)
    grouped = []
    for (relation, role), entries in sorted(grouped_values.items()):
        mean = finite_mean(entries, axis=0)
        direction = mean[:, 0]
        layers = direction.reshape(35, 3).sum(axis=1)
        gain = mean[:, 2]
        grouped.append(dict(relation=relation, role=role, questions=len(entries),
                            direction_squared_by_merge=finite_list(direction),
                            direction_squared_by_layer=finite_list(layers),
                            direction_total=float(layers.sum()),
                            direction_layer_share=finite_list(layers / layers.sum())
                            if layers.sum() else [None] * 35,
                            gain_squared_by_merge=finite_list(gain),
                            gain_total=float(gain.sum()) if np.isfinite(gain).all() else None))
    failures = {cid for cid, c in cases.items() if c['role'] == 'failure'}
    for cid, case in cases.items():
        if cid in failures:
            case['matched_success_positions'] = len(covered[cid])
            case['positions_without_success'] = sorted(set(roles[cid]) - covered[cid])
            case['matched_success_by_passage'] = dict(Counter(roles[cid][p] for p in covered[cid]))
    report = dict(
        population=dict(eligible_cases=len(cases), roles=dict(counts),
                        predictions=sum(c['predictions'] for c in cases.values()),
                        merge_observations=105 * sum(c['predictions'] for c in cases.values()),
                        excluded_cases=sorted(set(source['cases']) - cases.keys())),
        cases=cases, pairs=pair_records, excluded_pairs=excluded_pairs, questions=questions, grouped=grouped,
        metric_names=METRICS, self_comparisons=self_count, matched_pair_positions=total_matched,
        coverage=dict(failures_without_same_card_success=sorted(cid for cid in failures if not covered[cid]),
                      failure_positions_with_success=sum(len(covered[cid]) for cid in failures),
                      failure_positions_without_success=sum(cases[cid]['predictions'] - len(covered[cid])
                                                            for cid in failures)),
        source=dict(index=str(correspondence_index.resolve()), sha256=digest(correspondence_index),
                    original_root=str(original_root.resolve())),
        weighting='Positions within each pair/passage, then mates within each question, then equal questions. Self comparisons are retained but excluded from group summaries. No cross-card latent contrasts.',
        definition='Direction is unit(merged)-unit(incoming); local gain is log(outgoing norm/incoming norm). Both preserve operation order. Full coordinate sources and graph bindings are referenced, not replaced by these scalar summaries.',
        limits=['This map measures contextual differences; a numerical difference is not automatically an error.',
                'Unknown and unmatched positions remain explicit. No donor state or zero residual is invented.',
                'Role labels and final numerical success do not prove every explanatory sentence is correct.',
                'Layers are not selected and no significance or edit target is inferred by this report.',
                'Within-group reference comparisons are not independent statistical controls; shared templates and references are disclosed.'],
        complete=True)
    publish(destination / 'index.json', report)
    with (destination / 'layer-passages.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['relation', 'passage', 'questions', 'layer', 'direction_squared', 'share_of_passage_gap',
                         'attention', 'feedforward', 'per_layer'])
        for group in grouped:
            for layer in range(35):
                writer.writerow([group['relation'], group['role'], group['questions'], layer,
                                 group['direction_squared_by_layer'][layer], group['direction_layer_share'][layer],
                                 *group['direction_squared_by_merge'][3 * layer:3 * layer + 3]])
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--correspondence', type=Path, default=RUN / 'response-correspondence/index.json')
    parser.add_argument('--original', type=Path,
                        default=RUN / 'shared/observation/original-model-traces/full-response-turns')
    parser.add_argument('--output', type=Path, default=RUN.parents[3]/'studies/04-round-2-overlap/runs/overlap/response-map')
    args = parser.parse_args()
    result = map_population(args.correspondence, args.original, args.output)
    print(json.dumps(dict(population=result['population'], coverage=result['coverage'])), flush=True)
