"""Overlay original success/failure progressions separately before contrasting.

All full vectors stay in their own merge coordinates and every member remains
addressable. A group reference supplies correspondence, never desired values.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from pipeline.acquisition.run_a_baseline import RUN
from pipeline.native_records.group_observation import publish
from pipeline.observation.map_a_response_divergence import read, digest, finite_list, finite_mean


def overlay(reference, members, original, output):
    reference_index = read(original/reference/'index.json')
    shape = np.load(original/reference/'turns.npy', mmap_mode='r').shape
    output.mkdir(parents=True)
    mean = np.lib.format.open_memmap(output/'mean.npy', mode='w+', dtype='<f4', shape=shape)
    dispersion = np.full(shape[:2], np.nan)
    gains = np.full(shape[:2], np.nan)
    gain_variance = np.full(shape[:2], np.nan)
    maps, arrays, source_members = defaultdict(list), {}, []
    for member, mapping_path in members:
        mapping = read(mapping_path)
        for row in mapping['predictions']:
            maps[row['left_prediction']].append((member, row['right_prediction']))
        arrays[member] = (np.load(original/member/'turns.npy', mmap_mode='r'),
                          np.load(original/member/'norms.npy', mmap_mode='r'))
        source_members.append(dict(case=member, correspondence=str(mapping_path.resolve()),
                                   correspondence_sha256=digest(mapping_path),
                                   vectors=str((original/member/'turns.npy').resolve()),
                                   norms=str((original/member/'norms.npy').resolve()),
                                   index=str((original/member/'index.json').resolve()),
                                   unmatched=mapping['unmatched_right']))
    rows = []
    for prediction in range(shape[0]):
        points = maps[prediction]
        if len(points) != len({cid for cid, _ in points}):
            raise ValueError('An aligned group position counts a question twice')
        if not points:
            mean[prediction] = np.nan
            rows.append(dict(prediction=prediction, members=[], count=0))
            continue
        data = np.stack([arrays[cid][0][p] for cid, p in points]).astype('f8')
        norms = np.stack([arrays[cid][1][p] for cid, p in points]).astype('f8')
        center = data.mean(axis=0)
        mean[prediction] = center
        valid = np.all(np.isfinite(norms) & (norms > 0), axis=2)
        with np.errstate(divide='ignore', invalid='ignore'):
            gain = np.log(norms[..., 1]) - np.log(norms[..., 0])
        gain[~valid] = np.nan
        gains[prediction] = finite_mean(gain, axis=0)
        row = dict(prediction=prediction, members=[[cid, p] for cid, p in points], count=len(points))
        if len(points) > 1:
            dispersion[prediction] = np.sum((data - center) ** 2, axis=2).mean(axis=0)
            gain_variance[prediction] = finite_mean((gain - gains[prediction]) ** 2, axis=0)
            other = (data.sum(axis=0) - data) / (len(points) - 1)
            lengths = np.linalg.norm(data, axis=2) * np.linalg.norm(other, axis=2)
            cos = np.divide(np.sum(data * other, axis=2), lengths,
                            out=np.full(lengths.shape, np.nan), where=lengths > 0)
            row['member_leave_other_cosine'] = [dict(case=cid, prediction=p, by_merge=finite_list(cos[i]))
                                                for i, (cid, p) in enumerate(points)]
        rows.append(row)
    mean.flush()
    np.save(output/'dispersion.npy', dispersion)
    np.save(output/'log-gain.npy', gains)
    np.save(output/'log-gain-variance.npy', gain_variance)
    result = dict(reference=reference, rows=rows, members=source_members, shape=list(shape),
                  reference_graph=reference_index['connections'],
                  mean_vectors=str((output/'mean.npy').resolve()),
                  dispersion=str((output/'dispersion.npy').resolve()),
                  log_gain=str((output/'log-gain.npy').resolve()),
                  log_gain_variance=str((output/'log-gain-variance.npy').resolve()),
                  definition='Equal observed member weight at each corresponding prediction and merge. Dispersion is mean squared vector distance from the full-vector mean. One-member dispersion is undefined, not evidence of agreement. Full records and unmatched positions remain in member sources.')
    publish(output/'index.json', result)
    return result


def contrast(left, right, mapping, output):
    output.mkdir(parents=True)
    a = np.load(left['mean_vectors'], mmap_mode='r')
    b = np.load(right['mean_vectors'], mmap_mode='r')
    av, bv = [np.load(x['dispersion']) for x in [left, right]]
    ag, bg = [np.load(x['log_gain']) for x in [left, right]]
    frames = [f for f in mapping['frames'] if f.get('left_decision') is not None
              and f.get('right_decision') is not None]
    differences = np.lib.format.open_memmap(output/'contrast.npy', mode='w+', dtype='<f4',
                                            shape=(len(frames), *a.shape[1:]))
    stats = np.full((len(frames), a.shape[1], 4), np.nan)
    rows = []
    for i, frame in enumerate(frames):
        p, q = frame['left_decision'], frame['right_decision']
        difference = b[q].astype('f8') - a[p].astype('f8')
        differences[i] = difference
        squared = np.sum(difference ** 2, axis=1)
        pooled = (av[p] + bv[q]) / 2
        ratio = np.divide(squared, pooled, out=np.full(pooled.shape, np.nan), where=pooled > 0)
        stats[i] = np.stack([squared, pooled, ratio, bg[q] - ag[p]], axis=1)
        rows.append(dict(failure_reference_prediction=p, success_reference_prediction=q,
                         role=frame['role'], reason=frame['reason'],
                         failure_count=left['rows'][p]['count'], success_count=right['rows'][q]['count']))
    differences.flush()
    np.save(output/'metrics.npy', stats)
    summaries = []
    for role in sorted({r['role'] for r in rows}):
        selected = [i for i, r in enumerate(rows) if r['role'] == role]
        supported = [i for i in selected if rows[i]['failure_count'] > 1 and rows[i]['success_count'] > 1]
        values = finite_mean(stats[selected], axis=0)
        by_layer = values[:, 0].reshape(35, 3).sum(axis=1)
        summaries.append(dict(role=role, positions=len(selected),
                              positions_with_multiple_members_in_both_groups=len(supported),
                              direction_squared_by_merge=finite_list(values[:, 0]),
                              direction_squared_by_layer=finite_list(by_layer),
                              direction_total=float(by_layer.sum()),
                              pooled_within_group_squared_by_merge=finite_list(values[:, 1]),
                              right_minus_left_log_gain=finite_list(values[:, 3])))
    result = dict(left=left['reference'], right=right['reference'], rows=rows, summaries=summaries,
                  contrast_vectors=str((output/'contrast.npy').resolve()),
                  metrics=str((output/'metrics.npy').resolve()),
                  metric_names=['between_group_squared', 'pooled_within_group_squared',
                                'between_over_within_squared', 'success_minus_failure_log_gain'],
                  mapping=mapping,
                  limits=['Centroids can reflect shared wording and varied physical parameters, not only a failing procedure.',
                          'Inspect individual connected members and context contrasts before prescribing a correction.',
                          'No empirical significance threshold or capability identity follows from these descriptive ratios.'])
    publish(output/'index.json', result)
    return result


def build_groups(correspondence_index, original_root, reference_root, destination):
    correspondence_index, original_root, reference_root, destination = map(
        Path, (correspondence_index, original_root, reference_root, destination))
    if destination.exists():
        raise ValueError(f'Destination already exists: {destination}')
    source = read(correspondence_index)
    eligible = {cid for cid, c in source['cases'].items() if c['role'] in ['failure','natural_numerical_success']}
    group_members = defaultdict(list)
    refs = {}
    for pair in source['pairs']:
        if pair['relation'] not in ['failure_overlay', 'natural_numerical_success_overlay']:
            continue
        a, b = pair['left'], pair['right']
        if a not in eligible or b not in eligible:
            continue
        ca, cb = source['cases'][a], source['cases'][b]
        if not pair['same_hardware'] or ca['gpu'] != cb['gpu'] or ca['role'] != cb['role']:
            raise ValueError('A separate group crosses card or outcome populations')
        group = (ca['gpu'], ca['role'])
        if group in refs and refs[group] != a:
            raise ValueError('Inconsistent group reference')
        refs[group] = a
        group_members[group].append((b, correspondence_index.parent / pair['file']))
    observed = [cid for members in group_members.values() for cid, _ in members]
    if set(observed) != eligible or len(observed) != len(eligible):
        raise ValueError('Separate groups must include every eligible member exactly once')
    results, reports, comparisons = {}, [], []
    for group, members in sorted(group_members.items()):
        gpu, role = group
        name = f'gpu{gpu}-{role}'
        local = destination / name
        results[group] = overlay(refs[group], members, original_root, local)
        reports.append(dict(gpu=gpu, role=role, members=len(members), reference=refs[group],
                            record=str((local/'index.json').resolve())))
        print(json.dumps(dict(group=name, members=len(members))), flush=True)
    for gpu in sorted({g for g, _ in group_members}):
        path = reference_root/f'gpu{gpu}-failure-success/comparison.json'
        mapping = read(path)['correspondence']
        left, right = results[(gpu,'failure')], results[(gpu,'natural_numerical_success')]
        if mapping['left'] != left['reference'] or mapping['right'] != right['reference'] or not mapping['same_hardware']:
            raise ValueError('Group reference correspondence differs from recorded population')
        local = destination/f'gpu{gpu}-comparison'
        contrast(left, right, mapping, local)
        comparisons.append(dict(gpu=gpu, record=str((local/'index.json').resolve())))
    report = dict(groups=reports, comparisons=comparisons, members=len(observed),
                  source=str(correspondence_index.resolve()), source_sha256=digest(correspondence_index),
                  purpose='Whole-response separate natural success/failure observations, then same-card group contrasts. Neither group means nor pairwise means prescribe edits.',
                  complete=True)
    publish(destination/'index.json', report)
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=RUN.parents[3]/'studies/04-round-2-overlap/runs/overlap/group-map')
    args=parser.parse_args()
    build_groups(RUN/'response-correspondence/index.json',
                 RUN/'shared/observation/original-model-traces/full-response-turns',
                 RUN/'group-observation', args.output)
