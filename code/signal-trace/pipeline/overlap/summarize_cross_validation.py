"""Report held-out evidence with population coverage and exact-speed eligibility."""
import argparse
import csv
import gzip
from pathlib import Path
import numpy as np
from pipeline.overlap.analyze import read, write, finite
from pipeline.research_guard.verify_round2 import verify, ROOT


def average(a):
    good = np.isfinite(a)
    return float(a[good].mean()) if good.any() else None


def summarize(rows, ids, metrics, matched, adjustment, merges):
    eligible = [i for i in ids if rows[i]['matched_available']]
    a = metrics[np.ix_(ids, merges)]
    b = matched[np.ix_(eligible, merges)]
    c = adjustment[np.ix_(eligible, merges)]
    numerator = float(np.nansum(c[..., 0]))
    denominator = float(np.nansum(c[..., 1]))
    dot = float(np.nansum(c[..., 4] * c[..., 1]))
    result = dict(positions=len(ids), held_out_positions=sum(rows[i]['unmatched_available'] for i in ids),
                  matched_positions=len(eligible), merges=len(merges))
    for name, values in [('all_members', a), ('within_speed', b)]:
        result[name] = dict(
            own_side_failure_events=float(np.nansum(values[..., 0])),
            valid_failure_events=float(np.nansum(values[..., 1])),
            own_side_success_events=float(np.nansum(values[..., 2])),
            valid_success_events=float(np.nansum(values[..., 3])),
            mean_failure_own_side_fraction=average(np.divide(values[..., 0], values[..., 1],
                out=np.full_like(values[..., 0], np.nan), where=values[..., 1] > 0)),
            mean_success_own_side_fraction=average(np.divide(values[..., 2], values[..., 3],
                out=np.full_like(values[..., 2], np.nan), where=values[..., 3] > 0)),
            mean_balanced_fraction=average(values[..., 4]),
            mean_failure_margin=average(values[..., 5]),
            mean_success_margin=average(values[..., 6]),
            mean_held_out_separation=average(values[..., 7]),
            mean_direction_stability=average(values[..., 8]))
    result['speed_adjustment'] = dict(
        standardized_squared=numerator, eligible_unadjusted_squared=denominator,
        norm_ratio=np.sqrt(numerator / denominator) if denominator else None,
        cosine=dot / np.sqrt(numerator * denominator) if numerator * denominator else None,
        parallel_coefficient=dot / denominator if denominator else None,
        eligible_speeds=sorted({s['speed'] for i in eligible for s in rows[i]['strata']}))
    for key in ['failure_members', 'success_members', 'eligible_failure_members', 'eligible_success_members']:
        counts = [len(rows[i].get(key, [])) for i in (eligible if key.startswith('eligible') else ids)]
        result[key + '_range'] = [min(counts), max(counts)] if counts else []
    return result


def main(output):
    if verify(ROOT):
        raise ValueError('Round-two verifier refused')
    output = Path(output)
    dest = output / 'cross-validated'
    passages, layers, details = [], [], []
    for gpu in (0, 1):
        folder = dest / f'gpu{gpu}'
        index = read(folder / 'index.json')
        rows = index['rows']
        m = np.load(folder / 'metrics.npy')
        c = np.load(folder / 'matched-metrics.npy')
        a = np.load(folder / 'adjustment.npy')
        for coverage in ('all_aligned', 'full_population'):
            for role in dict.fromkeys(r['role'] for r in rows):
                ids = [i for i, r in enumerate(rows) if r['role'] == role and
                       (coverage == 'all_aligned' or r['full_population'])]
                if not ids:
                    continue
                identity = dict(gpu=gpu, coverage=coverage, passage=role)
                passages.append(dict(**identity, **summarize(rows, ids, m, c, a, list(range(m.shape[1])))))
                for layer in range(m.shape[1] // 3):
                    layers.append(dict(**identity, layer=layer,
                                       **summarize(rows, ids, m, c, a, list(range(3 * layer, 3 * layer + 3)))))
        for i, row in enumerate(rows):
            for merge in range(m.shape[1]):
                record = dict(gpu=gpu, position=i, role=row['role'], merge=merge,
                              full_population=row['full_population'],
                              failure_members=len(row['failure_members']), success_members=len(row['success_members']),
                              matched_failure_members=len(row.get('eligible_failure_members', [])),
                              matched_success_members=len(row.get('eligible_success_members', [])),
                              eligible_speeds='|'.join(str(s['speed']) for s in row['strata']))
                for prefix, array in [('unmatched', m), ('matched', c)]:
                    record.update({prefix + '_' + key: float(value) for key, value in zip(index['metrics'], array[i, merge])})
                record.update({key: float(value) for key, value in zip(index['adjustment_metrics'], a[i, merge])})
                details.append(finite(record))
    write(dest / 'summary.json', finite(dict(passages=passages, layers=layers,
        aggregation='Mean of per-position/per-merge class-balanced fractions; cells receive equal weight. Member-event counts are repeated measurements, not independent trials.',
        margin='Signed distance from the training midpoint along the omitted-member training direction. Positive means on the member own group side.',
        stability='Unmatched: pairwise cosine of all leave-one-out training directions. Matched: member-count-weighted mean of within-speed pairwise cosines, not agreement between speeds.',
        standardization='Identical pooled-member speed weights for the two outcome centroids; compare on exactly the eligible member pool. The norm ratio and cosine sum over all included positions and merges.',
        limitations=['Only speed is held equal; other parameters, wording and generated history remain uncontrolled.',
                     'GPU 1 only supports speed 8; its standardized-to-eligible ratio of one is an identity, not a confound-removal result.',
                     'Held-out classification does not purify the full-data difference or identify a causal editing direction.'])))
    with gzip.open(dest / 'position-merge-metrics.csv.gz', 'wt') as f:
        writer = csv.DictWriter(f, fieldnames=list(details[0])); writer.writeheader(); writer.writerows(details)
    flat = []
    for row in passages:
        record = {k: v for k, v in row.items() if not isinstance(v, (dict, list))}
        for family in ('all_members', 'within_speed', 'speed_adjustment'):
            record.update({family + '_' + k: v for k, v in row[family].items() if not isinstance(v, list)})
        flat.append(record)
    with (dest / 'passage-summary.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    print(dict(passage_rows=len(passages), layer_rows=len(layers), position_merge_rows=len(details)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('output', type=Path)
    main(parser.parse_args().output)
