"""Give each recorded response one score over its own aligned passage positions."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
from pipeline.overlap.analyze import read, write, SHARED
from pipeline.native_records.run_declaration import check_native_run

PASSAGES = ('destination', 'descent')
MODES = ('all_members', 'within_speed')


def collect_members(card_indexes, folds_by_card, inventory, parameters):
    bags, available = defaultdict(dict), defaultdict(set)
    for gpu, index in card_indexes.items():
        folds = folds_by_card[gpu]
        for row in index['rows']:
            passage = row['role']
            if passage not in PASSAGES:
                continue
            for case, prediction in row['failure_members'] + row['success_members']:
                available[(case, passage)].add(prediction)
            batches = []
            if row['unmatched_available']:
                batches.append(('all_members', row['failure_members'] + row['success_members'],
                                f"p{row['position']}_unmatched_margins"))
            for stratum in row['strata']:
                batches.append(('within_speed', stratum['failure_members'] + stratum['success_members'],
                                stratum['margins_key']))
            for mode, members, key in batches:
                margins = folds[key]
                if len(members) != len(margins):
                    raise ValueError('Member order must match stored fold rows')
                for (case, prediction), vector in zip(members, margins):
                    if inventory[case]['gpu'] != gpu:
                        raise ValueError('A response cannot cross cards')
                    bag = bags[(case, passage, mode)]
                    if prediction in bag:
                        raise ValueError('Repeated own prediction would double-weight a response position')
                    good = np.isfinite(vector)
                    bag[prediction] = (float(vector[good].mean()) if good.any() else None,
                                       float((vector[good] > 0).mean()) if good.any() else None,
                                       int(good.sum()))
    records = []
    for case, meta in sorted(inventory.items(), key=lambda x: (x[1]['gpu'], x[1]['role'], x[0])):
        if meta['role'] not in ('failure', 'natural_numerical_success'):
            continue
        for mode in MODES:
            for passage in PASSAGES:
                bag = bags[(case, passage, mode)]
                valid = {p: v for p, v in bag.items() if v[0] is not None}
                records.append(dict(case=case, gpu=meta['gpu'], outcome=meta['role'],
                    speed=parameters[case]['v'], mode=mode, passage=passage,
                    available_aligned_positions=len(available[(case, passage)]),
                    scored_aligned_positions=len(bag), positions=len(valid), predictions=sorted(valid),
                    valid_merge_observations=sum(v[2] for v in valid.values()),
                    mean_signed_margin=float(np.mean([v[0] for v in valid.values()])) if valid else None,
                    mean_own_side_fraction=float(np.mean([v[1] for v in valid.values()])) if valid else None,
                    unavailable_reason=None if valid else 'No valid held-out observations for this member and stratum'))
    return records


def summarize(records):
    groups = defaultdict(list)
    for r in records:
        groups[(r['gpu'], r['mode'], r['passage'], r['outcome'])].append(r)
    result = []
    for (gpu, mode, passage, outcome), members in sorted(groups.items()):
        margins = np.array([m['mean_signed_margin'] for m in members if m['mean_signed_margin'] is not None])
        result.append(dict(gpu=gpu, mode=mode, passage=passage, outcome=outcome,
            total_members=len(members), observed_members=len(margins), missing_members=len(members)-len(margins),
            positive_members=int((margins>0).sum()), negative_members=int((margins<0).sum()),
            zero_members=int((margins==0).sum()),
            equal_response_mean_margin=float(margins.mean()) if len(margins) else None,
            median_margin=float(np.median(margins)) if len(margins) else None,
            minimum_margin=float(margins.min()) if len(margins) else None,
            maximum_margin=float(margins.max()) if len(margins) else None))
    return result


def report(records, groups, speed_groups):
    lines = ['# Held-out scores, one response at a time', '',
        'Requested in Claude’s acceptance review of the overlap analysis. These are summaries of the already saved held-out margins; no new direction, fit, inference or acceptance threshold is introduced.', '',
        'For each response and passage, first average its signed margin across valid merges at each of its own aligned prediction positions, then average those positions equally. Positive means its own outcome-group side of the direction estimated without that response. Each response receives one score. Null is missing evidence, never zero. Both cards keep their original members.', '',
        'The within-speed view can only score supported speed strata. GPU 1’s speed-4 and speed-6 responses remain listed with no score. “Positions” is the count scored, not the number of independent trials. Training sets and questions still overlap.', '',
        '## Count of responses with a positive mean margin', '',
        '| Card | Comparison | Passage | Outcome | Positive / observed | Missing | Mean margin, equal responses | Median |',
        '|---|---|---|---|---:|---:|---:|---:|']
    for g in groups:
        mean = f"{g['equal_response_mean_margin']:+.6f}" if g['observed_members'] else '—'
        median = f"{g['median_margin']:+.6f}" if g['observed_members'] else '—'
        lines.append(f"| {g['gpu']} | {g['mode']} | {g['passage']} | {g['outcome']} | {g['positive_members']}/{g['observed_members']} | {g['missing_members']} | {mean} | {median} |")
    lines += ['', '## Same-speed climb scores by response', '',
        'This breakdown retains the numerical strata instead of treating a combined score as uniform support.', '',
        '| Card | Speed | Outcome | Positive / observed | Missing | Equal-response mean margin |',
        '|---|---:|---|---:|---:|---:|']
    for g in speed_groups:
        if g['mode'] != 'within_speed' or g['passage'] != 'destination':
            continue
        mean = f"{g['equal_response_mean_margin']:+.6f}" if g['observed_members'] else '—'
        lines.append(f"| {g['gpu']} | {g['speed']} | {g['outcome']} | {g['positive_members']}/{g['observed_members']} | {g['missing_members']} | {mean} |")
    for gpu in sorted({r['gpu'] for r in records}):
        for mode in MODES:
            subset = {(r['case'], r['passage']): r for r in records if r['gpu']==gpu and r['mode']==mode}
            lines += ['', f'## GPU {gpu} — {mode}', '',
                '| Response | Outcome | Speed | Climb positions | Climb mean margin | Descent positions | Descent mean margin |',
                '|---|---|---:|---:|---:|---:|---:|']
            for case in dict.fromkeys(c for c, _ in subset):
                climb, descent = [subset[(case, p)] for p in PASSAGES]
                def value(r):
                    return f"{r['mean_signed_margin']:+.6f}" if r['mean_signed_margin'] is not None else '—'
                outcome = 'failure' if climb['outcome']=='failure' else 'success'
                lines.append(f"| {case} | {outcome} | {climb['speed']} | {climb['positions']} | {value(climb)} | {descent['positions']} | {value(descent)} |")
    return '\n'.join(lines)+'\n'


def main(output):
    output = Path(output)
    check_native_run(output/'cross-validated', output/'declaration.json')
    root = output/'cross-validated'
    inventory = read(SHARED/'observation/response-correspondence/index.json')['cases']
    parameters = read(root/'index.json')['parameters_by_case']
    indexes = {gpu: read(root/f'gpu{gpu}/index.json') for gpu in sorted({inventory[c]['gpu'] for c in parameters})}
    with_arrays = {gpu: np.load(root/f'gpu{gpu}/held-out-members.npz') for gpu in indexes}
    try:
        records = collect_members(indexes, with_arrays, inventory, parameters)
    finally:
        for a in with_arrays.values():
            a.close()
    groups = summarize(records)
    speed_groups = [dict(speed=speed, **g) for speed in sorted({r['speed'] for r in records})
                    for g in summarize([r for r in records if r['speed']==speed])]
    write(root/'per-response.json', dict(records=records, summaries=groups, speed_summaries=speed_groups,
        definition='Mean over each own prediction of its mean signed margin across valid merges; summaries then give each response equal weight.',
        source='Stored held-out-members.npz and member order in each card index; no centroids or directions recomputed.',
        timing='Added after the accepted group-level analysis at Claude review request. Descriptive addition, not an acceptance gate.'))
    with (root/'per-response.csv').open('w') as f:
        writer=csv.DictWriter(f, fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    (root/'PER-RESPONSE.md').write_text(report(records, groups, speed_groups))
    print(dict(responses=len(parameters), records=len(records), summaries=len(groups)), flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path)
    main(parser.parse_args().output)
