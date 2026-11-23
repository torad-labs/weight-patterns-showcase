"""Bind A's wider working streams to the same original natural executions."""
import argparse
from collections import defaultdict
import hashlib
import json
import math

from pipeline.acquisition.run_a_baseline import RUN
from pipeline.native_records.group_observation import publish


def qualified_labels(gpu):
    relative = ('projection/block-changed' if gpu == 0 else 'gpu1/expanded-043-h3')
    path = RUN / 'stream-qualification-2026-11-08' / relative / 'stream.jsonl'
    labels, position = set(), None
    with path.open() as records:
        for line in records:
            record = json.loads(line)
            if record['phase'] != 'decode':
                continue
            if position is not None and record['input_position'] != position:
                break
            position = record['input_position']
            if record['label'] in labels:
                raise ValueError('Duplicate output in the independently qualified stream')
            labels.add(record['label'])
    if not labels:
        raise ValueError('Missing independent stream qualification')
    return labels


def check_case(member, directory, expected_labels):
    from pathlib import Path
    baseline_path = Path(member['baseline'])
    raw = baseline_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != member['baseline_sha256']:
        raise ValueError('Declared original natural response changed')
    original = json.loads(raw)
    observed = json.loads((directory / 'result.json').read_text())
    if observed['teacher_token_count'] or observed['conditioned_continuation']:
        raise ValueError('Coverage must be an independent natural execution')
    tokens_equal = observed['tokens'] == original['tokens']
    scores_equal = [x['logits_sha256'] for x in observed['steps']] == [x['logits_sha256'] for x in original['steps']]
    if not tokens_equal or not scores_equal:
        raise ValueError(f'{member["case"]}: natural reproduction differs from its same-card original')
    predictions = {step['input_position'] for step in original['steps']}
    counts, offset, events = defaultdict(set), 0, 0
    with (directory / 'stream.jsonl').open() as records:
        for line in records:
            record = json.loads(line)
            if (record['offset_bytes'] != offset or math.prod(record['shape']) != record['values'] or
                    len(record['positions']) != record['shape'][record['token_axis']]):
                raise ValueError('Stream byte layout or token axis is incomplete')
            offset += record['values'] * 4
            events += 1
            for position in record['positions']:
                if position not in predictions:
                    continue
                if record['label'] in counts[position]:
                    raise ValueError('Duplicate output at a recorded prediction')
                counts[position].add(record['label'])
    if (set(counts) != predictions or any(value != expected_labels for value in counts.values()) or
            offset != (directory / 'stream.f32').stat().st_size or offset != observed['stream']['bytes']):
        raise ValueError('Incomplete wider trace recording')
    return dict(case=member['case'], gpu=member['gpu'], tokens_equal=tokens_equal,
        all_score_hashes_equal=scores_equal, predictions=len(predictions),
        labels_at_every_prediction=len(expected_labels), events=events, bytes=offset,
        original_baseline=str(baseline_path), baseline_sha256=member['baseline_sha256'],
        stream=str(directory), scope='Same natural tokens and full score hashes. Complete qualified projection output set at every prediction; not a fresh full-tensor comparison or complete native-operation capture.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    root = RUN / 'shared/observation/original-model-traces'
    declaration = json.loads((root / 'declaration.json').read_text())
    destination = root / 'qualification'
    destination.mkdir(exist_ok=True)
    expected = {g: qualified_labels(g) for g in (0, 1)}
    cases, pending = [], []
    for member in declaration['population']:
        directory = root / f'original-gpu{member["gpu"]}' / member['case']
        if not (directory / 'result.json').exists():
            pending.append(member['case'])
            continue
        target = destination / (member['case'] + '.json')
        if target.exists():
            row = json.loads(target.read_text())
        else:
            row = check_case(member, directory, expected[member['gpu']])
            publish(target, row)
            print(json.dumps(dict(case=row['case'], predictions=row['predictions'], matched=True)), flush=True)
        cases.append(row)
    complete = not pending and all((root / f'original-gpu{g}/execution.json').exists() for g in (0, 1))
    if not args.partial and not complete:
        raise ValueError('Both original batches must finish before complete qualification')
    summary = dict(complete=complete, eligible=len(declaration['population']), cases=cases, pending=pending,
        predictions=sum(row['predictions'] for row in cases),
        all_tokens_equal=all(row['tokens_equal'] for row in cases),
        all_score_hashes_equal=all(row['all_score_hashes_equal'] for row in cases),
        scope='Wider working retention of the same original 45 natural executions; no new discovery questions, supplied answers or edited model.')
    publish(destination / ('index.json' if complete else 'partial.json'), summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['cases','pending']}))


if __name__ == '__main__':
    main()
