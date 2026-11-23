"""Qualify the reduced-retention stream recorder against unchanged native inference.

Checks, per case: (1) every generated token and full-vocabulary score hash equals
the baseline response; (2) every streamed output that also exists in the verified
full-recorder excerpt is bit-identical. Streamed outputs outside the excerpt are
counted, not assumed. Observation only; nothing is written to a model.
"""
import argparse
import io
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

from pipeline.native_records.native_excerpts import ExcerptReader


def stream_values(directory, record):
    with (directory/'stream.f32').open('rb') as f:
        f.seek(record['offset_bytes'])
        return np.frombuffer(f.read(4*record['values']), dtype='<f4')


def full_label(event):
    if event['operation'] == 'ADD':
        for x in event['inputs']:
            if x and re.fullmatch(r'pe_in-\d+', x['layout']['name']):
                return 'pe_out-' + x['layout']['name'].split('-')[1]
    return event['name']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stream', type=Path, help='observe-stream case directory')
    p.add_argument('baseline', type=Path, help='baseline result.json')
    p.add_argument('excerpt', type=Path, help='full-recorder excerpt directory (events.jsonl.zst + native/)')
    p.add_argument('output', type=Path)
    a = p.parse_args()

    result = json.loads((a.stream/'result.json').read_text())
    base = json.loads(a.baseline.read_text())
    toks = result['tokens'] == base['tokens']
    hashes = [s['logits_sha256'] for s in result['steps']] == [s['logits_sha256'] for s in base['steps']]

    records = [json.loads(l) for l in (a.stream/'stream.jsonl').open()]
    wanted = defaultdict(list)
    prefill_seen = defaultdict(int)
    for r in records:
        if r['phase'] == 'prefill':
            key = ('prefill', r['label'], prefill_seen[r['label']]); prefill_seen[r['label']] += 1
        else:
            key = ('decode', r['label'], r['input_position'])
        wanted[key].append(r)

    # Every decision must carry the same complete set of streamed outputs.
    per_position = defaultdict(list)
    for r in records:
        if r['phase'] == 'decode':
            per_position[r['input_position']].append(r['label'])
    shapes = {tuple(sorted(v)) for v in per_position.values()}
    complete = len(shapes) == 1
    labels_per_decision = len(next(iter(shapes))) if shapes else 0

    reader = ExcerptReader(a.excerpt/'native')
    proc = subprocess.Popen(['zstd', '-dc', str(a.excerpt/'events.jsonl.zst')], stdout=subprocess.PIPE)
    equal = differ = outside = 0
    first_diff = []
    prefill_count = defaultdict(int)
    matched = set()
    for line in io.TextIOWrapper(proc.stdout):
        e = json.loads(line)
        label = full_label(e)
        if e['phase'] == 'prefill':
            key = ('prefill', label, prefill_count[label]); prefill_count[label] += 1
        else:
            key = ('decode', label, e['input_position'])
        if key not in wanted or key in matched:
            continue
        try:
            full = reader.values(e['output'])
        except ValueError:
            outside += 1
            continue
        matched.add(key)
        mine = stream_values(a.stream, wanted[key][0])
        if mine.shape == full.shape and np.array_equal(mine.view('<u4'), full.view('<u4')):
            equal += 1
        else:
            differ += 1
            if len(first_diff) < 10:
                first_diff.append({'key': list(key), 'stream_values': int(mine.size), 'full_values': int(full.size)})
    proc.wait()
    summary = {
        'scope': 'Reduced-retention stream qualification; not a capability finding',
        'tokens_equal': toks, 'all_score_hashes_equal': hashes, 'decisions': len(result['steps']),
        'stream_records': len(records), 'stream_bytes': result['stream']['bytes'],
        'decode_positions': len(per_position), 'labels_per_decision': labels_per_decision,
        'every_decision_has_identical_label_set': complete,
        'compared_bit_identical': equal, 'compared_different': differ,
        'full_event_outside_excerpt': outside,
        'stream_records_without_full_counterpart_in_excerpt': len(wanted) - len(matched),
        'first_differences': first_diff,
    }
    a.output.write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
