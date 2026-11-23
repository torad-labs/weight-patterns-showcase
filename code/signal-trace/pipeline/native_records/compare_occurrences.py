"""Compare explicitly paired recorded occurrences without collapsing a passage.

Goal: preserve the incoming and outgoing vector contrasts, branches, and source
references at every recorded operation of a supplied pair of executions. This
is an observation tool, not an alignment algorithm or a capability detector.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from pipeline.native_records.native_trace import snapshot_values


def frame_index(path):
    index = {}
    digest = hashlib.sha256()
    with path.open('rb') as f:
        while True:
            start = f.tell()
            line = f.readline()
            if not line:
                break
            digest.update(line)
            fid = json.loads(line)['graph_execution']
            if fid not in index:
                index[fid] = [start, f.tell()]
            else:
                index[fid][1] = f.tell()
    return index, digest.hexdigest()


def read_frame(path, bounds):
    with path.open('rb') as f:
        f.seek(bounds[0])
        return [json.loads(line) for line in f.read(bounds[1] - bounds[0]).splitlines()]


def structure(frame):
    ids = {e['event']: i for i, e in enumerate(frame)}
    rows = []
    for e in frame:
        ports = []
        for x in e['inputs']:
            if x is None:
                ports.append(None)
                continue
            # A cache view can have both a producer and a storage version.
            # Preserve both relationships instead of replacing one with the other.
            ports.append({'weight': x.get('weight', {}).get('name'),
                          'producer': ids.get(x.get('producer_event')),
                          'cache': x.get('cache_storage'),
                          'unresolved': x['layout']['name'] if x.get('producer_event') is None
                          and 'weight' not in x and x.get('cache_state') is None else None})
        rows.append({'name': e['name'], 'operation': e['operation'], 'ports': ports})
    return rows


def source(x):
    return {k: x[k] for k in ('producer_event', 'cache_state', 'cache_storage',
                              'cache_view_offset', 'source_status', 'weight',
                              'weight_view_offset') if k in x}


def compare_frames(left, right, mapping, output):
    left, right, output = Path(left), Path(right), Path(output)
    if not mapping.get('relation') or not mapping.get('frames'):
        raise ValueError('Explicit relation and execution correspondence required')
    for pair in mapping['frames']:
        if not pair.get('reason'):
            raise ValueError('Every execution pair needs a correspondence reason')
    paths = [left / 'signal-events.jsonl', right / 'signal-events.jsonl']
    li, lh = frame_index(paths[0])
    ri, rh = frame_index(paths[1])
    output.mkdir(exist_ok=False, parents=True)
    paired_left, paired_right = set(), set()
    count = 0
    with (output / 'contrasts.bin').open('wb') as data, (output / 'contrasts.jsonl').open('w') as records:
        def contrast(ls, rs, reason=None):
            ref = {'left_snapshot': ls, 'right_snapshot': rs}
            if reason or ls['shape'] != rs['shape']:
                return dict(ref, status='unmatched', reason=reason or 'Different native shape; no row mapping supplied')
            if ls['type'] not in ('f32', 'f16', 'bf16') or rs['type'] not in ('f32', 'f16', 'bf16'):
                return dict(ref, status='control_values', reason='Integer runtime indices are not signal-vector coordinates')
            l = snapshot_values(left / 'signal-data.bin', ls).astype(np.float64)
            r = snapshot_values(right / 'signal-data.bin', rs).astype(np.float64)
            if not np.all(np.isfinite(l)) or not np.all(np.isfinite(r)):
                return dict(ref, status='nonfinite', reason='Native nonfinite values retained; no finite contrast invented')
            delta = r - l
            raw = delta.astype('<f4').tobytes()
            start = data.tell()
            data.write(raw)
            nl, nr, nd = float(np.linalg.norm(l)), float(np.linalg.norm(r)), float(np.linalg.norm(delta))
            return dict(ref, status='compared', offset_bytes=start, bytes=len(raw),
                        shape=ls['shape'], sha256=hashlib.sha256(raw).hexdigest(),
                        left_norm=nl, right_norm=nr, difference_norm=nd,
                        relative_to_left=nd / nl if nl else None,
                        cosine=float(np.dot(l, r) / (nl * nr)) if nl and nr else None)

        for pair_id, pair in enumerate(mapping['frames']):
            lf, rf = pair['left_execution'], pair['right_execution']
            if lf not in li or rf not in ri:
                raise ValueError('Mapped execution is absent from original recording')
            le, re = read_frame(paths[0], li[lf]), read_frame(paths[1], ri[rf])
            if structure(le) != structure(re):
                raise ValueError(f'Operation/source-port structure differs for executions {lf}, {rf}')
            paired_left.add(lf)
            paired_right.add(rf)
            for local, (l, r) in enumerate(zip(le, re)):
                cache_output = l['output']['name'].startswith('cache_') or r['output']['name'].startswith('cache_')
                inputs = []
                for port, (lx, rx) in enumerate(zip(l['inputs'], r['inputs'])):
                    if lx is None:
                        continue
                    item = {'port': port, 'left': source(lx), 'right': source(rx)}
                    if 'snapshot' in lx and 'snapshot' in rx:
                        reason = 'Cache slots require explicit cross-context token correspondence' if 'cache_storage' in lx or 'cache_storage' in rx else None
                        item['contrast'] = contrast(lx['snapshot'], rx['snapshot'], reason)
                    inputs.append(item)
                row = {'pair': pair_id, 'local_operation': local, 'name': l['name'],
                       'operation': l['operation'], 'left_event': l['event'], 'right_event': r['event'],
                       'left_execution': lf, 'right_execution': rf,
                       'left_input_position': l['input_position'], 'right_input_position': r['input_position'],
                       'inputs': inputs, 'output': contrast(l['output'], r['output'],
                           'Cache slots require explicit cross-context token correspondence' if cache_output else None)}
                records.write(json.dumps(row, allow_nan=False) + '\n')
                count += 1
    report = {'relation': mapping['relation'], 'mapping': mapping, 'left': str(left.resolve()),
              'right': str(right.resolve()), 'left_events_sha256': lh, 'right_events_sha256': rh,
              'operations': count, 'left_unpaired_executions': sorted(set(li) - paired_left),
              'right_unpaired_executions': sorted(set(ri) - paired_right),
              'delta_convention': 'right minus left, derived float32; original native snapshots remain referenced',
              'scope': 'Connected operation contrasts at supplied correspondences; no inferred capability, causal attribution, or edit selection',
              'cache_boundary': 'Cache versions and raw values retained; no cross-context slot alignment inferred'}
    (output / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('left', 'right', 'mapping', 'output'):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    report = compare_frames(a.left, a.right, json.loads(a.mapping.read_text()), a.output)
    print(json.dumps({'operations': report['operations'], 'scope': report['scope']}))
