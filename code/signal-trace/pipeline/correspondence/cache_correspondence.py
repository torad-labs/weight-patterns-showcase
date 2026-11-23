"""Compare recorded attention history using explicit token correspondences.

The observation must identify which earlier supplied token wrote each cache row.
It must not compare equal slot numbers after histories have diverged, include
future writes, or interpolate unmatched history. No model state is modified.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from pipeline.native_records.compare_occurrences import read_frame
from pipeline.native_records.native_trace import snapshot_values


def history_pairs(prompt_pairs, prediction_pairs, left_prompt_length, right_prompt_length):
    def ordered(pairs):
        pairs = [tuple(p) for p in pairs]
        previous = (-1, -1)
        for left, right in pairs:
            if left <= previous[0] or right <= previous[1]:
                raise ValueError('History correspondence must be nonnegative, unique and order preserving')
            previous = (left, right)
        return pairs
    prompt_pairs, prediction_pairs = ordered(prompt_pairs), ordered(prediction_pairs)
    if any(l >= left_prompt_length or r >= right_prompt_length for l, r in prompt_pairs):
        raise ValueError('Prompt correspondence extends beyond the supplied prompt')
    # A selected token joins the next prediction's input history. Its cache row
    # is not the row which produced its prediction.
    return prompt_pairs + [(left_prompt_length + l, right_prompt_length + r)
                           for l, r in prediction_pairs]


def written_pairs(pairs, left_slots, right_slots):
    left_slots, right_slots = set(left_slots), set(right_slots)
    return [(l, r) for l, r in pairs if l in left_slots and r in right_slots]


def cache_registry(events_path):
    """Index actual append-only writes, refusing slot reuse in this A observation."""
    stores, versions = {}, {}
    with events_path.open() as f:
        for line in f:
            e = json.loads(line)
            if 'cache_write' not in e:
                continue
            w = e['cache_write']
            rows = stores.setdefault(w['storage'], [])
            slots = w['slots']
            if slots != list(range(len(rows), len(rows) + len(slots))):
                raise ValueError('Cache reuses/reorders slots; this observation needs an explicit position map')
            if len(slots) == 1 and e['phase'] == 'decode' and slots[0] != e['input_position']:
                raise ValueError('Cache slot is not the recorded supplied-token position')
            rows.extend({'write_event': e['event'], 'source_row': i,
                         'source_producer': e['inputs'][w['source_input']].get('producer_event')}
                        for i in range(len(slots)))
            versions[e['event']] = {'storage': w['storage'], 'written_count': len(rows)}
    return stores, versions


def cache_rows(path, operand):
    snap, root = operand['snapshot'], operand['cache_root_layout']
    # The observed E2B attention view has [head dimension, token slots, 1, 1].
    # Refuse other layouts instead of guessing a token axis or flattening heads.
    if (snap['shape'][2:] != [1, 1] or root['shape'][2:] != [1, 1]
            or snap['shape'][0] != root['shape'][0]
            or operand['cache_view_offset'] % root['strides'][1]):
        raise ValueError('Cache view needs a different explicit row mapping')
    base = operand['cache_view_offset'] // root['strides'][1]
    values = snapshot_values(path, snap).reshape(snap['shape'], order='F')[:, :, 0, 0]
    return base, values


def compare(left, right, left_audit, right_audit, mapping, out):
    if 'prompt_token_pairs' not in mapping:
        raise ValueError('Explicit prompt-token correspondence required')
    cases = [json.loads((p/'case.json').read_text()) for p in (left, right)]
    results = [json.loads((p/'result.json').read_text()) for p in (left, right)]
    prompt_lengths = [len(c['tokens']) for c in cases]
    pp = [(f['left_decision'], f['right_decision']) for f in mapping['frames']
          if f.get('left_decision') is not None and f.get('right_decision') is not None]
    pairs = history_pairs(mapping['prompt_token_pairs'], pp, *prompt_lengths)
    for l, r in pairs:
        if l >= prompt_lengths[0]+len(results[0]['tokens']) or r >= prompt_lengths[1]+len(results[1]['tokens']):
            raise ValueError('History correspondence extends past the recorded response')
    indices = [{f['execution']: f for f in a['frames']} for a in (left_audit, right_audit)]
    registries = [cache_registry(p/'signal-events.jsonl') for p in (left, right)]
    out.mkdir(parents=True, exist_ok=False)
    (out/'cache-writers.json').write_text(json.dumps({'left': registries[0][0], 'right': registries[1][0]})+'\n')
    token_rows = []
    tokens = [c['tokens'] + r['tokens'] for c, r in zip(cases, results)]
    for l, r in pairs:
        token_rows.append({'left_position': l, 'right_position': r,
                           'left_token': tokens[0][l], 'right_token': tokens[1][r]})
    (out/'history-pairs.json').write_text(json.dumps(token_rows)+'\n')
    unique, read_count = {}, 0
    with (out/'row-differences.bin').open('wb') as data, (out/'rows.jsonl').open('w') as rows_out, (out/'reads.jsonl').open('w') as reads_out:
        for pair_id, pair in enumerate(mapping['frames']):
            frames = []
            for side, p in enumerate((left, right)):
                f = indices[side][pair['left_execution' if side == 0 else 'right_execution']]
                frames.append(read_frame(p/'signal-events.jsonl', [f['byte_start'], f['byte_end']]))
            for le, re in zip(*frames):
                if le['operation'] != 'FLASH_ATTN_EXT':
                    continue
                if re['operation'] != 'FLASH_ATTN_EXT':
                    raise ValueError('Attention occurrences do not correspond')
                for port in (1, 2):
                    lo, ro = le['inputs'][port], re['inputs'][port]
                    lv, rv = [reg[1][op['cache_state']] for reg, op in zip(registries, (lo, ro))]
                    if lv['storage'] != rv['storage']:
                        raise ValueError('Attention reads different cache storage')
                    active = written_pairs(pairs, range(lv['written_count']), range(rv['written_count']))
                    unseen = [(l, r) for l, r in active if (lv['storage'], l, r) not in unique]
                    if unseen:
                        lb, lvalues = cache_rows(left/'signal-data.bin', lo)
                        rb, rvalues = cache_rows(right/'signal-data.bin', ro)
                        for l, r in unseen:
                            if not (0 <= l-lb < lvalues.shape[1] and 0 <= r-rb < rvalues.shape[1]):
                                raise ValueError('Written token is outside this cache view')
                            delta = rvalues[:, r-rb].astype(np.float32)-lvalues[:, l-lb].astype(np.float32)
                            if not np.all(np.isfinite(delta)):
                                raise ValueError('Nonfinite value in a written cache row')
                            raw = delta.astype('<f4').tobytes()
                            row_id = len(unique)
                            record = {'id': row_id, 'storage': lv['storage'],
                                      'left_position': l, 'right_position': r,
                                      'left_write': registries[0][0][lv['storage']][l],
                                      'right_write': registries[1][0][rv['storage']][r],
                                      'left_read_event': le['event'], 'right_read_event': re['event'],
                                      'left_snapshot': lo['snapshot'], 'right_snapshot': ro['snapshot'],
                                      'offset_bytes': data.tell(), 'bytes': len(raw),
                                      'shape': list(delta.shape), 'sha256': hashlib.sha256(raw).hexdigest()}
                            data.write(raw)
                            rows_out.write(json.dumps(record)+'\n')
                            unique[(lv['storage'], l, r)] = row_id
                    reads_out.write(json.dumps({'pair': pair_id, 'port': port,
                        'left_event': le['event'], 'right_event': re['event'],
                        'left_cache_version': lo['cache_state'], 'right_cache_version': ro['cache_state'],
                        'rows': [unique[(lv['storage'], l, r)] for l, r in active],
                        'left_written_unmatched': sorted(set(range(lv['written_count']))-{l for l, r in active}),
                        'right_written_unmatched': sorted(set(range(rv['written_count']))-{r for l, r in active})})+'\n')
                    read_count += 1
    report = {'scope': 'Explicit token correspondence of native K/V at actual attention reads; not attention probabilities or causal attribution',
              'left': str(left), 'right': str(right), 'left_events_sha256': left_audit['events_sha256'],
              'right_events_sha256': right_audit['events_sha256'], 'mapping': mapping,
              'paired_history_tokens': len(pairs), 'attention_operand_reads': read_count,
              'unique_cache_row_pairs': len(unique), 'native_position_encoding_retained': True,
              'masked_rows': 'Written does not mean attended; original query and attention mask remain in each source event',
              'reuse': 'A stored row difference is reused only after verifying that cache writes are append-only, with no overwritten slot'}
    (out/'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('left', 'right', 'left_audit', 'right_audit', 'mapping', 'out'):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    r = compare(a.left, a.right, json.loads(a.left_audit.read_text()), json.loads(a.right_audit.read_text()),
                json.loads(a.mapping.read_text()), a.out)
    print(json.dumps({k: v for k, v in r.items() if k != 'mapping'}))
