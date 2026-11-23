"""Audit a completed capture against its saved plain run, without rerunning GPU work.

Reuse the recorder qualification's producer/cache checks on complete responses.
Also retain byte ranges for every execution and its actual prediction identity,
so later observation can stream the trace instead of loading it all into RAM.
This checks acquisition, not the residual-trace hypothesis.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def audit(capture, baseline):
    capture, baseline = Path(capture), Path(baseline)
    observed = json.loads((capture/'result.json').read_text())
    original = json.loads((baseline/'result.json').read_text())
    assert observed['tokens'] == original['tokens'], 'Token sequence changed under capture'
    assert [s['logits_sha256'] for s in observed['steps']] == [
        s['logits_sha256'] for s in original['steps']], 'Full score bytes changed under capture'
    seen, caches, weights = {}, {}, set()
    operations, gaps = Counter(), Counter()
    frames, current = [], None
    writes = reads = total = 0
    digest = hashlib.sha256()
    prediction_by_position = {s['input_position']: s['step'] for s in observed['steps']}
    with (capture/'signal-events.jsonl').open('rb') as f:
        while line := f.readline():
            start, end = f.tell()-len(line), f.tell()
            digest.update(line)
            e = json.loads(line)
            assert e['event'] == total
            total += 1
            assert e['before'] < e['after']
            if current is None or current['execution'] != e['graph_execution']:
                seen.clear()
                current = {'execution': e['graph_execution'], 'phase': e['phase'],
                           'call': e['call'], 'input_position': e['input_position'],
                           'first_event': e['event'], 'event_count': 0,
                           'byte_start': start, 'byte_end': end,
                           'prediction': None}
                frames.append(current)
            current['byte_end'] = end
            current['event_count'] += 1
            operations[e['operation']] += 1
            if e['name'] == 'result_output' and e['output']['shape'][1] > 0:
                current['prediction'] = prediction_by_position.get(e['input_position'])
                if current['prediction'] is not None:
                    # Prefill microbatches without requested outputs have a zero
                    # second dimension. Only the materialized output that really
                    # supplied this prediction is linked to its token decision.
                    snap = e['output']
                    assert snap['type'] == 'f32' and snap['shape'] == [observed['vocabulary_size'], 1, 1, 1]
                    score_hash = hashlib.sha256()
                    with (capture/'signal-data.bin').open('rb') as data:
                        for chunk in snap['chunks']:
                            data.seek(chunk['offset'])
                            raw = data.read(chunk['bytes'])
                            assert len(raw) == chunk['bytes']
                            assert hashlib.sha256(raw).hexdigest() == chunk['sha256']
                            score_hash.update(raw)
                    assert score_hash.hexdigest() == observed['steps'][current['prediction']]['logits_sha256']
            if 'cache_write' in e:
                writes += 1
                w = e['cache_write']
                assert w['slots']
                if w['previous_event'] is not None:
                    assert caches[w['previous_event']] == w['storage']
                caches[e['event']] = w['storage']
            if e['operation'] == 'FLASH_ATTN_EXT':
                reads += 1
                assert e['inputs'][1].get('cache_state') is not None
                assert e['inputs'][2].get('cache_state') is not None
            for operand in e['inputs']:
                if operand is None:
                    continue
                assert 'weight' in operand or 'snapshot' in operand
                if 'weight' in operand:
                    weights.add(operand['weight']['name'])
                producer = operand.get('producer_event')
                if producer is not None:
                    assert producer in seen, (e['event'], 'producer outside execution')
                    assert seen[producer] == operand['snapshot'], (e['event'], 'preimage differs')
                if operand.get('cache_state') is not None:
                    assert caches[operand['cache_state']] == operand['cache_storage']
                if producer is None and 'weight' not in operand and operand.get('cache_state') is None:
                    gaps[operand.get('source_status', 'Unresolved input')] += 1
            seen[e['event']] = e['output']
    layers = sorted({int(w.split('.')[1]) for w in weights if w.startswith('blk.')})
    assert layers == list(range(35))
    assert total == observed['signal_trace']['events']
    assert dict(operations) == observed['signal_trace']['operations']
    assert (capture/'signal-data.bin').stat().st_size == observed['signal_trace']['unique_native_bytes']
    assert [f['prediction'] for f in frames if f['prediction'] is not None] == list(range(len(observed['steps'])))
    return {'scope': 'Full saved-response acquisition check; no capability finding or edit selection',
            'capture': str(capture), 'baseline': str(baseline), 'case': observed['id'],
            'decisions': len(observed['steps']), 'tokens_and_all_score_hashes_equal': True,
            'recorded_vocabulary_bytes_match_every_decision': True,
            'events': total, 'layers': layers, 'weight_tensors': len(weights),
            'cache_writes': writes, 'attention_reads': reads, 'unresolved_inputs': dict(gaps),
            'recorded_producer_and_cache_links_valid': True,
            'complete_end_to_end_lineage': False, 'fused_attention_internals': 'opaque',
            'native_bytes': (capture/'signal-data.bin').stat().st_size,
            'event_bytes': (capture/'signal-events.jsonl').stat().st_size,
            'events_sha256': digest.hexdigest(), 'operations': dict(operations), 'frames': frames}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('capture', 'baseline', 'out'):
        p.add_argument(n, type=Path)
    a = p.parse_args()
    report = audit(a.capture, a.baseline)
    with a.out.open('x') as f:
        json.dump(report, f, indent=2)
        f.write('\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'frames'}, indent=2))
