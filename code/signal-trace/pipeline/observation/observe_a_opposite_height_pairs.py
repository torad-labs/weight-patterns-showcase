"""Follow two opposite climb-height contrasts through their recorded operations.

The 3 kg pair changes rise 3 -> 1; the 4 kg pair changes rise 1 -> 3.
Both contrasts are oriented failure -> natural numerical success. Their complete
vector differences remain distinct. Their half-sum is a contextual interaction
observation, not a repair vector or proof of a capability. All compatible ports
in every corresponding operation are included, with unmatched history explicit.
"""
import collections
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from pipeline.native_records.compare_occurrences import read_frame, source, structure
from pipeline.native_records.native_trace import snapshot_values
from pipeline.acquisition.run_a_baseline import ROOT, RUN


@contextmanager
def compressed_writer(path):
    with path.open('xb') as output:
        proc = subprocess.Popen(['zstd', '-q', '-1', '-T1', '-c'], stdin=subprocess.PIPE, stdout=output)
        failed = False
        try:
            yield proc.stdin
        except BaseException:
            failed = True
            raise
        finally:
            proc.stdin.close()
            if proc.wait() != 0 and not failed:
                raise RuntimeError('Lossless derived-record compression failed')


@contextmanager
def decompressed(path):
    proc = subprocess.Popen(['zstd', '-q', '-d', '-c', str(path)], stdout=subprocess.PIPE)
    failed = False
    try:
        yield proc.stdout
    except BaseException:
        failed = True
        raise
    finally:
        proc.stdout.close()
        code = proc.wait()
        if code != 0 and not failed:
            raise RuntimeError('Compressed source could not be read completely')


class OrderedDeltas:
    def __init__(self, stream):
        self.stream, self.position = stream, 0

    def read(self, record, wanted):
        if record['status'] != 'compared':
            return None
        start, size = record['offset_bytes'], record['bytes']
        if start < self.position:
            raise ValueError('Source differences are not in recorded order')
        while self.position < start:
            raw = self.stream.read(min(8*1024**2, start-self.position))
            if not raw:
                raise ValueError('Truncated source difference stream')
            self.position += len(raw)
        raw = self.stream.read(size)
        if len(raw) != size:
            raise ValueError('Truncated source vector')
        self.position += size
        if not wanted:
            return None
        if hashlib.sha256(raw).hexdigest() != record['sha256']:
            raise ValueError('Source vector hash differs from the recorded contrast')
        return np.frombuffer(raw, dtype='<f4').astype(np.float64)


def main():
    index = json.loads((RUN/'response-correspondence/index.json').read_text())
    def response_pair(left, right, relation):
        item = next(x for x in index['pairs'] if x['left'] == left and x['right'] == right and x['relation'] == relation)
        return json.loads((RUN/'response-correspondence'/item['file']).read_text()), Path(item['file']).name
    across, _ = response_pair('block-changed', 'discovery-13-channel-h1', 'failure_overlay')
    _, second_name = response_pair('discovery-13-channel-h1', 'discovery-13-channel-h3', 'natural_failure_success')
    second = json.loads((RUN/'execution-correspondence'/second_name).read_text())
    second_by_prediction = {x['left_decision']: x for x in second['frames'] if 'left_decision' in x}
    pilot = json.loads((RUN/'pilot-connected-contrast/comparison.json').read_text())
    across_predictions = {x['left_prediction']: x['right_prediction'] for x in across['predictions']}
    quad = {}
    for i, pair in enumerate(pilot['mapping']['frames']):
        if 'left_decision' not in pair:
            other = next((x for x in second['frames'] if x['role'] == 'prefill' and x['left_execution'] == pair['left_execution']), None)
        else:
            other = second_by_prediction.get(across_predictions.get(pair['left_decision']))
        if other is not None:
            quad[i] = dict(first=pair, second=other)
    left = ROOT/index['cases']['discovery-13-channel-h1']['capture']
    right = ROOT/index['cases']['discovery-13-channel-h3']['capture']
    cohort = json.loads((ROOT/'research/experiments/signal-trace-repair/A-cohort.json').read_text())
    selected = [next(x for x in cohort['cases'] if x['id'] == cid) for cid in ['discovery-13-channel-h1','discovery-13-channel-h3']]
    audits = [json.loads((ROOT/x['audit']).read_text()) for x in selected]
    frame_indices = [{x['execution']: x for x in a['frames']} for a in audits]
    output = RUN/'opposite-height-observation'
    output.mkdir(exist_ok=False)
    counts = collections.Counter()
    recorded_weights = set()
    offset, loaded = 0, None
    prefill_embedding = []
    with decompressed(RUN/'pilot-connected-contrast/contrasts.jsonl.zst') as lines, \
            decompressed(RUN/'pilot-connected-contrast/contrasts.bin.zst') as binary, \
            compressed_writer(output/'mean-differences.f32.zst') as data, \
            compressed_writer(output/'observations.jsonl.zst') as records:
        deltas = OrderedDeltas(binary)
        for line in lines:
            row = json.loads(line)
            pair_id = row['pair']
            wanted = pair_id in quad
            first_values = {}
            for port in row['inputs']:
                if 'contrast' in port:
                    first_values[port['port']] = deltas.read(port['contrast'], wanted)
            first_output = deltas.read(row['output'], wanted)
            if not wanted:
                counts['pilot_operations_without_four_way_correspondence'] += 1
                continue
            if pair_id != loaded:
                q = quad[pair_id]['second']
                frames = []
                for side, base in enumerate([left,right]):
                    entry = frame_indices[side][q['left_execution' if side == 0 else 'right_execution']]
                    frames.append(read_frame(base/'signal-events.jsonl',(entry['byte_start'],entry['byte_end'])))
                if structure(frames[0]) != structure(frames[1]):
                    raise ValueError('Second pair has different operation/source-port structure')
                loaded = pair_id
                counts['paired_execution_quadruples'] += 1
                if counts['paired_execution_quadruples'] % 25 == 0:
                    print(json.dumps(dict(execution_quadruples=counts['paired_execution_quadruples'], compared_vectors=counts['compared_vectors'])),flush=True)
            e, f = frames[0][row['local_operation']], frames[1][row['local_operation']]
            if (e['name'],e['operation']) != (row['name'],row['operation']):
                raise ValueError('Four-way operation identities differ')
            def observe(first, description, a, b):
                nonlocal offset
                if first is None:
                    counts['not_compared'] += 1
                    return dict(status='not_compared', first_status=description['status'])
                if a is None or b is None or a['shape'] != b['shape'] or a['shape'] != description['shape']:
                    counts['different_shapes_or_missing_snapshot'] += 1
                    return dict(status='different_shapes_or_missing_snapshot')
                av = snapshot_values(left/'signal-data.bin',a).astype(np.float64)
                bv = snapshot_values(right/'signal-data.bin',b).astype(np.float64)
                if not np.all(np.isfinite(av)) or not np.all(np.isfinite(bv)):
                    counts['nonfinite'] += 1
                    return dict(status='native_nonfinite_values_retained')
                second_delta = bv-av
                n1, n2 = float(np.linalg.norm(first)), float(np.linalg.norm(second_delta))
                raw = ((first+second_delta)/2).astype('<f4').tobytes()
                result = dict(status='compared', shape=a['shape'], first_offset=description['offset_bytes'],
                              first_sha256=description['sha256'], second_left_snapshot=a, second_right_snapshot=b,
                              first_norm=n1, second_norm=n2,
                              direction_agreement=float(np.dot(first,second_delta)/(n1*n2)) if n1 and n2 else None,
                              mean_offset=offset, mean_bytes=len(raw), mean_sha256=hashlib.sha256(raw).hexdigest(),
                              mean_norm=float(np.linalg.norm((first+second_delta)/2)))
                data.write(raw);offset+=len(raw);counts['compared_vectors']+=1
                return result
            ports = []
            for p in row['inputs']:
                port = p['port'];a,b=e['inputs'][port],f['inputs'][port]
                if a is None or b is None:
                    raise ValueError('A pilot source port is absent in the second pair')
                entry=dict(port=port, first_left=p['left'], first_right=p['right'],
                           second_left=source(a), second_right=source(b))
                for value in (a,b):
                    if value and 'weight' in value:
                        recorded_weights.add(value['weight']['name'])
                if 'contrast' in p:
                    entry['contrast']=observe(first_values.get(port),p['contrast'],a.get('snapshot'),b.get('snapshot'))
                ports.append(entry)
            out=observe(first_output,row['output'],e['output'],f['output'])
            observation=dict(pilot_pair=pair_id,quadruple=quad[pair_id],local_operation=row['local_operation'],
                             name=e['name'],operation=e['operation'],first_left_event=row['left_event'],
                             first_right_event=row['right_event'],second_left_event=e['event'],second_right_event=f['event'],
                             inputs=ports,output=out)
            records.write((json.dumps(observation,allow_nan=False)+'\n').encode())
            counts['compared_operation_occurrences']+=1
            if e['name']=='embd' and quad[pair_id]['first'].get('role')=='prefill':
                prefill_embedding.append(dict(pilot_pair=pair_id,**{k:out.get(k) for k in ['status','first_norm','second_norm','direction_agreement','mean_norm']}))
        # Verify the source decompressor reached its real end, not just the last
        # selected vector. An unexpected trailing byte means the index is incomplete.
        if binary.read(1):
            raise ValueError('Unindexed bytes remain in the pilot vector stream')
    report=dict(scope='Two opposing input contrasts through compatible recorded operations; not a capability finding or a repair candidate',
                counts=dict(counts),distinct_failed_questions=2,distinct_natural_successes=2,
                total_construction_failures=21,full_cohort_analysis_complete=False,new_weight_edits=0,
                changed_input='First pair rise 3 -> 1; second pair rise 1 -> 3; other parameters differ between pairs',
                mean_interpretation='Half-sum in each operation coordinate system; a contextual interaction, not an edit direction',
                mean_uncompressed_bytes=offset,weight_bindings=len(recorded_weights),weights=sorted(recorded_weights),
                prefill_embedding=prefill_embedding,quadruples=quad,
                source_pilot='pilot-connected-contrast',source_second_left=str(left.relative_to(ROOT)),source_second_right=str(right.relative_to(ROOT)),
                source_second_event_hashes=[a['events_sha256'] for a in audits],
                cache_limit='Unmatched cache views remain unmatched here. Original writers/masks/history and the first pair cache comparison remain available; second pair history comparison is separate.')
    (output/'observation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['weights','quadruples']},indent=2))


if __name__=='__main__':
    main()
