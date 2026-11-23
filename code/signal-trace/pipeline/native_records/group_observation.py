"""Observe complete corresponding vectors in one natural-response group.

The group mean is a descriptive view at a recorded operation, not the trace
itself, a capability label, or an edit target. Members retain their original
native vectors and the actual producer/cache/weight connections in their
immutable source recordings. No different operations or decisions are averaged
together. Cache rows without a group history mapping remain source observations.
"""


from collections import OrderedDict
from contextlib import contextmanager, nullcontext
import hashlib
import itertools
import json
import math
import mmap
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from pipeline.native_records.compare_occurrences import structure
from pipeline.native_records.native_trace import _values


def publish(path, value):
    temporary = path.with_suffix(path.suffix + '.writing')
    temporary.write_text(json.dumps(value, separators=(',', ':')) + '\n')
    temporary.replace(path)


@contextmanager
def event_stream(path):
    path = Path(path)
    if path.suffix != '.zst':
        with path.open('rb') as stream:
            yield stream
        return
    process = subprocess.Popen(['zstd', '-q', '-dc', str(path)], stdout=subprocess.PIPE)
    try:
        yield process.stdout
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.terminate()
        result = process.wait()
        if result not in (0, -15, -13):
            raise ValueError('Could not read the lossless event recording')


def frames(path, expected_sha=None):
    digest = hashlib.sha256()
    def rows(stream):
        for line in stream:
            digest.update(line)
            yield json.loads(line)
    with event_stream(path) as stream:
        for fid, group in itertools.groupby(rows(stream), key=lambda e: e['graph_execution']):
            yield fid, list(group)
    if expected_sha and digest.hexdigest() != expected_sha:
        raise ValueError('Event recording differs from its acquisition audit')


def layout_of(events):
    outputs = []
    offset = change_offset = 0
    for event in events:
        out = event['output']
        reason = None
        if out['name'].startswith('cache_'):
            reason = 'Cache rows need their separate history correspondence; original vectors and writers retained'
        elif out['type'] not in ('f32', 'f16', 'bf16'):
            reason = 'Runtime control values, not signal-vector coordinates'
        count = 0 if reason else math.prod(out['shape'])
        item = dict(name=event['name'], operation=event['operation'], offset=offset,
                    elements=count, reason=reason)
        if not reason:
            item.update(shape=out['shape'], native_type=out['type'])
        incoming = event['inputs'][0] if event['inputs'] else None
        item['addition_change'] = bool(count and event['operation']=='ADD' and incoming
            and incoming.get('snapshot', {}).get('shape')==out['shape'])
        if item['addition_change']:
            item['change_offset'] = change_offset
            change_offset += count
        outputs.append(item)
        offset += count
    return dict(schema=structure(events), outputs=outputs, elements=offset, change_elements=change_offset)


class NativeReader:
    def __init__(self, path):
        self.file = Path(path).open('rb')
        self.data = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        self.verified = OrderedDict()

    def values(self, snapshot):
        chunks = []
        for chunk in snapshot['chunks']:
            start, size = chunk['offset'], chunk['bytes']
            if start < 0 or start + size > len(self.data):
                raise ValueError('Native chunk is outside the recorded file')
            raw = self.data[start:start+size]
            key = (start, size, chunk['sha256'])
            if key not in self.verified:
                if hashlib.sha256(raw).hexdigest() != chunk['sha256']:
                    raise ValueError('Native chunk differs from the recorded snapshot')
                self.verified[key] = True
                if len(self.verified) > 16384:
                    self.verified.popitem(last=False)
            chunks.append(raw)
        raw = b''.join(chunks)
        if len(raw) != snapshot['bytes']:
            raise ValueError('Snapshot byte count differs from its chunks')
        return _values(raw, snapshot['type'], snapshot['shape'], snapshot['strides'])

    def close(self):
        self.data.close()
        self.file.close()


class GroupObservation:
    def __init__(self, destination):
        self.path = Path(destination)
        self.layout = json.loads((self.path/'layout.json').read_text())
        self.state = json.loads((self.path/'state.json').read_text())
        self.sums = np.memmap(self.path/'vector-sums.f32', dtype='<f4', mode='r+',
                              shape=(self.layout['elements'],))
        self.stats = np.memmap(self.path/'location-statistics.f64', dtype='<f8', mode='r+',
                               shape=(self.layout['locations'], 3))
        self.changes = (np.memmap(self.path/'addition-change-sums.f32', dtype='<f4', mode='r+',
                       shape=(self.layout['change_elements'],)) if self.layout['change_elements'] else None)

    @classmethod
    def create(cls, destination, events, identity, storage=None):
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=False)
        (destination/'members').mkdir()
        templates, frame_layouts = {}, {}
        elements = change_elements = locations = 0
        for fid, frame in frames(events, identity.get('reference_events_sha256')):
            template = layout_of(frame)
            encoded = json.dumps(template, separators=(',', ':')).encode()
            key = hashlib.sha256(encoded).hexdigest()
            templates.setdefault(key, template)
            frame_layouts[str(fid)] = dict(template=key, value_offset=elements,
                row_offset=locations, source_first_event=frame[0]['event'],
                phase=frame[0]['phase'], input_position=frame[0]['input_position'],
                change_offset=change_elements)
            elements += template['elements']
            change_elements += template['change_elements']
            locations += len(frame)
        if not elements:
            raise ValueError('No recorded signal-vector outputs')
        layout = dict(identity=identity, reference_events=str(events), frames=frame_layouts,
                      templates=templates, elements=elements, change_elements=change_elements, locations=locations,
                      representation='Full float32 vector sums at each corresponding output; each addition computes in float64 and rounds to float32 storage. Divide by its member count. Original member inputs/outputs remain exact lazy source records.',
                      rounding_limit='These descriptive group sums have ordinary float32 accumulation error; original native vectors are unchanged and must be used to verify any proposed small difference.')
        publish(destination/'layout.json', layout)
        files = [('vector-sums.f32', elements*4), ('location-statistics.f64', locations*3*8),
                 ('addition-change-sums.f32', change_elements*4)]
        required = sum(((size+4095)//4096)*4096 for _, size in files)
        with storage(destination, required) if storage else nullcontext():
            for name, size in files:
                with (destination/name).open('xb') as f:
                    # Reserve real blocks before observing a member. Truncation
                    # alone hides future writes from concurrent workers and df.
                    if size:
                        os.posix_fallocate(f.fileno(), 0, size)
        publish(destination/'state.json', dict(members=[], applying=None, complete=False))
        return cls(destination)

    def _row(self, execution, local):
        frame = self.layout['frames'][str(execution)]
        output = self.layout['templates'][frame['template']]['outputs'][local]
        return frame, output, frame['row_offset']+local

    def count(self, execution, local):
        return int(self.stats[self._row(execution, local)[2], 0])

    def schema(self, execution):
        frame = self.layout['frames'][str(execution)]
        return self.layout['templates'][frame['template']]['schema']

    def mean(self, execution, local):
        frame, output, row = self._row(execution, local)
        count = self.stats[row, 0]
        if not count:
            return None
        offset = frame['value_offset'] + output['offset']
        return np.asarray(self.sums[offset:offset+output['elements']], dtype=np.float64)/count

    def dispersion(self, execution, local):
        mean = self.mean(execution, local)
        if mean is None:
            return None
        row = self._row(execution, local)[2]
        return max(0.0, float(self.stats[row, 1]/self.stats[row, 0] - np.dot(mean, mean)))

    def addition_change(self, execution, local):
        frame, output, row = self._row(execution, local)
        count = self.stats[row, 0]
        if not count or not output['addition_change']:
            return None
        offset = frame['change_offset']+output['change_offset']
        return np.asarray(self.changes[offset:offset+output['elements']], dtype=np.float64)/count

    def add(self, member, events, native_data, correspondence, source=None, progress=None):
        if member in self.state['members']:
            raise ValueError('This question already contributed to its group')
        if self.state['applying'] or self.state['complete']:
            raise ValueError('Group is complete or has an interrupted contribution; do not silently resume partial sums')
        if not correspondence.get('relation') or correspondence.get('same_hardware') is False:
            raise ValueError('Within-group observation requires an explicit same-card relation')
        mapping = {}
        left_ids = set()
        for pair in correspondence['frames']:
            if not pair.get('reason'):
                raise ValueError('Every execution correspondence requires its reason')
            fid, reference = pair['right_execution'], pair['left_execution']
            if fid in mapping or reference in left_ids:
                raise ValueError('A question cannot contribute twice at one corresponding execution')
            mapping[fid] = reference
            left_ids.add(reference)
        expected_sha = correspondence.get('right_events_sha256')
        self.state['applying'] = member
        publish(self.path/'state.json', self.state)
        reader = NativeReader(native_data)
        missing, seen, matched = [], set(), 0
        norms = np.full(self.layout['locations'], np.nan, dtype='<f8')
        started = time.monotonic()
        try:
            for fid, frame in frames(events, expected_sha):
                if fid not in mapping:
                    continue
                reference = mapping[fid]
                target = self.layout['frames'][str(reference)]
                template = self.layout['templates'][target['template']]
                if structure(frame) != template['schema']:
                    raise ValueError(f'Actual operation/source-port structure differs at member execution {fid}, reference {reference}')
                cache = {}
                for local, (event, output) in enumerate(zip(frame, template['outputs'])):
                    if not output['elements']:
                        continue
                    snapshot = event['output']
                    row = target['row_offset']+local
                    if snapshot['shape'] != output['shape'] or snapshot['type'] not in ('f32','f16','bf16'):
                        missing.append(dict(execution=fid, reference=reference, local_operation=local,
                                            reason='Different native shape or unsupported value type', source_event=event['event']))
                        continue
                    key = (snapshot['type'], tuple(snapshot['shape']), tuple(snapshot['strides']),
                           tuple((c['offset'],c['bytes'],c['sha256']) for c in snapshot['chunks']))
                    values = cache.get(key)
                    if values is None:
                        values = reader.values(snapshot).astype(np.float64)
                        cache[key] = values
                    if not np.all(np.isfinite(values)):
                        self.stats[row, 2] += 1
                        missing.append(dict(execution=fid, reference=reference, local_operation=local,
                                            reason='Nonfinite native values retained; no finite mean invented', source_event=event['event']))
                        continue
                    change = None
                    if output['addition_change']:
                        if event['inputs'][0]['snapshot']['shape'] != snapshot['shape']:
                            missing.append(dict(execution=fid, reference=reference, local_operation=local,
                                reason='Addition input shape differs; no broadcast coordinate subtraction invented', source_event=event['event']))
                            continue
                        incoming = reader.values(event['inputs'][0]['snapshot']).astype(np.float64)
                        if not np.all(np.isfinite(incoming)):
                            missing.append(dict(execution=fid, reference=reference, local_operation=local,
                                reason='Nonfinite addition input; output and change remain source-only', source_event=event['event']))
                            self.stats[row,2] += 1
                            continue
                        change = values-incoming
                    offset = target['value_offset']+output['offset']
                    self.sums[offset:offset+output['elements']] += values
                    self.stats[row, 0] += 1
                    self.stats[row, 1] += float(np.dot(values, values))
                    norms[row] = np.linalg.norm(values)
                    if change is not None:
                        start = target['change_offset']+output['change_offset']
                        self.changes[start:start+output['elements']] += change
                seen.add(fid)
                matched += 1
                if progress and matched % 25 == 0:
                    progress(member=member, executions=matched, total=len(mapping), elapsed_seconds=time.monotonic()-started)
            if seen != set(mapping):
                raise ValueError('Correspondence names an execution absent from the recording')
            self.sums.flush()
            self.stats.flush()
            if self.changes is not None:
                self.changes.flush()
            np.save(self.path/'members'/(member+'-output-norms.npy'), norms)
            publish(self.path/'members'/(member+'.json'), dict(member=member,
                events=str(events), native_data=str(native_data), source=source,
                correspondence=correspondence, unmatched_locations=missing,
                matched_executions=matched, duration_seconds=time.monotonic()-started,
                addition_change='At actual compatible ADD operations only: complete output minus input port zero, retaining the other recorded operand and actual branches. Never a subtraction between projection coordinate spaces.',
                full_vectors='For every mapped execution/local operation use that member\'s original output and inputs, with their native chunk hashes, producer/cache references and weight bindings; summaries do not replace them.'))
            self.state['members'].append(member)
            self.state['applying'] = None
            publish(self.path/'state.json', self.state)
        finally:
            reader.close()

    def finish(self):
        if self.state['applying']:
            raise ValueError('An incomplete contribution is not a completed observation')
        self.sums.flush()
        self.stats.flush()
        if self.changes is not None:
            self.changes.flush()
        self.state['complete'] = True
        publish(self.path/'state.json', self.state)
        report = dict(identity=self.layout['identity'], members=self.state['members'],
                      location_count=self.layout['locations'], full_vector_elements=self.layout['elements'],
                      observed_locations=int(np.count_nonzero(self.stats[:,0])),
                      nonfinite_member_locations=int(self.stats[:,2].sum()),
                      scope='Separate natural-response group observation. Full vectors at corresponding recorded operations; no pair-difference fit, capability verdict or weight edit.',
                      variance='At each observed location: mean squared norm minus squared norm of mean. Counts and complete vectors remain attached to original members and connections.',
                      cache_limit='Cache vectors/writers/versions remain in original source records; no group cache mean without explicit history-row alignment.')
        publish(self.path/'observation.json', report)
        return report
