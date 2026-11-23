"""Index supplied token rows, verified against the recorded native i32 bytes."""
import hashlib
import itertools
import json
import struct


def verify_i32(snapshot, values):
    """Verify the entire candidate sequence against every native chunk hash.

    This can operate after raw capture archiving: the actual recorder's snapshot
    hashes, not only the expected input history, decide whether the mapping holds.
    Unsupported layouts are not guessed.
    """
    raw = struct.pack('<' + 'i' * len(values), *values)
    if (snapshot['type'] != 'i32' or snapshot['shape'] != [len(values), 1, 1, 1]
            or snapshot['strides'][0] != 4 or snapshot['bytes'] != len(raw)):
        raise ValueError('Unsupported native input-index layout')
    offset = 0
    for chunk in snapshot['chunks']:
        part = raw[offset:offset + chunk['bytes']]
        if len(part) != chunk['bytes'] or hashlib.sha256(part).hexdigest() != chunk['sha256']:
            raise ValueError('Expected supplied sequence differs from the native snapshot hash')
        offset += len(part)
    if offset != len(raw):
        raise ValueError('Native snapshot hashes do not cover the complete index buffer')


def index_context(events_path, audit, prompt_tokens, generated_tokens):
    history = prompt_tokens + generated_tokens
    audited = {f['execution']: f for f in audit['frames']}
    frames, predictions, seen, prefill_rows = [], {}, set(), 0
    digest = hashlib.sha256()
    with open(events_path, 'rb') as source:
        def events():
            for line in source:
                digest.update(line)
                yield json.loads(line)
        for execution, group in itertools.groupby(events(), key=lambda e: e['graph_execution']):
            if execution in seen or execution not in audited:
                raise ValueError('Repeated or unaudited graph execution')
            seen.add(execution)
            frame = list(group)
            entry = audited[execution]
            if len(frame) != entry['event_count'] or any(e['phase'] != entry['phase'] for e in frame):
                raise ValueError('Execution disagrees with its completed capture audit')
            token_inputs = [(e['event'], i, p['snapshot']) for e in frame
                            for i, p in enumerate(e['inputs']) if p and p['layout']['name'] == 'inp_tokens'
                            and 'snapshot' in p]
            positions_inputs = [(e['event'], 1, e['inputs'][1]['snapshot']) for e in frame
                                if e['operation'] == 'ROPE' and len(e['inputs']) > 1
                                and e['inputs'][1] and 'snapshot' in e['inputs'][1]]
            if not token_inputs or not positions_inputs:
                raise ValueError('Input tokens and native rotary-position buffers are both required')
            n = token_inputs[0][2]['shape'][0]
            if entry['phase'] == 'prefill':
                positions = list(range(prefill_rows, prefill_rows + n))
                prefill_rows += n
                if prefill_rows > len(prompt_tokens):
                    raise ValueError('Prefill rows exceed the supplied prompt')
            else:
                if entry['prediction'] is None or n != 1 or prefill_rows != len(prompt_tokens):
                    raise ValueError('Unsupported decode input correspondence')
                positions = [len(prompt_tokens) + entry['prediction'] - 1]
            tokens = [history[p] for p in positions]
            for _, _, snapshot in token_inputs:
                verify_i32(snapshot, tokens)
            for _, _, snapshot in positions_inputs:
                verify_i32(snapshot, positions)
            if entry['prediction'] is not None:
                expected_last = len(prompt_tokens) + entry['prediction'] - 1
                if positions[-1] != expected_last:
                    raise ValueError('Prediction does not follow its supplied history')
                predictions[str(entry['prediction'])] = execution
            frames.append(dict(execution=execution, phase=entry['phase'], prediction=entry['prediction'],
                               input_positions=positions, tokens=tokens,
                               token_snapshot_events=[[e,p] for e,p,_ in token_inputs],
                               position_snapshot_events=[[e,p] for e,p,_ in positions_inputs],
                               native_index_hashes_verified=True))
    if seen != set(audited) or prefill_rows != len(prompt_tokens):
        raise ValueError('The context index does not cover the complete audited execution')
    if 'events_sha256' in audit and audit['events_sha256'] != digest.hexdigest():
        raise ValueError('Event file differs from the completed capture audit')
    return dict(frames=frames, prediction_executions=predictions,
                events_sha256=digest.hexdigest(), supplied_prompt_rows=prefill_rows,
                scope='Actual input-token and positional-index buffers verified against native snapshot hashes')
