"""Join the frozen case manifest to independent, blinded readings and native receipts."""
from collections import Counter
from hashlib import sha256
from pathlib import Path
import json

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[3]
OPERATIONS = ('initial', 'downhill', 'bottom', 'friction', 'available', 'required', 'verdict', 'full_numerical_chain')


def load(path):
    return json.loads(path.read_text())


def status(row, operation):
    if operation == 'full_numerical_chain':
        return row[operation]
    return row[operation]['status']


def main():
    declaration = load(RUN / 'declaration.json')
    panel = load(RUN / 'panel.json')
    states = declaration['states']
    cases = [row['id'] for row in panel]
    held = [row['id'] for row in panel if 'physics' in row]
    readings = load(RUN / 'blinded-readings.json')
    responses = load(RUN / 'blinded-responses.json')
    keys = load(RUN / 'unblinding.json')
    expected = {f'{state}/{case}' for state in states for case in cases}
    if set(keys) != expected or set(keys.values()) != set(readings) or set(responses) != set(readings):
        raise ValueError('Incomplete blind readings or join')
    rows = {}
    records = {}
    declared = sha256((RUN / 'declaration.json').read_bytes()).hexdigest()
    for key, blind_id in keys.items():
        verified = load(RUN / key / 'verified.json')
        if f"{verified['state']}/{verified['member']}" != key or verified['declaration_sha256'] != declared:
            raise ValueError(f'Wrong measurement identity for {key}')
        ref = verified['records']['result.json']
        raw = (ROOT / ref['record']).read_bytes()
        result = json.loads(raw)
        row = readings[blind_id]
        if (sha256(raw).hexdigest() != ref['sha256'] or
                sha256(result['content'].encode()).hexdigest() != row['response_sha256'] or
                result['content'] != responses[blind_id]['generated'] or
                row['ended'] != result['ended'] or row['stopped_at_limit'] != result['stopped_limit']):
            raise ValueError(f'Blind judgment does not match the native generation for {key}')
        rows[key] = row
        records[key] = {'blind_id': blind_id, 'verified_sha256': sha256((RUN / key / 'verified.json').read_bytes()).hexdigest(), 'records': verified['records']}
    totals = {}
    transitions = {}
    per_family = {}
    for state in states:
        totals[state] = {}
        transitions[state] = {}
        for op in OPERATIONS:
            statuses = [status(rows[f'{state}/{c}'], op) for c in held]
            totals[state][op] = dict(Counter(statuses))
            transitions[state][op] = dict(Counter(f"{status(rows['original/'+c], op)} -> {s}" for c,s in zip(held,statuses)))
        per_family[state] = {}
        for row in panel:
            if 'physics' in row:
                p = row['physics']
                family = f"v{p['v']}-d{p['drop']}"
                per_family[state].setdefault(family, {})[row['id']] = status(rows[state+'/'+row['id']], 'full_numerical_chain')
    summary = {
        'declaration_sha256': declared, 'states': states, 'novel_cases': held, 'control_cases': [c for c in cases if c not in held],
        'novel_operation_counts': totals, 'paired_novel_changes_from_original': transitions, 'novel_chains_grouped_by_family': per_family,
        'physical_configurations': len(per_family['original']), 'verified_native_record_count': len(rows), 'unique_blind_readings': len(readings),
        'capped': [key for key, row in rows.items() if row['stopped_at_limit']], 'records': records,
    }
    (RUN / 'paired-outcomes.json').write_text(json.dumps(summary, indent=1) + '\n')


if __name__ == '__main__':
    main()
