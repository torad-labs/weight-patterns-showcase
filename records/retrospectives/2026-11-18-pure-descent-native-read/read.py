"""Read the already generated native computations at the audited descent decision."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUNS = ROOT / 'studies/04-round-2-overlap/runs'
ARM = RUNS / 'early-27-prechoice-pullback/ffn20'
SELECTIONS = {'origin': (313, 327), 'edited': (391, 402)}


def read(path):
    return json.loads(path.read_bytes())


def checked(binding):
    path = ROOT / binding['record']
    with path.open('rb') as source:
        if hashlib.file_digest(source, 'sha256').hexdigest() != binding['sha256']:
            raise ValueError(f'Changed source {path}')
    return path


def projections(raw, requests, geometry):
    wanted = {(event, geometry[merge][role]) for _, event, merge, _, _ in requests
              for role in ('incoming', 'connection')}
    values = {}
    with (raw/'stream.f32').open('rb') as data, (raw/'stream.jsonl').open() as index:
        for line in index:
            rec = json.loads(line)
            if len(rec['positions']) != 1:
                continue
            key = (rec['positions'][0], rec['label'])
            if key not in wanted:
                continue
            if key in values or rec['shape'] != [1536, 1, 1, 1] or rec['strides'][0] != 4:
                raise ValueError(f'Ambiguous native row {key}')
            data.seek(rec['offset_bytes'])
            values[key] = np.frombuffer(data.read(1536*4), dtype='<f4').astype(np.float64)
    results = {}
    for name, event, merge, axis, target in requests:
        a, b = [values[event, geometry[merge][role]] for role in ('incoming', 'connection')]
        turn = b/np.linalg.norm(b) - a/np.linalg.norm(a)
        value = float(turn @ axis)
        results.setdefault(name, []).append(dict(event=event, merge=merge, projection=value,
                                               target=target, signed_residual=value-target))
    return results


def main():
    target = read(RUNS/'descent-presign-witness-target/target.json')
    ff_axis = np.load(checked(target['axis']))
    geometry = read(checked(read(RUNS/'same-input-operation-comparison/prefix-declaration.json')['geometry']))['merges']
    study = read(ARM.parent/'declaration.json')
    matcher = read(checked(study['matcher']))
    with np.load(checked(matcher['field'])) as field:
        cells = [(matcher['stage_order'][stage], merge, axis, offset)
                 for (stage, merge), axis, offset in zip(field['cells'], field['axes'], field['offsets'], strict=True)]
    output = {}
    for arm, (symbolic, numeric) in SELECTIONS.items():
        raw = ARM/arm/'raw/discovery-05-channel-h1'
        generation = read(raw/'result.json')
        steps = {s['input_position']: s for s in generation['steps']}
        if any(steps[pos]['piece'].strip() != '-' for pos in (symbolic, numeric)):
            raise ValueError('Audited signs do not match the bound generation')
        req = [('historical_FFN20', symbolic-1, target['merge'], ff_axis, target['absolute_turn_projection'])]
        for stage, merge, axis, offset in cells:
            row = (symbolic if stage['phase']=='symbolic' else numeric) + stage['offset']
            req.append((stage['phase'], row, int(merge), axis, float(offset)))
        measured = projections(raw, req, geometry)
        sign_logits = {x['id']: x['logit'] for x in steps[symbolic]['top']}
        sign_margin = sign_logits[900]-sign_logits[753] if {900,753} <= sign_logits.keys() else None
        output[arm] = dict(audited_symbolic_prediction=symbolic, audited_numeric_prediction=numeric,
                           diagnostic_add_minus_subtract_logit_margin=sign_margin,
                           generation=hashlib.sha256((raw/'result.json').read_bytes()).hexdigest(),
                           historical_FFN20=measured['historical_FFN20'][0],
                           mean_squared={key: float(np.mean([s['signed_residual']**2 for s in values]))
                                         for key,values in measured.items()},
                           symbolic_cells=measured['symbolic'])
    initial = read(ARM/'origin-measurement.json')['readings']
    for a, b in [('historical_FFN20','ffn20'),('symbolic','early8')]:
        if not np.isclose(output['origin']['mean_squared'][a],initial[b]['mean_squared_residual'],rtol=1e-10,atol=1e-15):
            raise ValueError('Independent row read differs from primary source')
    json.dump(output,sys.stdout,indent=2,allow_nan=False)
    print()


if __name__=='__main__':
    main()
