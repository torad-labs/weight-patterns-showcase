"""Inspect A's recorded progressions with required climb energy held fixed.

Enumerate every same-card, one-factor contrast from the already selected cohort.
Speed or dissipation varies; mass and both heights remain fixed. These are
contextual observations, never identical-input counterfactuals or edit targets.
"""
import itertools
import json
from pathlib import Path

import numpy as np

from pipeline.native_records.group_observation import publish
from pipeline.acquisition.prepare_a_native_excerpts import OUTPUT


def enumerate_pairs(cases):
    pairs=[]
    for a,b in itertools.combinations(cases,2):
        if a['role']=='other_failure' or b['role']=='other_failure' or a['gpu']!=b['gpu']:
            continue
        changed=[k for k in a['parameters'] if a['parameters'][k]!=b['parameters'][k]]
        if len(changed)!=1 or changed[0] not in ['v','loss']:
            continue
        if a['role']!=b['role']:
            if a['role']!='failure':a,b=b,a
            relation='failure_success'
        else:
            relation='failure_failure' if a['role']=='failure' else 'success_success'
            if a['parameters'][changed[0]]<b['parameters'][changed[0]]:a,b=b,a
        pairs.append(dict(left=a['id'],right=b['id'],gpu=a['gpu'],factor=changed[0],relation=relation,
                          left_parameters=a['parameters'],right_parameters=b['parameters'],
                          fixed_required_energy=a['expected']['requiredJ'],
                          delta_available_energy=a['expected']['availableJ']-b['expected']['availableJ']))
    return pairs


def unit(x):
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-30)


def main():
    source=OUTPUT/'turn-fields'
    index=json.loads((source/'index.json').read_text())
    fields=np.load(source/'changes.npy',mmap_mode='r')
    records=index['records']
    lookup={(r['case'],r['anchor'],r['offset']):i for i,r in enumerate(records)}
    cohort=json.loads(Path('research/experiments/signal-trace-repair/A-cohort.json').read_text())
    pairs=enumerate_pairs(cohort['cases'])
    samples=sorted({(r['anchor'],r['offset']) for r in records})
    out=OUTPUT/'factor-contrasts'
    out.mkdir(exist_ok=True)
    rows=[]; missing=[]; transfer=[]
    for anchor,offset in samples:
        for gpu in [0,1]:
            available=[]; diffs=[]
            for number,p in enumerate(pairs):
                if p['gpu']!=gpu:continue
                a,b=(lookup.get((p[k],anchor,offset)) for k in ['left','right'])
                if a is None or b is None:
                    missing.append(dict(pair=number,anchor=anchor,offset=offset))
                    continue
                available.append(number)
                diffs.append(fields[a].astype('f8')-fields[b].astype('f8'))
            if not diffs:continue
            delta=np.asarray(diffs);directions=unit(delta)
            for factor in ['loss','v']:
                fs=[i for i,n in enumerate(available) if pairs[n]['relation']=='failure_success' and pairs[n]['factor']==factor]
                ss=[i for i,n in enumerate(available) if pairs[n]['relation']=='success_success' and pairs[n]['factor']==factor]
                if not fs:continue
                mean=directions[fs].mean(0);axis=unit(mean)
                projection=np.einsum('pnd,nd->pn',delta,axis)
                for j,c in enumerate(index['connections']):
                    rows.append(dict(anchor=anchor,offset=offset,gpu=gpu,factor=factor,layer=c['layer'],kind=c['kind'],
                                     failure_success_pairs=[available[i] for i in fs],success_success_pairs=[available[i] for i in ss],
                                     directional_coherence=float(np.linalg.norm(mean[j])),
                                     failure_success_projection_median=float(np.median(projection[fs,j])),
                                     success_success_absolute_projection_median=float(np.median(np.abs(projection[ss,j]))) if ss else None))
            # Predict the sign of the other factor's native directional contrast.
            # All pairs sharing either response are removed from the reference.
            for i,n in enumerate(available):
                p=pairs[n]
                if p['relation']!='failure_success':continue
                excluded={p['left'],p['right']}
                refs=[j for j,m in enumerate(available) if pairs[m]['relation']=='failure_success' and pairs[m]['factor']!=p['factor'] and not excluded.intersection({pairs[m]['left'],pairs[m]['right']})]
                if not refs:continue
                reference=unit(directions[refs].mean(0))
                cosine=np.einsum('nd,nd->n',directions[i],reference)
                transfer.append(dict(anchor=anchor,offset=offset,gpu=gpu,pair=n,references=[available[j] for j in refs],cosines=cosine.tolist()))
    publish(out/'index.json',dict(pairs=pairs,observations=rows,cross_factor_transfer=transfer,missing=missing,
                                 source=str(source/'index.json'),connections=index['connections'],
                                 observed_cases=sorted({r['case'] for r in records}),
                                 definitions=dict(direction='Difference between the two recorded merge direction changes, separately at each operation and prediction. Pair differences are normalized only for descriptive coherence; native magnitudes remain available.',
                                                  transfer='Cosine with the other factor contrast, excluding every pair sharing either response. Same card only; no fitting on token outputs.',
                                                  missing='Absent literal context or unavailable branch view; kept missing, never interpolated.'),
                                 limit='Descriptive recurrence and counterexamples, not proof of a capability, a causal repair, or a unique weight edit. Pairs sharing questions are dependent and are not independent sample counts.'))
    print(json.dumps(dict(pairs=len(pairs),observations=len(rows),transfer_checks=len(transfer),observed_cases=len({r['case'] for r in records}))),flush=True)


if __name__=='__main__':main()
