"""Summarize signed head/unit accounting without hiding cancellation or coverage."""
import argparse,csv
from pathlib import Path
import numpy as np
from pipeline.overlap.analyze import read,write,finite
from pipeline.research_guard.verify_round2 import verify,ROOT


def cosine(a,b):
    n=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/n) if n>0 else None


def main(output):
    if verify(ROOT):raise ValueError('Round-two verifier refused')
    output=Path(output);root=output/'components';index=read(root/'index.json');records=[];sharing=[];profiles={}
    maps={g:read(output/f'contrasts/gpu{g}/index.json') for g in (0,1)}
    metrics={g:np.load(output/f'contrasts/gpu{g}/metrics.npy') for g in maps}
    common={g:np.load(output/f'common-coordinates/gpu{g}/metrics.npy') for g in maps}
    coverage=[]
    merges=[op['merge'] for op in index['operators']]
    for gpu,meta in maps.items():
        for scope in ('all_aligned','full_population'):
            ids=[i for i,row in enumerate(meta['rows']) if scope=='all_aligned' or row['full_population']]
            raw=metrics[gpu][ids,:,0];shared=common[gpu][ids,:,1]
            coverage.append(dict(gpu=gpu,coverage=scope,raw_squared_share=raw[:,merges].sum()/raw.sum(),
                common_squared_share=shared[:,merges].sum()/shared.sum() if shared.sum() else None,
                merges_accounted=len(merges),merges_total=raw.shape[1]))
    for operator in index['operators']:
        layer,kind,merge=operator['layer'],operator['kind'],operator['merge'];folder=Path(operator['path'])
        for gpu,meta in maps.items():
            a=np.load(folder/f'gpu{gpu}.npz')
            for scope in ('all_aligned','full_population'):
                for axis in ('raw','common_axis'):
                    if axis=='raw':weights=metrics[gpu][:,merge,0];shares=a['raw_component_share'];context=a['raw_context_share']
                    else:weights=a['common_axis_raw_projection'];shares=a['common_axis_component_share'];context=a['common_axis_context_share']
                    role_vectors={}
                    for role in dict.fromkeys(r['role'] for r in meta['rows']):
                        ids=np.array([i for i,r in enumerate(meta['rows']) if r['role']==role and (scope=='all_aligned' or r['full_population'])],dtype=int)
                        if not len(ids):continue
                        w=weights[ids];valid=(w>0)&np.isfinite(shares[ids]).all(axis=1)&np.isfinite(context[ids]);denominator=w[valid].sum()
                        if denominator<=0:continue
                        chosen=ids[valid];w=weights[chosen]
                        vector=np.sum(shares[chosen]*w[:,None],axis=0)/denominator
                        residual=float(np.sum(context[chosen]*w)/denominator)
                        total_abs=float(np.abs(vector).sum());sq=float(np.sum(vector**2))
                        neff=total_abs**2/sq if sq else None
                        key=f'g{gpu}_l{layer}_{kind}_{scope}_{axis}_{role}';profiles[key]=vector
                        role_vectors[role]=vector
                        raw_sq=metrics[gpu][ids,merge,0]
                        rel=a['raw_reconstruction_relative'][ids]
                        rr=float(np.sqrt(np.nansum(raw_sq*rel**2)/raw_sq.sum())) if raw_sq.sum() else None
                        records.append(dict(gpu=gpu,layer=layer,kind=kind,merge=merge,coverage=scope,axis=axis,role=role,
                            positions=len(ids),positions_with_defined_axis=int(valid.sum()),component_count=len(vector),
                            effective_absolute_components=neff,absolute_component_sum=total_abs,signed_component_sum=float(vector.sum()),
                            context_share=residual,accounting_error=float(vector.sum()+residual-1),raw_relative_reconstruction=rr,
                            largest_positive_component=int(np.argmax(vector)),largest_positive_share=float(vector.max()),
                            largest_negative_component=int(np.argmin(vector)),largest_negative_share=float(vector.min()),profile_key=key))
                    if 'destination' in role_vectors:
                        for role,vector in role_vectors.items():
                            if role=='destination':continue
                            sharing.append(dict(gpu=gpu,layer=layer,kind=kind,coverage=scope,axis=axis,passage=role,
                                signed_component_profile_cosine_to_climb=cosine(vector,role_vectors['destination'])))
    write(root/'summary.json',finite(dict(records=records,sharing=sharing,coverage=coverage,
        definition='For each passage, sum each component projection over positions, divided by the summed raw contrast squared norm (raw axis) or summed raw contrast dot common gap (common axis). Signed components plus context should sum to one within native reconstruction error.',
        concentration='Effective absolute count = sum(abs(component shares)) squared / sum(component shares squared). Descriptive concentration, not a selected set of semantic units.',
        limitation='These profiles account for observed group differences. They do not predict ablating a component and are not yet controlled for speed.')))
    np.savez_compressed(root/'profiles.npz',**profiles)
    with (root/'summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);args=p.parse_args();main(args.output)
