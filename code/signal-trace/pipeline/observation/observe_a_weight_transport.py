"""Follow recorded signal differences through the actual native stored matrices.

For each matrix, c_j = dx_j * (W.T @ dy)_j records the signed contribution
of incoming coordinate j to the observed output difference's squared length.
It is a functional transport observation, not semantic weight ownership or a
causal attribution of the mistake. Full input/output vectors and links remain.
"""
import json
from pathlib import Path

import numpy as np

from pipeline.native_records.group_observation import publish
from pipeline.native_records.native_trace import checkpoint_values
from pipeline.observation.observe_a_factor_contrasts import enumerate_pairs
from pipeline.acquisition.prepare_a_native_excerpts import OUTPUT


MODEL=Path('models/gemma4/gemma4-e2b/gguf-bf16-bartowski/google_gemma-4-E2B-it-bf16.gguf')


def main():
    cohort=json.loads(Path('research/experiments/signal-trace-repair/A-cohort.json').read_text())
    eligible={c['id'] for c in cohort['cases'] if c['role']!='other_failure'}
    ready={p.parent.parent.name:p for p in OUTPUT.glob('*/weighted-paths/index.json')}
    if set(ready)!=eligible:raise ValueError('All eligible individual weighted paths are required')
    pairs=enumerate_pairs(cohort['cases']);views={};bindings={};operations=set()
    for cid,p in ready.items():
        d=json.loads(p.read_text());data=np.memmap(p.parent/'vectors.f32',dtype='<f4',mode='r')
        records={(r['anchor'],r['name']):r for r in d['records']}
        views[cid]=(records,data,p)
        for r in d['records']:
            key=(r['weight']['name'],r['name']);operations.add(key)
            if key in bindings and bindings[key]!=r['weight']:raise ValueError('Weight binding disagreement')
            bindings[key]=r['weight']
    out=OUTPUT/'weight-transport';out.mkdir(exist_ok=True)
    summaries=[]
    model_stat=MODEL.stat()
    for number,key in enumerate(sorted(operations)):
        destination=out/f'{number:03}.json'
        if destination.exists():
            summaries.append(json.loads(destination.read_text())['summary']);continue
        binding=bindings[key];shape=binding['shape']
        if int(np.prod(shape[2:]))!=1:raise ValueError('Unexpected batched matrix')
        weight=checkpoint_values(MODEL,binding).reshape(shape[1],shape[0]).astype(np.float64)
        rows=[];inputs=[];outputs=[]
        for pair_index,pair in enumerate(pairs):
            lr,ld,lp=views[pair['left']];rr,rd,rp=views[pair['right']]
            anchors=sorted({a for a,n in lr if n==key[1]}&{a for a,n in rr if n==key[1]})
            for anchor in anchors:
                l,r=lr[anchor,key[1]],rr[anchor,key[1]]
                if l['weight']!=binding or r['weight']!=binding:raise ValueError('Operation binding differs')
                def vector(data,record,which):
                    offset=record[which+'_offset'];count=record[which+'_elements']
                    return data[offset:offset+count].astype(np.float64)
                lx,rx=vector(ld,l,'input'),vector(rd,r,'input')
                ly,ry=vector(ld,l,'output'),vector(rd,r,'output')
                inputs.append(lx-rx);outputs.append(ly-ry)
                rows.append(dict(pair=pair_index,anchor=anchor,left_prediction=l['prediction'],right_prediction=r['prediction'],
                                 left_event=l['event'],right_event=r['event'],left_source=str(lp),right_source=str(rp),
                                 baseline_output_scale=float((np.linalg.norm(ly)+np.linalg.norm(ry))/2)))
        dx=np.asarray(inputs);dy=np.asarray(outputs)
        predicted=dx@weight.T
        energy=np.einsum('ij,ij->i',dy,dy)
        contribution=dx*(dy@weight)
        share=np.divide(contribution,energy[:,None],out=np.zeros_like(contribution),where=energy[:,None]>1e-30)
        error=np.linalg.norm(predicted-dy,axis=1)
        for i,row in enumerate(rows):
            row.update(output_difference_norm=float(np.sqrt(energy[i])),
                       reconstructed_difference_error=float(error[i]),
                       error_relative_to_original_output=float(error[i]/max(row['baseline_output_scale'],1e-30)),
                       signed_share_sum=float(share[i].sum()),zero_difference=bool(energy[i]<=1e-30))
        np.save(out/f'{number:03}-shares.npy',share.astype('<f4'))
        summary=dict(index=number,weight=key[0],operation=key[1],native_type=binding['native_type'],shape=shape,observations=len(rows),
                     largest_error_relative_to_original_output=max(r['error_relative_to_original_output'] for r in rows),
                     median_error_relative_to_difference=float(np.median(error/np.maximum(np.sqrt(energy),1e-30))))
        publish(destination,dict(summary=summary,weight=binding,rows=rows,contribution_shape=list(share.shape),
                                 definition='dx_j*(W.T@dy)_j / ||dy||^2 using actual recorded input/output differences and the stored native matrix. Shared residual-coordinate labels are not assumed across different operators.',
                                 limit='All context-induced differences contribute, including legitimate numerical and wording changes. This is not an error-specific direction or a recipe for changing weights.'))
        summaries.append(summary)
        print(json.dumps(summary),flush=True)
    end=MODEL.stat()
    if (end.st_size,end.st_mtime_ns,end.st_ino)!=(model_stat.st_size,model_stat.st_mtime_ns,model_stat.st_ino):raise ValueError('Checkpoint changed during observation')
    publish(out/'index.json',dict(model=str(MODEL),model_size=model_stat.st_size,model_mtime_ns=model_stat.st_mtime_ns,
                                 pairs=pairs,operations=summaries,new_weight_edits=0,
                                 scope='Every recorded layer-bound weighted vector projection at the declared context positions, for all enumerated same-card one-factor contrasts. Native BF16/F32 values read as declared; all work is read-only.'))


if __name__=='__main__':main()
