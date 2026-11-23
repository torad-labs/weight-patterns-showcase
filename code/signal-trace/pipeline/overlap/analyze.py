"""Build separate original-model group overlaps and descriptive whole-vector comparisons."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np

from pipeline.paths import ROOT
from pipeline.native_records.run_declaration import check_native_run
from pipeline.overlap.statistics import group_statistics,contrast_statistics

SHARED=ROOT/'studies/04-round-2-overlap/inputs'

def read(p):return json.loads(Path(p).read_text())
def write(p,value):Path(p).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def finite(v):
    if isinstance(v,np.ndarray):return finite(v.tolist())
    if isinstance(v,list):return [finite(x) for x in v]
    if isinstance(v,dict):return {k:finite(x) for k,x in v.items()}
    if isinstance(v,(float,np.floating)):return float(v) if np.isfinite(v) else None
    if isinstance(v,(np.integer,)):return int(v)
    if isinstance(v,(np.bool_,)):return bool(v)
    return v

def build_members(source,correspondence):
    groups={};eligible={c for c,row in source['cases'].items() if row['role'] in ('failure','natural_numerical_success')}
    observed=[]
    for pair in source['pairs']:
        if pair['relation'] not in ('failure_overlay','natural_numerical_success_overlay'):continue
        c=pair['right'];meta=source['cases'][c];key=(meta['gpu'],meta['role'])
        if c not in eligible or not pair['same_hardware']:raise ValueError('Invalid group population')
        group=groups.setdefault(key,dict(reference=pair['left'],members=[],positions=defaultdict(list)))
        if group['reference']!=pair['left']:raise ValueError('Group reference changed')
        group['members'].append(c);observed.append(c)
        for row in read(correspondence/pair['file'])['predictions']:
            group['positions'][row['left_prediction']].append([c,row['right_prediction']])
    if set(observed)!=eligible or len(observed)!=len(eligible):raise ValueError('Population must be enumerated exactly once')
    for group in groups.values():
        for rows in group['positions'].values():
            if len(rows)!=len({r[0] for r in rows}):raise ValueError('Repeated member at a group position')
    return groups

def main(output):
    output=Path(output).resolve();declaration=output/'declaration.json'
    check_native_run(output,declaration)
    if (output/'groups').exists():raise ValueError('Never overwrite an overlap run')
    correspondence=SHARED/'observation/response-correspondence';source=read(correspondence/'index.json')
    originals=SHARED/'observation/original-model-traces/full-response-turns'
    groups=build_members(source,correspondence)
    arrays={};gain={}
    for cid,row in source['cases'].items():
        if row['role'] not in ('failure','natural_numerical_success'):continue
        arrays[cid]=np.load(originals/cid/'turns.npy',mmap_mode='r')
        norms=np.load(originals/cid/'norms.npy',mmap_mode='r')
        if np.any(norms<=0) or not np.isfinite(norms).all():raise ValueError('Undefined recorded log gain')
        gain[cid]=np.log(norms[:,:,1].astype('f8')/norms[:,:,0].astype('f8'))
        if arrays[cid].shape[0]!=row['decisions']:raise ValueError('Prediction count changed')
    index=dict(groups=[],contrasts=[],population=len(arrays),source=str(correspondence/'index.json'))
    # First compute and retain each group independently. Neither a cross-group
    # question pair nor a desired answer is used to make these overlaps.
    for (gpu,role),group in sorted(groups.items()):
        folder=output/'groups'/f'gpu{gpu}-{role}';folder.mkdir(parents=True)
        shape=arrays[group['reference']].shape
        means=np.lib.format.open_memmap(folder/'mean.npy',mode='w+',dtype='f8',shape=shape)
        metrics=np.full((shape[0],shape[1],5),np.nan);rows=[]
        for pos in range(shape[0]):
            members=group['positions'].get(pos,[]);row=dict(prediction=pos,members=members)
            if not members:means[pos]=np.nan;rows.append(row);continue
            values=np.stack([arrays[c][p] for c,p in members]);stats=group_statistics(values)
            g=np.stack([gain[c][p] for c,p in members]);means[pos]=stats['mean']
            metrics[pos,:,0]=stats['dispersion'];metrics[pos,:,1]=g.mean(axis=0)
            metrics[pos,:,2]=g.var(axis=0)
            if len(members)>1:
                metrics[pos,:,3]=np.nanmin(stats['loo_cosine'],axis=0)
                metrics[pos,:,4]=np.mean(stats['loo_cosine'],axis=0)
            row['leave_one_out_cosine']=finite(stats['loo_cosine']);rows.append(row)
        means.flush();np.save(folder/'metrics.npy',metrics)
        write(folder/'index.json',dict(reference=group['reference'],members=group['members'],rows=rows,
            metrics=['dispersion_squared','mean_log_gain','log_gain_variance','minimum_loo_cosine','mean_loo_cosine'],
            coordinates='Each actual merge separately; mean of observed native direction changes.'))
        index['groups'].append(dict(gpu=gpu,role=role,reference=group['reference'],members=len(group['members']),path=str(folder)))
        print(json.dumps(dict(group=str(folder),members=len(group['members']))),flush=True)
    for gpu in sorted({key[0] for key in groups}):
        f,s=groups[(gpu,'failure')],groups[(gpu,'natural_numerical_success')]
        source_map=SHARED/f'observation/group-observation/gpu{gpu}-failure-success/comparison.json'
        bridge=read(source_map)['correspondence']
        if bridge['left']!=f['reference'] or bridge['right']!=s['reference'] or not bridge['same_hardware']:
            raise ValueError('Invalid bridge between group reference positions')
        frames=[r for r in bridge['frames'] if r.get('left_decision') is not None and r.get('right_decision') is not None]
        folder=output/'contrasts'/f'gpu{gpu}';folder.mkdir(parents=True)
        shape=(len(frames),*arrays[f['reference']].shape[1:])
        differences=np.lib.format.open_memmap(folder/'difference.npy',mode='w+',dtype='f8',shape=shape)
        supported=np.zeros(shape[:2],dtype=bool);metrics=np.full((*shape[:2],7),np.nan)
        gsupported=np.zeros(shape[:2],dtype=bool);gm=np.full((*shape[:2],4),np.nan);rows=[]
        for i,frame in enumerate(frames):
            fp,sp=frame['left_decision'],frame['right_decision']
            fm=f['positions'].get(fp,[]);sm=s['positions'].get(sp,[])
            row=dict(position=i,role=frame['role'],failure_prediction=fp,success_prediction=sp,failure_members=fm,success_members=sm,
                     full_population=(len(fm)==len(f['members']) and len(sm)==len(s['members'])))
            if not fm or not sm:differences[i]=np.nan;rows.append(row);continue
            fv=np.stack([arrays[c][p] for c,p in fm]);sv=np.stack([arrays[c][p] for c,p in sm])
            stats=contrast_statistics(fv,sv);differences[i]=stats['difference'];supported[i]=stats['unanimous']
            metrics[i]=np.stack([stats['squared'],stats['pooled_dispersion'],stats['sampling_noise'],
                stats['excess_squared'],np.min(stats['failure_margin'],axis=0),np.min(stats['success_margin'],axis=0),
                np.mean(np.concatenate([stats['failure_loo_difference_cosine'],stats['success_loo_difference_cosine']]),axis=0)],axis=-1)
            for k in ('failure_margin','success_margin','failure_loo_difference_cosine','success_loo_difference_cosine'):
                row[k]=finite(stats[k])
            gf=np.stack([gain[c][p] for c,p in fm])[:,:,None];sg=np.stack([gain[c][p] for c,p in sm])[:,:,None]
            g=contrast_statistics(gf,sg)
            # Gain group signs need not agree: the cross-group leave-one-out
            # separation is reported separately from direction recurrence.
            gsupported[i]=np.all(g['failure_margin']>0,axis=0)&np.all(g['success_margin']>0,axis=0)
            gm[i]=np.stack([g['difference'][:,0],g['pooled_dispersion'],np.min(g['failure_margin'],axis=0),np.min(g['success_margin'],axis=0)],axis=-1)
            rows.append(row)
        differences.flush();np.save(folder/'supported.npy',supported);np.save(folder/'metrics.npy',metrics)
        np.save(folder/'gain-supported.npy',gsupported);np.save(folder/'gain-metrics.npy',gm)
        summary=[]
        for role in dict.fromkeys(r['role'] for r in rows):
            ids=[i for i,r in enumerate(rows) if r['role']==role];m=metrics[ids];keep=supported[ids]
            bymerge=np.nanmean(m[:,:,0],axis=0);kept=np.nanmean(m[:,:,0]*keep,axis=0)
            summary.append(dict(role=role,positions=len(ids),full_population_positions=sum(rows[i]['full_population'] for i in ids),
                failure_member_range=[min(len(rows[i]['failure_members']) for i in ids),max(len(rows[i]['failure_members']) for i in ids)],
                success_member_range=[min(len(rows[i]['success_members']) for i in ids),max(len(rows[i]['success_members']) for i in ids)],
                difference_squared_by_merge=bymerge,whole_vector_supported_squared_by_merge=kept,
                supported_positions_by_merge=np.sum(keep,axis=0),pooled_dispersion_by_merge=np.nanmean(m[:,:,1],axis=0),
                gain_difference_by_merge=np.nanmean(gm[ids,:,0],axis=0),gain_supported_positions_by_merge=np.sum(gsupported[ids],axis=0)))
        write(folder/'index.json',finite(dict(gpu=gpu,failure_reference=f['reference'],success_reference=s['reference'],rows=rows,summaries=summary,
            population=dict(failures=len(f['members']),successes=len(s['members'])),
            bridge_source=str(source_map),bridge_sha256=hashlib.sha256(source_map.read_bytes()).hexdigest(),
            metrics=['difference_squared','pooled_dispersion','mean_sampling_noise','noise_corrected_squared','minimum_failure_loo_margin','minimum_success_loo_margin','mean_loo_difference_cosine'],
            support='Every included member agrees with the rest of its own group and is on its own class side of the leave-one-out group midpoint. This is descriptive unanimous support, not a significance test.')))
        index['contrasts'].append(dict(gpu=gpu,path=str(folder),positions=len(frames),supported_cells=int(supported.sum())))
        print(json.dumps(index['contrasts'][-1]),flush=True)
    write(output/'index.json',index)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);a=p.parse_args();main(a.output)
