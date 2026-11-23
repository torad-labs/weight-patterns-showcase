"""Enumerate response-level label assignments within recorded card/speed strata."""
import argparse
from pathlib import Path
import numpy as np
from pipeline.overlap.analyze import SHARED, read, write, finite
from pipeline.overlap.label_calibration import assignments, gram_margins
from pipeline.native_records.run_declaration import check_native_run

SCOPES=('all_aligned','full_population')
PASSAGES=('destination','descent')


def divide(a,b):
    return np.divide(a,b,out=np.full(np.broadcast_shapes(np.shape(a),np.shape(b)),np.nan),where=b>0)


def distribution(values,observed):
    a=np.asarray(values);total=a.size;a=a[np.isfinite(a)]
    return dict(assignments=len(a),total_assignments=total,undefined_assignments=total-len(a),observed=observed,
        mean=float(a.mean()) if len(a) else None,median=float(np.median(a)) if len(a) else None,
        minimum=float(a.min()) if len(a) else None,maximum=float(a.max()) if len(a) else None,
        greater_equal=int((a>=observed).sum()) if np.isfinite(observed) else None,
        greater=int((a>observed).sum()) if np.isfinite(observed) else None,
        lower=int((a<observed).sum()) if np.isfinite(observed) else None)


def group_metrics(margins,fractions,labels):
    result={}
    for name,mask in [('failure',labels),('success',~labels)]:
        good=mask&np.isfinite(margins)
        result[name+'_mean_margin']=divide(np.sum(np.where(good,margins,0),axis=1),good.sum(axis=1))
        result[name+'_positive_mean_fraction']=divide(((margins>0)&good).sum(axis=1),good.sum(axis=1))
        valid=mask&np.isfinite(fractions)
        result[name+'_own_side_fraction']=divide(np.sum(np.where(valid,fractions,0),axis=1),valid.sum(axis=1))
        result[name+'_observed_members']=good.sum(axis=1)
    for suffix in ('mean_margin','positive_mean_fraction','own_side_fraction'):
        result['balanced_'+suffix]=(result['failure_'+suffix]+result['success_'+suffix])/2
    return result


def main(output):
    output=Path(output);destination=output/'calibration'
    check_native_run(destination,output/'declaration.json')
    if destination.exists():raise ValueError('Do not overwrite a calibration')
    destination.mkdir()
    source=read(SHARED/'observation/response-correspondence/index.json')['cases']
    cvroot=output/'cross-validated';parameters=read(cvroot/'index.json')['parameters_by_case']
    original=SHARED/'observation/original-model-traces/full-response-turns'
    traces={c:np.load(original/c/'turns.npy',mmap_mode='r') for c in parameters}
    member_prior=read(cvroot/'per-response.json')['records']
    prior={(r['case'],r['passage']):r for r in member_prior if r['mode']=='within_speed'}
    strata=[];members=[];summaries=[];unavailable=[];joint_parts={};checks=[]
    for gpu in sorted({source[c]['gpu'] for c in parameters}):
        index=read(cvroot/f'gpu{gpu}/index.json')
        with np.load(cvroot/f'gpu{gpu}/held-out-members.npz') as saved:
            for speed in sorted({parameters[c]['v'] for c in parameters if source[c]['gpu']==gpu}):
                cases=sorted(c for c in parameters if source[c]['gpu']==gpu and parameters[c]['v']==speed)
                real=np.array([source[c]['role']=='failure' for c in cases]);nf=int(real.sum());ns=len(cases)-nf
                if min(nf,ns)<2:
                    unavailable.append(dict(gpu=gpu,speed=speed,cases=cases,failures=nf,successes=ns,reason='Full stratum cannot form both held-out centroids'))
                    continue
                labels=assignments(len(cases),nf);real_index=int(np.flatnonzero((labels==real).all(axis=1))[0])
                case_index={c:i for i,c in enumerate(cases)};shape=labels.shape
                totals={(scope,role):dict(margin=np.zeros(shape),fraction=np.zeros(shape),positions=np.zeros(shape,dtype='i4'),merges=np.zeros(shape,dtype='i4')) for scope in SCOPES for role in PASSAGES}
                max_error=0.;compared=0;different_sign=0;different_valid=0;rows_count=0
                for row in index['rows']:
                    if row['role'] not in PASSAGES:continue
                    present=[(c,p) for c,p in row['failure_members']+row['success_members'] if c in case_index]
                    if not present:continue
                    ids=np.array([case_index[c] for c,p in present]);local_labels=labels[:,ids]
                    n=local_labels.sum(axis=1);valid_assignment=(n>=2)&((len(ids)-n)>=2)
                    good=np.flatnonzero(valid_assignment)
                    if not len(good):continue
                    x=np.stack([traces[c][p] for c,p in present]);m=gram_margins(x,local_labels[good])
                    valid=np.isfinite(m);counts=valid.sum(axis=-1)
                    mean=divide(np.nansum(m,axis=-1),counts);fraction=divide(((m>0)&valid).sum(axis=-1),counts)
                    for scope in SCOPES:
                        if scope=='full_population' and not row['full_population']:continue
                        t=totals[(scope,row['role'])];sel=np.ix_(good,ids)
                        t['margin'][sel]+=np.nan_to_num(mean);t['fraction'][sel]+=np.nan_to_num(fraction)
                        t['positions'][sel]+=(counts>0);t['merges'][sel]+=counts
                    if valid_assignment[real_index]:
                        stratum=next(s for s in row['strata'] if s['speed']==speed)
                        if present!=[tuple(x) for x in stratum['failure_members']+stratum['success_members']]:
                            raise ValueError('Saved fold member order changed')
                        direct=saved[stratum['margins_key']];here=m[int(np.flatnonzero(good==real_index)[0])]
                        both=np.isfinite(direct)&np.isfinite(here);compared+=int(both.sum())
                        different_valid+=int((np.isfinite(direct)!=np.isfinite(here)).sum())
                        different_sign+=int(((direct>0)!=(here>0))[both].sum())
                        if both.any():max_error=max(max_error,float(np.abs(direct[both]-here[both]).max()))
                    rows_count+=1
                folder=destination/f'gpu{gpu}-speed{speed}';folder.mkdir()
                packed={'failure_labels':labels};group_index={}
                for (scope,role),t in totals.items():
                    key=scope+'_'+role;margin=divide(t['margin'],t['positions']);fraction=divide(t['fraction'],t['positions'])
                    packed[key+'_margin']=margin;packed[key+'_fraction']=fraction;packed[key+'_positions']=t['positions'];packed[key+'_merges']=t['merges']
                    group=group_metrics(margin,fraction,labels);group_index[key]={}
                    joint_parts.setdefault((gpu,scope,role),[]).append((nf,ns,group,real_index))
                    for metric,values in group.items():
                        packed[key+'_'+metric]=values
                        if 'observed_members' not in metric:
                            stat=distribution(values,values[real_index]);group_index[key][metric]=stat
                            summaries.append(dict(gpu=gpu,speed=speed,scope=scope,passage=role,metric=metric,**stat))
                    for i,c in enumerate(cases):
                        conditional=labels[:,i]==real[i]
                        record=dict(gpu=gpu,speed=speed,case=c,outcome=source[c]['role'],scope=scope,passage=role,
                            observed_positions=int(t['positions'][real_index,i]),
                            null_position_range=[int(t['positions'][conditional,i].min()),int(t['positions'][conditional,i].max())],
                            same_outcome_null=distribution(margin[conditional,i],margin[real_index,i]),
                            all_assignments_null=distribution(margin[:,i],margin[real_index,i]),
                            own_side_fraction_null=distribution(fraction[conditional,i],fraction[real_index,i]))
                        members.append(record)
                        if scope=='all_aligned':
                            old=prior[(c,role)];value=margin[real_index,i]
                            if old['positions']!=record['observed_positions']:raise ValueError('Original per-response coverage did not reproduce')
                            checks.append(dict(case=c,passage=role,observed_positions=record['observed_positions'],
                                old_mean=old['mean_signed_margin'],new_mean=value,
                                absolute_difference=abs(value-old['mean_signed_margin']) if old['mean_signed_margin'] is not None else None))
                np.savez_compressed(folder/'distributions.npz',**packed)
                meta=dict(gpu=gpu,speed=speed,cases=cases,failures=nf,successes=ns,assignments=len(labels),real_index=real_index,
                    aligned_rows_evaluated=rows_count,conditional_assignments_per_failure=int(labels[:,real].sum(axis=0)[0]),
                    conditional_assignments_per_success=int((~labels[:,~real]).sum(axis=0)[0]),
                    direct_check=dict(compared=compared,max_absolute_error=max_error,different_sign=different_sign,different_valid=different_valid),
                    group_distributions=group_index)
                write(folder/'index.json',finite(meta));strata.append(meta)
                print(dict(gpu=gpu,speed=speed,assignments=len(labels),direct_check=meta['direct_check']),flush=True)
                if different_sign or different_valid:raise ValueError('Real-label Gram calculation changed saved score signs or validity')
    joint=[];joint_arrays={}
    for (gpu,scope,role),parts in joint_parts.items():
        total_f=sum(p[0] for p in parts);total_s=sum(p[1] for p in parts)
        for suffix in ('mean_margin','positive_mean_fraction','own_side_fraction'):
            combined=np.zeros(1);observed=0.
            for nf,ns,group,real_index in parts:
                local=(nf/total_f*group['failure_'+suffix]+ns/total_s*group['success_'+suffix])/2
                combined=(combined[:,None]+local[None,:]).ravel();observed+=float(local[real_index])
            key=f'gpu{gpu}_{scope}_{role}_{suffix}';joint_arrays[key]=combined
            joint.append(dict(gpu=gpu,scope=scope,passage=role,metric=suffix,
                eligible_failures=total_f,eligible_successes=total_s,**distribution(combined,observed)))
    np.savez_compressed(destination/'joint-distributions.npz',**joint_arrays)
    write(destination/'summary.json',finite(dict(strata=strata,unavailable=unavailable,per_response=members,
        stratum_statistics=summaries,joint_statistics=joint,reproduction=checks,
        null='Uniform enumeration of all response-level outcome assignments preserving each card/speed full-population count. Assignment stays fixed across positions.',
        per_response_null_description='Primary null conditions on the response keeping its observed label, preserving its own group size; unconditional null is also supplied.',
        inference_limit='Conditional on retained alignment and selected population. Labels were not randomized; template variants are dependent. Exact enumeration is a design diagnostic, not an unconditional causal p-value.',
        coverage='All-aligned eligibility is recomputed under each assignment. Full-population positions hold membership fixed. Undefined scores stay undefined.',
        joint='Cartesian product of independent within-speed enumerations. Equal-response weighting within each assigned class, balanced between classes. Counts and null coverage remain explicit.')))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path)
    main(parser.parse_args().output)
