"""Held-out group separation, repeated inside exactly matched speed strata."""
import argparse
from pathlib import Path
import numpy as np
from pipeline.overlap.analyze import SHARED,read,write,finite
from pipeline.overlap.cross_validation import held_out_difference,matched_speed,summary
from pipeline.native_records.run_declaration import check_native_run

METRICS=['failure_own_side','failure_valid','success_own_side','success_valid','balanced_own_side_fraction',
         'mean_failure_signed_margin','mean_success_signed_margin','held_out_separation','mean_pairwise_direction_cosine']


def main(output):
    output=Path(output);dest=output/'cross-validated';check_native_run(dest,output/'declaration.json')
    if dest.exists():raise ValueError('Never overwrite held-out observations')
    dest.mkdir()
    source=read(SHARED/'observation/response-correspondence/index.json');original=SHARED/'observation/original-model-traces/full-response-turns'
    arrays={};speeds={};parameters={}
    for case,meta in source['cases'].items():
        if meta['role'] not in ('failure','natural_numerical_success'):continue
        arrays[case]=np.load(original/case/'turns.npy',mmap_mode='r')
        item=read(SHARED/'acquisition/cohort-inputs'/f'{case}.json')
        item=item[0] if isinstance(item,list) else item
        parameters[case]=item['parameters'];speeds[case]=item['parameters']['v']
    cards=[]
    for gpu in sorted({source['cases'][c]['gpu'] for c in arrays}):
        base=output/f'contrasts/gpu{gpu}';meta=read(base/'index.json');raw=np.load(base/'difference.npy',mmap_mode='r')
        folder=dest/f'gpu{gpu}';folder.mkdir();shape=raw.shape
        metrics=np.full((*shape[:2],len(METRICS)),np.nan);matched=np.full_like(metrics,np.nan)
        adjustment=np.full((*shape[:2],5),np.nan)
        standardized=np.lib.format.open_memmap(folder/'standardized-difference.npy',mode='w+',dtype='f8',shape=shape);standardized[:]=np.nan
        pooled=np.lib.format.open_memmap(folder/'eligible-unadjusted-difference.npy',mode='w+',dtype='f8',shape=shape);pooled[:]=np.nan
        folds={};rows=[]
        for i,row in enumerate(meta['rows']):
            fm,sm=row['failure_members'],row['success_members'];f=np.stack([arrays[c][p] for c,p in fm]);s=np.stack([arrays[c][p] for c,p in sm])
            unmatched_available=min(len(f),len(s))>=2
            if unmatched_available:
                cv=held_out_difference(f,s);metrics[i]=summary(cv);folds[f'p{i}_unmatched_margins']=cv['signed_margin']
            fit=matched_speed(f,s,[speeds[c] for c,p in fm],[speeds[c] for c,p in sm])
            record={k:row[k] for k in ('position','role','failure_prediction','success_prediction','failure_members','success_members','full_population')}
            record.update(unmatched_available=unmatched_available,matched_available=fit['eligible'],strata=[],unavailable_strata=fit['unavailable'])
            # Aggregated matched scores weight each held-out member equally within
            # its outcome class. Every member appears in at most one speed stratum.
            all_margins=[];all_labels=[];weighted_stability=[];stability_weight=[]
            for k,stratum in enumerate(fit['strata']):
                sv=stratum['cv'];fi,si=stratum['failure_indices'],stratum['success_indices']
                key=f'p{i}_speed{stratum["speed"]}'
                folds[key+'_margins']=sv['signed_margin'];folds[key+'_metrics']=summary(sv)
                record['strata'].append(dict(speed=stratum['speed'],failure_members=[fm[j] for j in fi],success_members=[sm[j] for j in si],
                    standardization_weight=stratum['standardization_weight'],margins_key=key+'_margins',metrics_key=key+'_metrics'))
                all_margins.append(sv['signed_margin']);all_labels.extend([0]*len(fi)+[1]*len(si))
                weighted_stability.append(sv['direction_pairwise_cosine']);stability_weight.append(len(fi)+len(si))
            if fit['eligible']:
                adjusted,unadjusted=fit['standardized_difference'],fit['unadjusted_difference'];standardized[i]=adjusted;pooled[i]=unadjusted
                a2=np.sum(adjusted**2,axis=-1);b2=np.sum(unadjusted**2,axis=-1);dot=np.sum(adjusted*unadjusted,axis=-1)
                cosine=np.divide(dot,np.sqrt(a2*b2),out=np.full_like(a2,np.nan),where=a2*b2>0)
                ratio=np.divide(a2,b2,out=np.full_like(a2,np.nan),where=b2>0)
                parallel=np.divide(dot,b2,out=np.full_like(a2,np.nan),where=b2>0)
                adjustment[i]=np.stack([a2,b2,cosine,ratio,parallel],axis=-1)
                margins=np.concatenate(all_margins);labels=np.array(all_labels);groups=[]
                for label in (0,1):
                    a=margins[labels==label];valid=np.isfinite(a);count=valid.sum(axis=0);yes=(a>0).sum(axis=0)
                    prop=np.divide(yes,count,out=np.full(count.shape,np.nan,dtype='f8'),where=count>0)
                    mean=np.divide(np.nansum(a,axis=0),count,out=np.full(count.shape,np.nan,dtype='f8'),where=count>0)
                    groups.append((yes,count,prop,mean))
                st=np.array(weighted_stability);w=np.array(stability_weight)[:,None];valid=np.isfinite(st)
                stability=np.divide(np.nansum(st*w,axis=0),np.sum(w*valid,axis=0),out=np.full(st.shape[1:],np.nan),where=np.sum(w*valid,axis=0)>0)
                fstat,sstat=groups
                matched[i]=np.stack([fstat[0],fstat[1],sstat[0],sstat[1],(fstat[2]+sstat[2])/2,
                    fstat[3],sstat[3],fstat[3]+sstat[3],stability],axis=-1)
                record['eligible_failure_members']=[fm[j] for j in fit['failure_indices']]
                record['eligible_success_members']=[sm[j] for j in fit['success_indices']]
            rows.append(record)
        standardized.flush();pooled.flush();np.save(folder/'metrics.npy',metrics);np.save(folder/'matched-metrics.npy',matched);np.save(folder/'adjustment.npy',adjustment)
        np.savez_compressed(folder/'held-out-members.npz',**folds)
        write(folder/'index.json',finite(dict(gpu=gpu,population=meta['population'],rows=rows,metrics=METRICS,
            adjustment_metrics=['standardized_squared','eligible_unadjusted_squared','cosine','squared_norm_ratio','parallel_coefficient'],
            score_scope='Predictive support on omitted original members. No component purity, causal identity, held-out question generalization, or edit target is certified.',
            control_scope='Exactly the same speed inside each stratum. Mass, drop, loss, rise, wording and history remain uncontrolled. Unavailable strata are explicit.',
            weighting='Every member has equal influence within its group. Matched score averages members within outcome class. Standardized difference uses identical pooled-member speed weights for the two groups.')))
        cards.append(dict(gpu=gpu,positions=len(rows),matched_positions=sum(r['matched_available'] for r in rows),path=str(folder)))
        print(cards[-1],flush=True)
    write(dest/'index.json',dict(cards=cards,parameters_by_case=parameters,definition='Leave each member out before estimating either centroid, direction and midpoint; report its signed unit-direction distance and the training-direction stability.'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);args=p.parse_args();main(args.output)
