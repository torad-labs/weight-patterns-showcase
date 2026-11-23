"""Basis-free held-out predictive support for independently averaged groups.

This does not filter components out of the full-data difference. Each held-out
score uses only other members to estimate its direction and midpoint.
"""
import numpy as np


def held_out_difference(failures, successes):
    f,s=[np.asarray(v,dtype='f8') for v in (failures,successes)]
    if f.ndim!=3 or s.ndim!=3 or f.shape[1:]!=s.shape[1:]:raise ValueError('Member/merge/coordinate arrays required')
    if min(len(f),len(s))<2:raise ValueError('Each group needs another member after omission')
    if not np.isfinite(f).all() or not np.isfinite(s).all():raise ValueError('Nonfinite observed trace')
    fm,sm=f.mean(axis=0),s.mean(axis=0)
    # Exclude the member before summation, so even floating-point cancellation
    # cannot make its value enter its own training direction.
    f_other=np.stack([f[np.arange(len(f))!=i].mean(axis=0) for i in range(len(f))])
    s_other=np.stack([s[np.arange(len(s))!=i].mean(axis=0) for i in range(len(s))])
    directions=np.concatenate([sm-f_other,s_other-fm],axis=0)
    midpoints=np.concatenate([(sm+f_other)/2,(s_other+fm)/2],axis=0)
    length=np.linalg.norm(directions,axis=-1)
    unit=np.divide(directions,length[:,:,None],out=np.zeros_like(directions),where=length[:,:,None]>0)
    side=np.concatenate([-np.ones(len(f)),np.ones(len(s))])
    margins=side[:,None]*np.sum((np.concatenate([f,s])-midpoints)*unit,axis=-1)
    valid=length>0;margins=np.where(valid,margins,np.nan)
    n=valid.sum(axis=0)
    pairwise=np.divide(np.sum(unit.sum(axis=0)**2,axis=-1)-n,n*(n-1),
        out=np.full(n.shape,np.nan,dtype='f8'),where=n>1)
    return dict(difference=sm-fm,directions=directions,signed_margin=margins,
        own_side=valid&(margins>0),valid=valid,direction_pairwise_cosine=pairwise,
        failure_count=len(f),success_count=len(s))


def summary(cv):
    nf=cv['failure_count']; rows=[]
    for sl in (slice(None,nf),slice(nf,None)):
        valid=cv['valid'][sl];n=valid.sum(axis=0);yes=cv['own_side'][sl].sum(axis=0)
        proportion=np.divide(yes,n,out=np.full(n.shape,np.nan,dtype='f8'),where=n>0)
        margin=np.divide(np.nansum(cv['signed_margin'][sl],axis=0),n,
            out=np.full(n.shape,np.nan,dtype='f8'),where=n>0)
        rows.append((yes,n,proportion,margin))
    f,s=rows
    return np.stack([f[0],f[1],s[0],s[1],(f[2]+s[2])/2,f[3],s[3],
        f[3]+s[3],cv['direction_pairwise_cosine']],axis=-1)


def matched_speed(failures,successes,failure_speed,success_speed):
    f,s=[np.asarray(v,dtype='f8') for v in (failures,successes)]
    fs,ss=np.asarray(failure_speed),np.asarray(success_speed)
    if len(fs)!=len(f) or len(ss)!=len(s):raise ValueError('Speed metadata count differs')
    strata=[];unavailable=[]
    for speed in sorted(set(fs)|set(ss)):
        fi,si=np.flatnonzero(fs==speed),np.flatnonzero(ss==speed)
        if min(len(fi),len(si))<2:
            unavailable.append(dict(speed=float(speed),failure_count=len(fi),success_count=len(si),reason='Not enough members in both groups for held-out recurrence'))
            continue
        strata.append(dict(speed=float(speed),failure_indices=fi,success_indices=si,
            cv=held_out_difference(f[fi],s[si])))
    if not strata:
        return dict(eligible=False,strata=[],unavailable=unavailable,standardized_difference=None,unadjusted_difference=None)
    fi=np.concatenate([v['failure_indices'] for v in strata]);si=np.concatenate([v['success_indices'] for v in strata])
    total=len(fi)+len(si)
    adjusted=np.zeros(f.shape[1:])
    for v in strata:
        v['standardization_weight']=(len(v['failure_indices'])+len(v['success_indices']))/total
        adjusted+=v['standardization_weight']*v['cv']['difference']
    return dict(eligible=True,strata=strata,unavailable=unavailable,
        standardized_difference=adjusted,unadjusted_difference=s[si].mean(axis=0)-f[fi].mean(axis=0),
        failure_indices=fi,success_indices=si)
