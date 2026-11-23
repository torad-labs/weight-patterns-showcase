"""Group-first, per-merge statistics with every leave-one-out member retained."""
import numpy as np


def _cosine(a,b):
    denominator=np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)
    return np.divide(np.sum(a*b,axis=-1),denominator,
                     out=np.full(denominator.shape,np.nan),where=denominator>0)


def group_statistics(values):
    x=np.asarray(values,dtype='f8')
    if x.ndim!=3 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Finite member/merge/coordinate values required')
    mean=x.mean(axis=0)
    dispersion=np.mean(np.sum((x-mean)**2,axis=-1),axis=0)
    loo=np.full(x.shape[:2],np.nan)
    if len(x)>1:
        other=(x.sum(axis=0)-x)/(len(x)-1)
        loo=_cosine(x,other)
    return dict(mean=mean,dispersion=dispersion,loo_cosine=loo)


def contrast_statistics(failures,successes):
    f,s=[np.asarray(v,dtype='f8') for v in (failures,successes)]
    a,b=group_statistics(f),group_statistics(s)
    if f.shape[1:]!=s.shape[1:]:raise ValueError('Groups must share actual merge coordinates')
    difference=b['mean']-a['mean']
    fm=np.full(f.shape[:2],np.nan);sm=np.full(s.shape[:2],np.nan)
    fd=np.full(f.shape[:2],np.nan);sd=np.full(s.shape[:2],np.nan)
    if len(f)>1:
        others=(f.sum(axis=0)-f)/(len(f)-1)
        d=b['mean']-others;mid=(b['mean']+others)/2
        fm=-np.sum((f-mid)*d,axis=-1)
        fd=_cosine(d,difference)
    if len(s)>1:
        others=(s.sum(axis=0)-s)/(len(s)-1)
        d=others-a['mean'];mid=(others+a['mean'])/2
        sm=np.sum((s-mid)*d,axis=-1)
        sd=_cosine(d,difference)
    supported=(np.all(a['loo_cosine']>0,axis=0)&np.all(b['loo_cosine']>0,axis=0)
               &np.all(fm>0,axis=0)&np.all(sm>0,axis=0))
    squared=np.sum(difference**2,axis=-1)
    # Independent group-mean sampling variance estimate: population dispersion
    # divided by n-1. Report the untruncated value; it is not a significance test.
    noise=(a['dispersion']/(len(f)-1)+b['dispersion']/(len(s)-1)
           if len(f)>1 and len(s)>1 else np.full_like(squared,np.nan))
    return dict(difference=difference,squared=squared,pooled_dispersion=(a['dispersion']+b['dispersion'])/2,
                sampling_noise=noise,excess_squared=squared-noise,
                failure_loo_cosine=a['loo_cosine'],success_loo_cosine=b['loo_cosine'],
                failure_margin=fm,success_margin=sm,failure_loo_difference_cosine=fd,
                success_loo_difference_cosine=sd,unanimous=supported)
