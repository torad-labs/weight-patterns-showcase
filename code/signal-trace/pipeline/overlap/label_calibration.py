"""Exact size-preserving label calibration of the recorded held-out statistic."""
import numpy as np
from itertools import combinations
from math import comb


def assignments(n, failures):
    result=np.zeros((comb(n,failures),n),dtype=bool)
    for i,choice in enumerate(combinations(range(n),failures)):
        result[i,list(choice)]=True
    return result


def gram_margins(values, labels):
    x=np.asarray(values,dtype='f8');labels=np.asarray(labels,dtype=bool)
    if x.ndim!=3 or labels.ndim!=2 or labels.shape[1]!=len(x):
        raise ValueError('Member/merge/coordinate values and assignment/member labels required')
    f=labels.astype('f8');s=1-f
    nf=f.sum(axis=1)[:,None,None];ns=s.sum(axis=1)[:,None,None]
    if np.any(nf<2) or np.any(ns<2):raise ValueError('Both held-out centroids must exist')
    # A common translation cancels from held-out directions and midpoint
    # distances. Center without using labels to reduce Gram cancellation.
    centered=(x-x.mean(axis=0)).transpose(1,0,2)
    gram=centered@centered.transpose(0,2,1)
    diagonal=np.diagonal(gram,axis1=-2,axis2=-1)[None,:,:]
    fg=np.einsum('mij,bj->bmi',gram,f)
    sg=np.einsum('mij,bj->bmi',gram,s)
    ff=np.sum(fg*f[:,None,:],axis=-1,keepdims=True)
    ss=np.sum(sg*s[:,None,:],axis=-1,keepdims=True)
    fs=np.sum(fg*s[:,None,:],axis=-1,keepdims=True)
    fm2=(ff-2*fg+diagonal)/(nf-1)**2
    sm2=(ss-2*sg+diagonal)/(ns-1)**2
    f2=ff/nf**2;s2=ss/ns**2
    df2=s2+fm2-2*(fs-sg)/(ns*(nf-1))
    ds2=sm2+f2-2*(fs-fg)/(nf*(ns-1))
    fnum=-(sg/ns-(fg-diagonal)/(nf-1)-(s2-fm2)/2)
    snum=(sg-diagonal)/(ns-1)-fg/nf-(sm2-f2)/2
    length2=np.where(labels[:,None,:],df2,ds2)
    numerator=np.where(labels[:,None,:],fnum,snum)
    # Nonpositive computed lengths are undefined, never successful scores.
    length=np.sqrt(np.where(length2>0,length2,np.nan))
    return (numerator/length).transpose(0,2,1)
