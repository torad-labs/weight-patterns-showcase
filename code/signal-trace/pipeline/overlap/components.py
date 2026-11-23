"""Exact additive accounting of an observed normalized residual-direction turn.

For each recorded execution, not the centroid's fictitious execution:
  turn = r*(1/|y|-1/|r|) + sum_j (g W[:,j]) * x_j/(rms(Wx)*|y|).
The group difference is then the difference of these separately averaged parts.
"""
import numpy as np


def normalized_parts(x,w,gain,incoming,norms,epsilon):
    x,w,gain,incoming,norms=[np.asarray(v,dtype='f8') for v in (x,w,gain,incoming,norms)]
    z=x@w.T;rms=np.sqrt(np.mean(z*z,axis=1)+epsilon)
    if np.any(norms<=0) or np.any(rms<=0):raise ValueError('Undefined native normalization')
    alpha=x/(rms*norms[:,1])[:,None]
    context=incoming*(1/norms[:,1]-1/norms[:,0])[:,None]
    return alpha,context


def project_components(delta_alpha,basis,context_difference,difference):
    a,b,r,d=[np.asarray(x,dtype='f8') for x in (delta_alpha,basis,context_difference,difference)]
    squared=np.sum(d*d,axis=1)
    contribution=a*(d@b)
    shares=np.divide(contribution,squared[:,None],out=np.full_like(contribution,np.nan),where=squared[:,None]>0)
    context=np.divide(np.sum(r*d,axis=1),squared,out=np.full_like(squared,np.nan),where=squared>0)
    reconstructed=a@b.T+r
    residual=np.divide(np.linalg.norm(reconstructed-d,axis=1),np.sqrt(squared),out=np.full_like(squared,np.nan),where=squared>0)
    return dict(unit_share=shares,context_share=context,reconstruction_relative=residual)


def project_along(delta_alpha,basis,context_difference,raw_difference,direction):
    """Account for the observed MEAN difference along a descriptive common axis.

The envelope gap need not itself be a realizable group difference. Therefore
shares sum to the raw mean's projection, not to an invented execution of the gap.
"""
    a,b,r,d,g=[np.asarray(x,dtype='f8') for x in
        (delta_alpha,basis,context_difference,raw_difference,direction)]
    denominator=np.sum(d*g,axis=1)
    contribution=a*(g@b)
    shares=np.divide(contribution,denominator[:,None],
        out=np.full_like(contribution,np.nan),where=denominator[:,None]>0)
    context=np.divide(np.sum(r*g,axis=1),denominator,
        out=np.full_like(denominator,np.nan),where=denominator>0)
    return dict(unit_share=shares,context_share=context,projected_raw=denominator)
