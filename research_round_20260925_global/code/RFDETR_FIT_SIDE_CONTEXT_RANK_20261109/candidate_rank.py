"""Candidate-level ordinal context for point-score inputs."""
import numpy as np


def midranks(values):
    """Per-column average empirical ranks mapped symmetrically into (-1, 1)."""
    n=len(values)
    if not n:
        return values.copy()
    if not np.isfinite(values).all():
        raise ValueError("Finite neighborhood logits required")
    order=np.argsort(values,axis=0,kind="stable")
    sorted_x=np.take_along_axis(values,order,axis=0)
    breaks=sorted_x[1:]!=sorted_x[:-1]
    idx=np.arange(n)[:,None]
    left=np.concatenate((np.ones_like(sorted_x[:1],dtype=bool),breaks))
    right=np.concatenate((breaks,np.ones_like(sorted_x[:1],dtype=bool)))
    lo=np.maximum.accumulate(np.where(left,idx,0),axis=0)
    hi=np.minimum.accumulate(np.where(right,idx,n-1)[::-1],axis=0)[::-1]
    result=np.empty_like(values)
    np.put_along_axis(result,order,(lo+hi+1)/n-1,axis=0)
    return result


def ranked_inputs(x,inside,by_side):
    if x.ndim!=2 or x.shape[1]!=38 or inside.shape!=(len(x),):
        raise ValueError("Original 38 columns and matching side membership required")
    out=x.copy()
    if by_side:
        for flag in (False,True):
            keep=inside==flag
            out[keep,6:]=midranks(x[keep,6:])
    else:
        out[:,6:]=midranks(x[:,6:])
    return out
