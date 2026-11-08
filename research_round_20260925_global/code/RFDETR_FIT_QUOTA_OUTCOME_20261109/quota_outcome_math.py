"""Exact quota outcomes and equal-capacity anchored ridge heads."""
import numpy as np


def feasible(add,delete):
    return np.arange(max(0,8-len(delete)),min(8,len(add))+1)


def prefixes(values):
    return np.r_[0.,np.cumsum(values,dtype=np.float64)]


def realized(area,fp,fn,add_correct,delete_correct):
    k=feasible(add_correct,delete_correct)
    correct=prefixes(add_correct)[k]+prefixes(delete_correct)[8-k]
    f=(fp+k-correct).astype(np.int64)
    n=(fn+8-k-correct).astype(np.int64)
    if area<=0 or (f<0).any() or (n<0).any() or (n>area).any():
        raise ValueError("Exact paired action error counts")
    inter=area-n
    union=area+f
    iou=inter/union
    count=(100*inter[:,None]>=union[:,None]*np.arange(50,100,5)).sum(1)
    strict=(10*inter>=9*union)
    native_strict=10*(area-fn)>=9*(area+fp)
    lost=(~strict).astype(np.int64)*int(native_strict)
    quality=count+iou-2*lost
    margin=(area-10*n-9*f)/area
    b=np.flatnonzero(k==4).item()
    return dict(k=k,fp=f,fn=n,iou=iou,multi=quality-quality[b],margin=margin-margin[b],strict=strict,
                native_strict=native_strict,pass75=100*inter>=75*union)


def action_features(add,delete,pa,pd,area,score,rank,support_balance=0.0):
    k=feasible(add,delete)
    x=np.column_stack(((k-4)/4,np.abs(k-4)/4,
                       (prefixes(add)[k]-prefixes(add)[4])/4,
                       (prefixes(delete)[8-k]-prefixes(delete)[4])/4,
                       (prefixes(pa)[k]-prefixes(pa)[4])/4,
                       (prefixes(pd)[8-k]-prefixes(pd)[4])/4))
    context=np.array([1.,np.log1p(area)/8,score,rank/100,support_balance])
    return k,(x[:,:,None]*context[None,None,:]).reshape(len(k),-1)


def fit_ridge(x,y,w):
    scale=np.sqrt((x*x*w[:,None]).sum(0)).clip(1e-4)
    normalized=x/scale
    gram=normalized.T@(normalized*w[:,None])
    beta=np.linalg.solve(gram+.05*np.eye(x.shape[1]),normalized.T@(y*w[:,None]))
    def predict(rows):
        return (rows/scale)@beta
    return predict


def pick(k,values):
    best=float(np.max(values))
    options=k[np.abs(values-best)<=1e-10]
    return int(min(options,key=lambda x:(abs(x-4),x)))

