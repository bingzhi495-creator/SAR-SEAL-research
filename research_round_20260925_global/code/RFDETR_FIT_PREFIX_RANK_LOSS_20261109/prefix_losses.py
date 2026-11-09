"""List-local uniform pairwise and precision-at-four structured objectives."""
import torch
from torch.nn import functional as F


def first_sum(scores,valid,count):
    limit=min(4,scores.shape[1])
    values=torch.sort(scores.masked_fill(~valid,float("-inf")),dim=1,descending=True,stable=True).values[:,:limit]
    select=torch.arange(limit,device=scores.device)[None,:]<count[:,None]
    return torch.where(select,values,0).sum(1)


def precision_ramp(scores,labels,valid):
    size=valid.sum(1)
    k=size.clamp_max(4)
    pos=labels&valid
    neg=~labels&valid
    good=torch.minimum(pos.sum(1),k)
    bad=k-good
    unconstrained=first_sum(scores+neg.to(scores.dtype),valid,k)
    best=first_sum(scores,pos,good)+first_sum(scores,neg,bad)
    active=(size>k)&pos.any(1)&neg.any(1)
    loss=(unconstrained-best-bad).clamp_min(0)/k.clamp_min(1)
    return torch.where(active,loss,0)


def uniform_pairwise(scores,labels,valid):
    pos=labels&valid
    neg=~labels&valid
    pairs=pos[:,:,None]&neg[:,None,:]
    diff=scores[:,None,:]-scores[:,:,None]
    return (F.softplus(diff)*pairs).sum((1,2))/pairs.sum((1,2)).clamp_min(1)

