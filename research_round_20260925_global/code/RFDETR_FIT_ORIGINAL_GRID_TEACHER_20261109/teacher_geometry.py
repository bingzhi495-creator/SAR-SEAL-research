"""Deterministic original-grid support and integer geometric teacher gate."""
import numpy as np


def sampling_support(truth, native, seed):
    h,w=truth.shape
    y,x=np.indices((h,w))
    parity=((x//4+y//4)%2)==0
    if not (parity & truth).any():
        parity=~parity
    union=truth | native
    local=((x>=x[union].min()-8)&(x<=x[union].max()+8)
           &(y>=y[union].min()-8)&(y<=y[union].max()+8))
    rng=np.random.default_rng(seed)

    def pick(mask,budget):
        ids=np.flatnonzero(mask)
        return ids if len(ids)<=budget else np.sort(rng.choice(ids,budget,replace=False))

    pos=pick(parity & truth,1024)
    neg=np.concatenate((pick(parity & ~truth & local,768),pick(parity & ~truth & ~local,256)))
    if len(pos)==0 or len(neg)==0:
        raise ValueError("The fixed native match must supply both fit classes")
    index=np.concatenate((pos,neg))
    weights=np.r_[np.full(len(pos),.5/len(pos)),np.full(len(neg),.5/len(neg))]
    return dict(index=index,weights=weights,held=~parity,local=local)


def signed_counts(pred, truth):
    fp=int((pred & ~truth).sum())
    fn=int((~pred & truth).sum())
    tp=int((pred & truth).sum())
    area=tp+fn
    error=10*fn+9*fp
    return dict(fp=fp,fn=fn,tp=tp,area=area,error=error,strict=int(area>0 and error<=area))


def accepted(stock, proposal, held_error_change, area):
    if stock["strict"]:
        return False
    enough=stock["error"]-proposal["error"]>=max(1,(area+49)//50)
    iou_good=proposal["tp"]*(area+stock["fp"])>=stock["tp"]*(area+proposal["fp"])
    fp_good=proposal["fp"]-stock["fp"]<=max(1,area//50)
    return bool(enough and iou_good and fp_good and held_error_change<=0)
