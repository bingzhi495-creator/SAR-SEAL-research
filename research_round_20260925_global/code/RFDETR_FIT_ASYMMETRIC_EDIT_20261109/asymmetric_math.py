"""Exact-eight paired-prefix allocation and FIT-only monotone reliability."""
import numpy as np

BUDGET=8


def choose_split(add_gain,delete_gain):
    a=np.r_[0.,np.cumsum(add_gain)]
    d=np.r_[0.,np.cumsum(delete_gain)]
    possible=range(max(0,BUDGET-(len(d)-1)),min(BUDGET,len(a)-1)+1)
    values=[(k,float(a[k]+d[BUDGET-k])) for k in possible]
    best=max(v for _,v in values)
    return min((v for v in values if best-v[1]<=1e-10),key=lambda v:(abs(v[0]-4),v[0]))


def signed_gain(probability,inside):
    return 19*np.asarray(probability)-(10 if inside else 9)


def fit_reliability(scores,labels,weights,bins=8):
    z=np.asarray(scores)
    y=np.asarray(labels)
    w=np.asarray(weights)
    if not (z.ndim==y.ndim==w.ndim==1 and len(z)==len(y)==len(w)>0 and np.isfinite(z).all()
            and np.isin(y,[0,1]).all() and np.isfinite(w).all() and (w>0).all()):
        raise ValueError("Finite nonempty weighted binary calibration rows")
    edges=np.unique(np.quantile(z,np.arange(1,bins)/bins))
    index=np.searchsorted(edges,z,side="right")
    mass=np.bincount(index,weights=w,minlength=len(edges)+1)
    wins=np.bincount(index,weights=w*y,minlength=len(edges)+1)
    existing=np.flatnonzero(mass)
    blocks=[]
    for i in existing:
        blocks.append(([i],mass[i],wins[i]))
        while len(blocks)>1 and blocks[-2][2]/blocks[-2][1]>blocks[-1][2]/blocks[-1][1]:
            i0,w0,y0=blocks.pop()
            i1,w1,y1=blocks.pop()
            blocks.append((i1+i0,w1+w0,y1+y0))
    values={i:y0/w0 for rows,w0,y0 in blocks for i in rows}
    table=np.interp(np.arange(len(mass)),existing,[values[i] for i in existing])
    def apply(x):
        return table[np.searchsorted(edges,np.asarray(x),side="right")]
    return apply


def edit_mask(mask,add,delete):
    a=np.asarray(add,dtype=np.int64)
    d=np.asarray(delete,dtype=np.int64)
    native=mask.reshape(-1)
    if len(a)+len(d) not in (0,BUDGET) or len(np.unique(a))!=len(a) or len(np.unique(d))!=len(d):
        raise ValueError("Exact eight distinct edits or no action")
    if any(np.any((x<0)|(x>=len(native))) for x in (a,d)) or native[a].any() or not native[d].all():
        raise ValueError("Original side and pixel checks")
    out=mask.copy().reshape(-1)
    out[a]=True
    out[d]=False
    return out.reshape(mask.shape)

