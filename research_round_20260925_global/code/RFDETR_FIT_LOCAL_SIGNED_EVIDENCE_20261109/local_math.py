"""Prediction-only original-grid supports and exact signed edit accounting."""
import numpy as np
from scipy.ndimage import distance_transform_edt


def boundary_samples(mask, cap, seed):
    if not mask.any() or mask.all():
        return (np.empty((0,2)),)*3+(np.empty(0),np.empty(0,dtype=bool))
    y,x=np.nonzero(mask)
    y0=max(0,int(y.min())-3)
    x0=max(0,int(x.min())-3)
    crop=mask[y0:min(mask.shape[0],int(y.max())+4),x0:min(mask.shape[1],int(x.max())+4)]
    rng=np.random.default_rng(seed)
    groups=[]
    for inside in (False,True):
        side=crop if inside else ~crop
        distance,nearest=distance_transform_edt(side,return_indices=True)
        indices=np.flatnonzero(side & (distance<=2.0))
        if len(indices)>cap:
            indices=np.sort(rng.choice(indices,cap,replace=False))
        y,x=np.unravel_index(indices,crop.shape)
        xy=np.column_stack((x+x0,y+y0))
        other=np.column_stack((nearest[1,y,x]+x0,nearest[0,y,x]+y0))
        d=distance[y,x]
        ray=xy-other
        normal=(ray if inside else -ray)/d[:,None]
        boundary=other+.5*ray/np.abs(ray).max(axis=1,keepdims=True)
        groups.append((xy,boundary,normal,d,np.full(len(xy),inside,dtype=bool)))
    return tuple(np.concatenate(parts) for parts in zip(*groups,strict=True))


def signed_prefix(pixel,inside,truth,scores,budget):
    chosen=[]
    out={}
    for flag,name,benefit,harm in ((False,"add",10,9),(True,"delete",9,10)):
        available=np.flatnonzero(inside==flag)
        order=np.lexsort((pixel[available],-scores[available]))
        selected=available[order[:budget]]
        chosen.extend(selected)
        correct=int(np.count_nonzero(truth[selected]!=flag))
        out[name+"_count"]=len(selected)
        out[name+"_correct"]=correct
        out[name+"_margin"]=benefit*correct-harm*(len(selected)-correct)
    out["fp_delta"]=out["add_count"]-out["add_correct"]-out["delete_correct"]
    out["fn_delta"]=out["delete_count"]-out["delete_correct"]-out["add_correct"]
    out["selected"]=np.array(chosen,dtype=np.int64)
    return out

