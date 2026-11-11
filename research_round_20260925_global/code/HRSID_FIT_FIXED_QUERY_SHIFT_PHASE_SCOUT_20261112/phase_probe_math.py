"""Numerical components of a prelocked paired-shift boundary probe."""
import numpy as np


def shift_features(logits, signed_distance, outward_xy):
    values = np.asarray(logits,dtype=float)
    distance = np.asarray(signed_distance,dtype=float)
    outward = np.asarray(outward_xy,dtype=float)
    if values.ndim!=2 or values.shape[1]!=5 or distance.shape!=(len(values),) or outward.shape!=(len(values),2):
        raise ValueError("Expected registered columns and native per-point geometry")
    if not all(np.isfinite(x).all() for x in (values,distance,outward)):
        raise ValueError("Nonfinite registered probe inputs")
    base=np.column_stack((values[:,0],distance,values.mean(axis=1),values.std(axis=1)))
    odd=(values[:,[2,4]]-values[:,[1,3]])/2.
    normal=np.einsum("ij,ij->i",odd,outward)
    tangent=np.einsum("ij,ij->i",odd,np.column_stack((-outward[:,1],outward[:,0])))
    return {
        "generic":np.column_stack((base,np.abs(odd))),
        "normal":np.column_stack((base,normal,np.abs(normal))),
        "tangent":np.column_stack((base,tangent,np.abs(tangent))),
        "native_order":values[:,0],
        "mean_order":base[:,2],
    }


def fit_weighted(features, target, weight):
    x,y,w=(np.asarray(v,dtype=float) for v in (features,target,weight))
    if x.ndim!=2 or x.shape[1]!=6 or y.shape!=(len(x),) or w.shape!=(len(x),):
        raise ValueError("Matched 6-column features and per-point targets required")
    if not len(x) or np.any(w<=0) or not all(np.isfinite(v).all() for v in (x,y,w)):
        raise ValueError("Finite positive training support required")
    if np.any((y<0)|(y>1)):
        raise ValueError("Continuous occupancy must be a fraction")
    center=np.average(x,axis=0,weights=w)
    width=np.sqrt(np.average((x-center)**2,axis=0,weights=w))
    width=np.where(width>1e-12,width,1.)
    scaled=(x-center)/width
    average=np.average(y,weights=w)
    weighted=scaled*np.sqrt(w)[:,None]
    coefficients=np.linalg.solve(weighted.T@weighted+np.eye(6),weighted.T@((y-average)*np.sqrt(w)))
    return center,width,coefficients,average


def predict(features, fit):
    center,width,coef,average=fit
    return np.clip((features-center)/width@coef+average,0,1)


def edits(predicted, inside, count):
    score=np.asarray(predicted,dtype=float)
    inside=np.asarray(inside,dtype=bool)
    positive=np.flatnonzero(~inside)
    negative=np.flatnonzero(inside)
    if score.shape!=inside.shape or score.ndim!=1 or not np.isfinite(score).all() or not 0<=count<=min(len(positive),len(negative)):
        raise ValueError("Requested area-preserving edit budget exceeds point support")
    add=positive[np.argsort(-score[positive],kind="stable")[:count]]
    remove=negative[np.argsort(score[negative],kind="stable")[:count]]
    return add,remove

