"""Truth-only geometry bounds on the same two endpoint sets."""

import numpy as np


def bounds(pred_area, truth_area, intersection, target, signed_menu):
    q = np.asarray(target,dtype=np.float64)
    if q.ndim!=2 or q.shape[1]!=2 or not np.isfinite(q).all() or (q<0).any() or (q>1).any():
        raise ValueError("Finite disjoint two-column unit-cell targets required")
    k = min(8,pred_area//10,len(q))
    best = signed_menu(pred_area,truth_area,intersection,q[:,0],q[:,1],2*k)
    utility = q[:,0]-q[:,1]
    top = np.sort(utility[utility>0])[::-1][:k]
    repaired_i = float(np.clip(intersection+top.sum(dtype=float),0.,min(pred_area,truth_area)))
    paired = repaired_i/(pred_area+truth_area-repaired_i)
    return dict(raw=best["raw"][0],paired=paired,rematched=best["balanced"][0],
                unrestricted=best["unrestricted"][0])
