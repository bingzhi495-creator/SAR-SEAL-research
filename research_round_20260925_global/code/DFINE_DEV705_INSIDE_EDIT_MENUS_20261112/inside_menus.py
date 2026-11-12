"""Exact continuous-IoU maxima for nested sign/count constraints."""

import numpy as np


def menu(pred_area, truth_area, intersection, addition_q, removal_q, cap):
    if pred_area<=0 or truth_area<=0 or cap<0 or cap>=pred_area:
        raise ValueError("Positive areas and a nonempty mask after any removal required")
    if not 0<=intersection<=min(pred_area,truth_area)+1e-8:
        raise ValueError("Intersection must fit in both masks")
    qa = np.asarray(addition_q,dtype=np.float64)
    qd = np.asarray(removal_q,dtype=np.float64)
    for q in (qa,qd):
        if q.ndim!=1 or not np.isfinite(q).all() or (q<0).any() or (q>1).any():
            raise ValueError("Exact finite unit-cell occupancy expected")
    pa = np.r_[0.,np.cumsum(np.sort(qa)[::-1][:cap])]
    pd = np.r_[0.,np.cumsum(np.sort(qd)[:cap])]
    a,d = np.indices((len(pa),len(pd)))
    feasible = a+d<=cap
    I = np.clip(intersection+pa[a]-pd[d],0.,np.minimum(pred_area+a-d,truth_area))
    iou = I/(pred_area+a-d+truth_area-I)

    def best(allowed):
        ax,dx = np.nonzero(feasible & allowed)
        values = iou[ax,dx]
        idx = np.lexsort((ax,ax+dx,-values))[0]
        return float(values[idx]),int(ax[idx]),int(dx[idx])

    return dict(raw=best((a==0)&(d==0)),add_only=best(d==0),drop_only=best(a==0),
                balanced=best(a==d),unrestricted=best(np.ones_like(feasible)))
