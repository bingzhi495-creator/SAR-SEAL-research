"""Equal edit-count selection using a ratio of conditional mean pixel masses."""

from math import isfinite

import numpy as np


def expected_ratio_writes(coordinates, probabilities, k, area, intersection, truth):
    """Return native adds then drops for each approximate-IoU optimizer.

    The fixed cutoff control matches an IoU just-at-threshold decision at 0.75.
    The ratio rule enumerates sign counts at the same budget. Within each count,
    it ranks the two sides separately because the ratio increases in intersection.
    """
    coords = np.asarray(coordinates)
    q = np.asarray(probabilities, dtype=np.float64)
    if (coords.shape != q.shape or coords.ndim != 2 or coords.shape[1] != 2
            or coords.dtype.kind not in "iu" or not np.isfinite(q).all()
            or np.any(q < 0) or np.any(q > 1)
            or np.unique(coords).size != coords.size
            or type(k) is not int or not 0 <= k <= min(8, len(coords))):
        raise ValueError("Unique native support and finite paired probabilities at feasible k")
    if (not all(isfinite(x) for x in (area, intersection, truth))
            or not 0 <= intersection <= min(area,truth) or area < len(coords)):
        raise ValueError("Feasible original expected overlap")
    if k == 0:
        return {a:np.empty(0, dtype=np.int64) for a in ("T","R")}

    cutoff = 3/7
    records = [(float(q[j,side]-cutoff) * (1 if side==0 else -1), int(coords[j,side]), side)
               for j in range(len(coords)) for side in (0,1)]
    constant = sorted(records, key=lambda x:(-x[0],x[1]))[:2*k]

    def ordered_ids(rows):
        return np.asarray([idx for side in (0,1) for _,idx,s in rows if s==side],dtype=np.int64)

    add = np.lexsort((coords[:,0],-q[:,0]))
    drop = np.lexsort((coords[:,1],q[:,1]))
    incorp = np.r_[0.,np.cumsum(q[add,0])]
    discard = np.r_[0.,np.cumsum(q[drop,1])]
    feasible = []
    for n in range(max(0,2*k-len(coords)),min(2*k,len(coords))+1):
        d = 2*k-n
        ip = intersection+incorp[n]-discard[d]
        ap = area+n-d
        if ip < -1e-6 or ip > min(ap,truth)+1e-6 or ap+truth-ip <= 0:
            raise ValueError("Expected edited mass and union compatible with the support")
        ratio = ip/(ap+truth-ip)
        feasible.append((ratio,-abs(n-k),-n))
    n = -max(feasible)[2]
    ratio_ids = np.r_[coords[add[:n],0],coords[drop[:2*k-n],1]]
    return {"T":ordered_ids(constant),"R":ratio_ids}
