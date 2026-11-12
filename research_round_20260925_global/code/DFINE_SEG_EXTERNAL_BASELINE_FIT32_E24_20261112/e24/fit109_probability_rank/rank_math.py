"""Pred-only support, signed probability ranking and fixed-cost sham controls."""
from hashlib import sha256

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt

STRATA = ("add1", "add2", "drop1", "drop2")
SHAMS = 16
STRUCTURE = np.ones((3,3), dtype=bool)


def editable_pools(hard, box):
    yy, xx = np.nonzero(hard)
    h, w = hard.shape
    y0, x0 = max(0,int(yy.min())-2), max(0,int(xx.min())-2)
    y1, x1 = min(h,int(yy.max())+3), min(w,int(xx.max())+3)
    local = hard[y0:y1,x0:x1]
    outer1 = binary_dilation(local, structure=STRUCTURE, border_value=0)
    inner1 = binary_erosion(local, structure=STRUCTURE, border_value=0)
    outer2 = binary_dilation(outer1, structure=STRUCTURE, border_value=0)
    inner2 = binary_erosion(inner1, structure=STRUCTURE, border_value=0)
    near = outer1 & ~inner1
    far = outer2 & ~inner2 & ~near
    bx1,by1,bx2,by2 = box
    x, y = np.arange(x0,x1), np.arange(y0,y1)
    inside = ((x>=bx1)&(x<bx2))[None,:] & ((y>=by1)&(y<by2))[:,None]
    padded = np.pad(local,1,constant_values=False)
    pools = {}
    for sign, foreground in (("add",False),("drop",True)):
        same = local == foreground
        distance = distance_transform_edt(padded if foreground else ~padded)[1:-1,1:-1]
        for layer, band in enumerate((near,far),1):
            support = same & band & inside
            ys,xs = np.nonzero(support)
            pools[sign+str(layer)] = ((ys+y0)*w+xs+x0, distance[support])
    k = min(8,len(yy)//10,sum(len(pools[s][0]) for s in STRATA[:2]),
            sum(len(pools[s][0]) for s in STRATA[2:]))
    quota = {}
    for sign in ("add","drop"):
        near_count = min(k,len(pools[sign+"1"][0]))
        quota[sign+"1"],quota[sign+"2"] = near_count,k-near_count
    return pools,quota


def sample_probability(plane, coords, width):
    """Named CPU feature, not a claim of exact CUDA half-interpolation."""
    half = plane.astype(np.float16).astype(np.float32)
    yy,xx = np.divmod(coords,width)

    def source_index(target):
        position = np.maximum(np.float32(0), (target.astype(np.float32)+np.float32(.5))
                              *np.float32(.20)-np.float32(.5))
        base = np.floor(position).astype(np.intp)
        frac = position-base.astype(np.float32)
        return base, np.minimum(base+1,159), frac

    y0,y1,dy = source_index(yy)
    x0,x1,dx = source_index(xx)
    top = (np.float32(1)-dx)*half[y0,x0] + dx*half[y0,x1]
    bottom = (np.float32(1)-dx)*half[y1,x0] + dx*half[y1,x1]
    return ((np.float32(1)-dy)*top + dy*bottom).astype(np.float16)


def rng(group, rank, stratum, draw):
    seed = int.from_bytes(sha256(f"D-FINE-FIT109-boundary-v1|{group}|{rank}|{stratum}|{draw}".encode())
                          .digest()[:16],"little")
    return np.random.Generator(np.random.PCG64(seed))


def selected(coords, ordering, size):
    return coords[np.lexsort((coords,ordering))[:size]].tolist()


def choose(pools, quotas, scores, group, rank):
    geom,prob = [[],[]],[[],[]]
    perm = [[[],[]] for _ in range(SHAMS)]
    for stratum,(coords,distance) in pools.items():
        q = quotas[stratum]
        if not q:
            continue
        sign = int(stratum.startswith("drop"))
        p = scores[stratum].astype(np.float32)
        costs = p if sign else -p
        geom[sign].extend(selected(coords,distance,q))
        prob[sign].extend(selected(coords,costs,q))
        for i in range(SHAMS):
            shuffled = rng(group,rank,stratum,i).permutation(costs)
            perm[i][sign].extend(selected(coords,shuffled,q))
    return dict(geometry=geom,probability=prob,permuted=perm)


def oracle(pools, quotas, fractions):
    answer=[[],[]]
    for stratum,(coords,_) in pools.items():
        sign=int(stratum.startswith("drop"))
        values=fractions[stratum]
        answer[sign].extend(selected(coords,values if sign else -values,quotas[stratum]))
    return answer


def verify_actions(action, pools, quotas):
    for sign,items in zip(("add","drop"),action,strict=True):
        if len(items)!=len(set(items)):
            raise ValueError("Unique balanced pixels required")
        before=len(items)
        for layer in ("1","2"):
            key=sign+layer
            used=len(set(items).intersection(pools[key][0].tolist()))
            if used!=quotas[key]:
                raise ValueError("Exact same signed geometry-layer quotas")
            before-=used
        if before:
            raise ValueError("All pixels in the common author box and band")