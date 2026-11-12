"""Label-free native pair geometry, common appearance features, and action controls."""
from hashlib import sha256

import numpy as np
from scipy.ndimage import uniform_filter
from scipy.optimize import linear_sum_assignment

FEATURES=14
APPEARANCE=(0,9,10,11,12,13)
SALT="DFINE-FIT109-trained-pair-v1|"


def pair_assignment(add,drop,width):
    add=np.sort(add)
    drop=np.sort(drop)
    ay,ax=np.divmod(add,width)
    dy,dx=np.divmod(drop,width)
    cost=(ay[:,None]-dy[None,:])**2+(ax[:,None]-dx[None,:])**2
    i,j=linear_sum_assignment(cost)
    return np.column_stack((add[i],drop[j]))


def choose_pairs(gains,cap,forced=None):
    ranks=np.arange(len(gains))
    ordered=np.lexsort((ranks,-gains))
    if forced is not None:
        if not 0<=forced<=cap:
            raise ValueError("Exact common realized count")
        return ordered[:forced].tolist()
    return ordered[gains[ordered]>0][:cap].tolist()


def contexts(features,layers,group,rank):
    true=features[:,::-1].copy()
    shuffled=true.copy()
    for side in (0,1):
        for depth in (1,2):
            candidates=np.flatnonzero(layers[:,1-side]==depth)
            seed=int.from_bytes(sha256(f"{SALT}{group}|{rank}|{side}|{depth}".encode()).digest()[:16],"little")
            shuffled_rows=np.random.Generator(np.random.PCG64(seed)).permutation(candidates)
            shuffled[candidates[:,None],side,APPEARANCE]=true[shuffled_rows[:,None],side,APPEARANCE]
    return dict(independent=features,shuffled=shuffled,paired=true)


def image_features(rgb):
    gray=(rgb.astype(np.float32) @ np.asarray([.299,.587,.114],dtype=np.float32))/np.float32(255)
    padded=np.pad(gray,1,mode="edge")
    dx=(padded[1:-1,2:]-padded[1:-1,:-2])*np.float32(.5)
    dy=(padded[2:,1:-1]-padded[:-2,1:-1])*np.float32(.5)
    return np.stack((gray,uniform_filter(gray,3,mode="nearest"),uniform_filter(gray,5,mode="nearest"),dx,dy))


def box_pixels(box):
    v=np.clip(np.ceil(box).astype(np.int64),0,800)
    return max(1,(v[2]-v[0])*(v[3]-v[1]))


def paired_points(stock,box,area,score,prob,image,rank_math):
    pools,_=rank_math.editable_pools(stock,box)
    data={}
    for side,keys in enumerate((("add1","add2"),("drop1","drop2"))):
        coords=np.concatenate([pools[k][0] for k in keys])
        distance=np.concatenate([pools[k][1] for k in keys])
        p=rank_math.sample_probability(prob,coords,800)
        layers=np.repeat([1,2],[len(pools[k][0]) for k in keys])
        data[side]=(coords,p,distance,layers)
    n=min(32,len(data[0][0]),len(data[1][0]))
    if not n:
        return None
    available={}
    for side,(coords,p,dist,layer) in data.items():
        order=np.lexsort((coords,np.abs(p.astype(np.float32)-.5)))[:n]
        available[side]=coords[order],p[order],dist[order],layer[order]
    paired=pair_assignment(available[0][0],available[1][0],800)
    features=[]
    layers=[]
    discord=0
    half=0
    x1,y1,x2,y2=box
    for side in (0,1):
        coords,p,dist,depth=available[side]
        mapping={int(c):i for i,c in enumerate(coords)}
        ii=np.array([mapping[int(c)] for c in paired[:,side]])
        y,x=np.divmod(paired[:,side],800)
        f=np.column_stack((p[ii],np.full(n,side),depth[ii],dist[ii],(x-x1)/(x2-x1),(y-y1)/(y2-y1),
                           np.full(n,np.log2(area)),np.full(n,area/box_pixels(box)),np.full(n,score),
                           image[:,y,x].T))
        features.append(f.astype(np.float32))
        layers.append(depth[ii])
        discord+=int(((p[ii]>=.5)!=bool(side)).sum())
        half+=int((p[ii]==.5).sum())
    return paired.astype(np.int32),np.stack(features,axis=1),np.column_stack(layers).astype(np.int8),discord,half