"""Native rectangular collar support and exact additive-only geometry."""
import numpy as np
from scipy.ndimage import binary_propagation

EIGHT=np.ones((3,3),dtype=bool)


def collar_region(box, hw):
    height,width=hw
    x1,y1,x2,y2=box
    if not np.isfinite(box).all() or x2<x1 or y2<y1:
        raise ValueError("Exact finite original author rectangle")
    x0=int(np.clip(np.ceil(x1-2),0,width))
    xend=int(np.clip(np.ceil(x2+2),0,width))
    y0=int(np.clip(np.ceil(y1-2),0,height))
    yend=int(np.clip(np.ceil(y2+2),0,height))
    x,y=np.arange(x0,xend),np.arange(y0,yend)
    inside=((x>=x1)&(x<x2))[None,:]&((y>=y1)&(y<y2))[:,None]
    yy,xx=np.nonzero(~inside)
    return (y0,yend,x0,xend),(yy+y0)*width+xx+x0


def choose(stock,region,coords,probabilities):
    y0,yend,x0,xend=region
    seed=stock[y0:yend,x0:xend]
    width=stock.shape[1]
    if seed.sum()!=stock.sum() or np.any(stock.ravel(order="C")[coords]):
        raise ValueError("Original author stock lies wholly within its box")
    selected=probabilities>=.5
    qualifying=coords[selected]
    probs=probabilities[selected]
    yy,xx=np.divmod(qualifying,width)
    y,x=yy-y0,xx-x0
    permitted=seed.copy()
    permitted[y,x]=True
    reachable=binary_propagation(seed,structure=EIGHT,mask=permitted)
    connected=qualifying[reachable[y,x]]
    p_indices=np.lexsort((qualifying,-probs.astype(np.float32)))[:len(connected)]
    equal_count=qualifying[p_indices]
    return dict(ordinary=qualifying.tolist(),connected=connected.tolist(),equal_count=equal_count.tolist())


def geometry(pred_area,truth_area,raw_intersection,added,contribution):
    area=pred_area+added
    intersection=raw_intersection+contribution
    if not -1e-8<=intersection<=min(area,truth_area)+1e-8:
        raise ValueError("Valid continuous additive-only intersection")
    intersection=float(np.clip(intersection,0,min(area,truth_area)))
    return np.array([100*intersection/(area+truth_area-intersection),
                     100*(area-intersection)/truth_area,100*(truth_area-intersection)/truth_area])


def max_iou(pred_area,truth_area,raw_intersection,fractions):
    fractions=np.asarray(fractions,dtype=np.float64)
    n=np.arange(len(fractions)+1,dtype=np.float64)
    gained=np.concatenate(([0.],np.cumsum(np.sort(fractions)[::-1])))
    inter=raw_intersection+gained
    union=pred_area+truth_area+n-inter
    return float(np.max(inter/union))