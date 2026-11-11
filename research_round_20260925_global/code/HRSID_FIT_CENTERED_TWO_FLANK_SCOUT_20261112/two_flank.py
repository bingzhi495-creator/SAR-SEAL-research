"""Width-preserving opposite-flank amplitude displacement on native pixel cells."""
import numpy as np
from scipy.ndimage import map_coordinates

OFFSETS = np.array([-2., -1., -.5, 0., .5, 1., 2.])
ZERO = 3
RADIUS = np.array([.75, 1.25, 1.75])
STATIONS = np.array([.2, .3, .4, .6, .7, .8])


def from_mask(mask):
    y, x = np.nonzero(mask)
    if not 32 <= len(x) < 1024:
        return None
    h, w = mask.shape
    if min(x.min(),y.min(),w-1-x.max(),h-1-y.max()) < 4:
        return None
    cells = np.stack((x,y),axis=-1).astype(float)
    center = cells.mean(axis=0)+.5
    delta = cells+.5-center
    values, vectors = np.linalg.eigh((delta.T@delta)/len(cells) + np.eye(2)/12.)
    if values[1] < 4*values[0]:
        return None
    tangent = vectors[:,1]
    if tangent[np.argmax(np.abs(tangent))] < 0:
        tangent = -tangent
    normal = np.array((-tangent[1],tangent[0]))
    longitudinal = delta@tangent
    pad = .5*np.abs(tangent).sum()
    lower,upper = longitudinal.min()-pad,longitudinal.max()+pad
    starts = center + (lower + STATIONS*(upper-lower))[:,None]*tangent
    sections = [interval(cells, point, normal) for point in starts]
    if any(s is None or s[1]-s[0]<2. for s in sections):
        return None
    return center,normal,starts,np.asarray(sections)


def interval(cells, origin, normal):
    lo = np.full(len(cells),-np.inf)
    hi = np.full(len(cells),np.inf)
    for axis in (0,1):
        lower = cells[:,axis]-origin[axis]
        upper = lower+1.
        v = normal[axis]
        if abs(v) <= 1e-14:
            # Use half-open native cells for an exactly aligned section.
            hi[(lower>0)|(upper<=0)] = -np.inf
        else:
            lo = np.maximum(lo,np.minimum(lower/v,upper/v))
            hi = np.minimum(hi,np.maximum(lower/v,upper/v))
    kept = lo < hi
    spans = sorted(zip(lo[kept],hi[kept]))
    if not spans:
        return None
    low, high = spans[0]
    for a,b in spans[1:]:
        if a > high+1e-10:
            return None
        high = max(high,b)
    return low,high


def contrasts(image, profile):
    _,normal,starts,sections = profile
    result = []
    for side,sign in ((0,1),(1,-1)):
        edge = sections[:,side,None,None]+OFFSETS[None,:,None]
        def sampled(inside):
            q = edge+inside*sign*RADIUS[None,None,:]
            xy = starts[:,None,None,:]+q[...,None]*normal
            return map_coordinates(image,(xy[...,1]-.5,xy[...,0]-.5),order=1,mode="nearest",prefilter=False)
        result.append((sampled(1)-sampled(-1)).mean(axis=2))
    return np.asarray(result)


def logit_contrasts(lowgrid, profile, shape):
    _,normal,starts,sections = profile
    rh,rw = lowgrid.shape
    h,w = shape
    result = []
    for side,sign in ((0,1),(1,-1)):
        edge = sections[:,side,None,None]+OFFSETS[None,:,None]
        def sampled(inside):
            q = edge+inside*sign*RADIUS[None,None,:]
            xy = starts[:,None,None,:]+q[...,None]*normal
            return map_coordinates(lowgrid,(xy[...,1]*rh/h-.5,xy[...,0]*rw/w-.5),
                                   order=1,mode="nearest",prefilter=False)
        result.append((sampled(1)-sampled(-1)).mean(axis=2))
    return np.asarray(result)


def gains(values):
    return values-values[:,:,ZERO,None]


def paired(values):
    return np.minimum(*gains(values))


def best_offset(scores):
    return min(range(len(OFFSETS)), key=lambda i:(-scores[i],abs(OFFSETS[i]),OFFSETS[i]))


def selected(values):
    score = paired(values)
    halves = np.stack((score[:3].mean(axis=0),score[3:].mean(axis=0)))
    a,b = map(best_offset,halves)
    if a==b and a!=ZERO and np.min(halves[:,a])>=1. and values[:,:,a].mean()>0.:
        return a
    return ZERO


def sign_control(scores, proposed):
    magnitude = abs(OFFSETS[proposed])
    choices = np.flatnonzero(np.abs(OFFSETS)==magnitude)
    return max(choices,key=lambda i:(scores[i],i==proposed))


def controls(image_values, logit_values, proposed):
    if proposed==ZERO:
        return {name:0. for name in ("paired","shuffle","sum","one_side","logit")}
    scrambled = image_values.copy()
    scrambled[1] = np.roll(scrambled[1],3,axis=0)
    side = int(np.argmax(image_values[:,:,ZERO].mean(axis=1)))
    scores = dict(shuffle=paired(scrambled).mean(axis=0),sum=gains(image_values).mean(axis=(0,1)),
                  one_side=gains(image_values)[side].mean(axis=0),logit=paired(logit_values).mean(axis=0))
    return dict(paired=float(OFFSETS[proposed]),**{k:float(OFFSETS[sign_control(v,proposed)]) for k,v in scores.items()})

