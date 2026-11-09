"""Signed two-view point evidence and spatial correspondence control."""
import numpy as np


def signed_scores(p0,p1,inside):
    polarity=np.where(inside,-1.,1.).astype(np.float32)
    base=polarity*(p0-.5)
    delta=polarity*(p1-p0)
    return base,delta,base+.5*delta


def shuffled(delta,p0,inside,seed):
    bands=np.minimum((np.abs(p0-.5)*8).astype(int),3)
    rng=np.random.default_rng(seed)
    control=delta.copy()
    for flag in (False,True):
        for b in range(4):
            rows=np.flatnonzero((inside==flag)&(bands==b))
            control[rows]=delta[rng.permutation(rows)]
    return control
