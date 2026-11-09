"""Grouped ridge and attainable bounds for fixed equal-area actions."""
from collections import Counter

import numpy as np


def fit_ridge(x,y,groups):
    x=np.asarray(x,dtype=np.float64)
    y=np.asarray(y,dtype=np.float64)
    g=Counter(groups)
    if len(x)!=len(y) or len(x)!=sum(g.values()) or len(g)<2 or not np.isfinite(x).all():
        raise ValueError("Finite group-excluded training rows and two content groups required")
    w=np.array([1/(len(g)*g[k]) for k in groups])
    mu=w@x
    center=x-mu
    scale=np.sqrt(w@center**2).clip(1e-6)
    a=center/scale
    target=float(w@y)
    gram=a.T@(w[:,None]*a)+.10*np.eye(a.shape[1])
    b=np.linalg.solve(gram,a.T@(w*(y-target)))
    return lambda z: target+((np.asarray(z)-mu)/scale)@b


def geometric_choice(native,raw,flip):
    either=raw|flip
    return dict(maximum_passed=either.sum(0).tolist(),gained_over_raw=(flip&~raw).sum(0).tolist(),
                lost_by_flip=(raw&~flip).sum(0).tolist(),
                minimum_native_ap90_losses=int((native[:,8]&~either[:,8]).sum()))
