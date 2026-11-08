"""Construct one matched-length finite perturbation per fixed gradient rule."""
import numpy as np

NAMES=("accepted","projected","scalarized","mixed")
EPS=1e-12


def make_directions(g_accepted,g_strict,g_mixed,budget):
    ga,gs,gm=(np.asarray(x,dtype=np.float64) for x in (g_accepted,g_strict,g_mixed))
    if not (ga.shape==gs.shape==gm.shape and all(np.isfinite(x).all() for x in (ga,gs,gm))
            and np.isfinite(budget) and budget>0):
        raise ValueError("Finite matching gradient vectors and positive step budget")
    norms=dict(accepted=float(np.linalg.norm(ga)),strict=float(np.linalg.norm(gs)),mixed=float(np.linalg.norm(gm)))
    descent=-ga
    strict_size=norms["strict"]
    conflict=float(gs@descent)
    correction=max(0.,conflict)/(strict_size**2) if strict_size>EPS else 0.
    projection=descent-correction*gs
    raw=dict(accepted=descent,projected=projection,scalarized=-(ga+gs),mixed=-gm)
    out={}
    degenerate={}
    for k,v in raw.items():
        length=float(np.linalg.norm(v))
        degenerate[k]=length<=EPS
        out[k]=np.zeros_like(v) if degenerate[k] else (budget/length)*v
    denom=norms["accepted"]*strict_size
    return out,dict(norms=norms,cosine=float(ga@gs/denom) if norms["accepted"]>EPS and strict_size>EPS else None,
                    degenerate=degenerate,projection_fraction=float(np.linalg.norm(projection)/norms["accepted"])
                    if norms["accepted"]>EPS else None,
                    derivative={k:dict(accepted=float(ga@v),strict=float(gs@v)) for k,v in out.items()})
