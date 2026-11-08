"""Deterministic prediction-only signed actions and a disjoint FIT risk rule."""
import numpy as np


def select_equal_count(scores,ranks,count):
    score=np.asarray(scores)
    rank=np.asarray(ranks)
    return np.lexsort((rank,-score))[:count]


def choose_points(pixels,inside,scores,count):
    out=[]
    for flag in (False,True):
        support=np.flatnonzero(inside==flag)
        order=select_equal_count(scores[support],pixels[support],count)
        out.append(support[order])
    return tuple(out)


def edited_mask(mask,add,delete):
    p=np.asarray(add,dtype=np.int64)
    m=np.asarray(delete,dtype=np.int64)
    native=mask.reshape(-1)
    if len(p)!=len(m) or len(np.unique(p))!=len(p) or len(np.unique(m))!=len(m):
        raise ValueError("Equal distinct add and delete edits")
    if any(np.any((a<0)|(a>=native.size)) for a in (p,m)) or native[p].any() or not native[m].all():
        raise ValueError("Edits agree with native foreground and background")
    out=mask.copy().reshape(-1)
    out[p]=True
    out[m]=False
    return out.reshape(mask.shape)


def risk_threshold(scores,useful,strict_lost,matched,margin,groups,min_rows=30,min_groups=15):
    z=np.asarray(scores)
    if not np.isfinite(z).all():
        raise ValueError("Finite candidate action scores")
    order=np.argsort(-z,kind="stable")
    summary=dict(threshold=None,count=0,groups=0,useful=0,unmatched=0,lost_strict=0,margin=0)
    seen=set()
    wins=misses=losses=total=0
    for i,j in enumerate(order,1):
        seen.add(groups[j])
        wins+=int(useful[j])
        misses+=int(not matched[j])
        losses+=int(strict_lost[j])
        total+=int(margin[j])
        tie=i<len(order) and z[order[i]]==z[j]
        if (not tie and i>=min_rows and len(seen)>=min_groups and losses==0
                and 5*wins>=3*i and 5*misses<=i and total>0):
            summary=dict(threshold=float(z[j]),count=i,groups=len(seen),useful=wins,
                         unmatched=misses,lost_strict=losses,margin=total)
    return summary

