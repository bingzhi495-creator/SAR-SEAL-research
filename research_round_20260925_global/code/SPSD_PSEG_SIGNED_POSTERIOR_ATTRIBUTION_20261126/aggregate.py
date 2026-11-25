"""Image-then-group aggregation for frozen signed probability evidence."""

from collections import defaultdict
import numpy as np


def group_table(rows, image_to_group, ordered_groups, width):
    groups=defaultdict(list)
    for image,group in image_to_group.items():
        vectors=rows.get((image,group),())
        groups[group].append(np.mean(vectors,axis=0) if len(vectors) else np.zeros(width))
    if set(groups)!=set(ordered_groups):
        raise ValueError("Original complete decoded-group order")
    return np.stack([np.mean(groups[g],axis=0) for g in ordered_groups])


def rate(group_values, draws):
    """Aggregate occupancy residual per exposed pixel at the fixed group weights."""
    sums=group_values.sum(axis=0)
    if sums[1]<=0:
        return None,None
    delta=(sums[0]-sums[2])/sums[1]
    draws_sum=group_values[draws].sum(axis=1)
    valid=draws_sum[:,1]>0
    if not valid.any():
        return float(delta),None
    errors=(draws_sum[valid,0]-draws_sum[valid,2])/draws_sum[valid,1]
    return float(delta),np.percentile(errors,[2.5,97.5]).tolist()


def intervals(values,draws,scale=1):
    """Same complete paired group draws for every signed column."""
    return ((scale*values.mean(axis=0)).tolist(),
           np.percentile(scale*values[draws].mean(axis=1),[2.5,97.5],axis=0).T.tolist())
