"""Partition false foreground changes without changing either scored mask."""
import numpy as np
from scipy import ndimage


def attachment(native,fixed,variable,reference,neighbors):
    if neighbors not in (4,8) or len({x.shape for x in (native,fixed,variable,reference)})!=1:
        raise ValueError("One native pixel frame and four or eight neighbors")
    kept = fixed & variable
    gained = variable & ~fixed & ~reference
    removed = fixed & ~variable & ~reference
    links = ndimage.generate_binary_structure(2,1 if neighbors==4 else 2)
    labels,count = ndimage.label(variable,structure=links)
    area = np.bincount(labels.ravel(),minlength=count+1)
    shared = np.bincount(labels[kept],minlength=count+1)
    seeded = np.flatnonzero(shared>0)
    if not len(seeded):
        raise ValueError("Matched positive-budget masks must share native foreground")
    main = max(map(int,seeded),key=lambda j:(area[j],shared[j],-j))
    main_gains = gained & (labels==main)
    secondary = gained & (shared[labels]>0) & (labels!=main)
    unseeded = gained & (shared[labels]==0)
    novel = ~native
    g,r = int(gained.sum()),int(removed.sum())
    x,y,z = (int(mask.sum()) for mask in (main_gains,secondary,unseeded))
    net = int((variable & ~reference).sum())-int((fixed & ~reference).sum())
    if sum((x,y,z))!=g or g-r!=net:
        raise ValueError("Disjoint exhaustive pixel ledger")
    near_kept = ndimage.binary_dilation(kept & (labels==main),structure=links)
    reference_neighbor = ndimage.binary_dilation(reference,structure=links)
    novel_gain = int((gained & novel).sum())
    novel_removal = int((removed & novel).sum())
    extra = dict(all_gain=g,all_removal=r,net_fp=net,
        main_gain=x,secondary_gain=y,unseeded_gain=z,
        main_vs_other=x-y-z,main_less_all_removal=x-r,
        main_novel_gain=int((main_gains & novel).sum()),
        secondary_novel_gain=int((secondary & novel).sum()),
        unseeded_novel_gain=int((unseeded & novel).sum()),
        all_restored_gain=g-novel_gain,all_restored_removed=r-novel_removal,
        net_novel=novel_gain-novel_removal,
        net_restored=(g-novel_gain)-(r-novel_removal),
        near_kept_main=int((main_gains & near_kept).sum()),
        near_author_main=int((main_gains & reference_neighbor).sum()),
        net_fn=int((reference & ~variable).sum())-int((reference & ~fixed).sum()))
    if sum((extra["net_novel"],extra["net_restored"]))!=net:
        raise ValueError("Native restoration and novel edits sum to net false foreground")
    return extra
