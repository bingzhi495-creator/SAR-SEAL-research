"""Per-image selective fixed-budget adoption and exact paired gate checks."""

from collections import defaultdict
from math import isfinite


def choose(native_ranks, quotas, scores):
    """Fill the fixed one-third image budget while matching quota composition."""
    strata=defaultdict(list)
    seen=set()
    for rank,k,score in zip(native_ranks,quotas,scores,strict=True):
        if rank in seen or not 1<=rank<=100 or not 1<=k<=8 or not isfinite(score):
            raise ValueError("Unique original rank, original positive quota and finite rank score")
        seen.add(rank)
        strata[k].append((score,rank))
    counts={k:len(rows)//3 for k,rows in strata.items()}
    remain=len(seen)//3-sum(counts.values())
    for k in sorted(strata,key=lambda k:(-(len(strata[k])%3),k)):
        if remain==0:
            break
        if len(strata[k])%3:
            counts[k]+=1
            remain-=1
    if remain:
        raise ValueError("Complete original one-third image adoption allocation")
    return {rank for k,rows in strata.items()
            for _,rank in sorted(rows,key=lambda row:(-row[0],row[1]))[:counts[k]]}


def gate(refs, exposure):
    def compared(arm):
        return [refs[name]["selective_minus"][arm] for name in ("pseg","spsd")]

    def criteria(arm):
        effects=compared(arm)
        return dict(iou_magnitude=all(r["iou_points"]>=.15 for r in effects),
                    iou_lower=all(r["percentile95_points"][0]>0 for r in effects),
                    ap_guard=all(r["ap_points"]>=-.10 for r in effects),
                    exposure=exposure[arm][0]>=20 and exposure[arm][1]>=8)

    main=criteria("bce_robust")
    specificity=criteria("margin_ranked")
    contrasts={arm:min(r["iou_points"] for r in compared(arm)) for arm in ("bce_mean","static")}
    controls={arm:value>0 for arm,value in contrasts.items()}
    main_pass=all(main.values()) and all(controls.values())
    specificity_pass=all(specificity.values())
    return dict(advances=main_pass and specificity_pass, main_pass=main_pass,
                specificity_pass=specificity_pass, always_robust_checks=main,
                margin_ranked_checks=specificity, control_checks=controls,
                worst_reference_iou_pts_vs_controls=contrasts)
