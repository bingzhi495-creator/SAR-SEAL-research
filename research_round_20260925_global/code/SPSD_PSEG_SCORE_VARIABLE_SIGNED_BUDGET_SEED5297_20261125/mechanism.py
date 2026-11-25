"""Exact image-mask budget and per-reference IoU-difference algebra."""

import numpy as np


def iou_terms(old_area,old_overlap,new_area,new_overlap,ref_area):
    """Return signed total, overlap, and area components using the symmetric exact identity."""
    if (not 0 <= old_overlap <= min(old_area,ref_area)
            or not 0 <= new_overlap <= min(new_area,ref_area) or ref_area<=0):
        raise ValueError("Physical foreground, intersections and nonempty author reference required")
    old_union=old_area+ref_area-old_overlap
    new_union=new_area+ref_area-new_overlap
    d_overlap=new_overlap-old_overlap
    d_area=new_area-old_area
    intersection=(ref_area+(old_area+new_area)/2)*d_overlap/(old_union*new_union)
    area=-(old_overlap+new_overlap)*d_area/(2*old_union*new_union)
    total=new_overlap/new_union-old_overlap/old_union
    if abs(total-intersection-area)>1e-12:
        raise ValueError("Exact symmetric IoU identity")
    return total,intersection,area


def signed_changes(old,new,points,k):
    """Read actual signed pixel writes against an immutable native mask and support."""
    if old.shape!=new.shape or old.ndim!=2 or old.dtype!=bool or new.dtype!=bool:
        raise ValueError("Two native binary masks on identical geometry required")
    if points.ndim!=2 or points.shape[1]!=2 or points.dtype.kind not in "iu":
        raise ValueError("Original signed point support shape")
    n=old.size
    if np.any((points<0)|(points>=n)) or len(np.unique(points))!=points.size:
        raise ValueError("Unique in-frame signed point support")
    a,b=old.ravel(),new.ravel()
    if np.any(a[points[:,0]]) or not np.all(a[points[:,1]]):
        raise ValueError("Trusted add/drop support disagrees with old mask")
    adds=np.flatnonzero(b & ~a)
    drops=np.flatnonzero(a & ~b)
    if len(adds)+len(drops)!=2*k:
        raise ValueError("Exact old total flip budget")
    if not np.isin(adds,points[:,0]).all() or not np.isin(drops,points[:,1]).all():
        raise ValueError("Exact frozen add/drop support")
    return len(adds),len(drops)



def validate_pairwise_gain(gain,action_area,incumbent_area,common_area):
    """Bound expected intersection change by the two exact foreground set differences."""
    if (not 0<=common_area<=min(action_area,incumbent_area) or gain.shape!=(2,)
            or not np.isfinite(gain).all()
            or np.any(gain>action_area-common_area+1e-12)
            or np.any(gain<-(incumbent_area-common_area)-1e-12)):
        raise ValueError("Expected gain must lie within exact pairwise foreground set-difference bounds")


def gate(refs, exposure):
    def matchup(arm):
        return [refs[ref]["variable_iou_minus"][arm] for ref in ("pseg","spsd")]

    def effect(arm):
        cases=matchup(arm)
        return dict(iou_magnitude=all(r["iou_points"]>=.15 for r in cases),
                    iou_lower=all(r["percentile95_points"][0]>0 for r in cases),
                    ap_guard=all(r["ap_points"]>=-.10 for r in cases),
                    exposure=exposure[arm][0]>=20 and exposure[arm][1]>=8)

    efficacy=effect("bce_robust")
    specificity=effect("fixed_iou")
    minima={arm:min(r["iou_points"] for r in matchup(arm)) for arm in ("bce_mean","static")}
    controls={arm:d>0 for arm,d in minima.items()}
    efficacy_pass=all(efficacy.values()) and all(controls.values())
    specificity_pass=all(specificity.values())
    return dict(advances=efficacy_pass and specificity_pass,main_pass=efficacy_pass,specificity_pass=specificity_pass,
                main_vs_archived_robust_checks=efficacy,specificity_vs_fixed_iou_checks=specificity,
                control_checks=controls,worst_reference_iou_difference_vs_controls=minima)
