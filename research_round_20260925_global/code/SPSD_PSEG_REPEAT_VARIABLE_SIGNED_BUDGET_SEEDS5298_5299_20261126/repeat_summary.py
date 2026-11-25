"""Equal-seed paired group estimates on the common acquisition geometry."""

import numpy as np


def paired_seed_pool(deltas, ap_points, group_draws):
    vectors=tuple(np.asarray(v,dtype=np.float64) for v in deltas)
    if (not vectors or len(vectors)!=len(ap_points) or group_draws.ndim!=2
            or any(v.ndim!=1 or v.size!=group_draws.shape[1] or not np.isfinite(v).all() for v in vectors)
            or not np.isfinite(ap_points).all()):
        raise ValueError("Every seed requires the same finite ordered decoded group records")
    means=np.mean(np.stack(vectors),axis=0)
    return dict(iou_points=float(means.mean()), ap_points=float(np.mean(ap_points)),
                percentile95_points=np.percentile(means[group_draws].mean(axis=1),[2.5,97.5]).tolist())


def thresholds(refs,exposure):
    def compared(name):
        return [refs[ref]["variable_iou_minus"][name] for ref in ("pseg","spsd")]

    def check(arm):
        effects=compared(arm)
        return dict(iou_magnitude=all(row["iou_points"]>=.15 for row in effects),
                    iou_lower=all(row["percentile95_points"][0]>0 for row in effects),
                    ap_guard=all(row["ap_points"]>=-.1 for row in effects),
                    exposure=exposure[arm][0]>=20 and exposure[arm][1]>=8)

    efficacy=check("same_bce_robust")
    specificity=check("fixed_iou")
    minima={arm:min(row["iou_points"] for row in compared(arm)) for arm in ("same_bce_mean","static")}
    controls={arm:minimum>0 for arm,minimum in minima.items()}
    main=all(efficacy.values()) and all(controls.values())
    specific=all(specificity.values())
    return dict(advances=main and specific,main_pass=main,specificity_pass=specific,
                main_vs_same_seed_robust_checks=efficacy,specificity_vs_fixed_iou_checks=specificity,
                control_checks=controls,worst_reference_iou_difference_vs_controls=minima)
