# a3 = RepMask (inter-instance mask-repulsion loss) on HRSID, a3-alone arm.
# Base = a0_m2f_r50_hrsid (stock Mask2FormerHead) so a0_hrsid_seed0 IS the matched control (same protocol,
# 100872 iters/bs2/800^2). Adds the RepGT loss on matched per-query masks (carrier/rep_mask.py). lambda_repgt=0
# would be byte-identical to a0. Targets the inshore crowding residual a2 leaves (67% multi-ship / 4 ships/img).
_base_ = './a0_m2f_r50_hrsid.py'

custom_imports = dict(imports=['carrier.rep_mask'], allow_failed_imports=False)

model = dict(
    panoptic_head=dict(
        type='RepMaskM2FHead',
        lambda_repgt=0.5,   # swept in {0.25,0.5,1.0} on a held-out split, never per-seed
        rep_tau=0.15,       # neighbour gate: GT centroid dist in normalized [0,1] coords (~15% of side)
        rep_sigma=0.5),     # Smooth_ln knee (CVPR2018 default)
)
