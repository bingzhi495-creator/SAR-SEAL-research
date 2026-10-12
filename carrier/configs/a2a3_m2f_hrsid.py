# a2+a3 stacked on HRSID (the 异轴叠加 headline ablation): a2 = Gaussian-Wasserstein MATCHING cost +
# a3 = RepMask inter-instance mask-repulsion LOSS, in one head (RepMaskWassM2FHead). Compared against
# a2-alone (a2_m2f_wassmatch_hrsid, inshore .173) and a3-alone (a3_m2f_repmask_hrsid) and baseline (.125).
# Contribution claim = a2+a3 > a2-alone on inshore AP75/AP_m without losing a2's AP50/AP_s.
_base_ = './a2_m2f_wassmatch_hrsid.py'   # inherits the GaussianWassersteinCost match_costs + a2 plumbing

custom_imports = dict(imports=['carrier.wass_match', 'carrier.rep_mask'], allow_failed_imports=False)

model = dict(
    panoptic_head=dict(
        type='RepMaskWassM2FHead',   # a2 matching (inherited from WassMatchM2FHead) + a3 RepGT loss
        lambda_repgt=0.5,
        rep_tau=0.15,
        rep_sigma=0.5),
)
