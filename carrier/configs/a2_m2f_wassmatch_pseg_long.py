# A2: Gaussian-Wasserstein matching cost (cross-domain novelty, workflow w7qfmdh0u).
# Diagnosis: inshore small-ship confident mis-localization originates in the Hungarian MATCHING COST
# (cls+mask-focal+dice are IoU-like -> step/zero gradient for ~3.5px ships -> matcher mis-assigns).
# Cure: add a Gaussian-Wasserstein (NWD) term to the matching cost -> smooth, scale-invariant, nonzero
# gradient for small shifts (CPU-verified). Orthogonal to a1 (resolution axis) + all washed mechanisms.
# WassMatchM2FHead attaches full spatial masks so the cost can compute centroid/cov (M2F matches on
# point-sampled masks). GATE: also have a loss-only arm later to separate from O2Former (NWD-in-loss).
_base_ = './a0_m2f_r50_pseg_long.py'

custom_imports = dict(imports=['carrier.wass_match'], allow_failed_imports=False)

model = dict(
    panoptic_head=dict(type='WassMatchM2FHead'),
    train_cfg=dict(   # assigner lives at model.train_cfg (NOT panoptic_head.train_cfg)
        assigner=dict(
            match_costs=[
                dict(type='ClassificationCost', weight=2.0),
                dict(type='CrossEntropyLossCost', weight=5.0, use_sigmoid=True),
                dict(type='DiceCost', weight=5.0, pred_act=True, eps=1.0),
                dict(type='GaussianWassersteinCost', weight=2.0, C=0.5),
            ])),
)
