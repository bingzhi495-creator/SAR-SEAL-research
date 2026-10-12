# a2b = EXACT Bures-Wasserstein matching cost on HRSID (review w13gekell math fix): identical to a2 except
# GaussianWassersteinCost(exact=True) uses the exact non-commuting Bures covariance term instead of the
# commuting-case approximation (which over-estimates distance by up to ~35-63% for rotated/elongated ships --
# exactly the SAR regime). All reported a2/a1a2 numbers used the approximate form (exact=False default).
# NOTE (mmengine gotcha): lists are REPLACED not merged, so the full 4-cost list is restated here.
_base_ = './a0_m2f_r50_hrsid.py'

custom_imports = dict(imports=['carrier.wass_match'], allow_failed_imports=False)

model = dict(
    panoptic_head=dict(type='WassMatchM2FHead'),
    train_cfg=dict(
        assigner=dict(
            match_costs=[
                dict(type='ClassificationCost', weight=2.0),
                dict(type='CrossEntropyLossCost', weight=5.0, use_sigmoid=True),
                dict(type='DiceCost', weight=5.0, pred_act=True, eps=1.0),
                dict(type='GaussianWassersteinCost', weight=2.0, C=0.5, exact=True),
            ])),
)
