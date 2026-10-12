# A2 on HRSID = Gaussian-Wasserstein matching cost, on the REAL battleground (HRSID inshore).
# Survey verdict: HRSID inshore is where M2F is weakest (our 55ep baseline a0_hrsid_seed0 = 12.5 AP /
# AP_s 0.107 / AP75 0.096; O2Former reports vanilla M2F 17.9) -> biggest gain space, best place to show the
# mechanism. Same protocol as a0_m2f_r50_hrsid (batch2, 100872 iters, 800^2) so a0_hrsid_seed0 IS the matched
# baseline. Adds Gaussian-Wasserstein term to the Hungarian matching cost (carrier/wass_match.py, all-tested).
_base_ = './a0_m2f_r50_hrsid.py'

custom_imports = dict(imports=['carrier.wass_match'], allow_failed_imports=False)

model = dict(
    panoptic_head=dict(type='WassMatchM2FHead'),
    train_cfg=dict(   # assigner at model.train_cfg (inherited a0_m2f_r50_pseg -> mmdet base)
        assigner=dict(
            match_costs=[
                dict(type='ClassificationCost', weight=2.0),
                dict(type='CrossEntropyLossCost', weight=5.0, use_sigmoid=True),
                dict(type='DiceCost', weight=5.0, pred_act=True, eps=1.0),
                dict(type='GaussianWassersteinCost', weight=2.0, C=0.5),
            ])),
)
