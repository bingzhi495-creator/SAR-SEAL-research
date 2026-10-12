# c0 @ 55ep with COSINE LR (local recipe-validation vs c0-step .279). Same everything as c0 EXCEPT the LR
# schedule: cosine+warmup instead of MultiStepLR step-decay. Isolates the cosine gain on OUR data/regime
# before committing it to the AutoDL 500ep run. If c0_cosine > c0_step here, cosine transfers -> use on AutoDL.
_base_ = './c0_m2f_cocoinit_hrsid.py'
max_iters = 100872
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-3, by_epoch=False, begin=0, end=500),
    dict(type='CosineAnnealingLR', by_epoch=False, begin=500, end=max_iters, eta_min_ratio=1e-2),
]
