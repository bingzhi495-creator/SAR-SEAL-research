# B0 BASELINE: stock CondInst-R50, PSeg-SSDD, 36ep-equiv (8352 iters @bs4), COCO-init.
# The honest SAME-CARRIER bar (native parse, FULL COCO transfer incl. controller). Do NOT use
# the [UNVERIFIED] O2Former-cited CondInst 58.0 as the bar -- this self-run is the bar.
# lr MUST be sweep-confirmed {1.25e-3, 2.5e-3, 5e-3} (CondInst is lr-sensitive).
_base_ = './condinst_r50coco_pseg_smoke.py'

max_iters = 8352  # 928 imgs * 36ep / bs4
train_cfg = dict(_delete_=True, type='IterBasedTrainLoop', max_iters=max_iters, val_interval=1044)
optim_wrapper = dict(optimizer=dict(lr=2.5e-3))  # B0 default; sweep around this
param_scheduler = [
    dict(type='LinearLR', start_factor=0.01, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', by_epoch=False, milestones=[7400, 8000], gamma=0.1),
]
default_hooks = dict(
    checkpoint=dict(interval=1044, by_epoch=False, max_keep_ckpts=2,
                    save_best='coco/segm_mAP', rule='greater'),
    logger=dict(interval=50))
randomness = dict(seed=0, deterministic=False)

# P1 booster: EMA weights (+0.2..0.5, free regularization). Applied EQUALLY to all F1 arms
# (fair kernel-shape comparison). ExpMomentumEMA standard for mmdet detectors.
custom_hooks = [dict(type='EMAHook', ema_type='ExpMomentumEMA',
                     momentum=0.0002, update_buffers=True, priority=49)]
