# A0 carrier baseline: mmdet Mask2Former-R50, PSeg-SSDD, 36 epochs.
# Protocol: batch 4 (1x 3090 Ti), lr linearly scaled from the official
# 1e-4@bs16 -> 2.5e-5@bs4; LSJ at 512^2 (PSeg images ~400-500 px).
# This row anchors the PSeg-SSDD ladder (L1) of the v4.1 plan.
_base_ = './a0_m2f_r50_pseg_smoke.py'

# 928 train imgs * 36 epochs / batch 4 = 8352 iters
max_iters = 8352
train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters, val_interval=1044)  # ~every 4.5 epochs

optim_wrapper = dict(optimizer=dict(lr=2.5e-5))
param_scheduler = dict(end=max_iters, milestones=[7400, 8000])

default_hooks = dict(
    checkpoint=dict(interval=2088, by_epoch=False, max_keep_ckpts=2,
                    save_best='coco/segm_mAP', rule='greater'),
    logger=dict(interval=50))

randomness = dict(seed=0, deterministic=False)
