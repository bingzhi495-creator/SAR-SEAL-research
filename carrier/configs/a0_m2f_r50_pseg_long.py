# M2F-R50 PSeg, LONG schedule (100ep) to test the undertraining fix locally (36ep=0.518; M2F is
# data-hungry). batch4 @512^2 COCO-init. ~23200 iters (~4h on 3090Ti). A800 version (batch16/hires)
# in autodl_package is the definitive one; this is the local trend-confirm + fallback. Resumable.
_base_ = './a0_m2f_r50_pseg.py'

max_iters = 23200   # 928*100/4
train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters, val_interval=2320)
optim_wrapper = dict(optimizer=dict(lr=2.5e-5))
param_scheduler = dict(end=max_iters, milestones=[20000, 22000])
default_hooks = dict(
    checkpoint=dict(interval=2320, by_epoch=False, max_keep_ckpts=2,
                    save_best='coco/segm_mAP', rule='greater'),
    logger=dict(interval=50))
randomness = dict(seed=0, deterministic=False)
