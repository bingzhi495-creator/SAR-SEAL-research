# Refine head (SealM2FHead, validated +2.6 AP75 at 36ep) RE-TESTED on the proper 100ep base.
# M2F-long(100ep) revealed 36ep was undertrained: overall .518->.590, inshore .34->.462.
# Question: does the refine booster still help once the base is properly trained? Mirror a0_m2f_r50_pseg_long.
_base_ = './a0r_refine_pseg.py'

max_iters = 23200   # 928*100/4
train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters, val_interval=2320)
optim_wrapper = dict(optimizer=dict(lr=2.5e-5))
param_scheduler = dict(end=max_iters, milestones=[20000, 22000])
default_hooks = dict(
    checkpoint=dict(interval=2320, by_epoch=False, max_keep_ckpts=2,
                    save_best='coco/segm_mAP', rule='greater'),
    logger=dict(interval=50))
randomness = dict(seed=0, deterministic=False)
