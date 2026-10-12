# c0 (COCO-pretrained M2F-R50) under a LONG ~200-epoch protocol, for an equal-ish-budget
# comparison to O2Former's 500ep-from-scratch result. 5604 imgs / bs2 = 2802 it/ep ->
# 200 ep = 560,400 iters (~2.5 days on 1x 3090Ti at 0.39 s/it). COCO-init converges fast
# and likely peaks well before ep200 -> save_best on segm_mAP is the reported checkpoint
# ('200ep' is the protocol budget; best-val is used, guarding against late overfit).
# Milestones at our proven ~88%/95% relative schedule. Val every ~15 ep; ckpt every ~10 ep
# (max_keep 3) so a crash over the multi-day run loses little (watchdog + --resume).
_base_ = './c0_m2f_cocoinit_hrsid.py'

max_iters = 560400
train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters, val_interval=42030)
param_scheduler = dict(end=max_iters, milestones=[493000, 532000])
default_hooks = dict(
    checkpoint=dict(interval=28020, by_epoch=False, max_keep_ckpts=3,
                    save_best='coco/segm_mAP', rule='greater'))
