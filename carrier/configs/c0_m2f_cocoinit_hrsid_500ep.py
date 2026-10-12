# c0 under the O2Former 500-EPOCH protocol, for a fair equal-epoch head-to-head.
# Base c0 = COCO-pretrained M2F-R50 (our regime). 5604 imgs / bs2 = 2802 iters/epoch,
# so 500 ep = 1,401,000 iters (~14x the 36-ep anchor; ~6-7 days on 1x 3090Ti at 0.39 s/it).
# COCO-init converges fast and will likely peak well before ep500 -> save_best on segm_mAP
# is the reported checkpoint (the '500ep' label is the protocol budget, best-val is used).
# Milestones kept at our proven ~88%/95% relative schedule. Val every ~20 ep, ckpt every
# ~10 ep (max_keep 3) so a crash over the multi-day run loses little (watchdog + --resume).
_base_ = './c0_m2f_cocoinit_hrsid.py'

max_iters = 1401000
train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters, val_interval=56040)
param_scheduler = dict(end=max_iters, milestones=[1236000, 1334000])
default_hooks = dict(
    checkpoint=dict(interval=28020, by_epoch=False, max_keep_ckpts=3,
                    save_best='coco/segm_mAP', rule='greater'))
