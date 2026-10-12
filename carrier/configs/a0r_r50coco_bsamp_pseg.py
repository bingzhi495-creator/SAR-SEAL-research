# Phase-1a boundary loss (cheap, config-only): focus the point-sampled mask loss on
# boundary/uncertain points. Mask2Former samples points by uncertainty (|logit|~0 =
# boundary); raising num_points + importance_sample_ratio concentrates the CE/Dice on
# the mask boundary -> tighter high-IoU masks (the diagnosed inshore deficit). On the
# COCO-init R50 baseline; val on inshore.
_base_ = './a0r_r50_coco_pseg.py'

model = dict(train_cfg=dict(
    num_points=24576,              # 12544 -> 24576 (2x boundary resolution)
    oversample_ratio=3.0,
    importance_sample_ratio=0.9))  # 0.75 -> 0.9 (more boundary/uncertain points)

_ins = 'E:/SAR/seal/data/pseg_ssdd/annotations/test_inshore.json'
val_dataloader = dict(num_workers=0, persistent_workers=False, dataset=dict(ann_file=_ins))
val_evaluator = dict(ann_file=_ins)
default_hooks = dict(checkpoint=dict(interval=1044, by_epoch=False, max_keep_ckpts=2,
                                     save_best='coco/segm_mAP', rule='greater'))
