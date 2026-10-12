# Phase-1b: dedicated boundary loss (UNet++ teacher MaskBoundaryLoss principle) on the
# COCO-init R50 baseline. boundary_weight=0.2 (teacher's value) adds a boundary-Dice on
# matched masks -> sharper high-IoU boundaries. val on inshore.
_base_ = './a0r_r50_coco_pseg.py'

model = dict(panoptic_head=dict(boundary_weight=0.2, boundary_ksize=3))

_ins = 'E:/SAR/seal/data/pseg_ssdd/annotations/test_inshore.json'
val_dataloader = dict(num_workers=0, persistent_workers=False, dataset=dict(ann_file=_ins))
val_evaluator = dict(ann_file=_ins)
default_hooks = dict(checkpoint=dict(interval=1044, by_epoch=False, max_keep_ckpts=2,
                                     save_best='coco/segm_mAP', rule='greater'))
