# High-IoU mask quality lever: refine the final-layer mask feature 1/4 -> 1/1 (full
# input resolution) instead of 1/4 -> 1/2, so the rank-1 readout produces tighter
# masks at high IoU (the diagnosed inshore deficit on the strong COCO-init carrier).
# On the COCO-init R50 baseline; val on inshore. refine_scale=4.
_base_ = './a0r_r50_coco_pseg.py'

model = dict(panoptic_head=dict(refine_scale=4))

_ins = 'E:/SAR/seal/data/pseg_ssdd/annotations/test_inshore.json'
val_dataloader = dict(num_workers=0, persistent_workers=False, dataset=dict(ann_file=_ins))
val_evaluator = dict(ann_file=_ins)
default_hooks = dict(checkpoint=dict(interval=1044, by_epoch=False, max_keep_ckpts=2,
                                     save_best='coco/segm_mAP', rule='greater'))
