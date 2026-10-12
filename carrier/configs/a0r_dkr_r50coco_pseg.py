# SAR-DKR D1: per-instance dynamic-kernel mask branch (no rel-coords yet) on COCO-init R50.
# Gated (init 0) -> floor = A0R. Tests whether adding dynamic-conv rank to the rank-1 readout
# moves mask mAP (the literal DiffSAR mask mechanism grafted into our COCO carrier). val inshore.
_base_ = './a0r_r50_coco_pseg.py'

custom_imports = dict(
    imports=['carrier.seal_m2f_head', 'carrier.dkr_op', 'carrier.geofield_m2f_head'],
    allow_failed_imports=False)

model = dict(panoptic_head=dict(type='DKRMask2FormerHead', refine_scale=2, dkr_channels=8))

_ins = 'E:/SAR/seal/data/pseg_ssdd/annotations/test_inshore.json'
val_dataloader = dict(num_workers=0, persistent_workers=False, dataset=dict(ann_file=_ins))
val_evaluator = dict(ann_file=_ins)
default_hooks = dict(checkpoint=dict(interval=1044, by_epoch=False, max_keep_ckpts=2,
                                     save_best='coco/segm_mAP', rule='greater'))
