# SAR-DKR D2: dynamic-kernel mask branch + RELATIVE COORDINATES (the theory's load-bearing
# per-instance spatial DOF that DiffSAR lacks and D1 omitted). Gated (init 0) -> floor = A0R.
# coord origin = soft-mask centroid (D3 will swap in the SAR scatterer-weighted centroid).
_base_ = './a0r_r50_coco_pseg.py'

custom_imports = dict(
    imports=['carrier.seal_m2f_head', 'carrier.dkr_op', 'carrier.geofield_m2f_head'],
    allow_failed_imports=False)

model = dict(panoptic_head=dict(
    type='DKRMask2FormerHead', refine_scale=2, dkr_channels=8, dkr_relcoord=True))

_ins = 'E:/SAR/seal/data/pseg_ssdd/annotations/test_inshore.json'
val_dataloader = dict(num_workers=0, persistent_workers=False, dataset=dict(ann_file=_ins))
val_evaluator = dict(ann_file=_ins)
default_hooks = dict(checkpoint=dict(interval=1044, by_epoch=False, max_keep_ckpts=2,
                                     save_best='coco/segm_mAP', rule='greater'))
