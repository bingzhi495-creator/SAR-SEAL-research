# Combo: OAEM (orientation) + DenoDet (frequency denoise) together on COCO-init R50.
# Tests whether the two new modules (different pain points) STACK. Both gated init 0.
_base_ = './a0r_r50_coco_pseg.py'

custom_imports = dict(
    imports=['carrier.seal_m2f_head', 'carrier.oaem_op', 'carrier.ndm_op', 'carrier.geofield_m2f_head'],
    allow_failed_imports=False)

model = dict(panoptic_head=dict(
    type='OAEMMask2FormerHead', refine_scale=2,
    oaem_in_channels=(256, 512, 1024, 2048), oaem_k=7,
    use_oaem=True, use_denodet=True))

_ins = 'E:/SAR/seal/data/pseg_ssdd/annotations/test_inshore.json'
val_dataloader = dict(num_workers=0, persistent_workers=False, dataset=dict(ann_file=_ins))
val_evaluator = dict(ann_file=_ins)
default_hooks = dict(checkpoint=dict(interval=1044, by_epoch=False, max_keep_ckpts=2,
                                     save_best='coco/segm_mAP', rule='greater'))
