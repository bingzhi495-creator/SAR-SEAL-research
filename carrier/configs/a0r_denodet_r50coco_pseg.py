# New module #2: DenoDet NDM (frequency-domain speckle denoise) on COCO-init R50.
# Gated residual (init 0) -> safe. val on inshore. Targets speckle + small ships +
# high-freq mask detail (AP75), a different pain point than OAEM (orientation).
_base_ = './a0r_r50_coco_pseg.py'

custom_imports = dict(
    imports=['carrier.seal_m2f_head', 'carrier.oaem_op', 'carrier.ndm_op', 'carrier.geofield_m2f_head'],
    allow_failed_imports=False)

model = dict(panoptic_head=dict(
    type='OAEMMask2FormerHead', refine_scale=2,
    oaem_in_channels=(256, 512, 1024, 2048),
    use_oaem=False, use_denodet=True))

_ins = 'E:/SAR/seal/data/pseg_ssdd/annotations/test_inshore.json'
val_dataloader = dict(num_workers=0, persistent_workers=False, dataset=dict(ann_file=_ins))
val_evaluator = dict(ann_file=_ins)
default_hooks = dict(checkpoint=dict(interval=1044, by_epoch=False, max_keep_ckpts=2,
                                     save_best='coco/segm_mAP', rule='greater'))
