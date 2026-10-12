# GeoField-shape RE-TEST on the COCO-init R50 baseline. Key question: now that the
# head is no longer the 36ep bottleneck (COCO-init), does GeoField's near-shore
# geometry enrichment finally show its value on a properly-converged carrier?
# Same COCO-init as a0r_r50_coco; only adds the GeoField head (equal-param control
# would be gen_mode='capacity').
_base_ = './a0r_r50_coco_pseg.py'

custom_imports = dict(
    imports=['carrier.seal_m2f_head', 'carrier.geofield_op', 'carrier.geofield_m2f_head'],
    allow_failed_imports=False)

model = dict(panoptic_head=dict(
    type='GeoFieldMask2FormerHead',
    refine_scale=2, gen_mode='shape',
    geo_d=16, geo_r=64, geo_m=8, geo_k=5))

optim_wrapper = dict(paramwise_cfg=dict(custom_keys=dict(
    geofield=dict(lr_mult=50.0, decay_mult=0.0))))

# GATE-0 evals on the diagnosed axis (inshore). num_workers=0 avoids the Windows
# val-dataloader deadlock. checkpoint every 1044 iters -> resumable after a reboot.
_ins = 'E:/SAR/seal/data/pseg_ssdd/annotations/test_inshore.json'
val_dataloader = dict(num_workers=0, persistent_workers=False, dataset=dict(ann_file=_ins))
val_evaluator = dict(ann_file=_ins)
default_hooks = dict(checkpoint=dict(interval=1044, by_epoch=False, max_keep_ckpts=2,
                                     save_best='coco/segm_mAP', rule='greater'))
