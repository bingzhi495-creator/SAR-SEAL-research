# GeoField-shape on the strong baseline: COCO-init Swin-S. The intended full
# composition (strong carrier + our near-shore mechanism) before adding the P4
# SAR-physics module. Adds GeoField head on top of a0r_swins_coco; geofield
# lr_mult=50 merges with the inherited Swin optimizer custom_keys.
_base_ = './a0r_swins_coco_pseg.py'

custom_imports = dict(
    imports=['carrier.seal_m2f_head', 'carrier.geofield_op', 'carrier.geofield_m2f_head'],
    allow_failed_imports=False)

model = dict(panoptic_head=dict(
    type='GeoFieldMask2FormerHead',
    refine_scale=2, gen_mode='shape',
    geo_d=16, geo_r=64, geo_m=8, geo_k=5))

optim_wrapper = dict(paramwise_cfg=dict(custom_keys=dict(
    geofield=dict(lr_mult=50.0, decay_mult=0.0))))
