# GeoField R6 (FULL + RELATION): head forward IDENTICAL to R5 (full) -- same params,
# same inference (relation adds ZERO inference params). The inter-instance RKD
# relational regularizer on z (training-only) is wired by the training wrapper;
# until then this arm == R5. relation effective iff R6>R5. Full-ladder arm.
_base_ = './a0r_refine_pseg.py'

custom_imports = dict(
    imports=['carrier.seal_m2f_head', 'carrier.geofield_m2f_head'],
    allow_failed_imports=False)

model = dict(panoptic_head=dict(
    type='GeoFieldMask2FormerHead',
    refine_scale=2,
    gen_mode='full_rel',
    geo_d=16, geo_r=64, geo_m=8, geo_k=5))

optim_wrapper = dict(paramwise_cfg=dict(
    custom_keys={
        'backbone': dict(lr_mult=0.1, decay_mult=1.0),
        'geofield': dict(lr_mult=1.0, decay_mult=0.0),
        'query_embed': dict(lr_mult=1.0, decay_mult=0.0),
        'query_feat': dict(lr_mult=1.0, decay_mult=0.0),
        'level_embed': dict(lr_mult=1.0, decay_mult=0.0),
    }))
