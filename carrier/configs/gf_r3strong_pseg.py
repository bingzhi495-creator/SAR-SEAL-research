# Strengthened GeoField-shape (the one positive lever): bigger free basis (m=16) + larger
# oriented kernel (k=7) for more shape-following along the elongated hull. gen_mode='shape'
# (position was inert), lr_mult=50 to engage. val on INSHORE (the diagnosed axis).
_base_ = './a0r_refine_pseg.py'

custom_imports = dict(
    imports=['carrier.seal_m2f_head', 'carrier.geofield_op', 'carrier.geofield_m2f_head'],
    allow_failed_imports=False)

model = dict(panoptic_head=dict(
    type='GeoFieldMask2FormerHead',
    refine_scale=2,
    gen_mode='shape',
    geo_d=16, geo_r=64, geo_m=16, geo_k=7))

optim_wrapper = dict(paramwise_cfg=dict(
    custom_keys={
        'backbone': dict(lr_mult=0.1, decay_mult=1.0),
        'geofield': dict(lr_mult=50.0, decay_mult=0.0),
        'query_embed': dict(lr_mult=1.0, decay_mult=0.0),
        'query_feat': dict(lr_mult=1.0, decay_mult=0.0),
        'level_embed': dict(lr_mult=1.0, decay_mult=0.0),
    }))
