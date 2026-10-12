# GeoField R5 (FULL, relation OFF): coords + oriented free-basis kernel +
# non-separable shape-coupled salience. The full unified operator. Decisive arm:
#   fusion>capacity iff R5>R1 ;  entanglement iff R5>max(R2,R3).
_base_ = './a0r_refine_pseg.py'

custom_imports = dict(
    imports=['carrier.seal_m2f_head', 'carrier.geofield_m2f_head'],
    allow_failed_imports=False)

model = dict(panoptic_head=dict(
    type='GeoFieldMask2FormerHead',
    refine_scale=2,
    gen_mode='full',
    geo_d=16, geo_r=64, geo_m=8, geo_k=5))

optim_wrapper = dict(paramwise_cfg=dict(
    custom_keys={
        'backbone': dict(lr_mult=0.1, decay_mult=1.0),
        'geofield': dict(lr_mult=50.0, decay_mult=0.0),  # gamma can't move at base lr 2.5e-5; boost so the operator actually engages
        'query_embed': dict(lr_mult=1.0, decay_mult=0.0),
        'query_feat': dict(lr_mult=1.0, decay_mult=0.0),
        'level_embed': dict(lr_mult=1.0, decay_mult=0.0),
    }))
