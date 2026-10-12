# GeoStack: plain module stacking (NO-GO fallback). CoordConv+CBAM+oriented-conv blocks
# stacked independently on the refined mask feature. Diagnostic: do the principles help
# when given full independent capacity (vs the unified GeoField NO-GO)? geostack lr_mult=50
# so the residual gate engages.
_base_ = './a0r_refine_pseg.py'

custom_imports = dict(
    imports=['carrier.seal_m2f_head', 'carrier.geostack_op', 'carrier.geofield_m2f_head'],
    allow_failed_imports=False)

model = dict(panoptic_head=dict(
    type='GeoStackMask2FormerHead',
    refine_scale=2,
    geo_m=8, geo_k=5))

optim_wrapper = dict(paramwise_cfg=dict(
    custom_keys={
        'backbone': dict(lr_mult=0.1, decay_mult=1.0),
        'geostack': dict(lr_mult=50.0, decay_mult=0.0),  # let the residual gate engage
        'query_embed': dict(lr_mult=1.0, decay_mult=0.0),
        'query_feat': dict(lr_mult=1.0, decay_mult=0.0),
        'level_embed': dict(lr_mult=1.0, decay_mult=0.0),
    }))
