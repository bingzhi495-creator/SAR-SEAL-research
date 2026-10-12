# M3 ShapePrior -- STATIC foil: ONE shared learned oriented ellipse (geometry is
# an nn.Parameter, NOT predicted per-query); gate still per-query. Equal params,
# no per-instance geometry -> the truly-static capacity foil. Baseline loss.
_base_ = './a0r_refine_pseg.py'

custom_imports = dict(
    imports=['carrier.seal_m2f_head', 'carrier.geokernel_m2f_head'],
    allow_failed_imports=False)

model = dict(panoptic_head=dict(
    type='ShapePriorMask2FormerHead',
    refine_scale=2,
    gen_mode='static',
    prior_hidden=32,
    gate_bias_init=-4.0,
    k0=4.0,
    a_min=1.0))

optim_wrapper = dict(paramwise_cfg=dict(
    custom_keys={
        'backbone': dict(lr_mult=0.1, decay_mult=1.0),
        'geom_mlp': dict(lr_mult=1.0, decay_mult=0.0),
        'query_embed': dict(lr_mult=1.0, decay_mult=0.0),
        'query_feat': dict(lr_mult=1.0, decay_mult=0.0),
        'level_embed': dict(lr_mult=1.0, decay_mult=0.0),
    }))
