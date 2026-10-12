# M3 ShapePrior -- GEO arm: per-query geometry-generated oriented-ellipse prior,
# added gated in logit space on top of the kept rank-1 readout. Baseline loss.
# Inherits the REAL A0R-refine recipe (lr=2.5e-5, max_iters=8352, clip 0.01).
_base_ = './a0r_refine_pseg.py'

custom_imports = dict(
    imports=['carrier.seal_m2f_head', 'carrier.geokernel_m2f_head'],
    allow_failed_imports=False)

model = dict(panoptic_head=dict(
    type='ShapePriorMask2FormerHead',
    refine_scale=2,
    gen_mode='geo',
    prior_hidden=32,
    gate_bias_init=-4.0,
    k0=4.0,
    a_min=1.0))

# Per-module LR DoF for the geometry generator (default 1.0). 'geom_mlp' is a
# unique, longer-than-'backbone' key so the substring match fires first and only
# one rule applies. wd=0 on the geometry generator (don't decay theta/a/b logits).
optim_wrapper = dict(paramwise_cfg=dict(
    custom_keys={
        'backbone': dict(lr_mult=0.1, decay_mult=1.0),
        'geom_mlp': dict(lr_mult=1.0, decay_mult=0.0),
        'query_embed': dict(lr_mult=1.0, decay_mult=0.0),
        'query_feat': dict(lr_mult=1.0, decay_mult=0.0),
        'level_embed': dict(lr_mult=1.0, decay_mult=0.0),
    }))
