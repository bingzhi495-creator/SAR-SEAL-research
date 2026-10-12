# Teacher-style placement (user directive): CBAM + DSConv in the BACKBONE (upstream, where
# info still exists), CoordConv in the HEAD. Tests whether backbone enrichment beats the
# baseline where all head-slot enrichment was NO-GO. Enrichment is residual-gated (init 0 =
# starts == baseline); enrichment params get lr_mult=40 to engage (backbone base lr 2.5e-6
# is far too low). BatchFormer deferred to a follow-up.
_base_ = './a0r_refine_pseg.py'

custom_imports = dict(
    imports=['carrier.seal_m2f_head', 'carrier.dsconv_teacher',
             'carrier.enriched_resnet', 'carrier.geofield_m2f_head'],
    allow_failed_imports=False)

model = dict(
    backbone=dict(type='EnrichedResNet', cbam_stages=(0, 1, 2, 3), ds_stages=(1, 2), ds_kernel=9),
    panoptic_head=dict(type='CoordRefineHead', refine_scale=2))

optim_wrapper = dict(paramwise_cfg=dict(
    custom_keys={
        'backbone.cbam': dict(lr_mult=8.0, decay_mult=0.0),   # CBAM + cbam_gamma (longer key fires first)
        'backbone.ds': dict(lr_mult=8.0, decay_mult=0.0),     # DSConv + ds_gamma
        'coord_conv': dict(lr_mult=8.0, decay_mult=0.0),      # CoordConv head
        'backbone': dict(lr_mult=0.1, decay_mult=1.0),         # the pretrained ResNet-50
        'query_embed': dict(lr_mult=1.0, decay_mult=0.0),
        'query_feat': dict(lr_mult=1.0, decay_mult=0.0),
        'level_embed': dict(lr_mult=1.0, decay_mult=0.0),
    }))
