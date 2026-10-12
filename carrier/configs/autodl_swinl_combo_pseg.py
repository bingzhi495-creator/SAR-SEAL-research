# AutoDL Swin-L COMBO: strongest carrier (Swin-L COCO-panoptic) + GeoField (shape mask
# enrichment) + boundary loss — all mechanisms on the strong carrier. Inherits the AutoDL
# Swin-L config (backbone/batch/lr/data/load_from); only swaps the head + adds geofield lr.
_base_ = './autodl_swinl_pseg.py'

custom_imports = dict(
    imports=['carrier.seal_m2f_head', 'carrier.geofield_op', 'carrier.geofield_m2f_head'],
    allow_failed_imports=False)

model = dict(panoptic_head=dict(
    type='GeoFieldMask2FormerHead',
    refine_scale=2, gen_mode='shape',
    geo_d=16, geo_r=64, geo_m=8, geo_k=5,
    boundary_weight=0.2, boundary_ksize=3))      # GeoField + boundary loss

# add geofield lr_mult=50 (merges with the inherited Swin custom_keys)
optim_wrapper = dict(paramwise_cfg=dict(custom_keys=dict(
    geofield=dict(lr_mult=50.0, decay_mult=0.0))))
