# M3 ShapePrior -- GEO arm + SAR loss (Loss A intensity-weighted BCE + Loss B1
# boundary-band-weighted Dice). The cell we most want to win (mechanism x loss).
# Loss A up-weights DARK in-GT pixels (targets the dropped stern/bow); Loss B1 is
# a boundary-band Dice for high-IoU/AP75. Reuses the stock DiceLoss (band-masked).
_base_ = './a0r_refine_pseg.py'

custom_imports = dict(
    imports=['carrier.seal_m2f_head', 'carrier.geokernel_m2f_head',
             'carrier.transforms_sar'],
    allow_failed_imports=False)

model = dict(panoptic_head=dict(
    type='ShapePriorMask2FormerHead',
    refine_scale=2,
    gen_mode='geo',
    prior_hidden=32,
    gate_bias_init=-4.0,
    k0=4.0,
    a_min=1.0,
    # ---- SAR loss add-on (default off elsewhere; ON here) ----
    sar_loss_a_w=1.0,        # Loss A enabled
    sar_loss_b1_w=2.0,       # Loss B1 band-Dice weight
    sar_lam_int=1.0,
    sar_gamma_int=1.0,
    sar_band_k=3))

optim_wrapper = dict(paramwise_cfg=dict(
    custom_keys={
        'backbone': dict(lr_mult=0.1, decay_mult=1.0),
        'geom_mlp': dict(lr_mult=1.0, decay_mult=0.0),
        'query_embed': dict(lr_mult=1.0, decay_mult=0.0),
        'query_feat': dict(lr_mult=1.0, decay_mult=0.0),
        'level_embed': dict(lr_mult=1.0, decay_mult=0.0),
    }))

# ---- train pipeline: stash per-instance SAR amplitude for Loss A, pack via the
# SAR-aware packer so it rides into gt_instances.amp_maps. (Loss A falls back to
# neutral amplitude if amp_maps is absent, so this is the path that makes Loss A
# use the real SAR intensity.) ----
image_size = (512, 512)
data_root = 'E:/SAR/seal/data/pseg_ssdd/'
img_train = 'E:/PSeg_SSDD/voc_style/JPEGImages_train/'
metainfo = dict(classes=('ship',))

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomResize', scale=image_size, ratio_range=(0.5, 2.0),
         resize_type='Resize', keep_ratio=True),
    dict(type='RandomCrop', crop_size=image_size, crop_type='absolute',
         recompute_bbox=True, allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-5, 1e-5), by_mask=True),
    dict(type='StashLowResAmplitude'),
    dict(type='PackDetInputsSAR'),
]

train_dataloader = dict(
    dataset=dict(
        type='CocoDataset', metainfo=metainfo,
        data_root='', ann_file=data_root + 'annotations/train.json',
        data_prefix=dict(img=img_train),
        filter_cfg=dict(filter_empty_gt=True, min_size=8),
        pipeline=train_pipeline))
