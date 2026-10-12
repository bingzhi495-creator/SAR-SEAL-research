# P4 SAR module: SCOPE (shape_scope) on the COCO-init R50 strong baseline (best
# inshore carrier). Adds the dense SAR amplitude to the pipeline (StashDenseAmplitude
# post-aug + PackDetInputsSAR) so GeoField's salience + kernel-axis are driven by the
# measured scatterer-constellation extent. gen_mode='shape_scope' (= shape + SCOPE).
_base_ = './a0r_r50_coco_pseg.py'

custom_imports = dict(
    imports=['carrier.seal_m2f_head', 'carrier.geofield_op', 'carrier.geofield_m2f_head',
             'carrier.transforms_sar'],
    allow_failed_imports=False)

model = dict(panoptic_head=dict(
    type='GeoFieldMask2FormerHead',
    refine_scale=2, gen_mode='shape_scope',
    geo_d=16, geo_r=64, geo_m=8, geo_k=5))

image_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomResize', scale=image_size, ratio_range=(0.5, 2.0),
         resize_type='Resize', keep_ratio=True),
    dict(type='RandomCrop', crop_size=image_size, crop_type='absolute',
         recompute_bbox=True, allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-5, 1e-5), by_mask=True),
    dict(type='StashDenseAmplitude'),                       # post-aug, mask-aligned
    dict(type='PackDetInputsSAR'),
]
test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='Resize', scale=image_size, keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='StashDenseAmplitude'),
    dict(type='PackDetInputsSAR',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))

optim_wrapper = dict(paramwise_cfg=dict(custom_keys=dict(
    geofield=dict(lr_mult=50.0, decay_mult=0.0))))

# val on the diagnosed axis (inshore), matching B1's GATE-0 protocol; resumable ckpts
_ins = 'E:/SAR/seal/data/pseg_ssdd/annotations/test_inshore.json'
val_dataloader = dict(num_workers=0, persistent_workers=False,
                      dataset=dict(ann_file=_ins, pipeline=test_pipeline))
val_evaluator = dict(ann_file=_ins)
default_hooks = dict(checkpoint=dict(interval=1044, by_epoch=False, max_keep_ckpts=2,
                                     save_best='coco/segm_mAP', rule='greater'))
