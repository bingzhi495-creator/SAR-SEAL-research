# SAR-DKR D3 (full SAR-aware): dynamic-kernel + relative coords with SCATTERER-weighted centre
# (SAR amplitude -> anchored to bright superstructure, speckle-stable) + ORIENTATION rotation to
# the hull principal axis. Gated (init 0) -> floor = A0R. Needs the SAR amp pipeline.
_base_ = './a0r_r50_coco_pseg.py'

custom_imports = dict(
    imports=['carrier.seal_m2f_head', 'carrier.dkr_op', 'carrier.geofield_m2f_head',
             'carrier.transforms_sar'],
    allow_failed_imports=False)

model = dict(panoptic_head=dict(
    type='DKRMask2FormerHead', refine_scale=2, dkr_channels=8,
    dkr_relcoord=True, dkr_scatter=True, dkr_orient=True))

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
    dict(type='StashDenseAmplitude'),
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

_ins = 'E:/SAR/seal/data/pseg_ssdd/annotations/test_inshore.json'
val_dataloader = dict(num_workers=0, persistent_workers=False,
                      dataset=dict(ann_file=_ins, pipeline=test_pipeline))
val_evaluator = dict(ann_file=_ins)
default_hooks = dict(checkpoint=dict(interval=1044, by_epoch=False, max_keep_ckpts=2,
                                     save_best='coco/segm_mAP', rule='greater'))
