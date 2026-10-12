# BL baseline 3/3 on PSeg-SSDD: SOLOv2 R50-FPN from the COCO 3x ckpt under OUR
# SSDD protocol. Inherits model (num_classes=1 cls branch, load_from, native
# clip_grad + 1/3 warmup) from bl_solov2_hrsid.py; data + schedule identical
# to bl_maskrcnn_pseg.py (see its header for the SSDD protocol notes).
#  * optimizer: SOLOv2 native SGD 0.01 @ bs16 -> linear scale to bs4: 0.0025
#    (deliberately NOT the R-CNN 0.005 -- each model scales its own default).
#  * evaluator: segm only (SOLOv2 emits dummy zero bboxes).
_base_ = './bl_solov2_hrsid.py'

data_root = 'E:/SAR/seal/data/pseg_ssdd/'
img_train = 'E:/PSeg_SSDD/voc_style/JPEGImages_train/'
img_test = 'E:/PSeg_SSDD/voc_style/JPEGImages_test/'
metainfo = dict(classes=('ship',))
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
    dict(type='PackDetInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='Resize', scale=image_size, keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
]

train_dataloader = dict(
    batch_size=4, num_workers=2, persistent_workers=True,
    dataset=dict(type='CocoDataset', metainfo=metainfo, data_root='',
                 ann_file=data_root + 'annotations/train.json',
                 data_prefix=dict(img=img_train),
                 filter_cfg=dict(filter_empty_gt=True, min_size=8),
                 pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1, num_workers=2, persistent_workers=True,
    dataset=dict(type='CocoDataset', metainfo=metainfo, test_mode=True, data_root='',
                 ann_file=data_root + 'annotations/test.json',
                 data_prefix=dict(img=img_test),
                 pipeline=test_pipeline))
test_dataloader = val_dataloader
val_evaluator = dict(type='CocoMetric', ann_file=data_root + 'annotations/test.json',
                     metric=['segm'], format_only=False)
test_evaluator = val_evaluator

max_iters = 8352
train_cfg = dict(max_iters=max_iters, val_interval=1044)
# SGD 0.01 @ bs16 (SOLOv2 native) -> linear scale to bs4: 0.01 * 4/16 = 0.0025
optim_wrapper = dict(optimizer=dict(lr=0.0025))
param_scheduler = [
    dict(type='LinearLR', start_factor=1.0 / 3, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end=max_iters, by_epoch=False,
         milestones=[7400, 8000], gamma=0.1),
]
default_hooks = dict(
    checkpoint=dict(interval=1044, by_epoch=False, max_keep_ckpts=2,
                    save_best='coco/segm_mAP', rule='greater'),
    logger=dict(interval=50))
