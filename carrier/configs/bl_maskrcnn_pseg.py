# BL baseline 1/3 on PSeg-SSDD: Mask R-CNN R50-FPN from the COCO 3x ckpt under
# OUR SSDD protocol (mirrors a0_m2f_r50_pseg.py / a0_m2f_r50_pseg_smoke.py).
# Inherits model/heads/load_from from bl_maskrcnn_hrsid.py; only data + schedule
# change. Protocol notes (see bl_maskrcnn_hrsid.py for the shared ones):
#  * LSJ at 512^2 (PSeg images ~190-670 px), same geometry family as ours.
#  * NOTE the SSDD image dirs DIFFER between train and test:
#    JPEGImages_train/ vs JPEGImages_test/ (val/test loaders point at _test).
#  * 928 train imgs * 36 epochs / bs4 = 8352 iters, val_interval 1044,
#    step LR x0.1 at [7400, 8000] (same relative milestones as ours).
#  * optimizer: native SGD 0.02 @ bs16 -> linear scale to bs4: 0.005.
_base_ = './bl_maskrcnn_hrsid.py'

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
                     metric=['bbox', 'segm'], format_only=False)
test_evaluator = val_evaluator

# 928 * 36 / 4 = 8352 iters (bs4), same budget as the a0 SSDD anchor.
max_iters = 8352
train_cfg = dict(max_iters=max_iters, val_interval=1044)
# SGD 0.02 @ bs16 (native) -> linear scale to bs4: 0.02 * 4/16 = 0.005
optim_wrapper = dict(optimizer=dict(lr=0.005))
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end=max_iters, by_epoch=False,
         milestones=[7400, 8000], gamma=0.1),
]
default_hooks = dict(
    checkpoint=dict(interval=1044, by_epoch=False, max_keep_ckpts=2,
                    save_best='coco/segm_mAP', rule='greater'),
    logger=dict(interval=50))
