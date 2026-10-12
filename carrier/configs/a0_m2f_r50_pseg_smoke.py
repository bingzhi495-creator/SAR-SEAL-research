# A0 carrier smoke test: mmdet Mask2Former-R50 instance segmentation on
# PSeg-SSDD. Purpose: validate the full train/val loop on this machine
# (Windows + single 3090 Ti + torch 2.0.1+cu118 + mmcv 2.1.0 + mmdet 3.3.0).
# 300 iters only — NOT a baseline number.
_base_ = 'mmdet::mask2former/mask2former_r50_8xb2-lsj-50e_coco.py'

data_root = 'E:/SAR/seal/data/pseg_ssdd/'
img_train = 'E:/PSeg_SSDD/voc_style/JPEGImages_train/'
img_test = 'E:/PSeg_SSDD/voc_style/JPEGImages_test/'
metainfo = dict(classes=('ship',))

num_things_classes = 1
num_stuff_classes = 0
image_size = (512, 512)  # PSeg-SSDD images are ~400-500 px; 1024 LSJ wastes compute

model = dict(
    data_preprocessor=dict(pad_size_divisor=32, batch_augments=[
        dict(type='BatchFixedSizePad', size=image_size, img_pad_value=0,
             pad_mask=True, mask_pad_value=0, pad_seg=False)
    ]),
    panoptic_head=dict(
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        loss_cls=dict(class_weight=[1.0] * num_things_classes + [0.1])),
    panoptic_fusion_head=dict(
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes))

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
    dataset=dict(
        type='CocoDataset', metainfo=metainfo,
        data_root='', ann_file=data_root + 'annotations/train.json',
        data_prefix=dict(img=img_train),
        filter_cfg=dict(filter_empty_gt=True, min_size=8),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1, num_workers=2, persistent_workers=True,
    dataset=dict(
        type='CocoDataset', metainfo=metainfo, test_mode=True,
        data_root='', ann_file=data_root + 'annotations/test.json',
        data_prefix=dict(img=img_test),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/test.json',
    metric=['bbox', 'segm'], format_only=False)
test_evaluator = val_evaluator

# smoke schedule: 300 iters, eval once at the end
train_cfg = dict(type='IterBasedTrainLoop', max_iters=300, val_interval=300)
optim_wrapper = dict(optimizer=dict(lr=5e-5))
param_scheduler = dict(end=300, milestones=[250])
default_hooks = dict(
    checkpoint=dict(interval=300, by_epoch=False),
    logger=dict(interval=20))
log_processor = dict(by_epoch=False)
