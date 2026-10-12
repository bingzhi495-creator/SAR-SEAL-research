# BL baseline 2/3 on HRSID: Cascade Mask R-CNN R50-FPN, fine-tuned from the
# mmdet COCO 3x checkpoint (pretrained/cascade_maskrcnn_r50_3x_coco.pth, recipe
# cascade-mask-rcnn_r50_fpn_ms-3x_coco) under OUR protocol.
# See bl_maskrcnn_hrsid.py header for the full protocol-match notes; identical
# here except the model. Cascade-specific notes:
#  * bbox_head is a LIST (3 stages) -> mmengine replaces lists wholesale, so
#    the three stage dicts are copied VERBATIM from
#    mmdet configs/_base_/models/cascade-mask-rcnn_r50_fpn.py with ONLY
#    num_classes changed 80 -> 1.
#  * reg_class_agnostic=True in all 3 stages (native) -> fc_reg loads 1:1 from
#    COCO; only the 3x fc_cls (81->2) and 3x mask conv_logits (80->1) are
#    re-initialized.
#  * optimizer: native SGD 0.02 @ bs16 -> 0.0025 @ bs2, warmup LinearLR 0.001
#    x 500 iters (native).
_base_ = 'mmdet::cascade_rcnn/cascade-mask-rcnn_r50_fpn_1x_coco.py'

img_dir = 'D:/SAR/UNet++/HRSID_JPG/HRSID_JPG/JPEGImages1/'
ann = 'E:/SAR/seal/data/hrsid/annotations/'
metainfo = dict(classes=('ship',))
image_size = (800, 800)

load_from = 'E:/SAR/seal/pretrained/cascade_maskrcnn_r50_3x_coco.pth'

model = dict(
    roi_head=dict(
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0))
        ],
        mask_head=dict(num_classes=1)))

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
    batch_size=2, num_workers=2, persistent_workers=True,
    dataset=dict(type='CocoDataset', metainfo=metainfo, data_root='',
                 ann_file=ann + 'official_train.json', data_prefix=dict(img=img_dir),
                 filter_cfg=dict(filter_empty_gt=True, min_size=8),
                 pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1, num_workers=2, persistent_workers=True,
    dataset=dict(type='CocoDataset', metainfo=metainfo, test_mode=True, data_root='',
                 ann_file=ann + 'official_test.json', data_prefix=dict(img=img_dir),
                 pipeline=test_pipeline))
test_dataloader = val_dataloader
val_evaluator = dict(type='CocoMetric', ann_file=ann + 'official_test.json',
                     metric=['bbox', 'segm'], format_only=False)
test_evaluator = val_evaluator

max_iters = 100872
train_cfg = dict(_delete_=True, type='IterBasedTrainLoop',
                 max_iters=max_iters, val_interval=16808)
# SGD 0.02 @ bs16 (native) -> linear scale to bs2: 0.02 * 2/16 = 0.0025
optim_wrapper = dict(optimizer=dict(lr=0.0025))
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end=max_iters, by_epoch=False,
         milestones=[89000, 96000], gamma=0.1),
]
default_hooks = dict(
    checkpoint=dict(interval=4000, by_epoch=False, max_keep_ckpts=2,
                    save_best='coco/segm_mAP', rule='greater'),
    logger=dict(interval=100))
log_processor = dict(by_epoch=False)
randomness = dict(seed=0, deterministic=False)
