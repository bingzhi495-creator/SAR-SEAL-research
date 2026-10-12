# BL baseline 3/3 on HRSID: SOLOv2 R50-FPN, fine-tuned from the mmdet COCO 3x
# checkpoint (pretrained/solov2_r50_3x_coco.pth, recipe
# solov2_r50_fpn_ms-3x_coco) under OUR protocol.
# See bl_maskrcnn_hrsid.py header for the full protocol-match notes; identical
# here except the model. SOLOv2-specific notes:
#  * NATIVE optimizer default is SGD lr 0.01 @ bs16 (NOT the R-CNN 0.02) with
#    clip_grad max_norm=35 -> linear scale to bs2 gives lr 0.00125 (clip_grad
#    inherited from the base). DEVIATION from the R-CNN rows' 0.0025 is
#    deliberate: each model scales from ITS OWN mmdet default.
#  * NATIVE warmup of the 3x recipe kept: LinearLR start_factor=1/3, 500 iters
#    (the R-CNNs use 0.001; each model keeps its own default warmup).
#  * mask_head.num_classes=1: only conv_cls (the cls branch output conv,
#    80->1) is re-initialized; kernel/feature branches are class-agnostic and
#    load 1:1.
#  * scale_ranges / num_grids kept native (COCO-tuned); documented as
#    untouched rather than re-tuned for SAR.
#  * evaluator metric=['segm'] only: SOLOv2 emits dummy zero bboxes, so bbox
#    AP is meaningless (mmdet's own solov2 configs also eval segm only).
_base_ = 'mmdet::solov2/solov2_r50_fpn_1x_coco.py'

img_dir = 'D:/SAR/UNet++/HRSID_JPG/HRSID_JPG/JPEGImages1/'
ann = 'E:/SAR/seal/data/hrsid/annotations/'
metainfo = dict(classes=('ship',))
image_size = (800, 800)

load_from = 'E:/SAR/seal/pretrained/solov2_r50_3x_coco.pth'

model = dict(mask_head=dict(num_classes=1))

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
                     metric=['segm'], format_only=False)
test_evaluator = val_evaluator

max_iters = 100872
train_cfg = dict(_delete_=True, type='IterBasedTrainLoop',
                 max_iters=max_iters, val_interval=16808)
# SGD 0.01 @ bs16 (SOLOv2 native) -> linear scale to bs2: 0.01 * 2/16 = 0.00125
optim_wrapper = dict(optimizer=dict(lr=0.00125))
param_scheduler = [
    dict(type='LinearLR', start_factor=1.0 / 3, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end=max_iters, by_epoch=False,
         milestones=[89000, 96000], gamma=0.1),
]
default_hooks = dict(
    checkpoint=dict(interval=4000, by_epoch=False, max_keep_ckpts=2,
                    save_best='coco/segm_mAP', rule='greater'),
    logger=dict(interval=100))
log_processor = dict(by_epoch=False)
randomness = dict(seed=0, deterministic=False)
