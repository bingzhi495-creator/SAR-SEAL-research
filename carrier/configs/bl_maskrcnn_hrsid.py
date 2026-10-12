# BL baseline 1/3 on HRSID: Mask R-CNN R50-FPN, fine-tuned from the mmdet COCO
# 3x checkpoint (pretrained/maskrcnn_r50_3x_coco.pth, recipe
# mask-rcnn_r50_fpn_ms-poly-3x_coco) under OUR protocol so the comparison is
# same-data / same-iters / same-eval as the a0 M2F anchor
# (carrier/configs/a0_m2f_r50_hrsid.py).
#
# Protocol match / documented deviations vs a0_m2f_r50_hrsid.py:
#  * DATA + LSJ GEOMETRY IDENTICAL: same ann files, same img dir, same
#    train_pipeline (RandomFlip 0.5 -> RandomResize 800 ratio 0.5-2.0
#    keep_ratio -> RandomCrop absolute 800 -> FilterAnnotations by_mask) and
#    same test Resize 800 keep_ratio. This LSJ geometry is shared across ALL
#    methods in the paper, including ours.
#  * data_preprocessor: BatchFixedSizePad is an M2F-specific batch_augment
#    (fixed-size batching for its mask attention); Mask R-CNN uses its NATIVE
#    DetDataPreprocessor with pad_size_divisor=32 (same mean/std/bgr_to_rgb as
#    M2F). Padding only, image content identical.
#  * optimizer: NATIVE family, SGD 0.02 momentum 0.9 wd 1e-4 @ bs16 (mmdet
#    default) linearly scaled to bs2 -> lr 0.0025. Native warmup kept
#    (LinearLR start_factor 0.001, 500 iters).
#  * schedule: IterBasedTrainLoop 100872 iters @ bs2 (= a0 anchor;
#    ~55.4 epochs over the 3642 official_train images), val_interval 16808,
#    step LR x0.1 at [89000, 96000] = same relative milestones as ours.
#  * heads: num_classes=1 ('ship'); the 80-class cls/reg/mask-logit branches
#    are re-initialized (expected 80->1 shape mismatch at load_from time,
#    everything else loads 1:1).
#  * backbone frozen_stages=1 + norm_eval=True kept native (standard COCO
#    fine-tune convention for R-CNNs).
#  * sampler: mmdet-base DefaultSampler + AspectRatioBatchSampler kept; under
#    IterBasedTrainLoop mmengine wraps the loader with an infinite iterator
#    (same machinery our M2F runs use).
_base_ = 'mmdet::mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py'

img_dir = 'D:/SAR/UNet++/HRSID_JPG/HRSID_JPG/JPEGImages1/'
ann = 'E:/SAR/seal/data/hrsid/annotations/'
metainfo = dict(classes=('ship',))
image_size = (800, 800)

load_from = 'E:/SAR/seal/pretrained/maskrcnn_r50_3x_coco.pth'

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1),
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

# 100872 iters @ bs2 == the a0 HRSID anchor budget (identical wall-schedule).
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
