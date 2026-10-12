# DINO-4scale-R50 detection baseline on PSeg-SSDD (the seeding gate baseline).
# Detection-only (box AP); the oracle-seeding gate compares SeedDINO vs this.
_base_ = 'mmdet::dino/dino-4scale_r50_8xb2-12e_coco.py'

data_root = 'E:/SAR/seal/data/pseg_ssdd/'
img_train = 'E:/PSeg_SSDD/voc_style/JPEGImages_train/'
img_test = 'E:/PSeg_SSDD/voc_style/JPEGImages_test/'
metainfo = dict(classes=('ship',))
image_size = (512, 512)

model = dict(num_queries=300, bbox_head=dict(num_classes=1))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomResize', scale=image_size, ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=image_size, crop_type='absolute', allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='PackDetInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),  # BEFORE Resize so GT lands in img_shape frame
    dict(type='Resize', scale=image_size, keep_ratio=True),  # scales image AND gt boxes
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
]

train_dataloader = dict(
    batch_size=4, num_workers=2, persistent_workers=True,
    dataset=dict(type='CocoDataset', metainfo=metainfo, data_root='',
                 ann_file=data_root + 'annotations/train.json',
                 data_prefix=dict(img=img_train),
                 filter_cfg=dict(filter_empty_gt=True), pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1, num_workers=2,
    dataset=dict(type='CocoDataset', metainfo=metainfo, test_mode=True, data_root='',
                 ann_file=data_root + 'annotations/test.json',
                 data_prefix=dict(img=img_test), pipeline=test_pipeline))
test_dataloader = val_dataloader
val_evaluator = dict(type='CocoMetric', ann_file=data_root + 'annotations/test.json',
                     metric='bbox', format_only=False)
test_evaluator = val_evaluator

# 928*36/4 = 8352 iters; DINO default is epoch-based, switch to iter for parity
max_iters = 8352
train_cfg = dict(_delete_=True, type='IterBasedTrainLoop', max_iters=max_iters, val_interval=2088)
param_scheduler = [dict(type='MultiStepLR', begin=0, end=max_iters, by_epoch=False,
                        milestones=[7400], gamma=0.1)]
default_hooks = dict(
    checkpoint=dict(interval=2088, by_epoch=False, max_keep_ckpts=1,
                    save_best='coco/bbox_mAP', rule='greater'),
    logger=dict(interval=50))
log_processor = dict(by_epoch=False)
randomness = dict(seed=0)
