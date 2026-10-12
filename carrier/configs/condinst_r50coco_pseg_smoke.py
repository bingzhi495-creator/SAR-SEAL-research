# CondInst-R50 carrier SMOKE on PSeg-SSDD: validate full train/val loop + clean COCO-init
# transfer. 300 iters only (NOT a baseline). CondInst = native pure dynamic-kernel readout host
# for our DS-3x3 (no rank-1 to fight; D1/D2 dilution avoided). COCO-init = pretrained/condinst_r50_coco.pth.
_base_ = 'mmdet::condinst/condinst_r50_fpn_ms-poly-90k_coco_instance.py'

data_root = 'E:/SAR/seal/data/pseg_ssdd/'
img_train = 'E:/PSeg_SSDD/voc_style/JPEGImages_train/'
img_test = 'E:/PSeg_SSDD/voc_style/JPEGImages_test/'
metainfo = dict(classes=('ship',))
image_size = (512, 512)

load_from = 'E:/SAR/seal/pretrained/condinst_r50_coco.pth'

model = dict(bbox_head=dict(num_classes=1))

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
        type='CocoDataset', metainfo=metainfo, data_root='',
        ann_file=data_root + 'annotations/train.json', data_prefix=dict(img=img_train),
        filter_cfg=dict(filter_empty_gt=True, min_size=8), pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1, num_workers=2, persistent_workers=True,
    dataset=dict(
        type='CocoDataset', metainfo=metainfo, test_mode=True, data_root='',
        ann_file=data_root + 'annotations/test.json', data_prefix=dict(img=img_test),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric', ann_file=data_root + 'annotations/test.json',
    metric=['bbox', 'segm'], format_only=False)
test_evaluator = val_evaluator

# smoke schedule: 300 iters, eval once. lr 0.01@bs16 -> 2.5e-3@bs4 (CondInst is lr-sensitive).
train_cfg = dict(_delete_=True, type='IterBasedTrainLoop', max_iters=300, val_interval=300)
optim_wrapper = dict(optimizer=dict(lr=2.5e-3))
param_scheduler = [
    dict(type='LinearLR', start_factor=0.01, by_epoch=False, begin=0, end=100),
    dict(type='MultiStepLR', by_epoch=False, milestones=[250], gamma=0.1),
]
default_hooks = dict(
    checkpoint=dict(interval=300, by_epoch=False),
    logger=dict(interval=20))
log_processor = dict(by_epoch=False)
