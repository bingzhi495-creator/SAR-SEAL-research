# A0 baseline on HRSID (the real target). 800x800 native chips, official split.
# Mirrors the PSeg A0 config; this is the anchor row for the HRSID ladder (L2).
# ~5604 imgs are bigger than PSeg; 36 epochs on 1x 3090 Ti ~ 11-13h -> watchdog.
_base_ = './a0_m2f_r50_pseg.py'

img_dir = 'D:/SAR/UNet++/HRSID_JPG/HRSID_JPG/JPEGImages1/'
ann = 'E:/SAR/seal/data/hrsid/annotations/'
metainfo = dict(classes=('ship',))
image_size = (800, 800)

model = dict(
    data_preprocessor=dict(batch_augments=[
        dict(type='BatchFixedSizePad', size=image_size, img_pad_value=0,
             pad_mask=True, mask_pad_value=0, pad_seg=False)
    ]))

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

# 5604*36/2 = 100872 iters; eval every ~4 epochs
# bs2 + accum2 = effective bs4 (lr 2.5e-5), VRAM-safe ~7GB (bs4 OOM-crashed on
# dense HRSID images). 5604*36/2 = 100872 iters (~28h, safe overnight anchor).
max_iters = 100872
train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters, val_interval=16808)
optim_wrapper = dict(optimizer=dict(lr=2.5e-5), accumulative_counts=2)
param_scheduler = dict(end=max_iters, milestones=[89000, 96000])
default_hooks = dict(
    checkpoint=dict(interval=16808, by_epoch=False, max_keep_ckpts=2,
                    save_best='coco/segm_mAP', rule='greater'),
    logger=dict(interval=100))
