# cs: BACKBONE LEVER on pure c0. R50 M2F (COCO mask AP 42.9) -> Swin-S M2F (COCO 46.1, +3.2).
# Protocol-identical to c0_m2f_cocoinit_hrsid: HRSID official split, 800x800, LSJ 0.5-2.0,
# 36 ep = 100872 iters @ bs2+accum2 (eff bs4), lr 2.5e-5, milestones [89000, 96000], seed 0,
# FULL COCO instance-seg init via load_from (pretrained/m2f_swins_coco.pth, 649 keys verified).
#
# Optimizer merge (critical): we ONLY override optimizer.lr and accumulative_counts.
# The swin-s base's paramwise_cfg is PRESERVED by mmengine dict-merge:
#   custom_keys = backbone lr_mult=0.1/decay 1.0; backbone.patch_embed.norm, backbone.norm,
#   absolute_pos_embed, relative_position_bias_table -> lr_mult=0.1/decay_mult=0.0;
#   query_embed/query_feat/level_embed -> lr_mult=1.0/decay_mult=0.0; per-block norms for
#   depths [2,2,18,2] + 3 downsample norms; norm_decay_mult=0.0. AdamW wd 0.05, clip_grad 0.01.
# backbone.init_cfg=None: load_from supplies ALL weights; kills the ImageNet-URL download
# that the swin base would otherwise attempt at init_weights() (rate-limited network).
#
# VRAM: mmdet metafile @1024^2 bs2: R50 13.7 GB -> Swin-S 18.8 GB (x1.37). Local R50 @800 bs2
# measured ~7-8 GB -> Swin-S @800 bs2 estimate ~10-11 GB. Safe on 24 GB 3090 Ti.
# OOM FALLBACK (same effective batch 4, protocol preserved):
#   train_dataloader = dict(batch_size=1)
#   optim_wrapper = dict(accumulative_counts=4)
_base_ = 'mmdet::mask2former/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco.py'

img_dir = 'D:/SAR/UNet++/HRSID_JPG/HRSID_JPG/JPEGImages1/'
ann = 'E:/SAR/seal/data/hrsid/annotations/'
metainfo = dict(classes=('ship',))
image_size = (800, 800)

num_things_classes = 1
num_stuff_classes = 0

model = dict(
    backbone=dict(init_cfg=None),  # full ckpt comes via load_from
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

# 5604 train imgs * 36 ep / bs2 = 100872 iters; eval every ~4 ep (c0 protocol)
max_iters = 100872
train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters, val_interval=16808)
optim_wrapper = dict(optimizer=dict(lr=2.5e-5), accumulative_counts=2)
param_scheduler = dict(end=max_iters, milestones=[89000, 96000])
default_hooks = dict(
    checkpoint=dict(interval=16808, by_epoch=False, max_keep_ckpts=2,
                    save_best='coco/segm_mAP', rule='greater'),
    logger=dict(interval=100))
log_processor = dict(by_epoch=False)

load_from = 'pretrained/m2f_swins_coco.pth'
randomness = dict(seed=0, deterministic=False)
