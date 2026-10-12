# A0R (resolution lever test): A0 carrier trained at 800^2 instead of 512^2.
# PSeg ships are tiny (median 19px side, 76% small); at 1/4 mask features a
# 19px ship -> ~5px mask. Raising input res to 800 gives small instances more
# mask pixels WITHOUT any architecture change -- the cheapest test of the
# oracle finding that ~40% of the inshore AP75 gap is mask-resolution-bound.
# If inshore AP75 improves here, the per-instance refinement head (real code)
# is justified as the follow-up.
_base_ = './a0_m2f_r50_pseg.py'

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

train_dataloader = dict(batch_size=2, dataset=dict(pipeline=train_pipeline))  # 800^2 -> bs2 to fit 24GB
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

# bs2 (vs bs4): double the iters to keep epochs constant, halve lr
max_iters = 16704
train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters, val_interval=2088)
optim_wrapper = dict(optimizer=dict(lr=1.25e-5), accumulative_counts=2)  # grad-accum -> effective bs4
param_scheduler = dict(end=max_iters, milestones=[14800, 16000])
default_hooks = dict(
    checkpoint=dict(interval=4176, by_epoch=False, max_keep_ckpts=2,
                    save_best='coco/segm_mAP', rule='greater'),
    logger=dict(interval=50))
