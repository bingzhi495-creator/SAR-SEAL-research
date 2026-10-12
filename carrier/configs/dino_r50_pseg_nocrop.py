# DINO baseline with the no-crop pipeline (fair comparison vs SeedDINO-pred,
# which drops RandomCrop so seed boxes only need flip+resize alignment).
# RandomCrop is ~no-op on ~450px PSeg images at 512 crop, so this ≈ dino_r50_pseg.
_base_ = './dino_r50_pseg.py'

image_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Resize', scale=image_size, keep_ratio=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='PackDetInputs'),
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
