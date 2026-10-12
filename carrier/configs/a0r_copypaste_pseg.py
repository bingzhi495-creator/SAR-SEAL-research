# A0R + Simple Copy-Paste augmentation. The biggest DATA-side lever for a small
# (928-img) dataset: synthetic multi-ship images add real positional/scale signal.
# MultiImageMixDataset wraps a load-only CocoDataset; the LSJ block + Pad(512) live
# in the INNER load_pipeline so dest+source are both exactly 512x512 before CopyPaste.
_base_ = './a0r_refine_pseg.py'

image_size = (512, 512)
data_root = 'E:/SAR/seal/data/pseg_ssdd/'
img_train = 'E:/PSeg_SSDD/voc_style/JPEGImages_train/'
metainfo = dict(classes=('ship',))

# inner pipeline: load + LSJ + flip + pad-to-512 -> every sample is exactly 512x512
load_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomResize', scale=image_size, ratio_range=(0.5, 2.0),
         resize_type='Resize', keep_ratio=True),
    dict(type='RandomCrop', crop_size=image_size, crop_type='absolute',
         recompute_bbox=True, allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-5, 1e-5), by_mask=True),
    dict(type='Pad', size=image_size),
]
# mix pipeline: CopyPaste pulls extra 512x512 samples from the inner dataset
train_pipeline = [
    dict(type='CopyPaste', max_num_pasted=20, selected=True),
    dict(type='PackDetInputs'),
]

train_dataloader = dict(dataset=dict(
    _delete_=True,
    type='MultiImageMixDataset',
    dataset=dict(
        type='CocoDataset', metainfo=metainfo, data_root='',
        ann_file=data_root + 'annotations/train.json',
        data_prefix=dict(img=img_train),
        filter_cfg=dict(filter_empty_gt=True, min_size=8),
        pipeline=load_pipeline),
    pipeline=train_pipeline))
