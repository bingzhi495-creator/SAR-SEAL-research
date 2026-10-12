# -*- coding: utf-8 -*-
# P3 precheck ONLY: Swin-L Mask2Former COCO-panoptic teacher, CPU inference.
# Overrides the mmdet swin-l coco-panoptic config to (a) disable the backbone
# pretrained-URL init (we load the FULL m2f_swinl_cocopan.pth checkpoint via
# init_detector, so no download is needed -- avoids a network fetch on this
# rate-limited box) and (b) use a single-scale test pipeline sized to the HRSID
# 800x800 chips. Instance masks are already on (test_cfg.instance_on=True in base).
_base_ = ['mmdet::mask2former/mask2former_swin-l-p4-w12-384-in21k_16xb1-lsj-100e_coco-panoptic.py']

# kill the URL backbone init; the full teacher ckpt supplies every weight
model = dict(backbone=dict(init_cfg=None), init_cfg=None)

# single-scale test at 800 (HRSID chip size); keep_ratio so 800x800 stays 800x800
test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='Resize', scale=(800, 800), keep_ratio=True),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
]

# init_detector builds test_dataloader.dataset just to read metainfo; the base
# coco-PANOPTIC dataset pulls in panopticapi (not installed). Swap it for a plain
# CocoDataset with our simple pipeline so no panopticapi is needed. Inference itself
# uses inference_detector (image path in), so the dataset ann_file is irrelevant.
test_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type='CocoDataset',
        metainfo=dict(classes=('ship',)),
        data_root='',
        ann_file='data/hrsid/annotations/test_inshore.json',
        data_prefix=dict(img='D:/SAR/UNet++/HRSID_JPG/HRSID_JPG/JPEGImages1/'),
        test_mode=True,
        pipeline=test_pipeline))
