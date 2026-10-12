# Our method (GeoField + SealM2FHead refine) on R101 backbone = matched to DiffSAR's R101.
# R101 ImageNet backbone + head/pixel-decoder from COCO (m2f_head_coco). val inshore.
_base_ = './a0r_geo_r50coco_pseg.py'

model = dict(backbone=dict(
    _delete_=True,
    type='ResNet', depth=101, num_stages=4, out_indices=(0, 1, 2, 3),
    frozen_stages=-1, norm_cfg=dict(type='BN', requires_grad=False), norm_eval=True,
    style='pytorch',
    init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101')))

load_from = 'E:/SAR/seal/pretrained/m2f_head_coco.pth'
