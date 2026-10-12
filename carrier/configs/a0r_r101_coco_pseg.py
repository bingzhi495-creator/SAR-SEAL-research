# Same-backbone control vs DiffSAR (R101): is the gap the backbone or the mask head?
# R101 backbone (ImageNet) + head/pixel-decoder from COCO Mask2Former (R50/R101 share output
# channels -> partial COCO-init, the part that mattered). Expect ~61.x (Swin-L already showed
# backbone capacity is saturated) -> confirms the gap is the mask head, not the backbone.
_base_ = './a0r_r50_coco_pseg.py'

model = dict(backbone=dict(
    _delete_=True,
    type='ResNet', depth=101, num_stages=4, out_indices=(0, 1, 2, 3),
    frozen_stages=-1, norm_cfg=dict(type='BN', requires_grad=False), norm_eval=True,
    style='pytorch',
    init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101')))

# head + pixel decoder from COCO (backbone stripped); backbone uses the init_cfg above.
load_from = 'E:/SAR/seal/pretrained/m2f_head_coco.pth'

_ins = 'E:/SAR/seal/data/pseg_ssdd/annotations/test_inshore.json'
val_dataloader = dict(num_workers=0, persistent_workers=False, dataset=dict(ann_file=_ins))
val_evaluator = dict(ann_file=_ins)
default_hooks = dict(checkpoint=dict(interval=1044, by_epoch=False, max_keep_ckpts=2,
                                     save_best='coco/segm_mAP', rule='greater'))
