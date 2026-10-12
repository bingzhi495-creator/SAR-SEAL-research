# c0+maskiou: Mask-IoU rescoring head on the COCO-init strong base. Targets the B-score residual which GREW
# 6%->16% under COCO init (strong init makes MORE well-localized-but-underscored masks; mmdet fusion scores by
# mean-fg-sigmoid NOT localization quality). STRONGEST transfer argument in the set -- it fixes a mismatch that
# COCO init CREATED, not a weak-regime blindness a1/a2 targeted (which COCO closed). NO a1/a2/refine/boundary.
# GATE-0: maskiou_weight=0 + infer_iou_lambda=0 byte-identical to c0.
_base_ = './a0_m2f_r50_hrsid.py'
custom_imports = dict(imports=['carrier.maskiou_head'], allow_failed_imports=False)
model = dict(
    panoptic_head=dict(type='MSIoUM2FHead', refine_scale=1, learn_refine=False,
                       boundary_weight=0.0, maskiou_weight=0.5, maskiou_loss_type='bce'),
    panoptic_fusion_head=dict(type='MSIoUFusionHead', infer_iou_lambda=0.5))
load_from = 'pretrained/m2f_r50_coco.pth'
