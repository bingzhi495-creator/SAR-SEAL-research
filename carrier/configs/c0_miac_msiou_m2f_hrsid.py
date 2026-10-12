# c0+MIAC+MSIoU(v2): COMBINE both mask-quality mechanisms on the COCO-init strong base.
# The COCO-init cls score is IoU-BLIND (rho(cls, true-mask-IoU) = -0.213); an oracle rescore
# by true mask-IoU gives +7.66 AP inshore. Two complementary levers attack this:
#   MIAC  (training source): each matched query's ship-class cls target becomes (1-w)+w*u^gamma
#         (u = pred-mask-vs-GT mask-IoU) -> cls-prob calibrated to localization quality.
#   MSIoU (predictor + inference rescore): a rich 11-stat V2 mask-IoU predictor head is trained
#         (maskiou_weight>0) and used to rescore det = cls*mask_score^(1-lambda)*pred_iou^lambda.
# Values: iou_cls_weight=0.5 (matches c0_miac), maskiou_weight=0.5 + maskiou_v2=True (matches
# c0_maskiou_v2), infer_iou_lambda=0.2 (the v2 sweet spot per the lambda sweep).
# Seal siblings OFF: refine_scale=1, learn_refine=False, boundary_weight=0.0 (see combo_head.py
# reviewer w13gekell: SealM2FHead refine defaults ON and must be explicitly disabled). No a1/a2/cbam.
# GATE-0: iou_cls_weight=0 + maskiou_weight=0 + infer_iou_lambda=0 byte-identical to c0.
_base_ = './c0_m2f_cocoinit_hrsid.py'
custom_imports = dict(imports=['carrier.miac_msiou_head'], allow_failed_imports=False)
model = dict(
    panoptic_head=dict(
        type='MIAC_MSIoUM2FHead',
        iou_cls_weight=0.5, iou_cls_gamma=1.0,
        maskiou_weight=0.5, maskiou_loss_type='bce', maskiou_v2=True,
        refine_scale=1, learn_refine=False, boundary_weight=0.0),
    panoptic_fusion_head=dict(type='MSIoUFusionHead', infer_iou_lambda=0.2))
