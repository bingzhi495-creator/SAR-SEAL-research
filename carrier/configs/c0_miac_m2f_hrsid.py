# c0+MIAC: Mask-IoU-Aware Classification soft-target on the COCO-init strong base.
# The COCO-init cls score is IoU-BLIND (measured rho(cls, true-mask-IoU) = -0.213); an
# oracle rescore by true mask-IoU gives +7.66 AP inshore. MIAC folds the mask-IoU into the
# CLASSIFICATION TARGET during training (Align-DETR IA-BCE / QFL lineage): each matched
# query's ship-class target becomes (1-w) + w*u^gamma (u = pred-mask-vs-GT mask-IoU),
# calibrating cls-prob to localization quality at the source (vs MSIoU which rescored at
# inference). NO a1/a2/refine/boundary. GATE-0: iou_cls_weight=0 byte-identical to c0.
_base_ = './a0_m2f_r50_hrsid.py'
custom_imports = dict(imports=['carrier.miac_head'], allow_failed_imports=False)
model = dict(
    panoptic_head=dict(type='MIACM2FHead', iou_cls_weight=0.5, iou_cls_gamma=1.0))
load_from = 'pretrained/m2f_r50_coco.pth'
