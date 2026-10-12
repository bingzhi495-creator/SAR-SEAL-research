# c0_maskiou_v2: RICH-feature Mask-IoU rescoring on COCO-init strong base. Oracle precheck (2026-07-05) showed
# +7.66 AP / +11.82 AP75 inshore rescoring headroom on c0 -> the weak-regime Mask-IoU (+0.002) was a POOR
# IoU-PREDICTOR (query_feat + 2 stats), not a capped axis. v2 adds 11 mask-shape/sharpness stats so IoU is
# predicted better -> captures a fraction of the +7.66 ceiling. NO a1/a2/refine. GATE-0: maskiou_weight=0 +
# infer_iou_lambda=0 byte-identical to c0. Must BEAT v1-plain-maskiou at matched lambda + offshore-neutral.
_base_ = './a0_m2f_r50_hrsid.py'
custom_imports = dict(imports=['carrier.maskiou_head'], allow_failed_imports=False)
model = dict(
    panoptic_head=dict(type='MSIoUM2FHead', refine_scale=1, learn_refine=False,
                       boundary_weight=0.0, maskiou_weight=0.5, maskiou_loss_type='bce', maskiou_v2=True),
    panoptic_fusion_head=dict(type='MSIoUFusionHead', infer_iou_lambda=0.5))
load_from = 'pretrained/m2f_r50_coco.pth'
