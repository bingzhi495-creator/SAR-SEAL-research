# c0_v2 WIDE multi-scale TTA + mask-IoU Matrix-NMS merge (F3).
# CLONE of c0_v2_mstta_wide_hrsid.py: the ONLY change is tta_model.tta_cfg.merge,
# switched from the default 'bbox_nms' (class-wise bbox-IoU NMS) to 'matrix'
# (mask-IoU Matrix-NMS). Rationale: near-shore SAR ships are often two adjacent
# parallel hulls whose BBOXES overlap heavily (high bbox-IoU) but whose MASKS
# barely touch (low mask-IoU). bbox-NMS wrongly deletes the true neighbour;
# matrix-NMS decays by TRUE mask-IoU so both hulls survive while exact/near
# duplicates are decayed to ~0. Inference-side only, zero training.
#
# Geometry / TTA views / infer_iou_lambda sweep are all inherited unchanged from
# the wide base; nms + max_per_img are kept so a caller could still A/B them.
_base_ = './c0_v2_mstta_wide_hrsid.py'

tta_model = dict(
    type='MaskTTAModel',
    tta_cfg=dict(
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100,
        merge='matrix'))
