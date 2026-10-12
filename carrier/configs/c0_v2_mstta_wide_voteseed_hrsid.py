# F2b: consensus-vote clustering (keeps low-mask-IoU parallel-hull neighbours as separate
# clusters) but the fused mask = the top-score member's mask VERBATIM (vote_mask='seed'),
# NOT a pixelwise majority vote. Rationale: plain majority voting eroded offshore boundaries
# (soup x wide x vote(majority) inshore 0.391 but offshore 0.559 vs bbox_nms 0.642). 'seed'
# keeps the best single-view mask intact -> should preserve offshore while retaining the
# inshore neighbour-recovery from mask-IoU clustering.
_base_ = './c0_v2_mstta_wide_hrsid.py'

custom_imports = dict(
    imports=['carrier.maskiou_head', 'carrier.mask_tta'],
    allow_failed_imports=False)

tta_model = dict(
    type='MaskTTAModel',
    tta_cfg=dict(
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100,
        merge='vote',
        vote_iou=0.5,
        vote_mask='seed'))
