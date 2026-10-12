# c0_v2 WIDE multi-scale TTA + mask-CONSENSUS-VOTING merge (F2).
# CLONE of c0_v2_mstta_wide_hrsid.py: the ONLY change is tta_model.tta_cfg.merge,
# switched from the default 'bbox_nms' (class-wise bbox-IoU NMS) to 'vote'
# (consensus voting). Rationale: for TTA merge each object appears as ~N
# near-identical masks across the N augmented views. Score-decay (merge='matrix')
# leaves those duplicates alive at moderate scores -> precision collapse; bbox-NMS
# hard-deletes a low-mask-IoU parallel-hull neighbour. Voting clusters the N
# views that belong to ONE object by TRUE mask-IoU (within a class), emits ONE
# majority-vote fused mask per cluster, and keeps adjacent hulls as separate
# clusters. Inference-side only, zero training.
#
# Knobs (defaults shown): vote_iou=0.5 (cluster join threshold on true mask-IoU),
# vote_frac=0.5 (per-pixel majority for the fused mask), vote_gamma=0.0 (support
# weighting exponent; 0.0 => fused score is the plain per-cluster max).
# Geometry / TTA views / infer_iou_lambda sweep are inherited unchanged from the
# wide base; nms + max_per_img are kept so a caller could still A/B them.
_base_ = './c0_v2_mstta_wide_hrsid.py'

tta_model = dict(
    type='MaskTTAModel',
    tta_cfg=dict(
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100,
        merge='vote',
        vote_iou=0.5,
        vote_frac=0.5,
        vote_gamma=0.0))
