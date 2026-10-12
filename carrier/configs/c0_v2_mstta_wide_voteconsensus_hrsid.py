# Consensus-gated adaptive fusion: soup x wide-TTA x vote(consensus). Denoise (majority) ONLY
# moderate-view-agreement clusters (2 <= size <= vote_max_support = ambiguous/high-variance,
# near-shore-like); keep the sharp SEED mask for HIGH agreement (every view found it = clean
# offshore, whose long boundary majority would erode) and singletons. Fixes area-adaptive's
# offshore erosion (0.560) by using the ensemble's OWN view-agreement, not mask area.
_base_ = './c0_v2_mstta_wide_hrsid.py'
custom_imports = dict(imports=['carrier.maskiou_head', 'carrier.mask_tta'],
                      allow_failed_imports=False)
tta_model = dict(
    type='MaskTTAModel',
    tta_cfg=dict(nms=dict(type='nms', iou_threshold=0.6), max_per_img=100,
                 merge='vote', vote_iou=0.5, vote_mask='consensus',
                 vote_min_support=2, vote_max_support=6))
