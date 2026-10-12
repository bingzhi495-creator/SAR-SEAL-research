# F4 soft-probability fusion: soup x wide-TTA x vote(prob). Cluster EXACTLY like 'consensus'
# (greedy BINARY mask-IoU >= 0.5, band gate 2 <= size <= 6), but fuse each cluster by averaging
# the members' FLOAT sigmoid maps and thresholding at 0.5 -- sub-pixel-correct: the boundary moves
# toward the ensemble MAP estimate instead of being voted OUT by {0,1} masks from misaligned views
# (binary majority eroded offshore to 0.559). Needs model.test_cfg.export_mask_probs=True so
# MSIoUFusionHead attaches pred_instances.mask_probs (fp16, ori-res; maskformer.py copies
# model.test_cfg into the fusion head). prob_high_agreement: 'mean' (default) also prob-fuses
# high-agreement clusters (size > 6) where 'consensus' kept the seed; sweep 'seed' to mirror it.
_base_ = './c0_v2_mstta_wide_hrsid.py'
custom_imports = dict(imports=['carrier.maskiou_head', 'carrier.mask_tta'],
                      allow_failed_imports=False)
model = dict(test_cfg=dict(export_mask_probs=True))
tta_model = dict(
    type='MaskTTAModel',
    tta_cfg=dict(nms=dict(type='nms', iou_threshold=0.6), max_per_img=100,
                 merge='vote', vote_iou=0.5, vote_mask='prob',
                 vote_min_support=2, vote_max_support=6,
                 prob_thr=0.5, prob_high_agreement='mean'))
