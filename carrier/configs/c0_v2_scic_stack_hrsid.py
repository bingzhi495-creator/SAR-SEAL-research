# SCIC stack-dump config: the FINAL method (soup x wide-TTA x vote-consensus
# ms6) with two dump-only knobs for the Self-Corrective Inference Cascade:
#   * max_per_img 300 (vs 100): keep the low-score reject pool in the dump so
#     the R10 miss-rescue verifier / R11 re-ranker can harvest candidates in
#     [0.01, 0.35]. The top-100-by-score instances are unchanged (pure tail
#     extension of the same argsort).
#   * export_vote_stats=True: MaskTTAModel attaches per-instance consensus
#     features (support_count, n_views_present, mean/min pairwise mask-IoU,
#     seed_score, score_spread) as pred_instances.vote_stats (C, 6). GATE-0:
#     with the flag off outputs are byte-identical (carrier/tests/
#     cpu_test_scic.py).
# Score floor: NONE at dump time (the vote merge has no floor); the 0.01 floor
# is applied CPU-side at harvest/apply. vote_max_support=6 is baked in to match
# the BPR/final-stack setting (run_bpr_queue.sh passed it via --cfg-options).
_base_ = './c0_v2_mstta_wide_voteconsensus_hrsid.py'
custom_imports = dict(imports=['carrier.maskiou_head', 'carrier.mask_tta'],
                      allow_failed_imports=False)
tta_model = dict(
    type='MaskTTAModel',
    tta_cfg=dict(nms=dict(type='nms', iou_threshold=0.6), max_per_img=300,
                 merge='vote', vote_iou=0.5, vote_mask='consensus',
                 vote_min_support=2, vote_max_support=6,
                 export_vote_stats=True))
