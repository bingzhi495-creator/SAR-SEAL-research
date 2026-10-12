# R12 stack-dump config: the SCIC stack dump (max_per_img 300 +
# export_vote_stats, see ./c0_v2_scic_stack_hrsid.py) PLUS depth-consistency
# export:
#   * model.test_cfg.export_depth_stats=True: MSIoUM2FHead computes per-query
#     stats over the LAST 3 decoder layers (L7/L8/L9) -- consecutive-layer
#     binarized mask IoU at native low res + per-layer max fg cls softmax --
#     and MSIoUFusionHead attaches them per KEPT instance via the
#     [query_indices][is_thing] chain (results.depth_stats, (K, 5)).
#   * MaskTTAModel vote fusion then folds the SEED member's depth stats into
#     vote_stats -> (C, 8) = 6 consensus + [seed_depth_iou_mean,
#     seed_score_traj]. Both flags on is the ONLY combination that widens the
#     export; either off -> byte-identical legacy outputs (GATE-0,
#     carrier/tests/cpu_test_r12.py).
# Consumed by run_r12_queue.sh; R11 reranker autodetects the 16-dim features.
_base_ = './c0_v2_scic_stack_hrsid.py'
custom_imports = dict(imports=['carrier.maskiou_head', 'carrier.mask_tta'],
                      allow_failed_imports=False)
model = dict(test_cfg=dict(export_depth_stats=True))
