# P0-4 train-side hyper-parameter re-scan (run_post_queue.sh step 1).
# CLONE of the final HRSID stack cfg (soup x wide 8-view msTTA x consensus vote);
# the ONLY change is the test annotation: a fixed-seed 20% image holdout of the
# OFFICIAL TRAINING split (data/hrsid/annotations/train_holdout20.json, written by
# tools/make_train_holdout.py --frac 0.2 --seed 0). These images were seen by the
# soup during training; the re-scan asks whether the operator's knobs chosen on
# the test partitions (S_max=6, tau=0.5) are also the training-side optimum.
# Knobs are swept via --cfg-options tta_model.tta_cfg.vote_max_support (S_max)
# and tta_model.tta_cfg.vote_iou (tau, the mask-IoU clustering threshold of
# Eq. (4)); infer_iou_lambda=0.0 as in the final stack.
_base_ = './c0_v2_mstta_wide_voteconsensus_hrsid.py'

_holdout = 'E:/SAR/seal/data/hrsid/annotations/train_holdout20.json'
test_dataloader = dict(num_workers=0, persistent_workers=False,
                       dataset=dict(ann_file=_holdout))
test_evaluator = dict(ann_file=_holdout)
