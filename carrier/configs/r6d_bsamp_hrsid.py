# R6D: boundary-focused point sampling (bsamp), ported 1:1 from PSeg-SSDD.
# PSeg provenance (a0r_r50coco_bsamp_pseg.py, COCO-init R50, 8352it bs4 512^2):
# measured +1.3 inshore segm AP75 vs its COCO-init baseline, while the hires
# pure-upsample lever was neutral (51.7) -> the gain comes from WHERE the loss
# looks (boundary points), not from mask resolution alone.
# Mechanism (stock mmdet Mask2FormerHead, PointRend-style
# get_uncertain_point_coords_with_randomness; uncertainty = -|logit|, i.e. the
# mask boundary): raise num_points 12544 -> 24576 (2x boundary resolution) and
# importance_sample_ratio 0.75 -> 0.9 so CE/Dice concentrate on boundary points
# -> tighter high-IoU masks (the diagnosed inshore deficit). num_points also
# feeds the Hungarian matching cost (uniform rand points), exactly as in PSeg.
# Config-only: ZERO new parameters, COCO checkpoint loads 1:1 (only the usual
# cls_embed 80+1 -> 1+1 re-init) -- the preservation law is trivially satisfied
# (no architecture change; at t=0 the model IS the COCO-init c0 model).
# HRSID protocol = c0: full COCO-M2F init, 100872 iters, 800^2, bs2+accum2,
# lr 2.5e-5, milestones 89000/96000. Eval on official/inshore/offshore splits.
_base_ = './c0_m2f_cocoinit_hrsid.py'

model = dict(train_cfg=dict(
    num_points=24576,              # 12544 -> 24576 (2x boundary resolution)
    oversample_ratio=3.0,
    importance_sample_ratio=0.9))  # 0.75 -> 0.9 (more boundary/uncertain points)
