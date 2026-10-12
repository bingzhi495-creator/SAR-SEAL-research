# A0R + Swin-S, initialized from the FULL COCO-instance Mask2Former-Swin-S model.
# The scout's rank-1 baseline: strongest locally-runnable instance segmenter
# (COCO mask AP 46.1), exact lineage of O2Former/DiffSAR, pretrained head -> fast
# convergence on small SAR data. Keeps SealM2FHead refine (GeoField-compatible).
_base_ = './a0r_swins_pseg.py'

model = dict(backbone=dict(init_cfg=None))   # full model comes from load_from
load_from = 'E:/SAR/seal/pretrained/m2f_swins_coco.pth'
