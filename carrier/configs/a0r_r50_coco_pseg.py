# A0R + R50, initialized from the FULL COCO-instance Mask2Former-R50 (not just the
# backbone). Tests whether COCO-instance init alone (pretrained head -> fast
# convergence) beats our from-scratch baseline (overall segm AP75 0.637) at 36ep.
# Same R50 backbone as baseline -> isolates the COCO-init effect.
_base_ = './a0r_refine_pseg.py'

model = dict(backbone=dict(init_cfg=None))   # full model comes from load_from
load_from = 'E:/SAR/seal/pretrained/m2f_r50_coco.pth'
