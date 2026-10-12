# SSDD generalization: COCO-init M2F-R50 on PSeg-SSDD, same recipe as HRSID c0.
# Protocol inherited from a0_m2f_r50_pseg (512^2, bs4, lr 2.5e-5, 8352 iters = 36 ep
# nominal by the config's own arithmetic (928*36/4), milestones 7400/8000).
# Seed set {0,2,3} to mirror the HRSID soup discipline (override per run).
_base_ = './a0_m2f_r50_pseg.py'

load_from = 'pretrained/m2f_r50_coco.pth'
