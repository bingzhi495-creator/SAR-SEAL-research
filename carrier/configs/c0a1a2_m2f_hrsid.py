# c0a1a2: a1(stride-4) + a2(GW matching) on the COCO-init strong base (c0's early val ~0.425 ≈ b1's 0.417 =>
# b1's gain is COCO-pretraining, NOT SAR-specific; c0 is the cleaner strong base). THE regime-transfer test:
# does our headline mechanism (a2) still help once the base is strong? c0a1a2 vs c0 isolates a1+a2's add on a
# COCO-init base (the regime the paper's final AutoDL runs will use).
_base_ = './a1a2_m2f_hrsid.py'

load_from = 'pretrained/m2f_r50_coco.pth'
