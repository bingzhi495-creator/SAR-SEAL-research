# c0a2: a2 (Gaussian-Wasserstein matching) ONLY on COCO-init base, NO a1 (stride-4). Cleanly isolates whether
# OUR NOVEL mechanism (a2) transfers to the strong regime, separate from a1's stride-4 fresh-init convergence
# tax that confounds c0a1a2 (which dropped -2.3 inshore vs c0). c0a2 vs c0 = a2's true strong-regime effect.
_base_ = './a2_m2f_wassmatch_hrsid.py'
load_from = 'pretrained/m2f_r50_coco.pth'
