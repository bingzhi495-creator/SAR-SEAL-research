# R6B on HRSID: minimal-footprint fix for degenerate (empty) masked-attention masks.
# PSeg verdict (seed0, single-variable): ALL THREE SPLITS POSITIVE (+0.8 inshore / +0.6
# test / +0.3 offshore vs s0 baseline 53.1/60.2/63.3) — first all-positive architecture
# change in the project. Mechanism predicts LARGER effect here: HRSID inshore boats
# (9x18 px) vanish at stride-32 (0.3x0.6 px), so empty attention masks are far more
# frequent than on PSeg. min_footprint=7 opens a 7x7 window at the predicted-mask
# centroid instead of falling back to global attention. Zero new parameters;
# min_footprint=0 reproduces stock byte-for-byte (GATE-0 verified 24/24 on PSeg build).
_base_ = './c0_m2f_cocoinit_hrsid.py'
custom_imports = dict(imports=['carrier.minfoot_m2f_head'], allow_failed_imports=False)
model = dict(panoptic_head=dict(type='MinFootprintM2FHead', min_footprint=7))
load_from = 'pretrained/m2f_r50_coco.pth'
