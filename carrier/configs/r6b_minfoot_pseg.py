# R6B: masked-attention EMPTY-mask degeneracy fix (small-target pathology),
# PSeg quick-falsify. mmdet falls back to FULL-IMAGE attention when a query's
# attn mask is empty at a level (mask2former_head.py L437-439; replicated in
# seal_m2f_head.py L141) -- a 9x18 px ship is ~0.3x0.6 px at stride 32, so
# interpolate+threshold empties the footprint and the small-target query gets
# diluted over the whole scene. MinFootprintM2FHead instead opens a 7x7 window
# at the predicted-mask centroid ("minimal footprint floor"). ZERO new params;
# min_footprint=0 == stock bitwise (GATE-0: carrier/tests/cpu_test_minfoot.py).
# A/B vs the cp_m2f_cocoinit_pseg baseline (runs/cp_pseg/s0): same COCO-init
# protocol (8352 it, bs4, 512^2), single-knob difference.
_base_ = './cp_m2f_cocoinit_pseg.py'

custom_imports = dict(imports=['carrier.minfoot_m2f_head'],
                      allow_failed_imports=False)

model = dict(panoptic_head=dict(type='MinFootprintM2FHead', min_footprint=7))
