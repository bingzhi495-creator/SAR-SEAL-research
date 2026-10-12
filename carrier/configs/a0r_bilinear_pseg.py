# A0R-bilinear: SealM2FHead with learn_refine=False (final mask 1/4->1/2 by pure
# bilinear interpolation, NO learned sharpening conv). P0-1 ablation: isolates
# whether the refine-head gain (AP75 +2.8/inshore +4.6) comes from the higher
# RESOLUTION itself or from the learned residual sharpening. If bilinear-only
# already gets most of the gain -> resolution is the lever (cleaner claim); if
# the learned conv adds a lot -> it's a learned-module gain (must say so).
_base_ = './a0_m2f_r50_pseg.py'

custom_imports = dict(imports=['carrier.seal_m2f_head'], allow_failed_imports=False)

model = dict(panoptic_head=dict(type='SealM2FHead', refine_scale=2, learn_refine=False))
