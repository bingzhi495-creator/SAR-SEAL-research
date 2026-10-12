# A0R-caps: EQUAL-PARAMETER CONTROL for the refine head (reviewer blocker B1).
# Identical added capacity as A0R-refine (the SAME mask_refine conv module, same
# parameter count) but applied at the ORIGINAL 1/4 stride -- NO resolution lift
# (refine_scale=1 makes the interpolate a no-op). This is the control the
# bilinear ablation cannot be: bilinear still raises resolution. If A0R-caps
# recovers << +2.6 AP75, the gain is the 1/4->1/2 resolution axis (claim
# survives); if it recovers ~+2.6, the gain is just added final-layer capacity
# and the framing must change to 'capacity helps'. THE single most important
# missing control given the 'minimal surgery, not added capacity' positioning.
_base_ = './a0_m2f_r50_pseg.py'

custom_imports = dict(imports=['carrier.seal_m2f_head'], allow_failed_imports=False)

# refine_scale=1 -> _refine_features interpolate is identity; learn_refine=True
# keeps the same conv module -> equal parameters, no resolution change.
model = dict(panoptic_head=dict(type='SealM2FHead', refine_scale=1, learn_refine=True))
