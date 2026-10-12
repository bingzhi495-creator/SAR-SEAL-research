# A0R-caps on HRSID: equal-parameter control (same conv at 1/4, no resolution
# lift). Confirms cross-dataset whether the refine gain is capacity vs resolution.
_base_ = './a0_m2f_r50_hrsid.py'
custom_imports = dict(imports=['carrier.seal_m2f_head'], allow_failed_imports=False)
model = dict(panoptic_head=dict(type='SealM2FHead', refine_scale=1, learn_refine=True))
