# A0R-refine: A0 Mask2Former-R50 + SealM2FHead (final-layer mask 1/4 -> 1/2).
# Tests the oracle finding that mask resolution bounds ~40% of the inshore AP75
# gap. Ablation vs A0 (a0_m2f_r50_pseg) isolates the resolution effect.
_base_ = './a0_m2f_r50_pseg.py'

custom_imports = dict(imports=['carrier.seal_m2f_head'], allow_failed_imports=False)

model = dict(panoptic_head=dict(type='SealM2FHead', refine_scale=2))
