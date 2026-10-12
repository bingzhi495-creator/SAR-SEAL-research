# A0R-refine on HRSID (real-target validation of the refinement head).
# Run only if the PSeg refine gate passes (AP75/Boundary-IoU-d2 lift). Mirrors
# the HRSID A0 anchor config + SealM2FHead final-layer 1/4->1/2 mask. ~9h bs2.
_base_ = './a0_m2f_r50_hrsid.py'

custom_imports = dict(imports=['carrier.seal_m2f_head'], allow_failed_imports=False)

model = dict(panoptic_head=dict(type='SealM2FHead', refine_scale=2))
