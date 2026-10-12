# GATE-0 CAPACITY CONTROL (variant b): DKRCondInstMaskHead, 1x1-only readout but WIDENED
# (feat_channels 8->11) so #params (265) ~= ds3x3 (269) WITHOUT any spatial extent.
# Make-or-break isolation: if ds3x3 > this on inshore AP75/Boundary-AP, the gain is the 3x3
# SPATIAL DOF, not just more capacity (kills the "CondInst Table-1 capacity is saturated" attack).
_base_ = './condinst_r50coco_pseg.py'
custom_imports = dict(imports=['carrier.dkr_condinst_head'], allow_failed_imports=False)
model = dict(
    bbox_head=dict(num_params=265),
    mask_head=dict(type='DKRCondInstMaskHead', kernel_mode='1x1', feat_channels=11))
