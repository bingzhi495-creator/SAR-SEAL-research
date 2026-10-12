# Per-instance boundary loss (SealM2FHead._boundary_dice on matched queries) ON the refine head.
# Tests the UNet++-evidenced boundary-AP75 booster on PSeg-SSDD (= SSDD-analog where boundary gave
# conn-comp AP75 +0.0235). Control = a0r_refine_pseg (boundary_weight=0, same model, zero extra params).
# Compare INSHORE AP75, multi-seed (UNet++ discipline: boundary gain is inshore-AP75 not overall).
_base_ = './a0r_refine_pseg.py'
model = dict(panoptic_head=dict(boundary_weight=0.2))
