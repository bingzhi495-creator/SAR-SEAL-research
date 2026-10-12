# F1 KERNEL-SHAPE (THE proposal): DS-3x3 dynamic first layer (per-instance spatial DOF at 1/3 the
# dense-3x3 total-readout cost, 269 vs 809 params; ~1/4.2 on the dynamic 3x3 weight tensor alone, 170
# vs 720). On a0r_dkr3 (relcoord+scatter+orient, gated init0). vs 1x1(=a0r_dkr3) vs dense3x3.
_base_ = './a0r_dkr3_r50coco_pseg.py'
model = dict(panoptic_head=dict(dkr_kernel_mode='ds3x3'))
