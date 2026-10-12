# F1 KERNEL-SHAPE control: dense-3x3 dynamic first layer (spatial DOF but ~16x more generated
# params -> overfit risk on 928 imgs). Isolates "is DS-3x3's cheapness necessary?".
_base_ = './a0r_dkr3_r50coco_pseg.py'
model = dict(panoptic_head=dict(dkr_kernel_mode='dense3x3'))
