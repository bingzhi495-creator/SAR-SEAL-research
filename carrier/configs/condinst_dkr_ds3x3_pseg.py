# F1 ablation arm 'ds3x3' (THE proposal): DKRCondInstMaskHead with a depthwise-separable dynamic
# 3x3 FIRST layer -> per-instance spatial DOF in the boundary band. 269 params (vs 1x1=169, dense3x3=809).
# COCO transfer: backbone/FPN/FCOS-trunk/mask_feature_head load; controller (169->269) freshly inited.
_base_ = './condinst_r50coco_pseg.py'
custom_imports = dict(imports=['carrier.dkr_condinst_head'], allow_failed_imports=False)
model = dict(
    bbox_head=dict(num_params=269),
    mask_head=dict(type='DKRCondInstMaskHead', kernel_mode='ds3x3'))
