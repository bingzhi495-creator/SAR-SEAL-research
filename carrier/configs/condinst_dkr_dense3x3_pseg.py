# F1 ablation arm 'dense3x3' (overfit control): DKRCondInstMaskHead with a DENSE dynamic 3x3 first
# layer (9*Ci*Co generated params = 809, ~3x the ds3x3 readout). Tests whether ds3x3's cheapness is
# necessary: claim holds IFF ds3x3 >= dense3x3 within noise (cheap stencil not beaten by the dense one).
_base_ = './condinst_r50coco_pseg.py'
custom_imports = dict(imports=['carrier.dkr_condinst_head'], allow_failed_imports=False)
model = dict(
    bbox_head=dict(num_params=809),
    mask_head=dict(type='DKRCondInstMaskHead', kernel_mode='dense3x3'))
