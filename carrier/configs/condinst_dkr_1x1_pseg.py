# F1 ablation arm '1x1' (sanity): DKRCondInstMaskHead with the SAME 3x 1x1 readout as stock
# CondInst, but routed through dkr_op._apply_dynamic (fresh controller). Should ~= B0 minus the
# COCO-controller transfer; isolates "kernel shape" with the other two arms (all fresh controller).
_base_ = './condinst_r50coco_pseg.py'
custom_imports = dict(imports=['carrier.dkr_condinst_head'], allow_failed_imports=False)
model = dict(
    bbox_head=dict(num_params=169),
    mask_head=dict(type='DKRCondInstMaskHead', kernel_mode='1x1'))
