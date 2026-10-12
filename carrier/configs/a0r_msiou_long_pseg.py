# P0 mechanism: Mask-IoU quality head on the proper 100ep base.
# Diagnosis (source-verified): M2F ranks by softmax(cls)*foreground-avg-confidence, NOT
# predicted mask-IoU (maskformer_fusion_head.py:170-173) -> AP75/inshore mismatch
# (Mask Scoring R-CNN, arXiv:1903.00241). This head learns per-query IoU and ranks by it.
# maskiou_weight>0 trains the IoU regressor (BCE on matched queries); infer_iou_lambda is
# swept on the SINGLE trained checkpoint at eval time (lambda=0 == byte-identical to the
# M2F-long/refine baseline -> de-risked operating-point selection, like infer_blend).
_base_ = './a0r_refine_long_pseg.py'

custom_imports = dict(imports=['carrier.maskiou_head'], allow_failed_imports=False)

model = dict(
    panoptic_head=dict(type='MSIoUM2FHead', maskiou_weight=0.5),
    panoptic_fusion_head=dict(type='MSIoUFusionHead', infer_iou_lambda=0.0),
)
