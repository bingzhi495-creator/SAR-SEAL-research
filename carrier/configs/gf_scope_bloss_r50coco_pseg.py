# Phase-2 combo C2: GeoField + SCOPE + boundary loss (all three). SCOPE helped offshore,
# GeoField helps inshore AP75, boundary sharpens high-IoU -> test if the combination nets
# an overall gain. Inherits the SCOPE config (COCO-init, SAR pipeline, val inshore).
_base_ = './gf_scope_r50coco_pseg.py'

model = dict(panoptic_head=dict(boundary_weight=0.2, boundary_ksize=3))
