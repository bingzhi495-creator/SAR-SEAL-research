# Phase-2 combo C1: GeoField-shape + boundary loss (GeoField's inshore AP75 mechanism
# + boundary-Dice for sharper high-IoU). Inherits the GeoField re-test config (COCO-init,
# val inshore, geofield lr_mult=50); just adds boundary_weight (passed through to SealM2FHead).
_base_ = './a0r_geo_r50coco_pseg.py'

model = dict(panoptic_head=dict(boundary_weight=0.2, boundary_ksize=3))
