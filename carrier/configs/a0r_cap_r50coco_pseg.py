# Equal-parameter CAPACITY control for GeoField, on COCO-init R50. Same module/params
# as the GeoField-shape arm but gen_mode='capacity' (uniform kernel, no oriented
# geometry, no coords) -> isolates whether GeoField's inshore AP75 gain comes from the
# GEOMETRY mechanism or merely from the added capacity. val on inshore (the diagnosed axis).
_base_ = './a0r_geo_r50coco_pseg.py'

model = dict(panoptic_head=dict(gen_mode='capacity'))
