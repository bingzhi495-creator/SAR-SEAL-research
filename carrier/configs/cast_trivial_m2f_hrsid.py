# TRIVIAL-paste control (GATE-0 for CAST): same count of pasted instances, random positions, no contrast
# recalibration/water/clutter constraints. If CAST-hard > trivial on inshore, the gain is TARGETING not volume.
_base_ = './a1a2_m2f_hrsid.py'

train_dataloader = dict(dataset=dict(ann_file='data/hrsid/annotations/train_cast_trivial.json'))
