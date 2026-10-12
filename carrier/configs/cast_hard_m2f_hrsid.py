# CAST-hard arm on HRSID: a1a2 + physics-consistent hard-example copy-paste training data (CAST v1.1,
# tools_cast_engine.py: water-constrained clutter-anchored clustered paste, contrast recalibrated to the
# measured hard band 0.7-1.5 sigma). Train ann = original 3642 imgs + 120 augmented copies (12609 anns).
# Images resolve via junction JPEGImages1/cast_images -> data/hrsid/cast_images (238 files). Controls:
# a1a2 (no CAST, .186) + trivial-paste arm (TODO) to prove targeting matters, not just more instances.
_base_ = './a1a2_m2f_hrsid.py'

train_dataloader = dict(
    dataset=dict(ann_file='data/hrsid/annotations/train_cast_hard.json'))
