# A0 Mask2Former-R50 on HRSID SCENE-LEVEL split (P0-3: official-vs-scene Delta).
# Same model/schedule as the official-split HRSID A0 (segm 0.348), but trained on
# the leakage-free scene_train.json and evaluated on scene_test.json (whole source
# scenes in/out, no 92.7% chip overlap). The Delta (official 0.348 vs scene) is the
# evidence the protocol contribution needs -- quantifies how much the 0.348 was
# inflated by the official-split leakage. Without this the leakage claim is half.
_base_ = './a0_m2f_r50_hrsid.py'

ann = 'E:/SAR/seal/data/hrsid/annotations/'
train_dataloader = dict(dataset=dict(ann_file=ann + 'scene_train.json'))
val_dataloader = dict(dataset=dict(ann_file=ann + 'scene_test.json'))
test_dataloader = val_dataloader
val_evaluator = dict(ann_file=ann + 'scene_test.json')
test_evaluator = dict(ann_file=ann + 'scene_test.json')
