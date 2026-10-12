# A0R-refine on HRSID SCENE-LEVEL (leakage-free) split. Tests the hypothesis
# that the 'neutral' refine result on the official (92.7%-leaky) split is a
# leakage artifact (no headroom: test ~= train), not a real mechanism failure.
# If refine lifts AP75 more on the leakage-free scene split, the dataset-
# dependence is partly a leakage story (ties to our protocol contribution).
_base_ = './a0_m2f_r50_hrsid_scene.py'
custom_imports = dict(imports=['carrier.seal_m2f_head'], allow_failed_imports=False)
model = dict(panoptic_head=dict(type='SealM2FHead', refine_scale=2))
