# A0R + DCNv2: modulated deformable conv (ModulatedDeformConv2dPack) in
# ResNet-50 stages c3-c5. Pure-config, no extra download (reuses cached
# torchvision://resnet50 = resnet50-0676ba61.pth; DCN offset/mask convs are
# learned from scratch). Same schedule/lr/weights as A0R baseline.
_base_ = './a0r_refine_pseg.py'

model = dict(
    backbone=dict(
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)))
