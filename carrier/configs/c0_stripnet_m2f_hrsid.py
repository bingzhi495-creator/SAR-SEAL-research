# StripNet-lite on the STRONG COCO-init base (c0).
#
# Capacity/shape-prior axis: a zero-gated anisotropic strip-convolution add-on
# (DWConv 1xk + kx1, k=7) bolted onto each ResNet-50 backbone output BEFORE the pixel
# decoder. SAR ships are ELONGATED; square 3x3 kernels miss that shape prior. The add-on
# encodes a long-and-thin receptive field as a residual: out = x + gamma * StripBlock(x),
# with gamma (per-stage, learnable, init 0) AND the final expand conv zero-initialised.
# => at init the add-on contributes EXACTLY zero, so this is byte-identical to c0 (GATE-0).
#
# Base is c0 (full COCO M2F init, ORIGINAL COCO backbone) -- preserving the strong base is
# the whole point. The stock ResNet conv/bn keys are unchanged (StripResNet subclasses
# ResNet), so the COCO checkpoint loads 1:1; only the NEW strip_blocks.* params are fresh
# (absent from the ckpt -> tolerated by mmdet's strict=False load). During training gamma /
# expand grow only if the anisotropic prior helps.
_base_ = './c0_m2f_cocoinit_hrsid.py'

custom_imports = dict(imports=['carrier.strip_addon'], allow_failed_imports=False)

# Swap the backbone TYPE only; every other ResNet arg (depth=50, num_stages=4,
# out_indices=(0,1,2,3), frozen_stages=-1, norm_cfg BN requires_grad=False, norm_eval,
# style='pytorch', init_cfg Pretrained torchvision://resnet50) is inherited unchanged
# from the base mask2former_r50 config via config dict merging.
model = dict(
    backbone=dict(
        type='StripResNet',
        strip_k=7,               # 1x7 + 7x1 depthwise anisotropic kernels
        strip_gate_init=0.0,     # per-stage learnable gate gamma, init 0 -> GATE-0
        strip_enable=True,
    ))

# strong base: full COCO Mask2Former init (backbone + heads), same as c0.
load_from = 'pretrained/m2f_r50_coco.pth'
