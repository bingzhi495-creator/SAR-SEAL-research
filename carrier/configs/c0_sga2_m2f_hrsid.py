# SGA-v2 (Strip-Gated Add-on v2) on the STRONG COCO-init base (c0).
#
# Research-backed upgrade of StripNet-lite (c0_stripnet, running as s0), same carrier
# (StripResNet residual add-on on the 4 R50 outputs, before the pixel decoder), three
# evidence-driven knob changes vs s0:
#   * strip_k=11             -- InceptionNeXt / Strip R-CNN band-kernel sweet spot (vs 7).
#   * strip_order='sequential' -- Strip R-CNN: DW 1xk THEN kx1 chained beats the summed
#                                parallel pair (+0.21 in their ablation). Same params,
#                                only the dataflow between the two DW convs changes.
#   * gamma_mode='channel'   -- CaiT LayerScale: per-CHANNEL learnable gamma vector (C,)
#                                per stage > per-stage scalar.
#   * strip_gate_init=0.1    -- CaiT warm-start epsilon for shallow stacks. NOTE: with
#                                0.1 this config is deliberately NOT byte-identical to c0
#                                at init (gamma != 0 lets the zero-init expand conv
#                                receive gradient at step 0 -- the warm-start mechanism).
#                                GATE-0 remains PROVABLE: set strip_gate_init=0.0 in ANY
#                                mode and out == x bit-for-bit again (expand conv weight
#                                AND bias stay zero-initialised, belt-and-suspenders).
#
# Base is c0 (full COCO M2F init, ORIGINAL COCO backbone). Stock ResNet conv/bn keys are
# unchanged (StripResNet subclasses ResNet) -> the COCO checkpoint loads 1:1; only the NEW
# strip_blocks.* params are fresh (absent from the ckpt, tolerated by strict=False load).
_base_ = './c0_m2f_cocoinit_hrsid.py'

custom_imports = dict(imports=['carrier.strip_addon'], allow_failed_imports=False)

# Swap the backbone TYPE only; every other ResNet arg (depth=50, num_stages=4,
# out_indices=(0,1,2,3), frozen_stages=-1, norm_cfg BN requires_grad=False, norm_eval,
# style='pytorch', init_cfg Pretrained torchvision://resnet50) is inherited unchanged
# from the base mask2former_r50 config via config dict merging.
model = dict(
    backbone=dict(
        type='StripResNet',
        strip_k=11,                # 1x11 + 11x1 depthwise anisotropic kernels
        strip_order='sequential',  # DW 1x11 THEN 11x1 chained (Strip R-CNN)
        gamma_mode='channel',      # per-channel LayerScale gamma vector (CaiT)
        strip_gate_init=0.1,       # CaiT warm-start (NOT byte-identical at init, see above)
        strip_enable=True,
    ))

# strong base: full COCO Mask2Former init (backbone + heads), same as c0.
load_from = 'pretrained/m2f_r50_coco.pth'
