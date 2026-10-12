# Offset-Adjusted Mask2Former on the STRONG COCO-init base (c0).
#
# Mechanism axis: a differentiable constraint on the MSDeformAttn sampling OFFSETS in
# the pixel decoder, pulling diverging sampling points back toward the compact
# foreground ("query nearby", arXiv:2506.05897). Targets near-shore small-ship misses
# where deformable sampling drifts into bright pier/harbour clutter. This is a
# query/sampling-DESIGN mechanism, orthogonal to the killed capacity/feature/data/
# supervision families -- NO backbone swap, NO aux/refine head (the a1 family is dead).
#
# Base is c0 (full COCO M2F init, ORIGINAL COCO backbone) -- preserving the strong base
# is the whole point of this ablation. Best variant per the paper: strategy 3,
# softmax x scale with constant 2 ("Sigmoid*2"), so offset_constraint='softmax_scale'.
#
# NOTE: unlike a gated-residual add-on, this constraint is ACTIVE from step 0 (it is not
# identity at init). Byte-exactness is guaranteed only at offset_constraint='none'; the
# stock COCO weights still load 1:1 (only the 6 new scalar `scale` params are fresh).
# The honest test is the c0-vs-offsetadj ablation on the inshore/small-ship slice.
_base_ = './c0_m2f_cocoinit_hrsid.py'

custom_imports = dict(imports=['carrier.offset_adjust'], allow_failed_imports=False)

model = dict(
    panoptic_head=dict(
        pixel_decoder=dict(
            type='OffsetAdjustMSDeformAttnPixelDecoder',
            offset_constraint='softmax_scale',   # paper-best strategy 3
            softmax_scale=2.0,                    # learnable, init 2 ("Sigmoid*2")
        )))

# strong base: full COCO Mask2Former init (backbone + heads), same as c0.
load_from = 'pretrained/m2f_r50_coco.pth'
