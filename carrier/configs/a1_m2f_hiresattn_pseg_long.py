# A1: stride-4 query attention (diagnosis-driven novelty, workflow wveuy1juz).
# Inshore small-obj failure = confident mis-localization: M2F queries attend only strides 8/16/32 where a
# typical small ship is ~3.5px (stride-8); they can't localize it -> mask drifts out of IoU0.5. CURE = give
# queries a stride-4 attention level (footprint 3.5->7px). This is the query-LOCALIZATION-resolution axis,
# orthogonal to every washed mechanism (refine/maskiou/boundary = mask-readout/ranking axis; oracle caps that
# axis at 19% recoverable). CHEAP & config-only: pixel_decoder ALWAYS builds outs=[s32,s16,s8,s4] (s4 via FPN);
# num_outs 3->4 just returns s4 too as a 4th query-attention memory level. Deformable encoder num_levels stays
# 3 (only s8/16/32 through O(HW^2)) -> no OOM, local 3090Ti bs4@512^2 feasible. NO code change (base head's
# forward already loops over num_transformer_feat_level). Base configs unchanged -> default-off, zero risk.
_base_ = './a0_m2f_r50_pseg_long.py'

model = dict(
    panoptic_head=dict(
        num_transformer_feat_level=4,
        pixel_decoder=dict(
            num_outs=4,
            # encoder num_levels 3->4: head asserts num_levels==num_transformer_feat_level.
            # deformable attn is O(N·K) linear (NOT O(HW^2)) so stride-4 in the encoder is affordable
            # (M2F-long was only ~5.5GB; ~4x encoder tokens still fits 24GB). Bonus: query gets a
            # deformable-ENRICHED stride-4 memory, better for small-obj localization.
            encoder=dict(layer_cfg=dict(self_attn_cfg=dict(num_levels=4))),
        ),
    ),
)
