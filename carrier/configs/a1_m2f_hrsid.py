# a1-alone on HRSID = RESOLUTION axis only (stride-4 query attention), NO a2 matching. Stock Mask2FormerHead
# + the a1 pixel-decoder overrides. Completes the GATE-0 2x2: baseline (a0) / a1 / a2 / a1+a2 (a1a2), so a1's
# INDEPENDENT contribution (a1 vs baseline) and additivity (a1+a2 vs a1,a2) are attributable.
# Same protocol as a0_m2f_r50_hrsid (100872 iters/bs2/800^2) -> a0_hrsid_seed0 is the matched baseline.
_base_ = './a0_m2f_r50_hrsid.py'

model = dict(
    panoptic_head=dict(
        num_transformer_feat_level=4,
        pixel_decoder=dict(
            num_outs=4,
            encoder=dict(layer_cfg=dict(self_attn_cfg=dict(num_levels=4))))))
