# a1+a2 dual-axis on HRSID (design-doc thesis = 异轴增益叠加): stack the RESOLUTION axis (a1, stride-4 query
# attention) ON TOP of the proven METRIC axis (a2, Gaussian-Wasserstein matching, HRSID inshore +4.8 single-seed).
# Diagnosis: inshore small ships are sub-pixel at stride-8/16/32; a1 gives the query stride-4 features (sees the
# ship), a2 ensures those features get matched to the right GT. Orthogonal axes -> may compound past a2-alone.
# a1 part = config-only: num_transformer_feat_level=4 + pixel_decoder num_outs=4 + encoder num_levels=4 (M2F head
# asserts encoder num_levels == num_transformer_feat_level). a2 part inherited (WassMatchM2FHead + NWD cost).
_base_ = './a2_m2f_wassmatch_hrsid.py'

model = dict(
    panoptic_head=dict(
        num_transformer_feat_level=4,
        pixel_decoder=dict(
            num_outs=4,
            encoder=dict(layer_cfg=dict(self_attn_cfg=dict(num_levels=4))))))
