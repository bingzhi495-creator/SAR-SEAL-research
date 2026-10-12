# A0R + stride-4 4th deformable-encoder level. The MSDeformAttn encoder normally
# skips stride-4 (where tiny SSDD ships live); feeding all 4 backbone levels makes
# mask_feature the encoder-refined stride-4 map. Config-only; measured ~12-13GB at bs4.
_base_ = './a0r_refine_pseg.py'

model = dict(panoptic_head=dict(
    num_transformer_feat_level=4,
    pixel_decoder=dict(
        num_outs=4,
        encoder=dict(layer_cfg=dict(self_attn_cfg=dict(num_levels=4))))))
