# A0R + Swin-S backbone. Same embed_dims=96 as Swin-T (so in_channels identical),
# deeper stage-3 (depths [2,2,18,2]) -> stronger lever. Keeps SealM2FHead refine
# (GeoField-compatible). Needs swin_small weights (cached via proxy).
_base_ = './a0r_refine_pseg.py'

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth'  # noqa
depths = [2, 2, 18, 2]
model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=96, depths=depths, num_heads=[3, 6, 12, 24],
        window_size=7, mlp_ratio=4, qkv_bias=True, qk_scale=None,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.3,
        patch_norm=True, out_indices=(0, 1, 2, 3), with_cp=False,
        convert_weights=True, frozen_stages=-1,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    panoptic_head=dict(in_channels=[96, 192, 384, 768]))

_bn = dict(lr_mult=0.1, decay_mult=0.0)
_be = dict(lr_mult=0.1, decay_mult=0.0)
_em = dict(lr_mult=1.0, decay_mult=0.0)
custom_keys = {
    'backbone': dict(lr_mult=0.1, decay_mult=1.0),
    'backbone.patch_embed.norm': _bn, 'backbone.norm': _bn,
    'absolute_pos_embed': _be, 'relative_position_bias_table': _be,
    'query_embed': _em, 'query_feat': _em, 'level_embed': _em,
}
custom_keys.update({f'backbone.stages.{s}.blocks.{b}.norm': _bn
                    for s, n in enumerate(depths) for b in range(n)})
custom_keys.update({f'backbone.stages.{s}.downsample.norm': _bn
                    for s in range(len(depths) - 1)})
optim_wrapper = dict(paramwise_cfg=dict(custom_keys=custom_keys, norm_decay_mult=0.0))
