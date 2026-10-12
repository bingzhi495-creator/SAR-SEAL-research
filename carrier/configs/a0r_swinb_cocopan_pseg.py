# Swin-B carrier (fallback if Swin-L OOMs), COCO-panoptic full-model init. Lighter than
# Swin-L (~107M), still much stronger than R50/Swin-S. val on OVERALL test.json.
_base_ = './a0r_refine_pseg.py'

depths = [2, 2, 18, 2]
model = dict(
    backbone=dict(
        _delete_=True, type='SwinTransformer', pretrain_img_size=384,
        embed_dims=128, depths=depths, num_heads=[4, 8, 16, 32],
        window_size=12, mlp_ratio=4, qkv_bias=True, qk_scale=None,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.3,
        patch_norm=True, out_indices=(0, 1, 2, 3), with_cp=False,
        convert_weights=True, frozen_stages=-1, init_cfg=None),
    panoptic_head=dict(in_channels=[128, 256, 512, 1024], num_queries=200))
load_from = 'E:/SAR/seal/pretrained/m2f_swinb_cocopan.pth'  # COCO-panoptic uses 200 queries

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

val_dataloader = dict(num_workers=0, persistent_workers=False)
default_hooks = dict(checkpoint=dict(interval=1044, by_epoch=False, max_keep_ckpts=2,
                                     save_best='coco/segm_mAP', rule='greater'))
