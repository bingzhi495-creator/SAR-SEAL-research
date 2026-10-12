# ===== AutoDL Swin-L run (80GB GPU) — the heavy carrier to beat LFG-Net (64.2/81.2) =====
# Mask2Former-Swin-L COCO-panoptic init. Maximize the 80GB: big batch -> faster -> cheaper.
# EDIT the 3 data paths below to your AutoDL upload location, then run autodl_run.sh.
_base_ = './a0r_refine_pseg.py'

# ---- EDIT THESE to your AutoDL data location ----
data_root = '/root/autodl-tmp/pseg_ssdd/'                 # has annotations/{train,test,test_inshore,test_offshore}.json
img_train = '/root/autodl-tmp/PSeg_SSDD/JPEGImages_train/'
img_test  = '/root/autodl-tmp/PSeg_SSDD/JPEGImages_test/'
# -------------------------------------------------

depths = [2, 2, 18, 2]
model = dict(
    backbone=dict(
        _delete_=True, type='SwinTransformer', pretrain_img_size=384,
        embed_dims=192, depths=depths, num_heads=[6, 12, 24, 48],
        window_size=12, mlp_ratio=4, qkv_bias=True, qk_scale=None,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.3,
        patch_norm=True, out_indices=(0, 1, 2, 3), with_cp=False,   # 80GB: cp OFF -> faster
        convert_weights=True, frozen_stages=-1, init_cfg=None),
    panoptic_head=dict(in_channels=[192, 384, 768, 1536], num_queries=200))
# auto-downloads on AutoDL (fast network); the COCO-panoptic Swin-L full model
load_from = 'https://download.openmmlab.com/mmdetection/v3.0/mask2former/mask2former_swin-l-p4-w12-384-in21k_16xb1-lsj-100e_coco-panoptic/mask2former_swin-l-p4-w12-384-in21k_16xb1-lsj-100e_coco-panoptic_20220407_104949-82f8d28d.pth'

# ---- 80GB: big batch. batch16 -> lr 1e-4 (mmdet's native). If OOM drop to 12/8 and
# scale lr = batch/16 * 1e-4. If nvidia-smi shows <70GB, push batch to 20-24. ----
train_dataloader = dict(
    batch_size=16, num_workers=8,
    dataset=dict(data_root='', ann_file=data_root + 'annotations/train.json',
                 data_prefix=dict(img=img_train)))
val_dataloader = dict(num_workers=4, persistent_workers=True,
    dataset=dict(data_root='', ann_file=data_root + 'annotations/test.json',
                 data_prefix=dict(img=img_test)))
test_dataloader = val_dataloader
val_evaluator = dict(ann_file=data_root + 'annotations/test.json')
test_evaluator = val_evaluator

# 50 epochs (928/16 = 58 it/ep -> 2900 iters), val every ~5ep
max_iters = 2900
train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters, val_interval=290)
optim_wrapper = dict(
    optimizer=dict(lr=1e-4),
    paramwise_cfg=dict(norm_decay_mult=0.0, custom_keys={
        'backbone': dict(lr_mult=0.1, decay_mult=1.0),
        'query_embed': dict(lr_mult=1.0, decay_mult=0.0),
        'query_feat': dict(lr_mult=1.0, decay_mult=0.0),
        'level_embed': dict(lr_mult=1.0, decay_mult=0.0),
        'backbone.patch_embed.norm': dict(lr_mult=0.1, decay_mult=0.0),
        'backbone.norm': dict(lr_mult=0.1, decay_mult=0.0),
        'absolute_pos_embed': dict(lr_mult=0.1, decay_mult=0.0),
        'relative_position_bias_table': dict(lr_mult=0.1, decay_mult=0.0)}))
param_scheduler = dict(end=max_iters, milestones=[2600, 2800])
default_hooks = dict(checkpoint=dict(interval=290, by_epoch=False, max_keep_ckpts=2,
                                     save_best='coco/segm_mAP', rule='greater'),
                     logger=dict(interval=20))
