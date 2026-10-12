# UNIVERSALITY MATRIX carrier 4/5: stock CondInst-R50 on PSeg-SSDD
# (single-stage dynamic-kernel family; runs/condinst/b0, ckpt
# best_coco_segm_mAP_iter_8352.pth, official 3-split segm mAP 41.2/54.3/60.1
# inshore/test/offshore per CKPT_INVENTORY sec.3) x wide 8-view msTTA +
# consensus-vote fusion.
#
# Base = the DUMPED run config (includes the EMAHook custom hook the carrier
# trained with -- harmless at test time, and identical to how the fresh
# runs/cmp evals of this checkpoint were executed). CAUTION: the dump's
# test_dataloader points at test_offshore.json (last eval override was
# re-dumped); the univ queue ALWAYS overrides ann_file+evaluator per split.
#
# GEOMETRY FACT (source-verified, mmdet 3.3.0 condinst_head.py
# _predict_masks_by_feat_single): rescale=True -> F.interpolate(mask_preds,
# (ori_h, ori_w)) > mask_thr, i.e. per-view masks at ori_shape. Same recipe
# geometry as cp_pseg_mstta_wide_votecons.py: scales {416,512,608,704} x
# {hflip,noflip}, keep_ratio=True, consensus band [2,6].
_base_ = 'E:/SAR/seal/runs/condinst/b0/condinst_r50coco_pseg.py'

custom_imports = dict(imports=['carrier.mask_tta'], allow_failed_imports=False)

tta_model = dict(
    type='MaskTTAModel',
    tta_cfg=dict(nms=dict(type='nms', iou_threshold=0.6), max_per_img=100,
                 merge='vote', vote_iou=0.5, vote_mask='consensus',
                 vote_min_support=2, vote_max_support=6))

tta_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale=(416, 416), keep_ratio=True),
                dict(type='Resize', scale=(512, 512), keep_ratio=True),
                dict(type='Resize', scale=(608, 608), keep_ratio=True),
                dict(type='Resize', scale=(704, 704), keep_ratio=True),
            ],
            [
                dict(type='RandomFlip', prob=1.0, direction='horizontal'),
                dict(type='RandomFlip', prob=0.0),
            ],
            [dict(type='LoadAnnotations', with_bbox=True, with_mask=True)],
            [
                dict(
                    type='PackDetInputs',
                    meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                               'scale_factor', 'flip', 'flip_direction')),
            ],
        ]),
]
