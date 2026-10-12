# UNIVERSALITY MATRIX carrier 1/5: Mask R-CNN R50-FPN on HRSID (two-stage
# RoI-paste family) x OUR inference-side recipe = wide 8-view msTTA +
# consensus-vote fusion (MaskTTAModel, model-agnostic: consumes only
# pred_instances masks/scores/labels).
#
# Base = the DUMPED run config of the trained carrier
# (runs/bl_maskrcnn_hrsid/s0/, ckpt best_coco_segm_mAP_iter_84040.pth,
# base 3-split segm mAP 28.3/44.8/57.8 inshore/official/offshore) so the model
# section is byte-identical to what was trained. Only ADDITIONS here:
# custom_imports + tta_model + tta_pipeline (exact HRSID recipe geometry from
# c0_v2_mstta_wide_voteconsensus_hrsid.py: scales {640,800,960,1088} x
# {hflip,noflip}, keep_ratio=True on native 800x800 chips; consensus band
# [2,6], vote_iou 0.5).
#
# GEOMETRY FACT (source-verified, mmdet 3.3.0): FCNMaskHead
# _predict_by_feat_single with rescale=True pastes every mask to
# ori_shape=(800,800), so all 8 views share the same HxW and MaskTTAModel
# only has to undo the flip -- same invariant as the M2F carrier.
_base_ = 'E:/SAR/seal/runs/bl_maskrcnn_hrsid/s0/bl_maskrcnn_hrsid.py'

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
                dict(type='Resize', scale=(640, 640), keep_ratio=True),
                dict(type='Resize', scale=(800, 800), keep_ratio=True),
                dict(type='Resize', scale=(960, 960), keep_ratio=True),
                dict(type='Resize', scale=(1088, 1088), keep_ratio=True),
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
