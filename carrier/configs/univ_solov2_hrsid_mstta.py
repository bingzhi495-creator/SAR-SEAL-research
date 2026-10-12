# UNIVERSALITY MATRIX carrier 6/6: SOLOv2 R50-FPN on HRSID (direct dynamic-
# kernel / grid-based mask family -- NO RoI crop, NO box branch) x OUR
# inference-side recipe = wide 8-view msTTA + consensus-vote fusion
# (MaskTTAModel, model-agnostic: consumes only pred_instances masks/scores/
# labels).
#
# WHY THIS CARRIER: it is the WEAKEST HRSID carrier by a wide margin
# (base 3-split segm mAP 19.0 / 38.2 / 54.0 inshore/official/offshore, see
# runs/_bl_hrsid.out BL_RESULT solov2 lines, ckpt best_coco_segm_mAP_iter_84040)
# versus Mask R-CNN 28.3/44.8/57.8 and Cascade 30.4/46.5/59.2. Prediction P9
# (recovery gain scales inversely with base strength) is therefore tested at its
# sharpest point here.
#
# Base = the DUMPED run config of the trained carrier
# (runs/bl_solov2_hrsid/s0/bl_solov2_hrsid.py) so the model section is
# byte-identical to what was trained and the checkpoint loads 1:1. Only
# ADDITIONS here: custom_imports + tta_model + tta_pipeline (exact HRSID recipe
# geometry from univ_mrcnn_hrsid_mstta.py / c0_v2_mstta_wide_voteconsensus_
# hrsid.py: scales {640,800,960,1088} x {hflip,noflip}, keep_ratio=True on
# native 800x800 chips; consensus band [2,6], vote_iou 0.5). NO model overrides
# are needed (unlike the Cascade carrier).
#
# GEOMETRY FACT (source-verified, mmdet 3.3.0,
# mmdet/models/dense_heads/solov2_head.py::SOLOv2Head._predict_by_feat_single):
# the head UNCONDITIONALLY resizes its mask logits to img_meta['ori_shape'][:2]
#     mask_preds = F.interpolate(mask_preds.unsqueeze(0), size=upsampled_size,
#                                mode='bilinear', ...)[:, :, :h, :w]
#     mask_preds = F.interpolate(mask_preds, size=img_meta['ori_shape'][:2],
#                                mode='bilinear', ...).squeeze(0)
#     masks = mask_preds > cfg.mask_thr
# and every early-exit returns empty_results(..., img_meta['ori_shape'][:2]).
# So under rescale=True (SingleStageInstanceSegmentor.predict default, which is
# what TestLoop uses) ALL 8 views emit masks at ori_shape=(800,800) -- the same
# invariant MaskTTAModel needs, identical to the M2F / Mask R-CNN carriers.
# Neither solo_head.py nor solov2_head.py nor single_stage_instance_seg.py
# reads the 'flip' meta key anywhere (grep: 0 hits), so no TTACascadeRoIHead-
# style fix is required -- MaskTTAModel undoes the flip ONCE, after pasting.
#
# DUMMY BOXES (source-verified, same function): SOLOv2 has no box branch and
# emits `results.bboxes = results.scores.new_zeros(len(scores), 4)` purely to
# keep downstream metric code happy. This is HARMLESS for our stack because the
# merge='vote' branch of MaskTTAModel (carrier/mask_tta.py) never reads
# pred_instances.bboxes: it clusters by TRUE pairwise mask-IoU computed from the
# flip-corrected mask tensors and then RE-DERIVES tight boxes from the fused
# masks with mask2bbox(). (Only the legacy merge='bbox_nms' path would consume
# input boxes, and it is not used here.) The BPR stage is likewise box-free:
# bpr_patches.py / bpr_apply.py read only scores + mask RLEs.
_base_ = 'E:/SAR/seal/runs/bl_solov2_hrsid/s0/bl_solov2_hrsid.py'

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
