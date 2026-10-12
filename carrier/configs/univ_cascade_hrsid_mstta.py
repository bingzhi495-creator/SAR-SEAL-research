# UNIVERSALITY MATRIX carrier 2/5: Cascade Mask R-CNN R50-FPN on HRSID
# (multi-stage refinement family) x wide 8-view msTTA + consensus-vote fusion.
#
# Base = the DUMPED run config of the trained carrier
# (runs/bl_cascade_hrsid/s0/, ckpt best_coco_segm_mAP_iter_84040.pth,
# base 3-split segm mAP 30.4/46.5/59.2 inshore/official/offshore).
# Additions identical to univ_mrcnn_hrsid_mstta.py (see its header for the
# geometry argument; Cascade's mask branch is the same FCNMaskHead paste path,
# so rescale=True -> masks at ori_shape=(800,800) per view).
#
# DOCUMENTED FIX (found by carrier/tests/cpu_test_univ.py): stock
# CascadeRoIHead.predict_mask crashes on flipped TTA views -- its per-stage
# merge_aug_masks consults meta 'flip' (mmdet-2.x aug-test leftover) and does
# negative-step slicing on a torch tensor (ValueError: step must be greater
# than zero); it would also be semantically wrong (per-stage un-flip of
# RoI-local logits that are pasted onto flipped-frame bboxes; MaskTTAModel
# undoes the flip ONCE after pasting). roi_head.type is therefore swapped to
# the PARAMETER-FREE TTACascadeRoIHead (carrier/mask_tta.py) whose only change
# is a plain per-stage mean -- output-identical to stock on non-flipped
# pipelines, ckpt keys untouched.
_base_ = 'E:/SAR/seal/runs/bl_cascade_hrsid/s0/bl_cascade_hrsid.py'

custom_imports = dict(imports=['carrier.mask_tta'], allow_failed_imports=False)

model = dict(roi_head=dict(type='TTACascadeRoIHead'))

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
