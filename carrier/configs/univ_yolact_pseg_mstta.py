# UNIVERSALITY MATRIX carrier 5/5: stock YOLACT-R50 on PSeg-SSDD
# (single-stage prototype family; runs/retrain/yolact, ckpt epoch_31.pth =
# best val epoch, official 3-split segm mAP 38.9/55.7/63.4 inshore/test/
# offshore per CKPT_INVENTORY sec.3) x wide 8-view msTTA + consensus-vote
# fusion.
#
# Base = the run's own dumped config; test loader redirected to the run's
# VAL loader (PSeg CocoDataset 'ship') exactly like runs/cmp/cfg/
# yolact_eval.py, because the copied test_dataloader targets the old
# cocossdd SAR_coco set.
#
# TWO documented protocol deviations vs the other PSeg carriers (both forced
# by YOLACT's native protocol, NOT by the recipe):
#  * keep_ratio=False square-stretch resizes -- YOLACT trains AND tests with
#    Resize(550,550) keep_ratio=False (fixed-anchor design: fractional
#    strides/centers in its AnchorGenerator are precomputed for square
#    inputs). The recipe's scale SET {416,512,608,704} is kept identical to
#    the other PSeg carriers (0.76x-1.28x of YOLACT's native 550, i.e. the
#    same wide band the recipe uses everywhere).
#  * LoadImageFromFile without to_float32 (native).
#
# GEOMETRY FACT (source-verified, mmdet 3.3.0 yolact_head.py YOLACTProtonet
# _predict_by_feat_single): rescale=True -> F.interpolate(mask_preds,
# (ori_h, ori_w)) > mask_thr, i.e. per-view masks at ori_shape, so
# MaskTTAModel's shared-HxW invariant holds.
_base_ = ['E:/SAR/seal/runs/retrain/yolact/yolact_r50_1xb8-55e_coco.py']
test_dataloader = {{_base_.val_dataloader}}
test_evaluator = {{_base_.val_evaluator}}

custom_imports = dict(imports=['carrier.mask_tta'], allow_failed_imports=False)

tta_model = dict(
    type='MaskTTAModel',
    tta_cfg=dict(nms=dict(type='nms', iou_threshold=0.6), max_per_img=100,
                 merge='vote', vote_iou=0.5, vote_mask='consensus',
                 vote_min_support=2, vote_max_support=6))

tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale=(416, 416), keep_ratio=False),
                dict(type='Resize', scale=(512, 512), keep_ratio=False),
                dict(type='Resize', scale=(608, 608), keep_ratio=False),
                dict(type='Resize', scale=(704, 704), keep_ratio=False),
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
