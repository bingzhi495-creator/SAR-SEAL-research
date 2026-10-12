# UNIVERSALITY MATRIX carrier 3/5: stock Mask R-CNN R50-FPN on PSeg-SSDD
# (runs/retrain/maskrcnn, ckpt epoch_41.pth = best val epoch, official 3-split
# segm mAP 40.8/57.8/65.1 inshore/test/offshore per CKPT_INVENTORY sec.3)
# x wide 8-view msTTA + consensus-vote fusion.
#
# Base = the run's own dumped config; like runs/cmp/cfg/maskrcnn_eval.py we
# must point test_dataloader/test_evaluator at the run's VAL loader (PSeg
# CocoDataset 'ship'), because the copied test_dataloader still targets the
# old cocossdd SAR_coco set. TTA geometry = the validated PSeg recipe from
# cp_pseg_mstta_wide_votecons.py: scales {416,512,608,704} x {hflip,noflip},
# keep_ratio=True (native protocol is Resize 512 keep_ratio=True), consensus
# band [2,6]. FCNMaskHead paste path -> per-view masks at ori_shape under
# rescale=True.
_base_ = ['E:/SAR/seal/runs/retrain/maskrcnn/mask-rcnn_r50_fpn_1x_coco.py']
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
