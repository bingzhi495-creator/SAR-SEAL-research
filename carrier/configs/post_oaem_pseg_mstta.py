# POST-REVIEW step 2 (docs/TGRS_SUBMISSION_REVIEW.md sec.7, R2 3.2): the
# recipe stacked on a SAR-module carrier. M2F + OAEM (our re-implementation,
# Table II row 53.2/60.8/64.0 from runs/cmp/_cmp.out, ckpt
# runs/overnight/oaem/best_coco_segm_mAP_iter_8352.pth) x wide 8-view msTTA +
# consensus-vote fusion, then a per-carrier BPR refiner (run_post_queue.sh).
#
# Base = the DUMPED run config that run_cmp_eval.sh evaluated for Table II
# (custom_imports already lists carrier.oaem_op / seal_m2f_head /
# geofield_m2f_head; its test_dataloader points at test_offshore.json, and the
# queue overrides ann_file + evaluator + img prefix per split). custom_imports
# is re-declared here in full because a child config REPLACES the list, and
# carrier.mask_tta must be added for MaskTTAModel.
#
# TTA geometry = the validated PSeg recipe (cp_pseg_mstta_wide_votecons.py,
# univ_*_pseg_mstta.py): scales {416,512,608,704} x {hflip,noflip},
# keep_ratio=True, consensus band [2,6], vote_iou 0.5. Mask2Former-family
# predict(rescale=True) returns per-view masks at ori_shape, which is the
# invariant MaskTTAModel needs (checked on CPU by carrier/tests/cpu_test_post.py).
_base_ = 'E:/SAR/seal/runs/overnight/oaem/a0r_oaem_r50coco_pseg.py'

custom_imports = dict(
    imports=['carrier.seal_m2f_head', 'carrier.oaem_op',
             'carrier.geofield_m2f_head', 'carrier.mask_tta'],
    allow_failed_imports=False)

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
