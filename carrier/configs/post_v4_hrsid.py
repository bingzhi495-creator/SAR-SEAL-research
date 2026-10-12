# POST-REVIEW step 5 (docs/TGRS_SUBMISSION_REVIEW.md sec.7, R2 5.2): number of
# TTA views. V=4 = scales {800, 960} x {hflip, noflip}, everything else
# byte-identical to the final HRSID stack (c0_v2_mstta_wide_voteconsensus_hrsid.py:
# soup x consensus vote, band [2,6], vote_iou 0.5; the queue passes
# infer_iou_lambda=0.0 and vote_max_support=6 exactly as run_bpr_queue.sh step4a).
# A child config REPLACES tta_pipeline wholesale, so the full pipeline is
# restated with only the Resize list shortened.
_base_ = './c0_v2_mstta_wide_voteconsensus_hrsid.py'

tta_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale=(800, 800), keep_ratio=True),
                dict(type='Resize', scale=(960, 960), keep_ratio=True),
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
