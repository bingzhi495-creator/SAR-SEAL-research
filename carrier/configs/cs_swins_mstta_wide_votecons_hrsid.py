# Swin-S base + the training-free recovery stack: wide 8-view msTTA {640,800,960,1088}x{flip,
# noflip} + vote-consensus merge (max_support=6). Mirrors c0_v2_mstta_wide_voteconsensus but on
# the Swin-S HRSID fine-tune (stock M2F head — no MSIoU/infer_iou_lambda knob; vote-consensus
# operates on masks/scores/labels only, so nothing else is needed).
_base_ = './cs_swins_m2f_cocoinit_hrsid.py'

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
