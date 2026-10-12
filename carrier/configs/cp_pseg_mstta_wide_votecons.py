# PSeg-SSDD recovery stack: wide 8-view msTTA + consensus-vote fusion, mirroring the
# HRSID final config but with scales proportional to the 512^2 protocol:
# HRSID {640,800,960,1088}/800 = {0.80,1.00,1.20,1.36} -> 512*ratios ~= {416,512,608,704}
# (multiples of 32, keep_ratio). Consensus band [2,6], vote_iou 0.5 unchanged.
_base_ = './cp_m2f_cocoinit_pseg.py'

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
