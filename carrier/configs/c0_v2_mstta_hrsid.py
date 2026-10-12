# c0_maskiou_v2 + MULTI-SCALE TTA. Extends the validated flip-only MaskTTAModel to
# 3 scales {736,800,864} x {flip,noflip} = 6 views. Inference-side only, zero training.
# Rationale: 5 training-side modules all subsumed by the strong COCO base; the ONLY real
# gains (v2 rescoring +0.017, flip-TTA +3) are inference-side, so we probe the highest-
# yield axis. All scales keep_ratio=True; rescale=True maps every view's masks back to
# ori_shape so the HxW matches and TTA only undoes the flip.
_base_ = './c0_maskiou_v2_m2f_hrsid.py'

custom_imports = dict(
    imports=['carrier.maskiou_head', 'carrier.mask_tta'],
    allow_failed_imports=False)

tta_model = dict(
    type='MaskTTAModel',
    tta_cfg=dict(nms=dict(type='nms', iou_threshold=0.6), max_per_img=100))

tta_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale=(736, 736), keep_ratio=True),
                dict(type='Resize', scale=(800, 800), keep_ratio=True),
                dict(type='Resize', scale=(864, 864), keep_ratio=True),
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
