# c0_maskiou_v2 + WIDER small-object-biased MULTI-SCALE TTA (G1).
# CLONE of c0_v2_mstta_hrsid.py: the ONLY change is the Resize scale set in
# transforms[0]. Replaces the measured 6-view {736,800,864} square pyramid with a
# WIDER, UPSAMPLE-BIASED 4-scale square pyramid {640,800,960,1088} x {flip,noflip}
# = 8 views. Motivation: HRSID ships are small/elongated; upsampling (960,1088 >
# native 800) enlarges tiny ships so the fusion head resolves them, while 640
# keeps a downsampled view for context. All keep_ratio=True on native 800x800
# square chips -> each square scale maps EXACTLY to that square size.
#
# GEOMETRY UNCHANGED vs the 6-view baseline: the fusion head predict(rescale=True)
# maps every view's masks back to ori_shape=(800,800), so all views share the same
# HxW and _merge_single_sample only undoes the flip. tta_model / flip branch /
# LoadAnnotations / PackDetInputs meta_keys are byte-identical to the baseline.
# Sweep infer_iou_lambda in {0.2,0.3} (larger upsampled scales add near-shore
# clutter, so the IoU-rescoring knob may want to lean harder on predicted IoU).
# The original c0_v2_mstta_hrsid.py is left intact to preserve the A/B reference.
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
