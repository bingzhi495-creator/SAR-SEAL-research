# NMS-attribution control for the +3 TTA gain. Identical to c0_maskiou_v2_tta EXCEPT the TestTimeAug has ONLY
# the no-flip branch -> MaskTTAModel runs its class-wise bbox-NMS merge on a SINGLE view (no augmentation).
# This isolates how much of the flip-TTA gain (in .312/off .488/offs .622 @lambda0) is actually just NMS
# dedup of M2F's overlapping query masks (baseline M2F fusion takes top-100 by score with NO bbox-NMS).
#   noflip+NMS ~= flip-TTA  -> the gain is NMS dedup (cheaper: no 2x inference, no flip)
#   noflip+NMS ~= baseline  -> the gain is genuinely from flip-TTA
_base_ = './c0_maskiou_v2_m2f_hrsid.py'

custom_imports = dict(
    imports=['carrier.maskiou_head', 'carrier.mask_tta'],
    allow_failed_imports=False)

tta_model = dict(
    type='MaskTTAModel',
    tta_cfg=dict(nms=dict(type='nms', iou_threshold=0.6), max_per_img=100))

img_scale = (800, 800)

tta_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [dict(type='Resize', scale=img_scale, keep_ratio=True)],
            [dict(type='RandomFlip', prob=0.0)],          # single no-flip view -> isolate NMS merge
            [dict(type='LoadAnnotations', with_bbox=True, with_mask=True)],
            [
                dict(
                    type='PackDetInputs',
                    meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                               'scale_factor', 'flip', 'flip_direction')),
            ],
        ]),
]
