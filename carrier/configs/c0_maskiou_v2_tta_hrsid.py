# c0_maskiou_v2 + INSTANCE-SEG TTA. mmdet's DetTTAModel merges only bboxes+scores and
# ASSERTS masks is None ("TTA of instance segmentation does not support now"), so we use
# carrier.mask_tta.MaskTTAModel: it flips each predicted mask back along flip_direction,
# pools (mask,score,label) across views, derives bboxes via mask2bbox, runs class-wise NMS
# reusing tta_cfg.nms, and KEEPS the surviving masks.
#
# KEY: the fusion head predicts masks with rescale=True -> every augmented view (flip AND
# multi-scale) returns masks already mapped back to ori_shape, so all views share the same
# HxW = ori image size; TTA only needs to undo the flip.
#
# Starts FLIP-ONLY (single scale 800x800, flip/noflip) to validate the mask merge. A
# multi-scale block [(736,736),(800,800),(864,864)] can be swapped in once flip-only is
# confirmed >= baseline.
_base_ = './c0_maskiou_v2_m2f_hrsid.py'

custom_imports = dict(
    imports=['carrier.maskiou_head', 'carrier.mask_tta'],
    allow_failed_imports=False)

tta_model = dict(
    type='MaskTTAModel',
    tta_cfg=dict(nms=dict(type='nms', iou_threshold=0.6), max_per_img=100))

# All three Resize scales use keep_ratio=True on square 800x800 chips.
img_scale = (800, 800)

tta_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [dict(type='Resize', scale=img_scale, keep_ratio=True)],
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
