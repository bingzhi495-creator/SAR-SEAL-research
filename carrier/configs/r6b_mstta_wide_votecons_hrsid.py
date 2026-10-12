# R6B (min-footprint head) + the full training-free recovery stack (wide 8-view TTA +
# consensus-vote). Tests whether the mechanism-level base gain (+0.5/+0.7/+0.9) SURVIVES
# the stack (variance-level gains historically get absorbed; mechanism-level may not).
_base_ = './r6b_minfoot_hrsid.py'
custom_imports = dict(imports=['carrier.minfoot_m2f_head', 'carrier.mask_tta'],
                      allow_failed_imports=False)
tta_model = dict(
    type='MaskTTAModel',
    tta_cfg=dict(nms=dict(type='nms', iou_threshold=0.6), max_per_img=100,
                 merge='vote', vote_iou=0.5, vote_mask='consensus',
                 vote_min_support=2, vote_max_support=6))
tta_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, backend_args=None),
    dict(type='TestTimeAug', transforms=[
        [dict(type='Resize', scale=(640, 640), keep_ratio=True),
         dict(type='Resize', scale=(800, 800), keep_ratio=True),
         dict(type='Resize', scale=(960, 960), keep_ratio=True),
         dict(type='Resize', scale=(1088, 1088), keep_ratio=True)],
        [dict(type='RandomFlip', prob=1.0, direction='horizontal'),
         dict(type='RandomFlip', prob=0.0)],
        [dict(type='LoadAnnotations', with_bbox=True, with_mask=True)],
        [dict(type='PackDetInputs',
              meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                         'scale_factor', 'flip', 'flip_direction'))]])]
