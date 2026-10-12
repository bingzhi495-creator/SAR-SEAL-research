# Round-2 #1: FreqFusion (TPAMI2024 real code) in the pixel decoder, on COCO-init R50.
# Targets high-IoU mask quality (AP75/AP90) via adaptive low/high-pass content-aware
# upsampling. val on inshore. carafe = mmcv compiled op.
_base_ = './a0r_r50_coco_pseg.py'

custom_imports = dict(
    imports=['carrier.seal_m2f_head', 'carrier.freqfusion', 'carrier.freqfusion_decoder'],
    allow_failed_imports=False)

model = dict(panoptic_head=dict(
    refine_scale=2,
    pixel_decoder=dict(type='FreqFusionPixelDecoder', ff_feat_channels=256)))

_ins = 'E:/SAR/seal/data/pseg_ssdd/annotations/test_inshore.json'
val_dataloader = dict(num_workers=0, persistent_workers=False, dataset=dict(ann_file=_ins))
val_evaluator = dict(ann_file=_ins)
default_hooks = dict(checkpoint=dict(interval=1044, by_epoch=False, max_keep_ckpts=2,
                                     save_best='coco/segm_mAP', rule='greater'))
