# c0 CONTROL (critical, found 2026-07-03): the whole HRSID family (a0/a2/a1a2/a2loss) trains from
# ImageNet-backbone + RANDOM heads (inherited mmdet COCO config has no load_from). b1's composed ckpt changes
# TWO things vs that baseline: backbone->SARDet AND heads->COCO-pretrained. c0 isolates the second: full COCO
# M2F init (same as b1 but with the ORIGINAL COCO backbone). b1 - c0 = the TRUE SAR-backbone contribution;
# c0 - a0 = the COCO-heads/undertraining contribution. Without c0, b1's gain is confounded.
_base_ = './a0_m2f_r50_hrsid.py'

load_from = 'pretrained/m2f_r50_coco.pth'
