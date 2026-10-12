# R6A: zero-init stem lateral -> high-res final mask (PSeg-SSDD, COCO-init).
#
# Mechanism: hires pure-Up2 was AP-neutral (51.7), bsamp +1.3, BPR (reads the
# ORIGINAL image) +10.9 inshore AP75 => stride-4 mask features already lost the
# boundary signal; new pre-downsampling signal must be injected. Here the R50
# stem feature (stride 2, pre-maxpool, 64ch, 256^2@512^2) enters the final-layer
# mask readout through a ControlNet-style ZERO 3x3 conv:
#     lat = ZeroConv3x3(proj_1x1(stem));  final logits = q . (Up2(F) + lat),
# implemented REASSOCIATED by linearity (train: Up2(q.F)+q.lat at 1/2 res;
# eval: Up4(q.F)+Up2(q.lat) at batch-input res) so the zero bypass is BITWISE
# exact in both modes (GATE-0: carrier/tests/gate0_stemlat.py, all numbers
# printed). Aux masks / attention masks stay stock 1/4.
#
# Protocol: identical to cp_m2f_cocoinit_pseg (COCO-init, 8352 it, bs4, 512^2,
# lr 2.5e-5, milestones 7400/8000). Only the backbone/head TYPES change; the
# new params (stem_proj normal init, stem_zero zero init) are absent from the
# COCO checkpoint and load strict=False (verified 1:1 for all COCO keys by the
# GATE-0 script, which prints missing/unexpected).
_base_ = './cp_m2f_cocoinit_pseg.py'

custom_imports = dict(
    imports=['carrier.stemlat_resnet', 'carrier.stemlat_head'],
    allow_failed_imports=False)

model = dict(
    backbone=dict(type='StemLatResNet', stemlat_push=True),
    panoptic_head=dict(
        type='StemLatM2FHead',
        refine_scale=2,          # final mask recomputed at 1/2 (256^2 @ 512^2)
        learn_refine=False,      # NO random-init refine module (COCO equivalence)
        stemlat_enable=True,
        stemlat_in_channels=64))

# load_from = 'pretrained/m2f_r50_coco.pth' is inherited from the base.
