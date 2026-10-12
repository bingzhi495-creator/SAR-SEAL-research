# Combined "kitchen-sink" model on HRSID: a1(resolution) + a2(GW matching) + refine + boundary + Mask-IoU + CBAM,
# in ONE training run (user strategy: train strong first, ablate later). Base = a1a2 (gives a1 config + a2
# assigner + WassMatch head); this swaps in SuperM2FHead (adds refine/boundary/maskiou via MRO), CBAM backbone,
# and the Mask-IoU fusion head. Every module is flag-gated so ablation = flip a flag (leave-one-out later).
# Matched control = a0_hrsid_seed0 (baseline .125) / a2 (.173) / a1a2 (.186); same 55ep protocol.
_base_ = './a1a2_m2f_hrsid.py'

custom_imports = dict(imports=['carrier.wass_match', 'carrier.combo_head'], allow_failed_imports=False)

model = dict(
    backbone=dict(type='ResNetCBAM', cbam_on=True),      # CBAM on the 4 ResNet levels (gamma-gate init 0 = identity)
    panoptic_head=dict(
        type='SuperM2FHead',                             # a2(inherited matching) + boundary + maskiou via MRO
        # refine OFF: refine_scale=1 + learn_refine=False -> mask stays 1/4 (the 1/2-res refine is the ~8GB memory
        # hog that pushes batch2 to 24GB OOM-edge on the 3090Ti). refine (diagnostic-aligned mask-quality) is
        # tested SEPARATELY (a0r_refine_hrsid). So combo = 5 modules: a1+a2+boundary+MaskIoU+CBAM (~16GB, reliable).
        refine_scale=1, learn_refine=False,
        boundary_weight=0.2,                             # boundary-Dice on matched masks
        maskiou_weight=0.5,                              # mask-IoU regression head
        maskiou_loss_type='bce'),
    panoptic_fusion_head=dict(
        type='MSIoUFusionHead',
        infer_iou_lambda=0.5),                           # rescore det = cls * mask_conf^(1-λ) * pred_iou^λ
)
train_cfg = dict(val_interval=5000)   # dense val (~every 2h) to watch metrics closely (user) + early combo-vs-a1a2
                                      # signal + early-abort if clearly < a1a2. val on official_test (overall).

# a1 (num_transformer_feat_level=4 + stride-4 pixel_decoder) + a2 (assigner GaussianWassersteinCost) inherited
# from _base_ a1a2_m2f_hrsid.py. Ablation configs (leave-one-out) will set the corresponding flag to 0/off.
