# F2c: AREA-ADAPTIVE consensus-vote fusion. Same mask-IoU clustering as
# vote/voteseed (keeps low-mask-IoU parallel-hull neighbours as separate
# clusters), but the PER-CLUSTER fused mask is chosen by object regime:
#   - SMALL seed (area < vote_area_thr=1024 px = 32x32) AND enough views
#     (cluster_size >= vote_min_support=2) -> pixelwise MAJORITY mean>=vote_frac
#     (near-shore ships are small + buried in speckle -> averaging denoises);
#   - otherwise -> SEED mask verbatim (large/offshore ships are clean; averaging
#     misaligned views erodes their boundary).
# Goal: get the best of both measured modes at once -- majority's inshore win
# (HRSID inshore 0.391) without its offshore erosion (0.559), keeping seed's
# offshore-sharp result (official 0.535 / offshore 0.648). vote_area_thr is the
# key knob and likely wants a sweep (e.g. {512,1024,2048}).
_base_ = './c0_v2_mstta_wide_hrsid.py'

custom_imports = dict(
    imports=['carrier.maskiou_head', 'carrier.mask_tta'],
    allow_failed_imports=False)

tta_model = dict(
    type='MaskTTAModel',
    tta_cfg=dict(
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100,
        merge='vote',
        vote_iou=0.5,
        vote_mask='adaptive',
        vote_area_thr=1024,
        vote_min_support=2))
