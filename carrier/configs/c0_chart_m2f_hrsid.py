# c0+CHART: Clutter-Hard Auxiliary Reinforced Targeting on the COCO-init strong base.
# Diagnosis: ~22% of near-shore ships are A-class COMPLETE MISSES (no query fires); the
# stock one-to-one Hungarian match gives each hard GT only a single, easily-out-voted
# positive query. Precheck PASSED: hard positives are cheaply identifiable (crowding AUC
# 0.749) and 52.5% of unfired misses carry a weak-but-real full-res response that more
# positive supervision can amplify. CHART adds an AUXILIARY one-to-many positive
# assignment -- each GT ALSO supervises its top-k best UNMATCHED queries -- with the
# extra budget CONCENTRATED on clutter-hard near-shore GTs (per-GT difficulty weight w_g
# from NN-crowding + small size). Training-side only; inference untouched (zero cost).
# NO a1/a2/refine/boundary/maskiou. GATE-0: aux_weight=0 byte-identical to c0.
_base_ = './c0_m2f_cocoinit_hrsid.py'
custom_imports = dict(imports=['carrier.chart_head'], allow_failed_imports=False)
model = dict(
    # aux_weight lowered 1.0->0.1: at parity (1.0) the loss-side top-k one-to-many on the SAME query set forced
    # over-firing -> precision collapse (val crashed to 0.18 vs c0 0.44). 0.1 = gentle auxiliary supervision.
    panoptic_head=dict(type='CHARTM2FHead', aux_weight=0.1, aux_topk=3, hard_focus=1.0))
load_from = 'pretrained/m2f_r50_coco.pth'
