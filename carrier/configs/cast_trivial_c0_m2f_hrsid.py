# CAST-trivial on the c0 (COCO-init) STRONG base -- the ONLY regime-valid CAST test (weak-base cast_trivial
# would answer a question the paper can't use, per workflow wlntrrlq1). = a1a2 + CAST-trivial data + COCO init.
# Control = c0a1a2 (no CAST). trivial-paste GATE-0 proves targeting != volume. Precheck-gated: only train if
# c0's median IoU on pasted instances <0.3 (c0 genuinely misses them). CAST touches 22%+B-mask-recall, NOT the
# 62% mask-quality residual -> honest coin-flip, expected +0..3.
_base_ = './cast_trivial_m2f_hrsid.py'
load_from = 'pretrained/m2f_r50_coco.pth'
