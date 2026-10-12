# CAST-hard on the PURE c0 (COCO-init) strong base -- the headline pure-data-axis arm.
# Precheck-gated (precheck_cast.py): only train if frozen c0 genuinely misses the pasted ships
# (recall@IoU0.5<0.5 OR median IoU<0.3). Control = the existing c0 3-seed (segm 0.457 / inshore 0.277
# / offshore 0.598) -- NO retrain needed. This arm changes exactly ONE thing vs c0: the train ann_file
# (original 3642 imgs + 120 CAST-hard augmented copies = 3762 imgs / 12609 anns; +145 pasted hard ships).
# CRITICAL: inherits c0_m2f_cocoinit_hrsid (pure c0), NOT a1a2 -- the a1a2 chain in
# cast_hard_c0_m2f_hrsid.py is -2.3 harmful and confounds the data axis. Everything else byte-inherits c0.
# Cast images resolve via junction JPEGImages1/cast_images -> data/hrsid/cast_images.
_base_ = './c0_m2f_cocoinit_hrsid.py'

load_from = 'pretrained/m2f_r50_coco.pth'

train_dataloader = dict(
    dataset=dict(ann_file='E:/SAR/seal/data/hrsid/annotations/train_cast_hard.json'))
