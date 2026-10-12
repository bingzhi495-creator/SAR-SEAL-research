# DAOM-Seed on HRSID SCENE-LEVEL split (leakage-free).
#
# ============================ TODO / BLOCKER ============================
# This config CANNOT be finalized yet: there is NO DINO base config for HRSID.
# A search of carrier/configs/ for hrsid found only Mask2Former bases
# (a0_m2f_r50_hrsid.py, a0_m2f_r50_hrsid_scene.py) and refine heads
# (a0r_*_hrsid*.py) -- none is a DINO detector config, so there is nothing for
# DAOMSeedDINO (a DINO subclass) to inherit the backbone/neck/head/schedule from.
#
# REQUIRED BEFORE USE -- create the DINO-HRSID-scene base, mirroring the PSeg
# pair (dino_r50_pseg.py -> dino_r50_pseg_nocrop.py -> seeddino_pred_pseg.py):
#   1. carrier/configs/dino_r50_hrsid.py
#        _base_ = 'mmdet::dino/dino-4scale_r50_8xb2-12e_coco.py'
#        data_root = 'E:/SAR/seal/data/hrsid/'  (img prefix per HRSID layout)
#        model = dict(num_queries=300, bbox_head=dict(num_classes=1))
#        metainfo = dict(classes=('ship',))
#        ann_file train/test = official HRSID annotations
#        (copy the iter-based schedule / evaluator pattern from dino_r50_pseg.py)
#   2. carrier/configs/dino_r50_hrsid_scene.py
#        _base_ = './dino_r50_hrsid.py'
#        train ann_file = .../annotations/scene_train.json
#        val/test ann_file = .../annotations/scene_test.json
#        (mirror a0_m2f_r50_hrsid_scene.py which already does the scene swap)
#   3. carrier/configs/seeddino_pred_hrsid_scene.py
#        _base_ = './dino_r50_hrsid_scene.py' + the LoadSeeds pipeline below,
#        then set this file's _base_ to it and drop the inline pipeline.
#
# SEED FORMAT NOTE: data/seeds/seeds_hrsid_train.jsonl is JSONL
#   ({"stem": ..., "seeds": [[x1,y1,x2,y2], ...]} per line), whereas LoadSeeds
#   reads a single JSON dict {stem: [...]}. Either (a) convert the jsonl to a
#   {stem: seeds} json (as was done for PSeg: seeds_pseg_train.json), or
#   (b) extend LoadSeeds to parse jsonl. Path to use once converted:
#       data/seeds/seeds_hrsid_train.jsonl  (train)
#       data/seeds/seeds_hrsid_test.jsonl   (val/test)
# =======================================================================
#
# Once dino_r50_hrsid_scene.py exists, switch the _base_ below to it and the
# model/pipeline blocks are ready to go. Left intentionally pointing at the
# PSeg DAOM config so this file imports cleanly for review, but it MUST be
# repointed before any HRSID run (the dataset/ann_file are PSeg here).

_base_ = './daom_seed_pseg.py'  # TODO: repoint to ./seeddino_pred_hrsid_scene.py

# --- HRSID DAOM model kwargs (ready; only the _base_/dataset/seeds are TODO) ---
seeds_dir = 'E:/SAR/seal/data/seeds/'
image_size = (800, 800)  # HRSID native chip size; adjust to the HRSID DINO base

model = dict(
    type='DAOMSeedDINO',
    seed_source='pred',
    max_seeds=200,
    aux_num_queries=300,
    k_o2m=6,
    lambda_aux=1.0,
    density_alpha=1.0,
    k_min=8,
    k_max=200,
)

# Pipeline sketch for when the HRSID-scene DINO base is wired up. Uses
# data/seeds/seeds_hrsid_train.jsonl per the spec (convert jsonl->json first, or
# extend LoadSeeds to read jsonl).
#
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(type='RandomFlip', prob=0.5),
#     dict(type='Resize', scale=image_size, keep_ratio=True),
#     dict(type='LoadSeeds', seeds_json=seeds_dir + 'seeds_hrsid_train.jsonl'),
#     dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
#     dict(type='PackDetInputs',
#          meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
#                     'scale_factor', 'flip', 'flip_direction', 'seeds')),
# ]
