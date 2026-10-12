# SeedDINO with REAL NQS evidence seeds (the actual seeding gate / A2-equivalent).
# Train + test with precomputed NQS seeds; compare inshore box AP vs the
# no-crop DINO baseline. Expected to beat baseline 0.474 inshore meaningfully
# (G2 82% inshore recall + 70% marginal coverage), well below oracle's 0.994.
_base_ = './dino_r50_pseg_nocrop.py'

custom_imports = dict(imports=['carrier.seed_dino', 'carrier.load_seeds'],
                      allow_failed_imports=False)

model = dict(type='SeedDINO', seed_source='pred', max_seeds=50, set_content=False)

seeds_dir = 'E:/SAR/seal/data/seeds/'
image_size = (512, 512)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Resize', scale=image_size, keep_ratio=True),
    dict(type='LoadSeeds', seeds_json=seeds_dir + 'seeds_pseg_train.json'),  # after flip+resize
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor',
                    'flip', 'flip_direction', 'seeds')),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=image_size, keep_ratio=True),
    dict(type='LoadSeeds', seeds_json=seeds_dir + 'seeds_pseg_test.json'),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor',
                    'flip', 'flip_direction', 'seeds')),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader
