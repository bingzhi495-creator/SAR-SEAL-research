# c0 @ 500ep with IMPROVED RECIPE (cosine LR + warmup + EMA) for AutoDL/A800 -- the paper's central number.
# Rationale (measured): c0 val rose monotonically to the milestone (0.425->0.456) = UNDERTRAINED @55ep, so
# the longer schedule is the #1 lever; cosine annealing (vs the stock step-decay) is a small transferable
# recipe gain (+0.2-0.5 typ), EMA is a free floor-safe add. FAIRNESS: this recipe must be used for the
# baseline AND any mechanism arm alike -- it lifts the whole tier, orthogonal to the mechanism delta.
# Protocol matched to O2Former (500ep). Assumes batch8 on A800: HRSID 3642/8 ~= 455 iter/ep -> 500ep ~= 227500.
_base_ = './a0_m2f_r50_hrsid.py'
load_from = 'pretrained/m2f_r50_coco.pth'   # COCO-head init (+15.4), the biggest lever

max_iters = 227500
train_dataloader = dict(batch_size=8, num_workers=8)
optim_wrapper = dict(optimizer=dict(lr=1e-4))   # O2Former LR @ batch8
train_cfg = dict(max_iters=max_iters, val_interval=11375)

# COSINE with linear warmup (replaces stock MultiStepLR step-decay):
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-3, by_epoch=False, begin=0, end=1000),          # warmup 1000 iter
    dict(type='CosineAnnealingLR', by_epoch=False, begin=1000, end=max_iters, eta_min_ratio=1e-2),
]
# EMA (free floor-safe; decay tuned for the long schedule):
custom_hooks = [dict(type='EMAHook', ema_type='ExpMomentumEMA', momentum=0.0002, update_buffers=True, priority=49)]
default_hooks = dict(checkpoint=dict(interval=8000, by_epoch=False, max_keep_ckpts=3))
