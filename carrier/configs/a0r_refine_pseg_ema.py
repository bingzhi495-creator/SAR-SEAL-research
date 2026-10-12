# A0R + EMA of weights. begin_iter=2000 skips the noisy warmup of the 100-query
# transformer (lr 2.5e-5, 928 imgs). decay = 1 - momentum = 0.9998. Near-free,
# stacks with everything; EMAHook auto-swaps EMA weights for val + ckpt save.
_base_ = './a0r_refine_pseg.py'

custom_hooks = [
    dict(type='EMAHook', ema_type='ExponentialMovingAverage',
         momentum=0.0002, priority=49, begin_iter=2000),
]
