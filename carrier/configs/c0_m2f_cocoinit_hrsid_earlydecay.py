# EARLY-DECAY continuation of the plateaued long run. The 200ep run's val converged by ~45ep
# (0.435->0.448->0.449, +0.001 over 30->45ep) — full-LR epochs stopped helping, so idling to
# ep176 for the scheduled decay is wasted. Instead: load_from the converged best checkpoint
# (0.449) and run a SHORT fresh schedule that decays LR quickly (x0.1 @4000, x0.1 @10000) to
# capture the post-decay settle. load_from (NOT --resume) so the NEW milestones actually apply
# (mmengine --resume would restore the old scheduler's milestones from the checkpoint state).
# ~14000 iters (~5 epochs, ~2h) vs ~2 days to reach the ep176 decay.
_base_ = './c0_m2f_cocoinit_hrsid.py'

load_from = 'E:/SAR/seal/runs/c0_200ep/s0/best_coco_segm_mAP_iter_126090.pth'

max_iters = 14000
train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters, val_interval=2800)
param_scheduler = dict(end=max_iters, milestones=[4000, 10000])
default_hooks = dict(
    checkpoint=dict(interval=2800, by_epoch=False, max_keep_ckpts=3,
                    save_best='coco/segm_mAP', rule='greater'))
