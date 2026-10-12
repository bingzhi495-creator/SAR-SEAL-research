# -*- coding: utf-8 -*-
"""CPU-ONLY preflight for run_post_queue.sh (post-review experiments,
docs/TGRS_SUBMISSION_REVIEW.md sec.7).  One step name per invocation; the
queue requires the line ``POST_PREFLIGHT <step> PASS`` before it spends GPU.

  hp     step 1  train-side hyper-parameter re-scan: post_hp_holdout_hrsid.py
                 builds, its tta_cfg is byte-identical to the final stack cfg
                 (c0_v2_mstta_wide_voteconsensus_hrsid.py), the knob names it
                 sweeps (vote_max_support / vote_iou) are the ones
                 carrier/mask_tta.py reads, the holdout json exists with 728
                 images whose files are on disk, and the 8-view pipeline is the
                 wide {640,800,960,1088} x flip set.
  oaem   step 2  post_oaem_pseg_mstta.py builds the OAEM Mask2Former on CPU,
                 the Table II checkpoint loads with ZERO missing / unexpected
                 keys, MaskTTAModel wraps it, and one real PSeg test image goes
                 through the 8-view pipeline: per-view masks at ori_shape and a
                 merged sample with aligned scores/labels/bboxes.  Also checks
                 the train-image root used for the BPR train dump.
  band   step 3  tools/bpr_apply_band.py --selftest passes, bpr_apply exposes
                 --band, and the HRSID stack dumps + refiner are on disk.
  seed   step 4  bpr_refiner exposes --seed, runs/bpr/patches is present.
  views  step 5  post_v2_hrsid.py / post_v4_hrsid.py emit 2 / 4 views on a real
                 HRSID test image and keep the final stack's tta_cfg.

Run (CPU, from E:/SAR/seal):
  KMP_DUPLICATE_LIB_OK=TRUE CUDA_VISIBLE_DEVICES=-1 \
  python carrier/tests/cpu_test_post.py hp|oaem|band|seed|views
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

ROOT = 'E:/SAR/seal'
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch  # noqa: E402

torch.set_grad_enabled(False)

from mmengine.config import Config, ConfigDict  # noqa: E402
from mmengine.dataset import Compose, pseudo_collate  # noqa: E402
from mmengine.registry import MODELS, init_default_scope  # noqa: E402

from mmdet.utils import register_all_modules  # noqa: E402

register_all_modules()
init_default_scope('mmdet')

FINAL_STACK = 'carrier/configs/c0_v2_mstta_wide_voteconsensus_hrsid.py'
EXPECTED_TTA_CFG = dict(nms=dict(type='nms', iou_threshold=0.6), max_per_img=100,
                        merge='vote', vote_iou=0.5, vote_mask='consensus',
                        vote_min_support=2, vote_max_support=6)


def _plain(d):
    """ConfigDict -> plain dict (recursive) for equality checks."""
    if isinstance(d, dict):
        return {k: _plain(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_plain(v) for v in d]
    return d


def scales_of(cfg):
    tta = cfg.tta_pipeline[1]
    assert tta['type'] == 'TestTimeAug', tta['type']
    return [tuple(t['scale']) for t in tta['transforms'][0]]


def n_views_of(cfg, img_path):
    pipeline = Compose(cfg.tta_pipeline)
    data = pipeline(dict(img_path=img_path, img_id=0, instances=[]))
    return len(data['inputs']), data


def first_image(ann_path, img_dir, k=0):
    with open(ann_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)
    info = coco['images'][k]
    return (os.path.join(img_dir, info['file_name']), int(info['height']),
            int(info['width']), len(coco['images']))


def source_has(path, needles):
    with open(path, encoding='utf-8') as f:
        src = f.read()
    return {n: (n in src) for n in needles}


# --------------------------------------------------------------------------
def check_hp():
    cfg = Config.fromfile('carrier/configs/post_hp_holdout_hrsid.py')
    base = Config.fromfile(FINAL_STACK)
    # (1) knob names / values identical to the final stack
    got, ref = _plain(cfg.tta_model), _plain(base.tta_model)
    assert got == ref, f'tta_model differs from final stack:\n{got}\n{ref}'
    assert _plain(cfg.tta_model['tta_cfg']) == EXPECTED_TTA_CFG, got
    print('[hp] tta_model identical to', FINAL_STACK, '->', got['tta_cfg'])
    # (2) the swept keys are the ones MaskTTAModel reads
    keys = source_has('carrier/mask_tta.py',
                      ["tta_cfg.get('vote_max_support'", "tta_cfg.get('vote_iou'",
                       "tta_cfg.get('vote_min_support'", "tta_cfg.get('vote_mask'"])
    assert all(keys.values()), keys
    print('[hp] carrier/mask_tta.py reads', sorted(keys))
    # (3) the fusion-head knob the queue pins (infer_iou_lambda=0.0) exists
    fh = cfg.model['panoptic_fusion_head']
    assert 'infer_iou_lambda' in fh, fh
    print('[hp] model.panoptic_fusion_head.infer_iou_lambda present '
          f"(cfg default {fh['infer_iou_lambda']}, queue pins 0.0)")
    # (4) holdout json wired into loader + evaluator, files on disk
    hold = cfg.test_dataloader['dataset']['ann_file']
    assert hold.endswith('train_holdout20.json'), hold
    assert cfg.test_evaluator['ann_file'] == hold, cfg.test_evaluator['ann_file']
    assert os.path.isfile(hold), f'holdout missing: {hold}'
    img_dir = cfg.test_dataloader['dataset']['data_prefix']['img']
    for k in range(3):
        p, h, w, n = first_image(hold, img_dir, k)
        assert os.path.isfile(p), f'holdout image missing: {p}'
    assert n == 728, f'holdout has {n} images, expected 728 (20% of 3642)'
    assert cfg.test_dataloader['num_workers'] == 0
    print(f'[hp] holdout {hold}: {n} images, img_dir={img_dir}, first files present')
    # (5) 8-view wide pipeline unchanged
    sc = scales_of(cfg)
    assert sc == [(640, 640), (800, 800), (960, 960), (1088, 1088)], sc
    p, h, w, _ = first_image(hold, img_dir, 0)
    nv, _ = n_views_of(cfg, p)
    assert nv == 8, nv
    print(f'[hp] pipeline scales {sc} x flip -> {nv} views on {os.path.basename(p)}')
    # (6) model builds on CPU (no forward: same detector as the validated stack)
    model = MODELS.build(cfg.model)
    print('[hp] BUILD ok ->', type(model).__name__,
          '/', type(model.panoptic_fusion_head).__name__)
    return True


def check_oaem():
    cfg_path = 'carrier/configs/post_oaem_pseg_mstta.py'
    ckpt = 'runs/overnight/oaem/best_coco_segm_mAP_iter_8352.pth'
    cfg = Config.fromfile(cfg_path)
    assert 'carrier.oaem_op' in cfg.custom_imports['imports'], cfg.custom_imports
    assert 'carrier.mask_tta' in cfg.custom_imports['imports'], cfg.custom_imports
    assert _plain(cfg.tta_model['tta_cfg']) == EXPECTED_TTA_CFG, _plain(cfg.tta_model)
    assert scales_of(cfg) == [(416, 416), (512, 512), (608, 608), (704, 704)]
    import carrier.mask_tta  # noqa: F401
    import carrier.oaem_op  # noqa: F401
    import carrier.seal_m2f_head  # noqa: F401
    import carrier.geofield_m2f_head  # noqa: F401
    from carrier.mask_tta import MaskTTAModel
    # (1) build
    model = MODELS.build(cfg.model)
    model.eval()
    head_cls = type(model.panoptic_head).__name__
    assert head_cls == 'OAEMMask2FormerHead', head_cls
    print(f'[oaem] BUILD ok: {type(model).__name__} / panoptic_head={head_cls}')
    # (2) ckpt 1:1
    assert os.path.isfile(ckpt), ckpt
    ck = torch.load(ckpt, map_location='cpu')
    sd = ck.get('state_dict', ck)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f'[oaem] CKPT {os.path.basename(ckpt)}: {len(sd)} keys, '
          f'missing={len(missing)} unexpected={len(unexpected)}')
    assert not missing and not unexpected, (sorted(missing)[:8], sorted(unexpected)[:8])
    # (3) wrap
    wrapper = MODELS.build(ConfigDict(**cfg.tta_model, module=cfg.model))
    assert isinstance(wrapper, MaskTTAModel), type(wrapper)
    wrapper.module.load_state_dict(sd, strict=True)
    wrapper.eval()
    print(f"[oaem] WRAP ok: merge={wrapper.tta_cfg.get('merge')} "
          f"band=[{wrapper.tta_cfg.get('vote_min_support')},"
          f"{wrapper.tta_cfg.get('vote_max_support')}] "
          f"vote_iou={wrapper.tta_cfg.get('vote_iou')}")
    # (4) forward, one real inshore test image
    img_dir = cfg.test_dataloader['dataset']['data_prefix']['img']
    assert img_dir.rstrip('/').endswith('JPEGImages_test'), img_dir
    ann = 'data/pseg_ssdd/annotations/test_inshore.json'
    img_path, ori_h, ori_w, _ = first_image(ann, img_dir, 0)
    assert os.path.isfile(img_path), img_path
    nv, data = n_views_of(cfg, img_path)
    assert nv == 8, nv
    batch = pseudo_collate([data])
    for vi in range(nv):
        view = {k: v[vi] for k, v in batch.items()}
        preds = wrapper.module.test_step(view)
        inst, meta = preds[0].pred_instances, preds[0].metainfo
        masks = inst.get('masks', None)
        assert masks is not None, f'view{vi}: masks None'
        assert tuple(masks.shape[-2:]) == (ori_h, ori_w), \
            (vi, tuple(masks.shape), (ori_h, ori_w))
        print(f"  view{vi}: img_shape={meta['img_shape']} flip={meta.get('flip')} "
              f'n={masks.shape[0]} masks_hw={tuple(masks.shape[-2:])} OK')
    merged = wrapper.test_step(batch)
    mi = merged[0].pred_instances
    n = mi.masks.shape[0]
    assert tuple(mi.masks.shape[-2:]) == (ori_h, ori_w)
    assert mi.scores.shape[0] == n and mi.labels.shape[0] == n and mi.bboxes.shape == (n, 4)
    assert mi.labels.dtype in (torch.int32, torch.int64), mi.labels.dtype
    assert mi.masks.dtype == torch.bool, mi.masks.dtype
    top = float(mi.scores.max()) if n else float('nan')
    print(f'[oaem] FORWARD ok: merged {n} instances at ({ori_h},{ori_w}), top score {top:.3f}')
    # (5) train-side dump inputs for the per-carrier BPR
    tr_ann = 'data/pseg_ssdd/annotations/train.json'
    tr_img = 'E:/PSeg_SSDD/voc_style/JPEGImages_train/'
    p, _, _, n_tr = first_image(tr_ann, tr_img, 0)
    assert os.path.isfile(p), p
    print(f'[oaem] BPR train dump inputs ok: {tr_ann} ({n_tr} images) under {tr_img}')
    base = 'runs/overnight/oaem/a0r_oaem_r50coco_pseg.py'
    bcfg = Config.fromfile(base)
    assert 'carrier.oaem_op' in bcfg.custom_imports['imports']
    print(f'[oaem] base dump cfg ok: {base}')
    return True


def check_band():
    r = subprocess.run([sys.executable, 'tools/bpr_apply_band.py', '--selftest'],
                       capture_output=True, text=True, cwd=ROOT)
    tail = (r.stdout + r.stderr).strip().splitlines()[-1] if (r.stdout + r.stderr).strip() else ''
    print('[band] selftest:', tail)
    assert r.returncode == 0 and 'PASS' in tail, tail
    flags = source_has('carrier/bpr_apply.py',
                       ["add_argument('--band'", "add_argument('--tag'", "add_argument('--thr'"])
    assert all(flags.values()), flags
    assert "band = make_band(mask, band_r)" in open('carrier/bpr_apply.py', encoding='utf-8').read()
    for f in ['runs/bpr/refiner.pth', 'runs/bpr/test_inshore.pkl',
              'runs/bpr/test_official.pkl', 'runs/bpr/test_offshore.pkl']:
        assert os.path.isfile(f), f
    print('[band] bpr_apply flags ok, HRSID stack dumps + refiner present')
    return True


def check_seed():
    flags = source_has('carrier/bpr_refiner.py',
                       ["add_argument('--seed'", 'torch.manual_seed(args.seed)',
                        'BPR_REFINER_DONE'])
    assert all(flags.values()), flags
    assert os.path.isfile('runs/bpr/patches/meta.json'), 'runs/bpr/patches/meta.json'
    meta = json.load(open('runs/bpr/patches/meta.json', encoding='utf-8'))
    print('[seed] bpr_refiner --seed present; patches meta keys:',
          sorted(meta)[:8] if isinstance(meta, dict) else type(meta).__name__)
    return True


def check_views():
    base = Config.fromfile(FINAL_STACK)
    ann = 'data/hrsid/annotations/test_inshore.json'
    img_dir = base.test_dataloader['dataset']['data_prefix']['img']
    p, h, w, _ = first_image(ann, img_dir, 0)
    assert os.path.isfile(p), p
    for v, exp_sc in ((2, [(800, 800)]), (4, [(800, 800), (960, 960)])):
        cfg = Config.fromfile(f'carrier/configs/post_v{v}_hrsid.py')
        assert _plain(cfg.tta_model) == _plain(base.tta_model), _plain(cfg.tta_model)
        assert _plain(cfg.model) == _plain(base.model), 'model differs from final stack'
        assert scales_of(cfg) == exp_sc, scales_of(cfg)
        nv, _ = n_views_of(cfg, p)
        assert nv == v, (v, nv)
        print(f'[views] post_v{v}_hrsid.py: scales {exp_sc} x flip -> {nv} views, '
              'tta_model/model identical to final stack')
    return True


CHECKS = dict(hp=check_hp, oaem=check_oaem, band=check_band, seed=check_seed,
              views=check_views)


def main():
    steps = sys.argv[1:] or list(CHECKS)
    rc = 0
    for s in steps:
        if s not in CHECKS:
            print(f'POST_PREFLIGHT {s} UNKNOWN (choose from {sorted(CHECKS)})')
            rc = 1
            continue
        try:
            ok = CHECKS[s]()
        except Exception:
            traceback.print_exc()
            ok = False
        print(f'POST_PREFLIGHT {s} {"PASS" if ok else "FAIL"}', flush=True)
        rc |= 0 if ok else 1
    return rc


if __name__ == '__main__':
    sys.exit(main())
