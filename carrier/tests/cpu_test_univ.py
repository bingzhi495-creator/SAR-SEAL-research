# -*- coding: utf-8 -*-
"""CPU-ONLY verification for the UNIVERSALITY MATRIX (run_univ_queue.sh).

Per carrier (5 total: HRSID MaskRCNN/Cascade, PSeg MaskRCNN/CondInst/YOLACT):

  (1) BUILD   the univ_*_mstta.py config builds and MODELS.build(cfg.model)
              constructs the detector on CPU.
  (2) CKPT    the trained checkpoint's state_dict loads with strict=False and
              has ZERO missing / ZERO unexpected keys (these are fine-tuned
              1-class checkpoints of exactly these architectures -- any
              mismatch is a bug, so we assert none).
  (3) WRAP    MODELS.build(dict(**cfg.tta_model, module=cfg.model)) yields a
              MaskTTAModel instance wrapping the right detector class.
  (4) FORWARD 1 real test image through the full TTA pipeline (8 views):
              a. per view: module.test_step -> pred_instances.masks is a
                 tensor with shape[-2:] == ori_shape (the invariant
                 MaskTTAModel needs under rescale=True);
              b. full wrapper.test_step -> one merged DetDataSample whose
                 masks are at ori_shape with aligned scores/labels/bboxes.

Run (CPU, from E:/SAR/seal):
  KMP_DUPLICATE_LIB_OK=TRUE CUDA_VISIBLE_DEVICES="" \
  python carrier/tests/cpu_test_univ.py
"""
from __future__ import annotations

import json
import os
import sys
import traceback

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ['CUDA_VISIBLE_DEVICES'] = ''

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

import carrier.mask_tta  # noqa: E402,F401  (registers MaskTTAModel)
from carrier.mask_tta import MaskTTAModel  # noqa: E402

CARRIERS = [
    # (name, univ cfg, ckpt, ann json for a real test image, expected detector)
    ('mrcnn_hrsid',
     'carrier/configs/univ_mrcnn_hrsid_mstta.py',
     'runs/bl_maskrcnn_hrsid/s0/best_coco_segm_mAP_iter_84040.pth',
     'data/hrsid/annotations/official_test.json',
     'MaskRCNN'),
    ('cascade_hrsid',
     'carrier/configs/univ_cascade_hrsid_mstta.py',
     'runs/bl_cascade_hrsid/s0/best_coco_segm_mAP_iter_84040.pth',
     'data/hrsid/annotations/official_test.json',
     'CascadeRCNN'),
    ('mrcnn_pseg',
     'carrier/configs/univ_mrcnn_pseg_mstta.py',
     'runs/retrain/maskrcnn/epoch_41.pth',
     'data/pseg_ssdd/annotations/test.json',
     'MaskRCNN'),
    ('condinst_pseg',
     'carrier/configs/univ_condinst_pseg_mstta.py',
     'runs/condinst/b0/best_coco_segm_mAP_iter_8352.pth',
     'data/pseg_ssdd/annotations/test.json',
     'CondInst'),
    ('yolact_pseg',
     'carrier/configs/univ_yolact_pseg_mstta.py',
     'runs/retrain/yolact/epoch_31.pth',
     'data/pseg_ssdd/annotations/test.json',
     'YOLACT'),
    # carrier 6/6 (2026-08-02): SOLOv2 = grid/dynamic-kernel family, no RoI
    # crop and NO box branch (emits dummy zero bboxes). Weakest HRSID carrier
    # (base 19.0/38.2/54.0) -> sharpest test of prediction P9.
    ('solov2_hrsid',
     'carrier/configs/univ_solov2_hrsid_mstta.py',
     'runs/bl_solov2_hrsid/s0/best_coco_segm_mAP_iter_84040.pth',
     'data/hrsid/annotations/official_test.json',
     'SOLOv2'),
]


def first_image(ann_path: str, img_dir: str):
    """(img_path, ori_h, ori_w) of the first image listed in a COCO json."""
    with open(ann_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)
    info = coco['images'][0]
    return (os.path.join(img_dir, info['file_name']),
            int(info['height']), int(info['width']))


def check_carrier(name, cfg_path, ckpt_path, ann_path, expected_cls):
    print(f'\n===== [{name}] =====')
    verdict = {'build': False, 'ckpt': False, 'wrap': False, 'forward': False}

    # (1) config builds + model constructs
    cfg = Config.fromfile(cfg_path)
    model = MODELS.build(cfg.model)
    model.eval()
    got_cls = type(model).__name__
    assert got_cls == expected_cls, \
        f'detector class {got_cls} != expected {expected_cls}'
    print(f'[{name}] BUILD ok: {cfg_path} -> {got_cls}')
    verdict['build'] = True

    # (2) checkpoint loads with ZERO mismatches
    assert os.path.isfile(ckpt_path), f'checkpoint missing: {ckpt_path}'
    ck = torch.load(ckpt_path, map_location='cpu')
    sd = ck.get('state_dict', ck)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f'[{name}] CKPT {os.path.basename(ckpt_path)}: '
          f'{len(sd)} keys, missing={len(missing)}, '
          f'unexpected={len(unexpected)}')
    if missing:
        print(f'    missing: {sorted(missing)[:8]}')
    if unexpected:
        print(f'    unexpected: {sorted(unexpected)[:8]}')
    assert not missing and not unexpected, \
        (f'fine-tuned ckpt must load 1:1; got {len(missing)} missing / '
         f'{len(unexpected)} unexpected keys')
    verdict['ckpt'] = True

    # (3) MaskTTAModel wraps this model class
    wrapper = MODELS.build(ConfigDict(**cfg.tta_model, module=cfg.model))
    assert isinstance(wrapper, MaskTTAModel), type(wrapper)
    wrapper.module.load_state_dict(sd, strict=True)
    wrapper.eval()
    print(f'[{name}] WRAP ok: MaskTTAModel(module={type(wrapper.module).__name__}), '
          f"merge={wrapper.tta_cfg.get('merge')}, "
          f"band=[{wrapper.tta_cfg.get('vote_min_support')},"
          f"{wrapper.tta_cfg.get('vote_max_support')}], "
          f"vote_iou={wrapper.tta_cfg.get('vote_iou')}")
    verdict['wrap'] = True

    # (4) 1-image CPU forward through the full 8-view TTA stack
    img_dir = cfg.test_dataloader['dataset']['data_prefix']['img']
    img_path, ori_h, ori_w = first_image(ann_path, img_dir)
    assert os.path.isfile(img_path), f'test image missing: {img_path}'
    pipeline = Compose(cfg.tta_pipeline)
    data = pipeline(dict(img_path=img_path, img_id=0, instances=[]))
    n_views = len(data['inputs'])
    assert n_views == 8, f'expected 8 TTA views, pipeline emitted {n_views}'
    batch = pseudo_collate([data])

    # (4a) per-view: masks present + at ori_shape (rescale=True invariant)
    for vi in range(n_views):
        view = {k: v[vi] for k, v in batch.items()}
        preds = wrapper.module.test_step(view)
        inst = preds[0].pred_instances
        meta = preds[0].metainfo
        masks = inst.get('masks', None)
        assert masks is not None, f'view{vi}: pred_instances.masks is None'
        assert tuple(meta['ori_shape']) == (ori_h, ori_w), \
            f"view{vi}: ori_shape {meta['ori_shape']} != json ({ori_h},{ori_w})"
        assert tuple(masks.shape[-2:]) == (ori_h, ori_w), \
            (f'view{vi}: masks {tuple(masks.shape)} not at ori_shape '
             f'({ori_h},{ori_w}) -- head does NOT rescale masks under TTA')
        bb = inst.get('bboxes', None)
        # SOLOv2 has no box branch: it emits all-zero dummy boxes. Report it
        # explicitly -- merge='vote' never reads them (it re-derives boxes from
        # the fused masks via mask2bbox), so zeros are harmless, but they must
        # be VISIBLE in the verification log rather than silently assumed.
        bb_note = ''
        if bb is not None and bb.numel():
            bb_note = (' bboxes=ALL_ZERO(dummy)' if float(bb.abs().max()) == 0
                       else ' bboxes=real')
        print(f'  view{vi}: img_shape={meta["img_shape"]} flip={meta.get("flip")}'
              f' n={masks.shape[0]} masks_hw={tuple(masks.shape[-2:])}'
              f' dtype={masks.dtype}{bb_note} OK')

    # (4b) merged output
    merged = wrapper.test_step(batch)
    assert len(merged) == 1
    mi = merged[0].pred_instances
    n = mi.masks.shape[0]
    assert tuple(mi.masks.shape[-2:]) == (ori_h, ori_w)
    assert mi.scores.shape[0] == n and mi.labels.shape[0] == n \
        and mi.bboxes.shape == (n, 4)
    # dtype contract: CocoMetric.results2json indexes self.cat_ids[label], so
    # a float label raises TypeError AFTER the whole inference pass. See the
    # dtype-normalisation comment in carrier/mask_tta.py.
    assert mi.labels.dtype in (torch.int32, torch.int64), \
        f'merged labels must be integer, got {mi.labels.dtype}'
    assert mi.masks.dtype == torch.bool, \
        f'merged masks must be bool, got {mi.masks.dtype}'
    if n:
        print(f'[{name}] FORWARD ok: merged {n} instances, masks '
              f'{tuple(mi.masks.shape)} dtype={mi.masks.dtype} at '
              f'ori_shape=({ori_h},{ori_w}), top score='
              f'{float(mi.scores.max()):.3f}, '
              f'bbox_max={float(mi.bboxes.abs().max()):.1f} '
              f'(re-derived from fused masks)')
    else:
        print(f'[{name}] FORWARD ok: merged 0 instances (empty path exercised)')
    verdict['forward'] = True
    return verdict


def empty_view_dtype_regression():
    """Model-free guard for the 2026-08-02 SOLOv2 crash.

    Feeds MaskTTAModel two views of one image: a normal one (bool masks, int64
    labels) and an EMPTY one typed exactly like SOLOv2's early-exit
    ``empty_results`` (float32 masks AND float32 labels). Before the dtype
    normalisation in ``_merge_single_sample`` the ``torch.cat`` promoted the
    pooled labels to float32, and CocoMetric later died with
    ``TypeError: list indices must be integers or slices, not numpy.float32``.
    """
    from mmengine.structures import InstanceData
    from mmdet.structures import DetDataSample

    H = W = 32
    meta = dict(img_id=0, ori_shape=(H, W), img_shape=(H, W), flip=False,
                flip_direction=None)

    full = DetDataSample(metainfo=meta)
    m = torch.zeros(2, H, W, dtype=torch.bool)
    m[0, 4:12, 4:12] = True
    m[1, 20:28, 20:28] = True
    full.pred_instances = InstanceData(
        masks=m, scores=torch.tensor([0.9, 0.7]),
        labels=torch.zeros(2, dtype=torch.long), bboxes=torch.zeros(2, 4))

    empty = DetDataSample(metainfo=dict(meta, flip=True,
                                        flip_direction='horizontal'))
    z = torch.zeros(0)                       # SOLOv2: cls_scores is float32
    empty.pred_instances = InstanceData(
        masks=z.new_zeros(0, H, W), scores=z.new_ones(0),
        labels=z.new_ones(0), bboxes=z.new_zeros(0, 4))

    # _merge_single_sample touches only self.tta_cfg, so skip __init__ (which
    # would insist on building a real detector we do not need here).
    wrapper = MaskTTAModel.__new__(MaskTTAModel)
    wrapper.tta_cfg = ConfigDict(
        nms=dict(type='nms', iou_threshold=0.6), max_per_img=100,
        merge='vote', vote_iou=0.5, vote_mask='consensus',
        vote_min_support=2, vote_max_support=6)
    out = wrapper._merge_single_sample([full, empty]).pred_instances
    assert out.labels.dtype == torch.int64, out.labels.dtype
    assert out.masks.dtype == torch.bool, out.masks.dtype
    assert out.masks.shape[0] == 2 and out.scores.shape[0] == 2
    print(f'EMPTY_VIEW_DTYPE_REGRESSION PASS: n={out.masks.shape[0]} '
          f'labels={out.labels.dtype} masks={out.masks.dtype}')
    return True


def main():
    # Optional argv filter: `python cpu_test_univ.py solov2_hrsid` checks only
    # the named carrier(s). No argv -> the full matrix, as before.
    want = set(sys.argv[1:])
    carriers = [c for c in CARRIERS if not want or c[0] in want]
    assert carriers, f'no carrier matches {sorted(want)}'
    results = {}
    try:
        results['empty_view_dtype'] = {'regression': empty_view_dtype_regression()}
    except Exception:
        traceback.print_exc()
        results['empty_view_dtype'] = {'FAILED': True}
    for name, cfg_path, ckpt_path, ann_path, exp_cls in carriers:
        try:
            results[name] = check_carrier(name, cfg_path, ckpt_path,
                                          ann_path, exp_cls)
        except Exception:
            traceback.print_exc()
            results[name] = {'FAILED': True}

    print('\n================ VERDICTS ================')
    all_ok = True
    for name, v in results.items():
        if v.get('FAILED'):
            print(f'UNIV_CPU_TEST {name}: FAIL')
            all_ok = False
        else:
            ok = all(v.values())
            all_ok &= ok
            print(f'UNIV_CPU_TEST {name}: '
                  + ' '.join(f'{k}={"PASS" if p else "FAIL"}'
                             for k, p in v.items())
                  + f'  -> {"PASS" if ok else "FAIL"}')
    print(f'UNIV_CPU_TEST OVERALL: {"ALL PASS" if all_ok else "FAIL"}')
    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
