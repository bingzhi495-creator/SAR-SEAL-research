# -*- coding: utf-8 -*-
"""CPU-ONLY sanity + GATE-0 tests for R12 depth-consistency features.

NO GPU. Checks:
  (a) GATE-0 head+fusion: a REAL (tiny) MSIoUM2FHead + MSIoUFusionHead on a
      random forward -- export_depth_stats OFF vs ON: every common output
      field byte-identical; OFF leaves NO _depth_stats on the data sample and
      NO depth_stats on the instances; ON attaches (Q,5)/(K,5).
  (b) depth_stats value + alignment correctness:
      b1 hand-computed per-query IoU/score values on synthetic 3-layer
         outputs, incl. the both-empty->1.0 convention and the 2x-refined
         final layer (SealM2FHead) downsample path;
      b2 fusion alignment: unique per-query masks -> every kept instance's
         depth row equals its SOURCE query's row (mapping recovered from the
         mask itself, independent of topk order), incl. truncation
         (max_per_image < Q).
  (c) mask_tta seed inheritance: fused cluster inherits the SEED member's
      depth stats; vote_stats (C, 8) hand-verified; backward compat (any
      view missing depth -> (C, 6)); export_vote_stats OFF + depth present
      -> no vote_stats; empty input -> (0, 8) / (0, 6).
  (d) scic_rerank feature-count autodetect: 6-col stack -> 14-dim npz
      (R11-identical), 8-col stack -> 16-dim npz with DEPTH_STAT_NAMES;
      16-dim fit/predict recovers a depth-driven target; apply preserves the
      candidate set; 14-dim ckpt on an 8-col stack is REFUSED (hard guard).

Run:
  KMP_DUPLICATE_LIB_OK=TRUE CUDA_VISIBLE_DEVICES="" \
  python carrier/tests/cpu_test_r12.py
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from pycocotools import mask as maskUtils

from mmdet.structures import DetDataSample

from carrier.mask_tta import MaskTTAModel
from carrier import scic_rerank
from carrier.maskiou_head import MSIoUFusionHead, MSIoUM2FHead
from carrier.scic_rerank import (DEPTH_STAT_NAMES, GEOM_NAMES,
                                 RERANK_FEAT_NAMES, feat_names_for, fit_gbdt)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def build_tiny_head(test_cfg):
    """Real MSIoUM2FHead, shrunk (Q=6, C=32, 3 decoder layers, 1-cls ship)."""
    torch.manual_seed(0)
    head = MSIoUM2FHead(
        in_channels=[8, 16, 32, 64],
        feat_channels=32,
        out_channels=32,
        num_things_classes=1,
        num_stuff_classes=0,
        num_queries=6,
        num_transformer_feat_level=3,
        refine_scale=1,
        learn_refine=False,       # match c0_maskiou_v2_m2f_hrsid.py
        pixel_decoder=ConfigDict(
            type='MSDeformAttnPixelDecoder',
            num_outs=3,
            norm_cfg=dict(type='GN', num_groups=8),
            act_cfg=dict(type='ReLU'),
            encoder=dict(
                num_layers=1,
                layer_cfg=dict(
                    self_attn_cfg=dict(embed_dims=32, num_heads=2,
                                       num_levels=3, num_points=2,
                                       dropout=0.0, batch_first=True),
                    ffn_cfg=dict(embed_dims=32, feedforward_channels=64,
                                 num_fcs=2, ffn_drop=0.0,
                                 act_cfg=dict(type='ReLU', inplace=True)))),
            positional_encoding=dict(num_feats=16, normalize=True)),
        enforce_decoder_input_project=False,
        positional_encoding=ConfigDict(num_feats=16, normalize=True),
        transformer_decoder=ConfigDict(
            return_intermediate=True,
            num_layers=3,
            layer_cfg=dict(
                self_attn_cfg=dict(embed_dims=32, num_heads=2, dropout=0.0,
                                   batch_first=True),
                cross_attn_cfg=dict(embed_dims=32, num_heads=2, dropout=0.0,
                                    batch_first=True),
                ffn_cfg=dict(embed_dims=32, feedforward_channels=64,
                             num_fcs=2, ffn_drop=0.0,
                             act_cfg=dict(type='ReLU', inplace=True))),
            init_cfg=None),
        loss_cls=ConfigDict(type='CrossEntropyLoss', use_sigmoid=False,
                            loss_weight=2.0, reduction='mean',
                            class_weight=[1.0] * 1 + [0.1]),
        loss_mask=ConfigDict(type='CrossEntropyLoss', use_sigmoid=True,
                             reduction='mean', loss_weight=5.0),
        loss_dice=ConfigDict(type='DiceLoss', use_sigmoid=True, activate=True,
                             reduction='mean', naive_dice=True, eps=1.0,
                             loss_weight=5.0),
        train_cfg=None,
        test_cfg=test_cfg)
    head.eval()
    return head


def build_fusion(max_per_image, export_depth=False):
    cfg = ConfigDict(panoptic_on=False, semantic_on=False, instance_on=True,
                     max_per_image=max_per_image)
    if export_depth:
        cfg.export_depth_stats = True
    return MSIoUFusionHead(num_things_classes=1, num_stuff_classes=0,
                           test_cfg=cfg, loss_panoptic=None, init_cfg=None)


def make_inputs():
    torch.manual_seed(1)
    x = [torch.randn(1, c, s, s) for c, s in
         zip([8, 16, 32, 64], [16, 8, 4, 2])]
    ds = DetDataSample()
    ds.set_metainfo(dict(batch_input_shape=(64, 64), img_shape=(64, 64),
                         ori_shape=(64, 64), flip=False))
    return x, [ds]


def make_tta_model(merge, **extra):
    m = MaskTTAModel.__new__(MaskTTAModel)
    cfg = dict(nms=dict(type='nms', iou_threshold=0.6), max_per_img=100,
               merge=merge)
    cfg.update(extra)
    m.tta_cfg = ConfigDict(cfg)
    return m


def make_view(masks_bool, scores, labels, depth=None, flip=False,
              flip_direction=None):
    inst = InstanceData()
    inst.masks = masks_bool
    inst.scores = scores
    inst.labels = labels
    if depth is not None:
        inst.depth_stats = depth
    ds = DetDataSample()
    ds.pred_instances = inst
    ds.set_metainfo(dict(flip=flip, flip_direction=flip_direction))
    return ds


def _square(r0, r1, c0, c1, H=100, W=100):
    m = np.zeros((H, W), dtype=bool)
    m[r0:r1, c0:c1] = True
    return m


def _rle(mask_bool):
    return maskUtils.encode(np.asfortranarray(mask_bool.astype(np.uint8)))


# --------------------------------------------------------------------------- #
# (a) GATE-0 on the real tiny head + fusion
# --------------------------------------------------------------------------- #
def check_gate0_head():
    print('=' * 70)
    print('(a) GATE-0 head+fusion: export_depth_stats OFF == byte-identical')
    print('=' * 70)
    head = build_tiny_head(ConfigDict())              # flag ABSENT
    x, samples_off = make_inputs()
    with torch.no_grad():
        cls_off, mask_off = head.predict(x, samples_off)
    no_attr = not hasattr(samples_off[0], '_depth_stats')

    head.test_cfg = ConfigDict(export_depth_stats=True)   # flag ON, same head
    x2, samples_on = make_inputs()                    # identical inputs (seeded)
    with torch.no_grad():
        cls_on, mask_on = head.predict(x2, samples_on)
    head_same = (torch.equal(cls_off, cls_on)
                 and torch.equal(mask_off, mask_on))
    has_attr = hasattr(samples_on[0], '_depth_stats')
    dshape = tuple(samples_on[0]._depth_stats.shape) if has_attr else None
    print(f'  head outputs byte-identical OFF vs ON: {head_same}')
    print(f'  OFF: no _depth_stats attr: {no_attr}; '
          f'ON: attr present {has_attr} shape={dshape} (expect (6, 5))')

    fusion = build_fusion(max_per_image=6)
    with torch.no_grad():
        r_off = fusion.predict(cls_off, mask_off, samples_off,
                               rescale=False)[0]['ins_results']
        r_on = fusion.predict(cls_on, mask_on, samples_on,
                              rescale=False)[0]['ins_results']
    fused_same = (torch.equal(r_off.masks, r_on.masks)
                  and torch.equal(r_off.scores, r_on.scores)
                  and torch.equal(r_off.labels, r_on.labels)
                  and torch.equal(r_off.bboxes, r_on.bboxes))
    off_clean = 'depth_stats' not in r_off
    on_has = 'depth_stats' in r_on
    kshape = tuple(r_on.depth_stats.shape) if on_has else None
    print(f'  fused common fields byte-identical: {fused_same}')
    print(f'  OFF: no instance depth_stats: {off_clean}; ON: present '
          f'{on_has} shape={kshape} (expect (K, 5), K={len(r_on.scores)})')
    finite = bool(torch.isfinite(r_on.depth_stats).all()) if on_has else False
    in01 = (bool((r_on.depth_stats[:, :2].min() >= 0)
                 and (r_on.depth_stats[:, :2].max() <= 1)) if on_has else
            False)
    print(f'  ON depth values finite: {finite}; IoU cols in [0,1]: {in01}')
    ok = (head_same and no_attr and has_attr and dshape == (6, 5)
          and fused_same and off_clean and on_has and kshape[1] == 5
          and finite and in01)
    print(f'\n  (a) {"PASS" if ok else "FAIL"}')
    return ok, head


# --------------------------------------------------------------------------- #
# (b) depth_stats values + fusion alignment
# --------------------------------------------------------------------------- #
def check_depth_values(head):
    print('\n' + '=' * 70)
    print('(b) depth_stats: hand-computed values + kept-instance alignment')
    print('=' * 70)
    B, Q, h, w = 1, 3, 8, 8
    POS, NEG = 3.0, -3.0

    def mk(cover):  # cover: list of (r0, r1, c0, c1) per query
        m = torch.full((B, Q, h, w), NEG)
        for q, box in enumerate(cover):
            if box is not None:
                r0, r1, c0, c1 = box
                m[0, q, r0:r1, c0:c1] = POS
        return m

    # q0: L7 4px vs L8 4px overlap 2 -> 2/6; L8 == L9 -> 1.0
    # q1: L7,L8 empty -> 1.0 (both-empty convention); L9 nonempty -> 0.0
    # q2: L7 16px, L8 = right half 8px (subset) -> 8/16; L9 disjoint -> 0.0
    m7 = mk([(0, 2, 0, 2), None, (4, 8, 0, 4)])
    m8 = mk([(0, 2, 1, 3), None, (4, 8, 2, 4)])
    m9_lo = mk([(0, 2, 1, 3), (6, 8, 6, 8), (0, 2, 6, 8)])
    # final layer at 2x res (SealM2FHead refine): nearest-up preserves the
    # binarization after the head's bilinear downsample (exact for 2x blocks)
    m9 = F.interpolate(m9_lo, scale_factor=2, mode='nearest')
    c = torch.zeros((B, Q, 2))
    c[0, :, 0] = torch.tensor([2.0, -1.0, 0.5])   # ship logit per layer L7
    c7, c8, c9 = c, c * 0.5, c * 2.0
    # lists have num_layers+1 entries; only the last 3 are read
    pad_c, pad_m = torch.zeros((B, Q, 2)), mk([None, None, None])
    with torch.no_grad():
        d = head._depth_stats([pad_c, c7, c8, c9], [pad_m, m7, m8, m9])[0]

    def fg(logits):
        return F.softmax(logits, dim=-1)[..., :-1].amax(-1)[0]

    exp_iou78 = torch.tensor([2 / 6, 1.0, 8 / 16])
    exp_iou89 = torch.tensor([1.0, 0.0, 0.0])
    exp = torch.stack([exp_iou78, exp_iou89, fg(c7), fg(c8), fg(c9)], dim=-1)
    v_ok = bool(torch.allclose(d, exp, atol=1e-6))
    print(f'  computed iou_L7L8 = {d[:, 0].tolist()}')
    print(f'  expected          = {exp_iou78.tolist()}')
    print(f'  computed iou_L8L9 = {d[:, 1].tolist()} (2x-refined L9 path)')
    print(f'  expected          = {exp_iou89.tolist()}')
    print(f'  scores match softmax-excl-bg reference: '
          f'{bool(torch.allclose(d[:, 2:], exp[:, 2:], atol=1e-6))}')
    print(f'  full (Q,5) match: {v_ok}')

    # b2: alignment through instance_postprocess (mapping recovered from the
    # MASK, so the check is independent of topk internal order)
    align_ok = True
    for max_per in (4, 2):
        Q2 = 4
        mask_cls = torch.zeros((Q2, 2))
        mask_cls[:, 0] = torch.tensor([3.0, 1.0, 2.0, 0.5])  # distinct ranks
        mask_pred = torch.full((Q2, 8, 8), -5.0)
        for q in range(Q2):
            mask_pred[q, 2 * q:2 * q + 2, :] = 5.0        # unique row band
        depth = torch.arange(Q2 * 5, dtype=torch.float32).reshape(Q2, 5)
        fusion = build_fusion(max_per_image=max_per)
        with torch.no_grad():
            res = fusion.instance_postprocess(mask_cls, mask_pred, None,
                                              depth)
        got_n = len(res.scores)
        this_ok = got_n == max_per and 'depth_stats' in res
        for i in range(got_n):
            q = int(torch.nonzero(res.masks[i][:, 0])[0]) // 2  # row -> query
            this_ok = this_ok and bool(torch.equal(res.depth_stats[i],
                                                   depth[q]))
        kept_q = sorted(int(torch.nonzero(res.masks[i][:, 0])[0]) // 2
                        for i in range(got_n))
        print(f'  max_per_image={max_per}: kept queries {kept_q}, every '
              f'instance row == its query row: {this_ok}')
        align_ok = align_ok and this_ok

    ok = v_ok and align_ok
    print(f'\n  (b) {"PASS" if ok else "FAIL"}')
    return ok


# --------------------------------------------------------------------------- #
# (c) TTA seed inheritance
# --------------------------------------------------------------------------- #
def check_tta_inheritance():
    print('\n' + '=' * 70)
    print('(c) mask_tta: fused instance inherits SEED depth stats -> (C, 8)')
    print('=' * 70)
    A0 = _square(10, 30, 10, 30)              # 400 px, view 0, score 0.9 SEED
    A1 = _square(10, 30, 10, 28)              # 360 px, view 1, score 0.7
    C = _square(70, 90, 70, 90)               # singleton, view 0, score 0.8
    iou_a = 360.0 / 400.0
    dA0 = torch.tensor([0.80, 0.90, 0.30, 0.40, 0.70])
    dC = torch.tensor([0.20, 0.40, 0.60, 0.55, 0.50])
    dA1 = torch.tensor([0.99, 0.98, 0.97, 0.96, 0.95])  # must be IGNORED

    def views(with_depth=(True, True)):
        v0 = make_view(torch.from_numpy(np.stack([A0, C])).bool(),
                       torch.tensor([0.90, 0.80]),
                       torch.zeros(2, dtype=torch.long),
                       depth=torch.stack([dA0, dC]) if with_depth[0] else None)
        v1 = make_view(torch.from_numpy(A1[None]).bool().flip(dims=[-1]),
                       torch.tensor([0.70]), torch.zeros(1, dtype=torch.long),
                       depth=dA1[None] if with_depth[1] else None,
                       flip=True, flip_direction='horizontal')
        return [v0, v1]

    out = make_tta_model('vote', vote_mask='consensus',
                         export_vote_stats=True).\
        _merge_single_sample(views()).pred_instances
    vs = out.vote_stats if 'vote_stats' in out else None
    exp = torch.tensor([
        [2.0, 2.0, iou_a, iou_a, 0.90, 0.90 - 0.70,
         float((dA0[0] + dA0[1]) / 2), float(dA0[4] - dA0[2])],
        [1.0, 1.0, 0.0, 0.0, 0.80, 0.0,
         float((dC[0] + dC[1]) / 2), float(dC[4] - dC[2])]])
    shape_ok = vs is not None and tuple(vs.shape) == (2, 8)
    val_ok = shape_ok and bool(torch.allclose(vs, exp, atol=1e-6))
    if shape_ok:
        print(f'  row0 (cluster A, seed=view0 A0) = {vs[0].tolist()}')
        print(f'  expected                        = {exp[0].tolist()}')
        print(f'  row1 (singleton C)              = {vs[1].tolist()}')
        print(f'  expected                        = {exp[1].tolist()}')
    print(f'  vote_stats (2, 8): {shape_ok}; SEED (not member) depth '
          f'inherited + hand-values match: {val_ok}')

    # backward compat: one view without depth -> (C, 6), values = R11
    out6 = make_tta_model('vote', vote_mask='consensus',
                          export_vote_stats=True).\
        _merge_single_sample(views(with_depth=(True, False))).pred_instances
    compat_ok = tuple(out6.vote_stats.shape) == (2, 6) and \
        bool(torch.allclose(out6.vote_stats, exp[:, :6], atol=1e-6))
    print(f'  one view missing depth -> (2, 6) with identical consensus '
          f'cols: {compat_ok}')

    # export flag OFF + depth present -> NO vote_stats, fields untouched
    out_off = make_tta_model('vote', vote_mask='consensus').\
        _merge_single_sample(views()).pred_instances
    off_ok = ('vote_stats' not in out_off
              and torch.equal(out_off.masks, out.masks)
              and torch.equal(out_off.scores, out.scores))
    print(f'  export_vote_stats OFF + depth present: no vote_stats, common '
          f'fields identical: {off_ok}')

    # empty input: (0, 8) with depth, (0, 6) without
    def empty_views(with_depth):
        return [make_view(torch.zeros((0, 50, 50), dtype=torch.bool),
                          torch.zeros(0), torch.zeros(0, dtype=torch.long),
                          depth=(torch.zeros((0, 5)) if with_depth
                                 else None))]
    e8 = make_tta_model('vote', export_vote_stats=True).\
        _merge_single_sample(empty_views(True)).pred_instances
    e6 = make_tta_model('vote', export_vote_stats=True).\
        _merge_single_sample(empty_views(False)).pred_instances
    empty_ok = (tuple(e8.vote_stats.shape) == (0, 8)
                and tuple(e6.vote_stats.shape) == (0, 6))
    print(f'  empty input: with depth {tuple(e8.vote_stats.shape)} '
          f'(expect (0, 8)); without {tuple(e6.vote_stats.shape)} '
          f'(expect (0, 6)): {empty_ok}')

    ok = val_ok and compat_ok and off_ok and empty_ok
    print(f'\n  (c) {"PASS" if ok else "FAIL"}')
    return ok


# --------------------------------------------------------------------------- #
# (d) scic_rerank 14/16-dim autodetect
# --------------------------------------------------------------------------- #
def build_world(tmp, n_stats):
    """1 image, 2 GT; stack pkl with 4 candidates + (4, n_stats) vote_stats."""
    H = W = 200
    g1 = _square(20, 40, 20, 40, H, W)
    g2 = _square(100, 130, 100, 140, H, W)

    def seg(mask):
        r = _rle(mask)
        return dict(size=[int(v) for v in r['size']],
                    counts=r['counts'].decode())

    ann = dict(
        images=[dict(id=1, file_name='im1.png', width=W, height=H)],
        annotations=[
            dict(id=i + 1, image_id=1, category_id=1, segmentation=seg(g),
                 area=float(maskUtils.area(_rle(g))),
                 bbox=[float(v) for v in maskUtils.toBbox(_rle(g))],
                 iscrowd=0) for i, g in enumerate([g1, g2])],
        categories=[dict(id=1, name='ship')])
    ann_path = os.path.join(tmp, f'ann{n_stats}.json')
    with open(ann_path, 'w') as f:
        json.dump(ann, f)

    cand = [g1, g2, g1, _square(160, 180, 160, 180, H, W)]
    base6 = np.array([[6, 6, 0.9, 0.8, 0.90, 0.1],
                      [2, 2, 0.7, 0.6, 0.20, 0.05],
                      [1, 1, 0.0, 0.0, 0.10, 0.0],
                      [1, 1, 0.0, 0.0, 0.05, 0.0]], dtype=np.float32)
    stats = base6 if n_stats == 6 else np.concatenate(
        [base6, np.array([[0.95, 0.30], [0.80, 0.10],
                          [0.40, -0.20], [0.10, -0.50]], np.float32)], axis=1)
    entry = dict(img_id=1, pred_instances=dict(
        masks=[_rle(c) for c in cand],
        scores=torch.tensor([0.90, 0.20, 0.10, 0.05]),
        labels=torch.zeros(4, dtype=torch.long),
        vote_stats=torch.from_numpy(stats)))
    pkl_path = os.path.join(tmp, f'stack{n_stats}.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump([entry], f)

    base = [dict(image_id=1, category_id=1, segmentation=seg(g1), score=0.90),
            dict(image_id=1, category_id=1, segmentation=seg(g1), score=0.10)]
    base_path = os.path.join(tmp, f'base{n_stats}.json')
    with open(base_path, 'w') as f:
        json.dump(base, f)
    return ann_path, pkl_path, base_path, base


def check_rerank_autodetect():
    print('\n' + '=' * 70)
    print('(d) scic_rerank: 14/16-dim autodetect, fit/predict, apply guard')
    print('=' * 70)
    tmp = tempfile.mkdtemp(prefix='r12_test_')

    dims_ok = True
    for n_stats, want in ((6, 14), (8, 16)):
        ann, pkl_p, _, _ = build_world(tmp, n_stats)
        npz = os.path.join(tmp, f'feats{n_stats}.npz')
        scic_rerank.main(['harvest', '--pred', pkl_p, '--ann', ann,
                          '--out', npz, '--floor', '0.01'])
        d = np.load(npz, allow_pickle=True)
        names = [str(n) for n in d['feat_names']]
        got = d['feats'].shape[1]
        name_ok = names == feat_names_for(n_stats)
        depth_named = (n_stats == 6) or names[6:8] == DEPTH_STAT_NAMES
        print(f'  {n_stats}-col stack -> harvest dim={got} (expect {want}), '
              f'feat_names correct: {name_ok and depth_named}')
        dims_ok = dims_ok and got == want and name_ok and depth_named

    # 16-dim fit/predict: target driven by seed_depth_iou_mean (col 6)
    rng = np.random.RandomState(0)
    n = 4000
    X = rng.randn(n, 16).astype(np.float32)
    y = np.clip(0.5 + 0.35 * X[:, 6] + 0.15 * X[:, 4]
                + 0.05 * rng.randn(n), 0, 1).astype(np.float32)
    m16 = fit_gbdt(X[:3000], y[:3000], seed=0, max_iter=100)
    pv = m16.predict(X[3000:])
    r = float(np.corrcoef(pv, y[3000:])[0, 1])
    fit_ok = pv.shape == (1000,) and r > 0.8
    print(f'  16-dim GBDT holdout pearson={r:.4f} (>0.8, target uses the '
          f'depth col): {fit_ok}')

    # apply with a 16-dim ckpt on the 8-col stack: candidate set preserved
    ann8, pkl8, base_path, base = build_world(tmp, 8)
    ck16 = os.path.join(tmp, 'rr16.pkl')
    with open(ck16, 'wb') as f:
        pickle.dump(dict(model=m16, alpha=0.5, feat_names=feat_names_for(8),
                         floor=0.01), f)
    out16 = os.path.join(tmp, 'r12.json')
    scic_rerank.main(['apply', '--pred', pkl8, '--base-json', base_path,
                      '--ann', ann8, '--model', ck16, '--out', out16,
                      '--tag', 'mini16', '--variant', 'R12', '--no-eval'])
    with open(out16) as f:
        r12 = json.load(f)
    apply_ok = (len(r12) == len(base)
                and all(a['segmentation'] == b['segmentation']
                        for a, b in zip(r12, base))
                and all(0.0 <= dd['score'] <= 1.0 for dd in r12)
                and sum(abs(a['score'] - b['score']) > 1e-9
                        for a, b in zip(r12, base)) >= 1)
    print(f'  16-dim apply: count/masks preserved, scores in [0,1], '
          f'>=1 rescored: {apply_ok}')

    # guard: 14-dim ckpt on the 8-col stack must be refused
    m14 = fit_gbdt(rng.randn(500, 14).astype(np.float32),
                   rng.rand(500).astype(np.float32), seed=0, max_iter=20)
    ck14 = os.path.join(tmp, 'rr14.pkl')
    with open(ck14, 'wb') as f:
        pickle.dump(dict(model=m14, alpha=0.5,
                         feat_names=RERANK_FEAT_NAMES, floor=0.01), f)
    try:
        scic_rerank.main(['apply', '--pred', pkl8, '--base-json', base_path,
                          '--ann', ann8, '--model', ck14,
                          '--out', os.path.join(tmp, 'bad.json'),
                          '--tag', 'bad', '--no-eval'])
        guard_ok = False
    except AssertionError as exc:
        guard_ok = 'expects 14 features' in str(exc)
        print(f'  14-dim ckpt on 8-col stack REFUSED: {guard_ok} '
              f'({str(exc)[:70]}...)')

    ok = dims_ok and fit_ok and apply_ok and guard_ok
    print(f'\n  (d) {"PASS" if ok else "FAIL"}')
    return ok


def main():
    a, head = check_gate0_head()
    b = check_depth_values(head)
    c = check_tta_inheritance()
    d = check_rerank_autodetect()
    print('\n' + '=' * 70)
    print(f'OVERALL: {"ALL PASS" if all([a, b, c, d]) else "FAIL"}  '
          f'[(a) GATE-0 head+fusion={"PASS" if a else "FAIL"}, '
          f'(b) depth values+alignment={"PASS" if b else "FAIL"}, '
          f'(c) TTA seed inheritance={"PASS" if c else "FAIL"}, '
          f'(d) rerank autodetect={"PASS" if d else "FAIL"}]')
    print('=' * 70)
    sys.exit(0 if all([a, b, c, d]) else 1)


if __name__ == '__main__':
    main()
