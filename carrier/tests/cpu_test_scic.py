# -*- coding: utf-8 -*-
"""CPU-ONLY sanity + GATE-0 tests for the Self-Corrective Inference Cascade.

NO GPU. Checks:
  (a) GATE-0 mask_tta: export_vote_stats ABSENT -> vote merge byte-identical
      to the PRE-EDIT vote branch (replicated inline) for consensus /
      majority / seed, multi-view flipped delivery included; no vote_stats
      attribute; default bbox_nms path carries no vote_stats even with the
      flag ON (vote-only feature); empty input with flag ON -> (0, 6) stats.
  (b) export flag ON: common fields byte-identical to flag OFF; vote_stats
      present, (C, 6), rows EXACTLY equal to hand-computed consensus features
      on a 2-view case (support, n_views, mean/min pairwise IoU, seed score,
      score spread), truncation-aligned.
  (c) R10 harvest label correctness on synthetic candidates: GT missed by
      kept preds -> pool candidate labeled 1; GT already covered by a kept
      pred -> duplicate pool candidate labeled 0; background candidate -> 0;
      features finite, patches 48x48.
  (d) verifier forward/backward: loss backprop, nonzero grads, <500k params.
  (e) reranker fit/predict: recovers a synthetic monotone target
      (holdout pearson > 0.8), predict shape/range.
  (f) apply-safety end-to-end on a synthetic mini-world (image file + ann +
      stack pkl + base json): R10 rescued scores STRICTLY < 0.35 kept-floor,
      base dets byte-unchanged, overlap-suppression works; R11 preserves the
      candidate set exactly (same count/order/masks, only scores change,
      all in [0, 1]).

Run:
  KMP_DUPLICATE_LIB_OK=TRUE CUDA_VISIBLE_DEVICES="" \
  python carrier/tests/cpu_test_scic.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import cv2
import numpy as np
import torch
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from pycocotools import mask as maskUtils

from mmdet.structures import DetDataSample
from mmdet.structures.mask import mask2bbox

from carrier.mask_tta import MaskTTAModel
from carrier import scic_rescue, scic_rerank
from carrier.scic_rescue import (RESCUE_FEAT_NAMES, build_verifier,
                                 harvest_image, train_verifier)
from carrier.scic_rerank import RERANK_FEAT_NAMES, fit_gbdt, rerank_dets


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def make_model(merge, **extra):
    m = MaskTTAModel.__new__(MaskTTAModel)
    cfg = dict(nms=dict(type='nms', iou_threshold=0.6), max_per_img=100,
               merge=merge)
    cfg.update(extra)
    m.tta_cfg = ConfigDict(cfg)
    return m


def make_sample(masks_bool, scores, labels, flip=False, flip_direction=None):
    inst = InstanceData()
    inst.masks = masks_bool
    inst.scores = scores
    inst.labels = labels
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


def old_vote_reference(merged_masks, merged_scores, merged_labels, n_views,
                       vote_iou, vote_frac, vote_gamma, vote_mask,
                       vote_min_support, vote_max_support, max_per_img):
    """Verbatim replication of the PRE-EDIT vote branch."""
    K = merged_masks.shape[0]
    flat = merged_masks.bool().reshape(K, -1).float()
    area = flat.sum(dim=1)
    inter = flat @ flat.t()
    union = area[:, None] + area[None, :] - inter
    iou = torch.where(union > 0, inter / union, torch.zeros_like(inter))
    order = torch.argsort(merged_scores, descending=True)
    used = torch.zeros(K, dtype=torch.bool)
    fused_masks, fused_scores, fused_labels = [], [], []
    for idx in order.tolist():
        if used[idx]:
            continue
        same = merged_labels == merged_labels[idx]
        members = (~used) & same & (iou[idx] >= vote_iou)
        members[idx] = True
        used |= members
        cluster_size = int(members.sum())
        if vote_mask == 'seed':
            fused = merged_masks[idx].bool()
        elif vote_mask == 'consensus':
            if vote_min_support <= cluster_size <= vote_max_support:
                member_masks = merged_masks[members].bool().float()
                fused = member_masks.mean(dim=0) >= vote_frac
            else:
                fused = merged_masks[idx].bool()
        else:
            member_masks = merged_masks[members].bool().float()
            fused = member_masks.mean(dim=0) >= vote_frac
        fused_masks.append(fused)
        support = (cluster_size / n_views) ** vote_gamma
        fused_scores.append(merged_scores[members].max() * support)
        fused_labels.append(merged_labels[idx])
    fused_masks = torch.stack(fused_masks, dim=0)
    fused_scores = torch.stack(fused_scores)
    fused_labels = torch.stack(fused_labels)
    keep = torch.argsort(fused_scores, descending=True)[:max_per_img]
    fused_masks = fused_masks[keep]
    return (fused_masks, fused_scores[keep], fused_labels[keep],
            mask2bbox(fused_masks.bool()))


def gate_views():
    """Two views (second flipped): 2 clusters + 1 singleton, mixed labels."""
    def band(offset, xlo, xhi):
        H = W = 100
        yy, xx = np.mgrid[0:H, 0:W]
        return (np.abs((yy - xx) - offset) <= 2) & (xx >= xlo) & (xx <= xhi)
    A0, A1 = band(6, 15, 75), band(6, 16, 74)
    B0, B1 = band(-6, 15, 75), band(-6, 16, 74)
    C0 = _square(80, 95, 5, 20)
    v0_masks = torch.from_numpy(np.stack([A0, B0, C0])).bool()
    v0_scores = torch.tensor([0.90, 0.85, 0.80])
    v0_labels = torch.tensor([0, 0, 1])
    v1_masks = torch.from_numpy(np.stack([A1, B1])).bool().flip(dims=[-1])
    v1_scores = torch.tensor([0.70, 0.65])
    v1_labels = torch.tensor([0, 0])
    v0 = make_sample(v0_masks, v0_scores, v0_labels)
    v1 = make_sample(v1_masks, v1_scores, v1_labels, flip=True,
                     flip_direction='horizontal')
    merged_m = torch.cat([torch.from_numpy(np.stack([A0, B0, C0])).bool(),
                          torch.from_numpy(np.stack([A1, B1])).bool()])
    merged_s = torch.cat([v0_scores, v1_scores])
    merged_l = torch.cat([v0_labels, v1_labels])
    return [v0, v1], merged_m, merged_s, merged_l


# --------------------------------------------------------------------------- #
# (a) GATE-0: flag off byte-identical, no attribute
# --------------------------------------------------------------------------- #
def check_gate0():
    print('=' * 70)
    print('(a) GATE-0 mask_tta: export_vote_stats off == pre-edit vote branch')
    print('=' * 70)
    ok = True
    for vote_mask in ('consensus', 'majority', 'seed'):
        samples, mm, ms, ml = gate_views()
        ref = old_vote_reference(mm, ms, ml, n_views=2, vote_iou=0.5,
                                 vote_frac=0.5, vote_gamma=0.0,
                                 vote_mask=vote_mask, vote_min_support=2,
                                 vote_max_support=5, max_per_img=100)
        out = make_model('vote', vote_mask=vote_mask, vote_iou=0.5).\
            _merge_single_sample(samples).pred_instances
        same = (torch.equal(out.masks, ref[0])
                and torch.equal(out.scores, ref[1])
                and torch.equal(out.labels, ref[2])
                and torch.equal(out.bboxes, ref[3]))
        no_attr = 'vote_stats' not in out
        print(f"  vote_mask='{vote_mask}': byte-identical={same} "
              f'no vote_stats attr={no_attr}')
        ok = ok and same and no_attr

    # default bbox_nms path: flag ON must be a no-op (vote-only feature)
    samples, _, _, _ = gate_views()
    out_nms = make_model('bbox_nms', export_vote_stats=True).\
        _merge_single_sample(samples).pred_instances
    nms_no_attr = 'vote_stats' not in out_nms
    print(f'  bbox_nms + flag ON: no vote_stats attr={nms_no_attr} '
          f'(n={len(out_nms.masks)})')
    ok = ok and nms_no_attr

    # empty input with flag ON -> (0, 6) stats, no crash
    empty = make_sample(torch.zeros((0, 50, 50), dtype=torch.bool),
                        torch.zeros(0), torch.zeros(0, dtype=torch.long))
    out_e = make_model('vote', export_vote_stats=True).\
        _merge_single_sample([empty]).pred_instances
    e_ok = ('vote_stats' in out_e
            and tuple(out_e.vote_stats.shape) == (0, 6))
    print(f'  empty input + flag ON: vote_stats shape (0,6)={e_ok}')
    ok = ok and e_ok
    print(f'\n  (a) {"PASS" if ok else "FAIL"}')
    return ok


# --------------------------------------------------------------------------- #
# (b) flag ON: fields unchanged + hand-computed stats
# --------------------------------------------------------------------------- #
def check_export_values():
    print('\n' + '=' * 70)
    print('(b) export_vote_stats ON: fields unchanged, values hand-verified')
    print('=' * 70)
    A0 = _square(10, 30, 10, 30)              # 400 px, view 0, score 0.9
    A1 = _square(10, 30, 10, 28)              # 360 px, view 1, score 0.7
    C = _square(70, 90, 70, 90)               # singleton, view 0, score 0.8
    iou_a = 360.0 / 400.0                     # inter 360 / union 400 = 0.9
    v0 = make_sample(torch.from_numpy(np.stack([A0, C])).bool(),
                     torch.tensor([0.90, 0.80]),
                     torch.zeros(2, dtype=torch.long))
    v1 = make_sample(torch.from_numpy(A1[None]).bool().flip(dims=[-1]),
                     torch.tensor([0.70]), torch.zeros(1, dtype=torch.long),
                     flip=True, flip_direction='horizontal')

    out_off = make_model('vote', vote_mask='consensus').\
        _merge_single_sample([v0, v1]).pred_instances
    v0b = make_sample(torch.from_numpy(np.stack([A0, C])).bool(),
                      torch.tensor([0.90, 0.80]),
                      torch.zeros(2, dtype=torch.long))
    v1b = make_sample(torch.from_numpy(A1[None]).bool().flip(dims=[-1]),
                      torch.tensor([0.70]), torch.zeros(1, dtype=torch.long),
                      flip=True, flip_direction='horizontal')
    out_on = make_model('vote', vote_mask='consensus',
                        export_vote_stats=True).\
        _merge_single_sample([v0b, v1b]).pred_instances

    fields_same = (torch.equal(out_on.masks, out_off.masks)
                   and torch.equal(out_on.scores, out_off.scores)
                   and torch.equal(out_on.labels, out_off.labels)
                   and torch.equal(out_on.bboxes, out_off.bboxes))
    has = 'vote_stats' in out_on
    vs = out_on.vote_stats if has else None
    print(f'  common fields byte-identical to flag OFF: {fields_same}; '
          f'vote_stats present: {has} shape='
          f'{tuple(vs.shape) if has else None}')
    # expected rows (score-desc order): cluster A then singleton C
    exp = torch.tensor([[2.0, 2.0, iou_a, iou_a, 0.90, 0.90 - 0.70],
                        [1.0, 1.0, 0.0, 0.0, 0.80, 0.0]])
    val_ok = has and vs.shape == (2, 6) and \
        bool(torch.allclose(vs, exp, atol=1e-6))
    if has:
        print(f'  row0 (cluster A) = {vs[0].tolist()}')
        print(f'  expected        = {exp[0].tolist()}')
        print(f'  row1 (singleton C) = {vs[1].tolist()}')
        print(f'  expected           = {exp[1].tolist()}')
    print(f'  values match hand computation: {val_ok}')
    ok = fields_same and val_ok
    print(f'\n  (b) {"PASS" if ok else "FAIL"}')
    return ok


# --------------------------------------------------------------------------- #
# (c) R10 harvest labels
# --------------------------------------------------------------------------- #
def check_harvest_labels():
    print('\n' + '=' * 70)
    print('(c) R10 harvest: labels on synthetic candidates')
    print('=' * 70)
    H = W = 200
    rng_img = np.random.RandomState(0)
    img = (rng_img.rand(H, W) * 60).astype(np.uint8)
    img[100:130, 100:140] = 200                       # bright missed ship
    g1 = _square(20, 40, 20, 40, H, W)                # GT1 (covered)
    g2 = _square(100, 130, 100, 140, H, W)            # GT2 (missed by kept)
    gt_rles = [_rle(g1), _rle(g2)]
    preds = [g1,                                       # K1 kept dup of GT1
             g2,                                       # P1 pool on GT2 -> 1
             g1,                                       # P2 pool dup GT1 -> 0
             _square(160, 180, 160, 180, H, W)]        # P3 background -> 0
    rles = [_rle(p) for p in preds]
    scores = np.array([0.90, 0.20, 0.10, 0.05])
    stats = np.arange(24, dtype=np.float32).reshape(4, 6)
    rng = np.random.RandomState(0)
    fs, ims, ms, ls = harvest_image(img, rles, scores, stats, gt_rles,
                                    floor=0.01, keep_thr=0.35, iou_thr=0.5,
                                    rng=rng, neg_per_img=30)
    # order: positives first (P1), then negatives (P2, P3)
    lab_ok = ls == [1, 0, 0]
    dim_ok = all(f.shape == (len(RESCUE_FEAT_NAMES),) for f in fs)
    fin_ok = all(np.isfinite(f).all() for f in fs)
    patch_ok = all(i.shape == (48, 48) and m.shape == (48, 48)
                   for i, m in zip(ims, ms))
    score_feat_ok = abs(fs[0][6] - 0.20) < 1e-6       # feat[6] = score of P1
    print(f'  labels={ls} (expect [1, 0, 0]): {lab_ok}')
    print(f'  feats dim {len(RESCUE_FEAT_NAMES)}: {dim_ok} finite: {fin_ok} '
          f'patches 48x48: {patch_ok} score-feature of P1==0.20: '
          f'{score_feat_ok}')
    # no-GT image: every pool candidate is a negative
    fs2, _, _, ls2 = harvest_image(img, rles, scores, stats, [], 0.01, 0.35,
                                   0.5, np.random.RandomState(0), 30)
    nogt_ok = ls2 == [0, 0, 0]
    print(f'  no-GT image labels={ls2} (expect [0, 0, 0]): {nogt_ok}')
    ok = lab_ok and dim_ok and fin_ok and patch_ok and score_feat_ok \
        and nogt_ok
    print(f'\n  (c) {"PASS" if ok else "FAIL"}')
    return ok


# --------------------------------------------------------------------------- #
# (d) verifier forward/backward
# --------------------------------------------------------------------------- #
def check_verifier():
    print('\n' + '=' * 70)
    print('(d) verifier: forward/backward, <500k params')
    print('=' * 70)
    torch.manual_seed(0)
    model = build_verifier(len(RESCUE_FEAT_NAMES))
    n_params = sum(p.numel() for p in model.parameters())
    patch = torch.rand(8, 2, 48, 48)
    feats = torch.randn(8, len(RESCUE_FEAT_NAMES))
    target = torch.randint(0, 2, (8,)).float()
    logits = model(patch, feats)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits,
                                                                target)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    gnorm = float(sum(g.abs().sum() for g in grads))
    ok = (logits.shape == (8,) and n_params < 500_000 and len(grads) > 0
          and gnorm > 0 and np.isfinite(float(loss)))
    print(f'  params={n_params} (<500k) logits shape={tuple(logits.shape)} '
          f'loss={float(loss):.4f} grad|sum|={gnorm:.4f}')
    print(f'\n  (d) {"PASS" if ok else "FAIL"}')
    return ok


# --------------------------------------------------------------------------- #
# (e) reranker fit/predict
# --------------------------------------------------------------------------- #
def check_reranker_fit():
    print('\n' + '=' * 70)
    print('(e) reranker: recovers synthetic monotone target')
    print('=' * 70)
    rng = np.random.RandomState(0)
    n = 4000
    X = rng.randn(n, len(RERANK_FEAT_NAMES)).astype(np.float32)
    y = np.clip(0.5 + 0.3 * X[:, 4] + 0.2 * X[:, 0]
                + 0.05 * rng.randn(n), 0, 1).astype(np.float32)
    m = fit_gbdt(X[:3000], y[:3000], seed=0, max_iter=100)
    pv = m.predict(X[3000:])
    r = float(np.corrcoef(pv, y[3000:])[0, 1])
    ok = pv.shape == (1000,) and r > 0.8
    print(f'  holdout pearson={r:.4f} (>0.8) predict shape={pv.shape}')
    print(f'\n  (e) {"PASS" if ok else "FAIL"}')
    return ok


# --------------------------------------------------------------------------- #
# (f) apply-safety end-to-end (synthetic mini-world)
# --------------------------------------------------------------------------- #
def build_mini_world(tmp):
    """1 image, 2 GT ships; stack pkl with 4 candidates + vote_stats;
    base json with 2 dets (1 kept, 1 tail)."""
    import pickle
    H = W = 200
    rng = np.random.RandomState(1)
    img = (rng.rand(H, W) * 60).astype(np.uint8)
    img[100:130, 100:140] = 200
    img_path = os.path.join(tmp, 'im1.png')
    cv2.imwrite(img_path, img)

    g1 = _square(20, 40, 20, 40, H, W)
    g2 = _square(100, 130, 100, 140, H, W)

    def seg(mask):
        r = _rle(mask)
        return dict(size=[int(v) for v in r['size']],
                    counts=r['counts'].decode())

    ann = dict(
        images=[dict(id=1, file_name='im1.png', width=W, height=H)],
        annotations=[
            dict(id=1, image_id=1, category_id=1, segmentation=seg(g1),
                 area=float(maskUtils.area(_rle(g1))),
                 bbox=[float(v) for v in maskUtils.toBbox(_rle(g1))],
                 iscrowd=0),
            dict(id=2, image_id=1, category_id=1, segmentation=seg(g2),
                 area=float(maskUtils.area(_rle(g2))),
                 bbox=[float(v) for v in maskUtils.toBbox(_rle(g2))],
                 iscrowd=0)],
        categories=[dict(id=1, name='ship')])
    ann_path = os.path.join(tmp, 'ann.json')
    with open(ann_path, 'w') as f:
        json.dump(ann, f)

    # stack pkl: K1 (kept, on GT1), P1 (pool, on GT2), P2 (pool, dup GT1),
    # P3 (pool, background)
    cand = [g1, g2, g1, _square(160, 180, 160, 180, H, W)]
    entry = dict(
        img_id=1,
        pred_instances=dict(
            masks=[_rle(c) for c in cand],
            scores=torch.tensor([0.90, 0.20, 0.10, 0.05]),
            labels=torch.zeros(4, dtype=torch.long),
            vote_stats=torch.tensor([[6, 6, 0.9, 0.8, 0.90, 0.1],
                                     [2, 2, 0.7, 0.6, 0.20, 0.05],
                                     [1, 1, 0.0, 0.0, 0.10, 0.0],
                                     [1, 1, 0.0, 0.0, 0.05, 0.0]],
                                    dtype=torch.float32)))
    pkl_path = os.path.join(tmp, 'stack.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump([entry], f)

    base = [dict(image_id=1, category_id=1, segmentation=seg(g1),
                 score=0.90),
            dict(image_id=1, category_id=1, segmentation=seg(g1),
                 score=0.10)]
    base_path = os.path.join(tmp, 'base.json')
    with open(base_path, 'w') as f:
        json.dump(base, f)
    return img_path, ann_path, pkl_path, base_path, base


def check_apply_safety():
    print('\n' + '=' * 70)
    print('(f) apply-safety: R10 scores < 0.35, base untouched; R11 '
          'preserves the candidate set')
    print('=' * 70)
    tmp = tempfile.mkdtemp(prefix='scic_test_')
    _, ann_path, pkl_path, base_path, base = build_mini_world(tmp)

    # random-init verifier ckpt (probs ~0.5 -> exercises the score cap)
    torch.manual_seed(0)
    model = build_verifier(len(RESCUE_FEAT_NAMES))
    ck_path = os.path.join(tmp, 'verifier.pth')
    torch.save(dict(model=model.state_dict(),
                    feat_mu=np.zeros(len(RESCUE_FEAT_NAMES), np.float32),
                    feat_sd=np.ones(len(RESCUE_FEAT_NAMES), np.float32),
                    n_feats=len(RESCUE_FEAT_NAMES), epoch=0, val_auc=0.5,
                    val_ap=0.5, feat_names=RESCUE_FEAT_NAMES), ck_path)

    r10_out = os.path.join(tmp, 'r10.json')
    scic_rescue.main([
        'apply', '--pred', pkl_path, '--base-json', base_path,
        '--ann', ann_path, '--img-dir', tmp, '--ckpt', ck_path,
        '--out', r10_out, '--tag', 'mini', '--device', 'cpu',
        '--min-prob', '0.0', '--no-eval'])
    with open(r10_out) as f:
        r10 = json.load(f)
    n_rescued = len(r10) - len(base)
    resc = r10[len(base):]
    base_same = r10[:len(base)] == base
    below = all(d['score'] < 0.35 for d in resc)
    # P2 (dup of kept GT1 det) must be overlap-suppressed -> 2 rescued
    n_ok = n_rescued == 2
    print(f'  R10: rescued={n_rescued} (expect 2: P1+P3, dup P2 suppressed) '
          f'-> {n_ok}')
    print(f'  R10: all rescued scores < 0.35: {below} '
          f'(scores={[round(d["score"], 4) for d in resc]})')
    print(f'  R10: base dets byte-unchanged: {base_same}')

    # R11: toy reranker on 14 feats
    rng = np.random.RandomState(0)
    X = rng.randn(500, len(RERANK_FEAT_NAMES)).astype(np.float32)
    y = np.clip(0.5 + 0.3 * X[:, 6], 0, 1).astype(np.float32)
    gb = fit_gbdt(X, y, seed=0, max_iter=50)
    import pickle as pk
    m_path = os.path.join(tmp, 'reranker.pkl')
    with open(m_path, 'wb') as f:
        pk.dump(dict(model=gb, alpha=0.5, feat_names=RERANK_FEAT_NAMES,
                     floor=0.01), f)
    r11_out = os.path.join(tmp, 'r11.json')
    scic_rerank.main([
        'apply', '--pred', pkl_path, '--base-json', base_path,
        '--ann', ann_path, '--model', m_path, '--out', r11_out,
        '--tag', 'mini', '--no-eval'])
    with open(r11_out) as f:
        r11 = json.load(f)
    count_ok = len(r11) == len(base)
    masks_ok = all(a['segmentation'] == b['segmentation']
                   for a, b in zip(r11, base))
    range_ok = all(0.0 <= d['score'] <= 1.0 for d in r11)
    changed = sum(1 for a, b in zip(r11, base)
                  if abs(a['score'] - b['score']) > 1e-9)
    print(f'  R11: det count preserved: {count_ok} ({len(r11)}) '
          f'masks identical in order: {masks_ok}')
    print(f'  R11: scores in [0,1]: {range_ok}; matched dets rescored: '
          f'{changed}/{len(base)}')
    # sanity: direct rerank_dets round-trip preserves candidate set too
    entry = scic_rescue.load_stack_pkl(pkl_path)[1]
    nd, nm = rerank_dets(base, entry['rles'], entry['scores'],
                         entry['stats'], gb, 0.5, 0.5)
    direct_ok = (len(nd) == len(base) and nm >= 1
                 and all(a['segmentation'] == b['segmentation']
                         for a, b in zip(nd, base)))
    print(f'  R11 direct rerank_dets: preserved={direct_ok} matched={nm}')
    ok = (n_ok and below and base_same and count_ok and masks_ok
          and range_ok and changed >= 1 and direct_ok)
    print(f'\n  (f) {"PASS" if ok else "FAIL"}')
    return ok


def main():
    a = check_gate0()
    b = check_export_values()
    c = check_harvest_labels()
    d = check_verifier()
    e = check_reranker_fit()
    f = check_apply_safety()
    print('\n' + '=' * 70)
    print(f'OVERALL: {"ALL PASS" if all([a, b, c, d, e, f]) else "FAIL"}  '
          f'[(a) GATE-0={"PASS" if a else "FAIL"}, '
          f'(b) export values={"PASS" if b else "FAIL"}, '
          f'(c) harvest labels={"PASS" if c else "FAIL"}, '
          f'(d) verifier={"PASS" if d else "FAIL"}, '
          f'(e) reranker fit={"PASS" if e else "FAIL"}, '
          f'(f) apply-safety={"PASS" if f else "FAIL"}]')
    print('=' * 70)
    sys.exit(0 if all([a, b, c, d, e, f]) else 1)


if __name__ == '__main__':
    main()
