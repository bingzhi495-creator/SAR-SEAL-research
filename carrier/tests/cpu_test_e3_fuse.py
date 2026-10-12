# -*- coding: utf-8 -*-
"""CPU-ONLY sanity test for E3 offline 3-seed ensemble fusion (ensemble_fuse.py).

NO GPU, no checkpoints. Builds THREE synthetic per-seed pkls in the exact
DumpDetResults format (list of to_dict()-ed DetDataSample dicts: top-level
'img_id' + 'pred_instances' {'masks': RLE list via pycocotools.mask.encode,
'scores'/'labels': CPU torch tensors}), runs the full load->fuse->json->COCOeval
pipeline, and asserts:

  (a) DUP-COLLAPSE   img1: the SAME 20x20 square predicted by all 3 seeds
      (scores .9/.8/.7) fuses to EXACTLY ONE instance with score == max (0.9).
  (b) NEIGHBOUR-KEEP img1: two parallel rects with mask-IoU ~0.33 (< vote_iou
      0.5) from different seeds BOTH survive as separate instances.
  (c) SINGLETON      img1: a distinct square only seed C predicted survives as
      a singleton with its own (seed) mask and score 0.6.
  (d) SIZE-2 BAND    img2: seeds A+B predict overlapping squares (IoU 0.667);
      consensus band [2,6] -> majority(frac=.5) of 2 == UNION (area 480),
      score == max (0.8); seed C has ZERO instances on img2 (empty handling).
  (e) VALID COCO     output json loads via COCO.loadRes, COCOeval segm runs,
      counts are str, bbox present, E3_RESULT line printed.

Run:
  KMP_DUPLICATE_LIB_OK=TRUE CUDA_VISIBLE_DEVICES="" \
  python carrier/tests/cpu_test_e3_fuse.py
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import tempfile

import numpy as np
import torch
from pycocotools import mask as mask_util
from pycocotools.coco import COCO

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))
from carrier.ensemble_fuse import run  # noqa: E402

H = W = 100


def rect(y0, y1, x0, x1):
    m = np.zeros((H, W), dtype=np.uint8)
    m[y0:y1, x0:x1] = 1
    return m


def rle(m):
    return mask_util.encode(np.asfortranarray(m))


def sample(img_id, masks, scores):
    """One to_dict()-ed DetDataSample as DumpDetResults writes it."""
    return dict(
        img_id=img_id, ori_shape=(H, W),
        pred_instances=dict(
            masks=[rle(m) for m in masks],
            scores=torch.tensor(scores, dtype=torch.float32),
            labels=torch.zeros(len(masks), dtype=torch.long),
            bboxes=torch.zeros(len(masks), 4, dtype=torch.float32)))


def iou_np(a, b):
    i = np.logical_and(a, b).sum()
    u = np.logical_or(a, b).sum()
    return i / u if u else 0.0


def main():
    torch.set_grad_enabled(False)
    tmp = tempfile.mkdtemp(prefix='e3fuse_')

    # ---------------- synthetic geometry ----------------
    dup = rect(10, 30, 10, 30)                 # img1: all 3 seeds (identical)
    n1 = rect(40, 48, 10, 40)                  # img1: seed A only
    n2 = rect(44, 52, 10, 40)                  # img1: seed B only; IoU(n1,n2)=1/3
    single = rect(70, 80, 70, 80)              # img1: seed C only
    sqA = rect(10, 30, 10, 30)                 # img2: seed A
    sqB = rect(10, 30, 14, 34)                 # img2: seed B; IoU(sqA,sqB)=2/3
    assert abs(iou_np(n1, n2) - 1 / 3) < 1e-6
    assert abs(iou_np(sqA, sqB) - 2 / 3) < 1e-6

    seedA = [sample(1, [dup, n1], [0.90, 0.85]), sample(2, [sqA], [0.80])]
    seedB = [sample(1, [dup, n2], [0.80, 0.75]), sample(2, [sqB], [0.70])]
    seedC = [sample(1, [dup, single], [0.70, 0.60]), sample(2, [], [])]

    pkls = []
    for name, data in [('s0', seedA), ('s2', seedB), ('s3', seedC)]:
        p = os.path.join(tmp, f'{name}.pkl')
        with open(p, 'wb') as f:
            pickle.dump(data, f)
        pkls.append(p)

    # minimal valid GT ann (1 box-ish object per image) just so COCOeval runs
    gt_rle1 = rle(dup)
    gt_rle2 = rle(sqA)
    ann = dict(
        images=[dict(id=1, width=W, height=H, file_name='im1.jpg'),
                dict(id=2, width=W, height=H, file_name='im2.jpg')],
        categories=[dict(id=1, name='ship')],
        annotations=[
            dict(id=1, image_id=1, category_id=1, iscrowd=0,
                 segmentation=dict(size=list(gt_rle1['size']),
                                   counts=gt_rle1['counts'].decode('utf-8')),
                 area=float(mask_util.area(gt_rle1)),
                 bbox=[float(v) for v in mask_util.toBbox(gt_rle1)]),
            dict(id=2, image_id=2, category_id=1, iscrowd=0,
                 segmentation=dict(size=list(gt_rle2['size']),
                                   counts=gt_rle2['counts'].decode('utf-8')),
                 area=float(mask_util.area(gt_rle2)),
                 bbox=[float(v) for v in mask_util.toBbox(gt_rle2)]),
        ])
    ann_path = os.path.join(tmp, 'gt.json')
    with open(ann_path, 'w') as f:
        json.dump(ann, f)

    out_json = os.path.join(tmp, 'e3.json')
    stats = run(pkls, ann_path, out_json, tag='cputest')  # defaults = E3 config

    dets = json.load(open(out_json))
    d1 = [d for d in dets if d['image_id'] == 1]
    d2 = [d for d in dets if d['image_id'] == 2]

    def dec(d):
        r = dict(size=d['segmentation']['size'],
                 counts=d['segmentation']['counts'].encode('utf-8'))
        return mask_util.decode(r).astype(bool)

    ok = True

    # (a) dup collapses to exactly 1, score = max
    hits = [d for d in d1 if iou_np(dec(d), dup.astype(bool)) > 0.9]
    print(f'(a) dup-collapse: {len(hits)} instance(s) match dup square, '
          f'scores={[round(d["score"], 3) for d in hits]}')
    if not (len(hits) == 1 and abs(hits[0]['score'] - 0.90) < 1e-6):
        print('  !! FAIL (a)'); ok = False

    # (b) both low-IoU parallel neighbours survive
    hitn1 = [d for d in d1 if iou_np(dec(d), n1.astype(bool)) > 0.9]
    hitn2 = [d for d in d1 if iou_np(dec(d), n2.astype(bool)) > 0.9]
    print(f'(b) neighbour-keep: n1={len(hitn1)} n2={len(hitn2)} '
          f'(both must be 1; IoU(n1,n2)=0.333 < vote_iou 0.5)')
    if not (len(hitn1) == 1 and len(hitn2) == 1):
        print('  !! FAIL (b)'); ok = False

    # (c) seed-C-only singleton survives verbatim with its score
    hs = [d for d in d1 if iou_np(dec(d), single.astype(bool)) > 0.99]
    print(f'(c) singleton: {len(hs)} instance(s), '
          f'scores={[round(d["score"], 3) for d in hs]}')
    if not (len(hs) == 1 and abs(hs[0]['score'] - 0.60) < 1e-6):
        print('  !! FAIL (c)'); ok = False

    # img1 total = dup + n1 + n2 + singleton = 4
    print(f'    img1 total fused = {len(d1)} (expect 4)')
    if len(d1) != 4:
        print('  !! FAIL img1 count'); ok = False

    # (d) img2: ONE fused instance, majority-of-2 == union, score = max
    union = np.logical_or(sqA, sqB)
    print(f'(d) size-2 band: img2 fused = {len(d2)} (expect 1)')
    if len(d2) == 1:
        fm = dec(d2[0])
        area = int(fm.sum())
        print(f'    fused area = {area} (union = {int(union.sum())}), '
              f'IoU(fused, union) = {iou_np(fm, union):.4f}, '
              f'score = {d2[0]["score"]:.3f} (expect 0.800)')
        if not (iou_np(fm, union) > 0.999 and abs(d2[0]['score'] - 0.80) < 1e-6):
            print('  !! FAIL (d) mask/score'); ok = False
    else:
        print('  !! FAIL (d) count'); ok = False

    # (e) output json is valid COCO det + COCOeval ran (stats returned)
    coco = COCO(ann_path)
    coco.loadRes(out_json)
    counts_ok = all(isinstance(d['segmentation']['counts'], str) for d in dets)
    bbox_ok = all(len(d['bbox']) == 4 for d in dets)
    print(f'(e) valid-coco: loadRes OK, counts-are-str={counts_ok}, '
          f'bbox-present={bbox_ok}, n_stats={len(stats)}')
    if not (counts_ok and bbox_ok and len(stats) == 12):
        print('  !! FAIL (e)'); ok = False

    print('=' * 70)
    print(f'OVERALL: {"ALL PASS" if ok else "FAIL"}')
    print('=' * 70)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
