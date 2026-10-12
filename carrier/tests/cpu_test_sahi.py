# -*- coding: utf-8 -*-
"""CPU-ONLY sanity test for R4-3 SAHI zoom-slice inference (sahi_infer.py).

NO GPU, no checkpoints, no mmdet import (sahi_infer imports mmdet lazily).
Exercises the pure geometry + the fuse-format contract:

  (a) ROUND-TRIP    100x100 'image' with a known 30x30 blob; scale 1.5 ->
      150x150; tile=100 stride=50 -> positions [0,50] per axis = 2x2 tiles.
      Every guard-surviving tile view of the blob pastes back to the ORIGINAL
      full-frame RLE with IoU >= 0.95.
  (b) GUARD         the blob cut by the INTERIOR top edge of the y0=50 tiles
      is dropped; a second blob touching the OUTER image border (row 0 of the
      y0=0 tile = image edge) is kept.
  (c) FUSE FORMAT   make_sample() dicts (base source + tiles source, 2 pkls)
      are accepted end-to-end by ensemble_fuse.run(): both-agree pair fuses to
      ONE instance (consensus band [2,6], majority of 2 at frac .5 == UNION,
      score == max); a tiles-only singleton SURVIVES with its own score (the
      SAHI gain mechanism); E3_RESULT stats returned.
  (d) TILING MATH   1200x1200 with tile=800 stride=400 -> [0,400] x [0,400]
      = exactly 4 tiles; 1200 with clamp (stride 500) still covers the border.

Run:
  KMP_DUPLICATE_LIB_OK=TRUE CUDA_VISIBLE_DEVICES="" \
  python carrier/tests/cpu_test_sahi.py
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import tempfile

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_util
from pycocotools.coco import COCO

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))
from carrier.ensemble_fuse import run  # noqa: E402
from carrier.sahi_infer import (  # noqa: E402
    interior_border_touch, make_sample, paste_back, tile_positions)

H = W = 100
SCALE, TILE, STRIDE = 1.5, 100, 50


def rect(y0, y1, x0, x1, h=H, w=W):
    m = np.zeros((h, w), dtype=np.uint8)
    m[y0:y1, x0:x1] = 1
    return m


def rle(m):
    return mask_util.encode(np.asfortranarray(m.astype(np.uint8)))


def iou_np(a, b):
    a, b = a.astype(bool), b.astype(bool)
    u = np.logical_or(a, b).sum()
    return np.logical_and(a, b).sum() / u if u else 0.0


def upscaled_view(mask_ori, up_w, up_h):
    """What the model 'sees' at zoom: the blob on the upscaled image."""
    up = cv2.resize(mask_ori.astype(np.float32), (up_w, up_h),
                    interpolation=cv2.INTER_LINEAR)
    return (up >= 0.5).astype(np.uint8)


def main():
    ok = True
    up_w, up_h = int(round(W * SCALE)), int(round(H * SCALE))   # 150x150

    # ---------------- (d) tiling math ----------------
    xs = tile_positions(1200, 800, 400)
    n_tiles = len(xs) ** 2
    print(f'(d) tiling: 1200/800/stride400 -> positions {xs} per axis, '
          f'{n_tiles} tiles (expect [0, 400] -> 4)')
    if xs != [0, 400] or n_tiles != 4:
        print('  !! FAIL (d)'); ok = False
    clamp = tile_positions(1200, 800, 500)
    print(f'    clamp check: stride 500 -> {clamp} (last must be 400)')
    if clamp[-1] != 400:
        print('  !! FAIL (d) clamp'); ok = False
    test_xs = tile_positions(up_w, TILE, STRIDE)
    print(f'    test grid: 150/100/stride50 -> {test_xs} (expect [0, 50])')
    if test_xs != [0, 50]:
        print('  !! FAIL (d) test grid'); ok = False

    # ---------------- (a) round-trip + (b) guard ----------------
    blob = rect(30, 60, 35, 65)               # interior blob (up: r45-90 c52-98)
    edge_blob = rect(0, 15, 40, 60)           # touches OUTER top image border
    blob_rle = rle(blob)

    offs = [(x0, y0) for y0 in tile_positions(up_h, TILE, STRIDE)
            for x0 in tile_positions(up_w, TILE, STRIDE)]
    up_blob = upscaled_view(blob, up_w, up_h)
    up_edge = upscaled_view(edge_blob, up_w, up_h)

    n_kept = n_dropped = 0
    for x0, y0 in offs:
        tile_m = up_blob[y0:y0 + TILE, x0:x0 + TILE]
        if not tile_m.any():
            continue
        if interior_border_touch(tile_m, x0, y0, up_w, up_h):
            n_dropped += 1
            print(f'(b) guard DROP blob view in tile ({x0},{y0}) '
                  f'(cut by interior edge)')
            continue
        full = paste_back(tile_m, x0, y0, up_w, up_h, W, H)
        v = iou_np(mask_util.decode(blob_rle), full)
        n_kept += 1
        print(f'(a) round-trip tile ({x0},{y0}): IoU(full-frame, original RLE) '
              f'= {v:.4f} (need >= 0.95)')
        if v < 0.95:
            print('  !! FAIL (a)'); ok = False
    # blob spans up-rows 45-90 -> whole in y0=0 tiles (2 kept), cut at the
    # interior top edge of the y0=50 tiles (2 dropped)
    print(f'    blob views: kept={n_kept} (expect 2), '
          f'guard-dropped={n_dropped} (expect 2)')
    if n_kept != 2 or n_dropped != 2:
        print('  !! FAIL (a/b) counts'); ok = False

    # edge blob: in tile (0,0) it touches ROW 0 == OUTER image border -> KEPT
    t00 = up_edge[0:TILE, 0:TILE]
    touched = interior_border_touch(t00, 0, 0, up_w, up_h)
    print(f'(b) outer-border blob in tile (0,0): touches row0(outer)='
          f'{bool(t00[0, :].any())}, guard-drop={touched} (must be False)')
    if touched or not t00[0, :].any():
        print('  !! FAIL (b) outer keep'); ok = False
    full_edge = paste_back(t00, 0, 0, up_w, up_h, W, H)
    v_edge = iou_np(edge_blob, full_edge)
    print(f'    outer blob round-trip IoU = {v_edge:.4f} (need >= 0.95)')
    if v_edge < 0.95:
        print('  !! FAIL (b) round-trip'); ok = False

    # ---------------- (c) fuse-format contract ----------------
    tmp = tempfile.mkdtemp(prefix='sahi_')
    sq_base = rect(10, 30, 10, 30)            # base source
    sq_tile = rect(10, 30, 14, 34)            # tiles source; IoU = 2/3 >= 0.5
    only_tile = rect(70, 80, 70, 80)          # tiles-only small ship (the gain)
    assert abs(iou_np(sq_base, sq_tile) - 2 / 3) < 1e-6

    base = [dict(img_id=1, ori_shape=(H, W),
                 pred_instances=dict(
                     masks=[rle(sq_base)],
                     scores=torch.tensor([0.90]),
                     labels=torch.zeros(1, dtype=torch.long),
                     bboxes=torch.zeros(1, 4)))]
    tiles = [make_sample(1, (H, W), [rle(sq_tile), rle(only_tile)],
                         [0.80, 0.30], [0, 0])]
    pkls = []
    for name, data in [('base', base), ('tiles', tiles)]:
        p = os.path.join(tmp, f'{name}.pkl')
        with open(p, 'wb') as f:
            pickle.dump(data, f)
        pkls.append(p)

    gt = rle(sq_base)
    ann = dict(images=[dict(id=1, width=W, height=H, file_name='im1.jpg')],
               categories=[dict(id=1, name='ship')],
               annotations=[dict(
                   id=1, image_id=1, category_id=1, iscrowd=0,
                   segmentation=dict(size=list(gt['size']),
                                     counts=gt['counts'].decode('utf-8')),
                   area=float(mask_util.area(gt)),
                   bbox=[float(v) for v in mask_util.toBbox(gt)])])
    ann_path = os.path.join(tmp, 'gt.json')
    with open(ann_path, 'w') as f:
        json.dump(ann, f)
    out_json = os.path.join(tmp, 'fused.json')
    stats = run(pkls, ann_path, out_json, tag='sahicputest')  # defaults [2,6]

    dets = json.load(open(out_json))

    def dec(d):
        return mask_util.decode(dict(
            size=d['segmentation']['size'],
            counts=d['segmentation']['counts'].encode('utf-8'))).astype(bool)

    union = np.logical_or(sq_base, sq_tile)
    pair = [d for d in dets if iou_np(dec(d), union) > 0.999]
    single = [d for d in dets if iou_np(dec(d), only_tile.astype(bool)) > 0.99]
    print(f'(c) fuse: {len(dets)} dets total (expect 2); '
          f'pair->union={len(pair)} score={[round(d["score"], 3) for d in pair]} '
          f'(expect 1 @ 0.9); tiles-only singleton={len(single)} '
          f'score={[round(d["score"], 3) for d in single]} (expect 1 @ 0.3)')
    if not (len(dets) == 2 and len(pair) == 1
            and abs(pair[0]['score'] - 0.90) < 1e-6
            and len(single) == 1 and abs(single[0]['score'] - 0.30) < 1e-6):
        print('  !! FAIL (c)'); ok = False
    coco = COCO(ann_path)
    coco.loadRes(out_json)
    if len(stats) != 12:
        print('  !! FAIL (c) stats'); ok = False

    print('=' * 70)
    print(f'OVERALL: {"ALL PASS" if ok else "FAIL"}')
    print('=' * 70)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
