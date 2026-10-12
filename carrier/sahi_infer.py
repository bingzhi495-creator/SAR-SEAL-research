# -*- coding: utf-8 -*-
"""R4-3: SAHI-style ZOOM-SLICE inference (Pass B of the offline zoom-slice fusion).

WHY: last live free lever for the inshore AP75 gap. SAHI-style slicing gives its
largest gains on small objects (MDPI Math 13(19):3079, 2025). We upscale each
800x800 HRSID chip 1.5x -> 1200x1200 (image-level bicubic; no logits involved),
slice into 800x800 tiles at stride 400 (positions 0,400 per axis -> 2x2 = 4
tiles, 25%+ overlap), run the EXISTING soup checkpoint per tile SINGLE-VIEW (no
TTA inside tiles), map masks back to full-frame original coords, and dump a pkl
in the exact DumpDetResults dict format that carrier/ensemble_fuse.py reads.
Fusion with the wide-TTA vote-consensus base predictions happens OFFLINE in
Pass C (ensemble_fuse.py with 2 sources) - the tile views therefore join the
vote-consensus pool without touching the TestTimeAug plumbing (tiles change the
per-image view count, which TestTimeAug cannot express).

GEOMETRY (pure, unit-tested in carrier/tests/cpu_test_sahi.py):
  original (W,H) --x1.5 bicubic--> (1.5W,1.5H) --crop tile (x0,y0,800,800)-->
  model -> mask (800,800, tile coords) --paste at (x0,y0) into 1.5W x 1.5H
  canvas--> --bilinear resize back to (W,H), threshold 0.5--> full-frame mask.
  Verified round-trip IoU == 1.0 on synthetic blobs (2026-07-20).

CENTRAL-CROP GUARD: an instance CUT by a tile edge that is INTERIOR to the
upscaled image is a border artifact (the object continues in the neighbouring
tile, which sees it whole thanks to the 400px overlap) -> dropped. Instances
touching the OUTER image border are genuinely image-truncated -> kept.

SCORE GATE: keep tile instances with score >= 0.05 (M2F emits 100/query soup;
below 0.05 is noise that would only bloat the fuse pool).

BASELINE RECIPE MATCH: the soup x wide x vote-consensus baseline evaluates with
model.panoptic_fusion_head.infer_iou_lambda=0.0 -> tile inference defaults to
the same lambda so the two fusion sources score on the same scale.

Usage (GPU pass; launched ONLY by run_sahi_queue.sh):
  python carrier/sahi_infer.py infer \
      --config carrier/configs/c0_maskiou_v2_m2f_hrsid.py \
      --ckpt runs/c0_maskiou_v2/soup_s023/iter_100872.pth \
      --ann data/hrsid/annotations/test_inshore.json \
      --out runs/sahi/tiles_inshore.pkl
  python carrier/sahi_infer.py concat --inputs A.pkl B.pkl --out AB.pkl
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_util
from pycocotools.coco import COCO


# --------------------------------------------------------------------------- #
# pure geometry (unit-tested on CPU, no model involved)
# --------------------------------------------------------------------------- #
def tile_positions(size_up: int, tile: int, stride: int):
    """Left/top offsets of the tiles covering one axis of the upscaled image.

    1200 with tile=800, stride=400 -> [0, 400] (2 per axis -> 2x2 = 4 tiles).
    The last position is clamped to size_up - tile so the far border is always
    covered exactly once.
    """
    if size_up <= tile:
        return [0]
    pos = list(range(0, size_up - tile + 1, stride))
    if pos[-1] != size_up - tile:
        pos.append(size_up - tile)
    return pos


def interior_border_touch(mask_tile: np.ndarray, x0: int, y0: int,
                          up_w: int, up_h: int) -> bool:
    """True iff the mask touches a tile border that is INTERIOR to the image.

    Args:
        mask_tile: (th, tw) bool/uint8 mask in TILE coordinates.
        x0, y0: tile offset in the upscaled image.
        up_w, up_h: upscaled image size.
    """
    th, tw = mask_tile.shape
    if x0 > 0 and mask_tile[:, 0].any():           # left edge is interior
        return True
    if y0 > 0 and mask_tile[0, :].any():           # top edge is interior
        return True
    if x0 + tw < up_w and mask_tile[:, -1].any():  # right edge is interior
        return True
    if y0 + th < up_h and mask_tile[-1, :].any():  # bottom edge is interior
        return True
    return False


def paste_back(mask_tile: np.ndarray, x0: int, y0: int, up_w: int, up_h: int,
               ori_w: int, ori_h: int) -> np.ndarray:
    """Tile-coords mask -> full-frame ORIGINAL-resolution bool mask.

    Paste into the up_h x up_w canvas at (x0, y0), bilinear-resize back to the
    original size, threshold at 0.5 (mirrors mmdet's bilinear+0.5 mask rescale
    convention). Returns (ori_h, ori_w) bool.
    """
    th, tw = mask_tile.shape
    canvas = np.zeros((up_h, up_w), dtype=np.float32)
    canvas[y0:y0 + th, x0:x0 + tw] = mask_tile.astype(np.float32)
    full = cv2.resize(canvas, (ori_w, ori_h), interpolation=cv2.INTER_LINEAR)
    return full >= 0.5


def make_sample(img_id: int, ori_shape, rles, scores, labels, img_path=''):
    """One image's predictions in the exact to_dict()-ed DetDataSample layout
    that mmdet DumpDetResults writes and ensemble_fuse.load_pkl asserts on:
    top-level 'img_id' + 'pred_instances' {'masks': RLE list, 'scores'/'labels'
    /'bboxes': CPU torch tensors}."""
    n = len(rles)
    assert n == len(scores) == len(labels)
    if n:
        bboxes = np.stack([mask_util.toBbox(r) for r in rles]).astype(np.float32)
        bboxes[:, 2:] += bboxes[:, :2]                       # xywh -> xyxy
    else:
        bboxes = np.zeros((0, 4), np.float32)
    return dict(
        img_id=int(img_id), ori_shape=tuple(ori_shape), img_path=img_path,
        pred_instances=dict(
            masks=list(rles),
            scores=torch.as_tensor(np.asarray(scores, np.float32)),
            labels=torch.as_tensor(np.asarray(labels, np.int64)),
            bboxes=torch.as_tensor(bboxes)))


def concat_pkls(inputs, output):
    """Concatenate per-split dumps (disjoint img ids) into one superset pkl.

    HRSID test_inshore/test_offshore are id-preserving subsets partitioning
    official_test (369+1593=1962, verified 2026-07-19), so inshore+offshore
    dumps concatenated == an official-split dump.
    """
    out, seen = [], set()
    for p in inputs:
        with open(p, 'rb') as f:
            data = pickle.load(f)
        assert isinstance(data, list) and data, f'{p}: not a non-empty list'
        ids = {int(d['img_id']) for d in data}
        assert not (ids & seen), f'{p}: img ids overlap earlier inputs'
        seen |= ids
        out.extend(data)
    with open(output, 'wb') as f:
        pickle.dump(out, f)
    print(f'[sahi-concat] {" + ".join(inputs)} -> {output} '
          f'({len(out)} imgs, ids disjoint)')


# --------------------------------------------------------------------------- #
# GPU pass (mmdet imported lazily so the CPU unit test never touches it)
# --------------------------------------------------------------------------- #
def run_split(config, ckpt, ann, out_pkl, img_dir=None, scale=1.5, tile=800,
              stride=400, score_thr=0.05, iou_lambda=0.0, device='cuda:0'):
    from mmcv.transforms import Compose
    from mmdet.apis import inference_detector, init_detector
    from mmdet.utils import get_test_pipeline_cfg

    model = init_detector(
        config, ckpt, device=device,
        cfg_options={'model.panoptic_fusion_head.infer_iou_lambda':
                     float(iou_lambda)})
    if img_dir is None:
        img_dir = model.cfg.test_dataloader.dataset.data_prefix['img']

    # ndarray input + no GT at inference: swap the loader, drop LoadAnnotations
    pipe = get_test_pipeline_cfg(model.cfg.copy())
    pipe[0].type = 'mmdet.LoadImageFromNDArray'
    pipe = [p for p in pipe if 'LoadAnnotations' not in p['type']]
    test_pipeline = Compose(pipe)

    coco = COCO(ann)
    img_ids = sorted(coco.getImgIds())
    out, t0 = [], time.time()
    n_tiles_total = n_kept = n_guard = 0
    for k, img_id in enumerate(img_ids):
        info = coco.loadImgs([img_id])[0]
        path = os.path.join(img_dir, info['file_name'])
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        assert img is not None, f'cannot read {path}'
        ori_h, ori_w = img.shape[:2]
        up_w, up_h = int(round(ori_w * scale)), int(round(ori_h * scale))
        up = cv2.resize(img, (up_w, up_h), interpolation=cv2.INTER_CUBIC)

        offs = [(x0, y0) for y0 in tile_positions(up_h, tile, stride)
                for x0 in tile_positions(up_w, tile, stride)]
        tiles = [up[y0:y0 + tile, x0:x0 + tile] for x0, y0 in offs]
        results = inference_detector(model, tiles, test_pipeline=test_pipeline)
        n_tiles_total += len(tiles)

        rles, scores, labels = [], [], []
        for (x0, y0), res in zip(offs, results):
            inst = res.pred_instances
            keep = inst.scores >= score_thr
            masks = inst.masks[keep].cpu().numpy().astype(np.uint8)
            scs = inst.scores[keep].cpu().numpy()
            lbs = inst.labels[keep].cpu().numpy()
            for m, s, lab in zip(masks, scs, lbs):
                if not m.any():
                    continue
                if interior_border_touch(m, x0, y0, up_w, up_h):
                    n_guard += 1
                    continue
                full = paste_back(m, x0, y0, up_w, up_h, ori_w, ori_h)
                if not full.any():
                    continue
                rles.append(mask_util.encode(
                    np.asfortranarray(full.astype(np.uint8))))
                scores.append(float(s))
                labels.append(int(lab))
        n_kept += len(rles)
        out.append(make_sample(img_id, (ori_h, ori_w), rles, scores, labels,
                               img_path=path))
        if (k + 1) % 50 == 0 or k + 1 == len(img_ids):
            dt = time.time() - t0
            print(f'[sahi] {k + 1}/{len(img_ids)} imgs  '
                  f'{n_tiles_total} tiles  kept={n_kept} guard-drop={n_guard}  '
                  f'{dt:.0f}s ({dt / (k + 1):.2f}s/img)', flush=True)

    with open(out_pkl, 'wb') as f:
        pickle.dump(out, f)
    print(f'[sahi] wrote {len(out)} imgs -> {out_pkl}  '
          f'(tiles={n_tiles_total}, kept={n_kept}, guard-drop={n_guard})')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='mode', required=True)
    ai = sub.add_parser('infer', help='tile inference -> DumpDetResults pkl')
    ai.add_argument('--config', required=True)
    ai.add_argument('--ckpt', required=True)
    ai.add_argument('--ann', required=True)
    ai.add_argument('--out', required=True)
    ai.add_argument('--img-dir', default=None,
                    help='default: config test_dataloader data_prefix.img')
    ai.add_argument('--scale', type=float, default=1.5)
    ai.add_argument('--tile', type=int, default=800)
    ai.add_argument('--stride', type=int, default=400)
    ai.add_argument('--score-thr', type=float, default=0.05)
    ai.add_argument('--iou-lambda', type=float, default=0.0,
                    help='matches the soup x wide x vote-consensus baseline')
    ai.add_argument('--device', default='cuda:0')
    ac = sub.add_parser('concat', help='concat disjoint per-split pkls')
    ac.add_argument('--inputs', nargs='+', required=True)
    ac.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    if a.mode == 'infer':
        run_split(a.config, a.ckpt, a.ann, a.out, img_dir=a.img_dir,
                  scale=a.scale, tile=a.tile, stride=a.stride,
                  score_thr=a.score_thr, iou_lambda=a.iou_lambda,
                  device=a.device)
    else:
        concat_pkls(a.inputs, a.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
