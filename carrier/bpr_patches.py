# -*- coding: utf-8 -*-
"""BPR step-2 (CPU): build boundary-patch training set from a train-split dump.

BPR-style (CVPR'21, arXiv 2104.05239) lightweight variant, from scratch:
  * load predictions pkl (mmdet 3.3 DumpDetResults format: list of dicts with
    img_id / img_path / pred_instances{scores, labels, masks(RLE list)}),
  * match preds to GT by mask-IoU >= iou_thr (pycocotools RLE iou),
  * along each matched PRED boundary sample KxK windows centered on boundary
    pixels, deduped on a stride-grid (one center per stride x stride cell),
  * store [image_gray_u8, coarse_mask_u8, gt_mask_u8] patch triplets at native
    K resolution as compressed .npz shards. The x2 (128x128) bicubic upscale
    for sub-pixel supervision happens ON-THE-FLY in the refiner dataloader
    (equivalent supervision, 4x less disk).

Pure numpy/cv2/pycocotools. No torch, no GPU.

Run (from E:/SAR/seal, PYTHONPATH=E:/SAR/seal):
  python carrier/bpr_patches.py --pred runs/bpr/train_preds.pkl \
    --ann data/hrsid/annotations/official_train.json \
    --img-dir D:/SAR/UNet++/HRSID_JPG/HRSID_JPG/JPEGImages1/ \
    --out runs/bpr/patches
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time

import cv2
import numpy as np
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO

KERN3 = np.ones((3, 3), np.uint8)


def boundary_centers(mask, stride):
    """Boundary pixels of a binary mask, deduped to one per stride-grid cell.

    Boundary = mask XOR erode(mask, 3x3, 1 iter). Iteration order is row-major
    (np.nonzero), first pixel in each cell wins -> deterministic.
    Returns list of (y, x) int tuples.
    """
    m = (mask > 0).astype(np.uint8)
    er = cv2.erode(m, KERN3, iterations=1)
    bnd = m.astype(bool) & ~er.astype(bool)
    ys, xs = np.nonzero(bnd)
    seen, out = set(), []
    for y, x in zip(ys.tolist(), xs.tolist()):
        c = (y // stride, x // stride)
        if c not in seen:
            seen.add(c)
            out.append((y, x))
    return out


def patch_window(cy, cx, K, H, W):
    """Top-left of a KxK window centered at (cy,cx), shifted to stay in-bounds."""
    t = min(max(cy - K // 2, 0), H - K)
    l = min(max(cx - K // 2, 0), W - K)
    return t, l


def extract_instance_patches(img_gray, coarse, gt, K, stride, max_per_inst=32):
    """KxK triplet patches along the coarse-mask boundary.

    Returns (list_of_(img_p, coarse_p, gt_p) uint8, list_of_(t, l)). gt may be
    None (inference use) -> gt_p omitted (2-tuples).
    """
    H, W = coarse.shape
    if H < K or W < K:
        return [], []
    centers = boundary_centers(coarse, stride)[:max_per_inst]
    patches, coords = [], []
    for cy, cx in centers:
        t, l = patch_window(cy, cx, K, H, W)
        sl = (slice(t, t + K), slice(l, l + K))
        if gt is None:
            patches.append((img_gray[sl], coarse[sl]))
        else:
            patches.append((img_gray[sl], coarse[sl], gt[sl]))
        coords.append((t, l))
    return patches, coords


def paste_average(patches, coords, K, H, W):
    """Paste KxK float patches back, averaging overlaps.

    Returns (avg float32 HxW, weight float32 HxW); avg is 0 where weight == 0.
    """
    acc = np.zeros((H, W), np.float32)
    wt = np.zeros((H, W), np.float32)
    for p, (t, l) in zip(patches, coords):
        acc[t:t + K, l:l + K] += p.astype(np.float32)
        wt[t:t + K, l:l + K] += 1.0
    avg = np.zeros_like(acc)
    nz = wt > 0
    avg[nz] = acc[nz] / wt[nz]
    return avg, wt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True)
    ap.add_argument('--ann', required=True)
    ap.add_argument('--img-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--K', type=int, default=64)
    ap.add_argument('--stride', type=int, default=32)
    ap.add_argument('--cap', type=int, default=60000)
    ap.add_argument('--score-thr', type=float, default=0.2)
    ap.add_argument('--iou-thr', type=float, default=0.5)
    ap.add_argument('--max-per-inst', type=int, default=32)
    ap.add_argument('--shard-size', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    coco = COCO(args.ann)
    with open(args.pred, 'rb') as f:
        preds = pickle.load(f)
    rng = np.random.RandomState(args.seed)
    order = rng.permutation(len(preds))

    buf_i, buf_c, buf_g = [], [], []
    shard_id, total, n_img, n_pair = 0, 0, 0, 0

    def flush():
        nonlocal shard_id, buf_i, buf_c, buf_g
        if not buf_i:
            return
        np.savez_compressed(
            os.path.join(args.out, 'shard_%03d.npz' % shard_id),
            img=np.stack(buf_i), coarse=np.stack(buf_c), target=np.stack(buf_g))
        shard_id += 1
        buf_i, buf_c, buf_g = [], [], []

    for idx in order:
        if total >= args.cap:
            break
        entry = preds[idx]
        img_id = int(entry['img_id'])
        info = coco.loadImgs([img_id])[0]
        path = os.path.join(args.img_dir, info['file_name'])
        if not os.path.exists(path):
            path = entry.get('img_path', path)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        H, W = img.shape
        pi = entry['pred_instances']
        scores = np.asarray(pi['scores'])
        keep = np.nonzero(scores >= args.score_thr)[0]
        if keep.size == 0:
            continue
        dt_rles = [pi['masks'][i] for i in keep]
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
        if not anns:
            continue
        gt_rles = [coco.annToRLE(a) for a in anns]
        ious = maskUtils.iou(dt_rles, gt_rles, [0] * len(gt_rles))
        ious = np.atleast_2d(np.asarray(ious))
        gt_dec = {}
        used = False
        for di in range(len(dt_rles)):
            gi = int(np.argmax(ious[di]))
            if ious[di, gi] < args.iou_thr:
                continue
            coarse = maskUtils.decode(dt_rles[di]).astype(np.uint8)
            if coarse.shape != (H, W):
                continue
            if gi not in gt_dec:
                gt_dec[gi] = maskUtils.decode(gt_rles[gi]).astype(np.uint8)
            gt = gt_dec[gi]
            patches, _ = extract_instance_patches(
                img, coarse, gt, args.K, args.stride, args.max_per_inst)
            for (p_img, p_c, p_g) in patches:
                buf_i.append(p_img)
                buf_c.append(p_c)
                buf_g.append(p_g)
                total += 1
                if len(buf_i) >= args.shard_size:
                    flush()
                if total >= args.cap:
                    break
            if patches:
                used = True
                n_pair += 1
            if total >= args.cap:
                break
        if used:
            n_img += 1
    flush()

    meta = dict(K=args.K, stride=args.stride, cap=args.cap,
                score_thr=args.score_thr, iou_thr=args.iou_thr,
                max_per_inst=args.max_per_inst, seed=args.seed,
                total=total, shards=shard_id, images_used=n_img,
                matched_instances=n_pair, pred=args.pred, ann=args.ann,
                secs=round(time.time() - t0, 1))
    with open(os.path.join(args.out, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    print('PATCHES_RESULT total=%d shards=%d images_used=%d matched_inst=%d secs=%.0f'
          % (total, shard_id, n_img, n_pair, time.time() - t0))


if __name__ == '__main__':
    main()
