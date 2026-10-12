# -*- coding: utf-8 -*-
"""BPR step-4: apply the boundary-patch refiner to a predictions pkl + eval.

For every instance: extract the same boundary patches as training (K=64,
stride 32, grid-deduped), run the UNet refiner at 128x128, downsample probs to
64x64 and paste back (averaging overlaps). SAFETY GUARD: the refined
probability may only flip pixels inside a boundary BAND =
dilate(pred, r) XOR erode(pred, r) (r=4, 3x3 kernel iterations); every pixel
outside the band keeps the ORIGINAL mask verbatim. Band pixels not covered by
any patch also keep the original. Empty refined masks fall back to original.

Writes a COCO-format segm results json and prints paired COCOeval lines
(kind=orig reproduces the incoming stack, kind=refined is BPR):
  BPR_RESULT tag=<t> kind=<orig|refined> AP=.. AP50=.. AP75=.. APs=.. APm=.. APl=..

Run (GPU or CPU; from E:/SAR/seal, PYTHONPATH=E:/SAR/seal):
  python carrier/bpr_apply.py --pred runs/bpr/test_inshore.pkl \
    --ann data/hrsid/annotations/test_inshore.json \
    --img-dir D:/SAR/UNet++/HRSID_JPG/HRSID_JPG/JPEGImages1/ \
    --ckpt runs/bpr/refiner.pth --out runs/bpr/refined_inshore.json --tag inshore
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time

import cv2
import numpy as np
import torch
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from carrier.bpr_patches import (KERN3, extract_instance_patches,
                                 paste_average)
from carrier.bpr_refiner import UNetRefiner, upscale_triplet


def make_band(mask, r=4):
    """Boundary band: dilate(mask, r) XOR erode(mask, r) (3x3 kernel, r iters)."""
    m = (mask > 0).astype(np.uint8)
    dil = cv2.dilate(m, KERN3, iterations=r)
    ero = cv2.erode(m, KERN3, iterations=r)
    return (dil ^ ero).astype(bool)


def refine_instance(img_gray, mask, model, device, K=64, stride=32,
                    band_r=4, thr=0.5, max_patches=64, _prob_override=None):
    """Refine one instance mask. Returns (refined uint8 HxW, n_patches).

    Pixels outside the boundary band are ALWAYS returned unchanged.
    _prob_override: test hook - callable(batch_np)->prob_np, bypasses model.
    """
    H, W = mask.shape
    mask = (mask > 0).astype(np.uint8)
    patches, coords = extract_instance_patches(
        img_gray, mask, None, K, stride, max_per_inst=max_patches)
    if not patches:
        return mask, 0
    xs = []
    for (p_img, p_c) in patches:
        im, co, _ = upscale_triplet(p_img, p_c, None, 2 * K)
        xs.append(np.stack([im, co]))
    batch = np.stack(xs).astype(np.float32)
    if _prob_override is not None:
        probs = _prob_override(batch)
    else:
        with torch.no_grad():
            x = torch.from_numpy(batch).to(device)
            probs = torch.sigmoid(model(x)).squeeze(1).cpu().numpy()
    probs_k = [cv2.resize(p, (K, K), interpolation=cv2.INTER_AREA)
               for p in probs]
    avg, wt = paste_average(probs_k, coords, K, H, W)
    band = make_band(mask, band_r)
    refined = mask.copy()
    sel = band & (wt > 0)
    refined[sel] = (avg[sel] >= thr).astype(np.uint8)
    if refined.sum() == 0:                       # safety fallback
        return mask, len(patches)
    return refined, len(patches)


def coco_stats(coco_gt, dets, tag, kind):
    if not dets:
        print('BPR_RESULT tag=%s kind=%s EMPTY' % (tag, kind), flush=True)
        return
    dt = coco_gt.loadRes(dets)
    ev = COCOeval(coco_gt, dt, 'segm')
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    s = ev.stats
    print('BPR_RESULT tag=%s kind=%s AP=%.4f AP50=%.4f AP75=%.4f '
          'APs=%.4f APm=%.4f APl=%.4f' %
          (tag, kind, s[0], s[1], s[2], s[3], s[4], s[5]), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True)
    ap.add_argument('--ann', required=True)
    ap.add_argument('--img-dir', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tag', default='split')
    ap.add_argument('--K', type=int, default=64)
    ap.add_argument('--stride', type=int, default=32)
    ap.add_argument('--band', type=int, default=4)
    ap.add_argument('--thr', type=float, default=0.5)
    ap.add_argument('--score-thr', type=float, default=0.05,
                    help='instances below are passed through unrefined')
    ap.add_argument('--limit', type=int, default=0, help='debug: first N imgs')
    ap.add_argument('--device', default=None)
    ap.add_argument('--alive', default='runs/_alive')
    args = ap.parse_args()

    dev = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(args.ckpt, map_location='cpu')
    model = UNetRefiner()
    model.load_state_dict(ck['model'] if 'model' in ck else ck)
    model.eval().to(dev)
    print('BPR_APPLY tag=%s ckpt=%s (val_loss=%s epoch=%s) device=%s' %
          (args.tag, args.ckpt, ck.get('val_loss'), ck.get('epoch'), dev),
          flush=True)

    coco_gt = COCO(args.ann)
    cat_id = sorted(coco_gt.getCatIds())[0]
    with open(args.pred, 'rb') as f:
        preds = pickle.load(f)
    if args.limit:
        preds = preds[:args.limit]

    orig_dets, ref_dets = [], []
    n_inst, n_ref, n_patch, n_chg = 0, 0, 0, 0
    t0 = time.time()
    for k, entry in enumerate(preds):
        img_id = int(entry['img_id'])
        info = coco_gt.loadImgs([img_id])[0]
        path = os.path.join(args.img_dir, info['file_name'])
        if not os.path.exists(path):
            path = entry.get('img_path', path)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        pi = entry['pred_instances']
        scores = np.asarray(pi['scores']).tolist()
        for i, rle in enumerate(pi['masks']):
            sc = float(scores[i])
            seg = dict(size=rle['size'],
                       counts=rle['counts'].decode()
                       if isinstance(rle['counts'], bytes) else rle['counts'])
            orig_dets.append(dict(image_id=img_id, category_id=cat_id,
                                  segmentation=seg, score=sc))
            n_inst += 1
            if img is None or sc < args.score_thr:
                ref_dets.append(dict(image_id=img_id, category_id=cat_id,
                                     segmentation=seg, score=sc))
                continue
            m = maskUtils.decode(rle).astype(np.uint8)
            refined, np_ = refine_instance(
                img, m, model, dev, K=args.K, stride=args.stride,
                band_r=args.band, thr=args.thr)
            n_patch += np_
            if np_ > 0:
                n_ref += 1
                n_chg += int((refined != m).sum())
            r2 = maskUtils.encode(np.asfortranarray(refined))
            r2['counts'] = r2['counts'].decode()
            ref_dets.append(dict(image_id=img_id, category_id=cat_id,
                                 segmentation=r2, score=sc))
        if k % 100 == 0:
            try:
                with open(args.alive, 'w') as f:
                    f.write(str(int(time.time())))
            except OSError:
                pass
            print('BPR_APPLY progress %d/%d inst=%d refined=%d secs=%.0f' %
                  (k, len(preds), n_inst, n_ref, time.time() - t0), flush=True)

    with open(args.out, 'w') as f:
        json.dump(ref_dets, f)
    print('BPR_APPLY tag=%s imgs=%d inst=%d refined=%d patches=%d '
          'changed_px=%d secs=%.0f out=%s' %
          (args.tag, len(preds), n_inst, n_ref, n_patch, n_chg,
           time.time() - t0, args.out), flush=True)
    coco_stats(coco_gt, orig_dets, args.tag, 'orig')
    coco_stats(coco_gt, ref_dets, args.tag, 'refined')


if __name__ == '__main__':
    main()
