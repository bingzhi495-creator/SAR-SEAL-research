# -*- coding: utf-8 -*-
"""SCIC R10: miss-rescue verifier (tiny CNN over the TTA vote REJECT pool).

Pattern: BPR (train a tiny component on the frozen main model's OWN train-split
errors). Here the unsaturated axis is MISSES: near-shore ships the stack finds
only at score < 0.35 (the reject pool). A <500k-param verifier re-scores pool
candidates from (a) the vote-consensus features exported by MaskTTAModel
(``export_vote_stats=True``, see carrier/mask_tta.py), (b) cheap geometry +
local-brightness (shore/clutter proxy) stats, and (c) a 48x48 gray image patch
+ its mask patch.

Subcommands
  harvest  CPU: train_stack.pkl (WITH vote_stats) + train GT ->
           candidates with score in [floor, keep_thr); label 1 iff the
           candidate mask-IoU>=iou_thr matches a GT ship NOT already covered
           by any kept (score>=keep_thr) prediction; npz shards.
  train    minutes (GPU or CPU): tiny CNN verifier, BCE(pos_weight), 10 ep.
  apply    test time: pool candidates from the stack dump that do not overlap
           (IoU>=iou_thr) any kept BASE detection get
           score' = sigmoid_output * rescue_scale and are APPENDED to the base
           detections. SAFETY: rescue_scale defaults to keep_thr (0.35) so a
           rescued score is STRICTLY < the kept floor (sigmoid < 1): rescues
           only fill the ranking tail and can never displace a confident det.
           (The plan sheet said "*0.5"; 0.5*sigmoid can exceed 0.35, which
           would violate its own "below kept range" invariant, so the scale is
           the kept floor itself. --rescue-scale overrides.)

Base detections at apply = the FINAL method's +BPR refined jsons
(runs/bpr/refined_*.json): BPR refines MASKS only (scores/order preserved
1:1 from the stack dump, see bpr_apply.py), and the rescue only APPENDS, so
composing is safe. Rescued masks are raw stack masks (not BPR-refined) --
documented tail-only approximation.

Run (from E:/SAR/seal, PYTHONPATH=E:/SAR/seal):
  python carrier/scic_rescue.py harvest --pred runs/scic/train_stack.pkl \
    --ann data/hrsid/annotations/official_train.json \
    --img-dir D:/SAR/UNet++/HRSID_JPG/HRSID_JPG/JPEGImages1/ \
    --out runs/scic/rescue_patches
  python carrier/scic_rescue.py train --patches runs/scic/rescue_patches \
    --out runs/scic
  python carrier/scic_rescue.py apply --pred runs/scic/test_official_stack.pkl \
    --base-json runs/bpr/refined_inshore.json \
    --ann data/hrsid/annotations/test_inshore.json \
    --img-dir D:/SAR/UNet++/HRSID_JPG/HRSID_JPG/JPEGImages1/ \
    --ckpt runs/scic/verifier.pth --out runs/scic/r10_inshore.json \
    --tag inshore --variant R10
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import time

import cv2
import numpy as np
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

VOTE_STAT_NAMES = ['support', 'n_views', 'mean_pair_iou', 'min_pair_iou',
                   'seed_score', 'score_spread']
GEOM_NAMES = ['log_area', 'log_w', 'log_h', 'aspect', 'fill', 'cx', 'cy']
BRIGHT_NAMES = ['mean_in', 'mean_ctx', 'std_ctx', 'contrast']
# rescue scalar features = vote(6) + score(1) + geom(7) + brightness(4) = 18
RESCUE_FEAT_NAMES = VOTE_STAT_NAMES + ['score'] + GEOM_NAMES + BRIGHT_NAMES
PATCH = 48


# --------------------------------------------------------------------------- #
# shared stack-dump helpers (also imported by scic_rerank.py)
# --------------------------------------------------------------------------- #
def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


def load_stack_pkl(path, need_stats=True):
    """DumpDetResults pkl -> {img_id: dict(rles, scores, labels[, stats])}.

    stats is the per-instance vote_stats float array exported by MaskTTAModel
    with export_vote_stats=True: (N, 6) consensus features (VOTE_STAT_NAMES)
    or, since R12 (dump ALSO run with model.test_cfg.export_depth_stats=True),
    (N, 8) = the 6 + [seed_depth_iou_mean, seed_score_traj]. The column count
    is AUTODETECTED from the array; consumers decide what they accept
    (R11 reranker: either; R10 rescue: slices to the first 6, frozen).
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)
    assert isinstance(data, list) and data, f'{path}: not a non-empty list'
    per_img = {}
    for ds in data:
        img_id = int(ds['img_id'])
        pred = ds['pred_instances']
        rles = list(pred.get('masks', []))
        scores = to_numpy(pred['scores']).astype(np.float64).reshape(-1)
        labels = to_numpy(pred['labels']).astype(np.int64).reshape(-1)
        entry = dict(rles=rles, scores=scores, labels=labels)
        if need_stats:
            assert 'vote_stats' in pred, (
                f'{path} img {img_id}: no vote_stats -- was the dump run with '
                'tta_model.tta_cfg.export_vote_stats=True '
                '(carrier/configs/c0_v2_scic_stack_hrsid.py)?')
            stats = to_numpy(pred['vote_stats']).astype(np.float32)
            if stats.ndim != 2:                      # legacy flat dumps
                stats = stats.reshape(-1, len(VOTE_STAT_NAMES))
            assert stats.shape[1] in (6, 8), (
                f'{path} img {img_id}: vote_stats has {stats.shape[1]} cols '
                '(expected 6 consensus or 8 = consensus + 2 depth)')
            assert stats.shape[0] == len(rles), (
                f'{path} img {img_id}: vote_stats rows {stats.shape[0]} != '
                f'{len(rles)} masks')
            entry['stats'] = stats
        per_img[img_id] = entry
    return per_img


def geom_feats(rle):
    """7 geometry features from one RLE (order = GEOM_NAMES)."""
    H, W = rle['size']
    area = float(maskUtils.area(rle))
    x, y, w, h = [float(v) for v in maskUtils.toBbox(rle)]
    return np.array([
        np.log1p(area),
        np.log1p(w),
        np.log1p(h),
        w / (h + 1e-6),
        area / (w * h + 1e-6),
        (x + w / 2.0) / max(W, 1),
        (y + h / 2.0) / max(H, 1),
    ], dtype=np.float32)


def brightness_feats(img_gray, mask_u8, rle):
    """4 local-brightness stats (shore/clutter proxy, order = BRIGHT_NAMES).

    mean_in: mean gray inside the mask; mean/std_ctx over a context window =
    bbox expanded by 0.5*max(w,h) per side; contrast = mean_in - mean_ctx.
    All in [0,1] gray units.
    """
    H, W = img_gray.shape
    x, y, w, h = [float(v) for v in maskUtils.toBbox(rle)]
    inside = img_gray[mask_u8 > 0]
    mean_in = float(inside.mean()) / 255.0 if inside.size else 0.0
    m = 0.5 * max(w, h) + 1.0
    x0, y0 = int(max(0, x - m)), int(max(0, y - m))
    x1, y1 = int(min(W, x + w + m)), int(min(H, y + h + m))
    ctx = img_gray[y0:y1, x0:x1].astype(np.float32) / 255.0
    mean_ctx = float(ctx.mean()) if ctx.size else 0.0
    std_ctx = float(ctx.std()) if ctx.size else 0.0
    return np.array([mean_in, mean_ctx, std_ctx, mean_in - mean_ctx],
                    dtype=np.float32)


def extract_patch(img_gray, mask_u8, rle, patch=PATCH):
    """48x48 gray + mask patches around the candidate (window centered on the
    bbox center, side = max(patch, 1.5*max(w,h)), clipped in-bounds, resized).
    Returns (img_p u8, mask_p u8)."""
    H, W = img_gray.shape
    x, y, w, h = [float(v) for v in maskUtils.toBbox(rle)]
    side = int(max(patch, np.ceil(1.5 * max(w, h, 1.0))))
    side = min(side, H, W)
    cy, cx = int(y + h / 2.0), int(x + w / 2.0)
    t = min(max(cy - side // 2, 0), H - side)
    l = min(max(cx - side // 2, 0), W - side)
    sl = (slice(t, t + side), slice(l, l + side))
    img_p = cv2.resize(img_gray[sl], (patch, patch),
                       interpolation=cv2.INTER_AREA)
    mask_p = cv2.resize(mask_u8[sl], (patch, patch),
                        interpolation=cv2.INTER_NEAREST)
    return img_p.astype(np.uint8), (mask_p > 0).astype(np.uint8)


def candidate_features(img_gray, rle, score, stat_row):
    """(18,) scalar feature vector + (48,48) img/mask patches for a candidate."""
    mask_u8 = maskUtils.decode(rle).astype(np.uint8)
    feats = np.concatenate([
        stat_row.astype(np.float32),
        np.array([score], dtype=np.float32),
        geom_feats(rle),
        brightness_feats(img_gray, mask_u8, rle),
    ])
    img_p, mask_p = extract_patch(img_gray, mask_u8, rle)
    return feats, img_p, mask_p


# --------------------------------------------------------------------------- #
# harvest
# --------------------------------------------------------------------------- #
def harvest_image(img_gray, rles, scores, stats, gt_rles, floor, keep_thr,
                  iou_thr, rng, neg_per_img):
    """One image -> (feats list, img_p list, mask_p list, labels list).

    label 1 iff the pool candidate (floor <= s < keep_thr) matches
    (mask-IoU >= iou_thr) a GT NOT covered by any kept (s >= keep_thr) pred.
    Negatives are subsampled to neg_per_img (all positives kept).
    """
    # R10 features are FROZEN at the 6 consensus stats (trained verifier
    # expects 18 scalars); an R12 (N, 8) depth-extended dump is sliced back.
    stats = np.asarray(stats)[:, :len(VOTE_STAT_NAMES)]
    scores = np.asarray(scores)
    kept_idx = np.nonzero(scores >= keep_thr)[0]
    pool_idx = np.nonzero((scores >= floor) & (scores < keep_thr))[0]
    if pool_idx.size == 0:
        return [], [], [], []
    covered = set()
    if gt_rles and kept_idx.size:
        iou_kg = np.atleast_2d(maskUtils.iou(
            [rles[i] for i in kept_idx], gt_rles, [0] * len(gt_rles)))
        for gi in range(len(gt_rles)):
            if iou_kg[:, gi].max() >= iou_thr:
                covered.add(gi)
    labels = np.zeros(pool_idx.size, dtype=np.uint8)
    if gt_rles:
        iou_pg = np.atleast_2d(maskUtils.iou(
            [rles[i] for i in pool_idx], gt_rles, [0] * len(gt_rles)))
        for k in range(pool_idx.size):
            gi = int(np.argmax(iou_pg[k]))
            if iou_pg[k, gi] >= iou_thr and gi not in covered:
                labels[k] = 1
    pos = np.nonzero(labels == 1)[0]
    neg = np.nonzero(labels == 0)[0]
    if neg.size > neg_per_img:
        neg = rng.choice(neg, size=neg_per_img, replace=False)
    sel = np.concatenate([pos, neg])
    out_f, out_i, out_m, out_l = [], [], [], []
    for k in sel:
        i = pool_idx[k]
        feats, img_p, mask_p = candidate_features(
            img_gray, rles[i], float(scores[i]), stats[i])
        out_f.append(feats)
        out_i.append(img_p)
        out_m.append(mask_p)
        out_l.append(int(labels[k]))
    return out_f, out_i, out_m, out_l


def cmd_harvest(args):
    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)
    coco = COCO(args.ann)
    stack = load_stack_pkl(args.pred, need_stats=True)
    rng = np.random.RandomState(args.seed)
    img_ids = sorted(set(coco.getImgIds()) & set(stack.keys()))

    buf_f, buf_i, buf_m, buf_l, buf_id = [], [], [], [], []
    shard_id, total, n_pos, n_img = 0, 0, 0, 0

    def flush():
        nonlocal shard_id, buf_f, buf_i, buf_m, buf_l, buf_id
        if not buf_f:
            return
        np.savez_compressed(
            os.path.join(args.out, 'shard_%03d.npz' % shard_id),
            feats=np.stack(buf_f), img=np.stack(buf_i),
            mask=np.stack(buf_m), label=np.asarray(buf_l, np.uint8),
            img_id=np.asarray(buf_id, np.int64))
        shard_id += 1
        buf_f, buf_i, buf_m, buf_l, buf_id = [], [], [], [], []

    for n, img_id in enumerate(img_ids):
        info = coco.loadImgs([img_id])[0]
        path = os.path.join(args.img_dir, info['file_name'])
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        e = stack[img_id]
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
        gt_rles = [coco.annToRLE(a) for a in anns]
        fs, ims, ms, ls = harvest_image(
            img, e['rles'], e['scores'], e['stats'], gt_rles,
            args.floor, args.keep_thr, args.iou_thr, rng, args.neg_per_img)
        if fs:
            n_img += 1
        for f, i_, m_, l_ in zip(fs, ims, ms, ls):
            buf_f.append(f)
            buf_i.append(i_)
            buf_m.append(m_)
            buf_l.append(l_)
            buf_id.append(img_id)
            total += 1
            n_pos += l_
            if len(buf_f) >= args.shard_size:
                flush()
        if args.alive and n % 200 == 0:
            _touch(args.alive)
        if args.cap and total >= args.cap:
            break
    flush()

    meta = dict(floor=args.floor, keep_thr=args.keep_thr, iou_thr=args.iou_thr,
                neg_per_img=args.neg_per_img, seed=args.seed, total=total,
                pos=n_pos, neg=total - n_pos, shards=shard_id,
                images_used=n_img, feat_names=RESCUE_FEAT_NAMES,
                pred=args.pred, ann=args.ann,
                secs=round(time.time() - t0, 1))
    with open(os.path.join(args.out, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    print('SCIC_RESCUE_HARVEST total=%d pos=%d neg=%d shards=%d imgs=%d '
          'secs=%.0f' % (total, n_pos, total - n_pos, shard_id, n_img,
                         time.time() - t0), flush=True)


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def build_verifier(n_feats=len(RESCUE_FEAT_NAMES)):
    """Tiny CNN verifier: 2 conv blocks on the 2ch 48x48 patch + MLP over
    [conv features (512) | scalar feats]. ~85k params (<500k budget)."""
    import torch
    import torch.nn as nn

    class TinyVerifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(2, 16, 3, padding=1, bias=False),
                nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1, bias=False),
                nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d(4))
            self.mlp = nn.Sequential(
                nn.Linear(32 * 16 + n_feats, 128), nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(128, 64), nn.ReLU(inplace=True),
                nn.Linear(64, 1))

        def forward(self, patch, feats):
            z = self.conv(patch).flatten(1)
            return self.mlp(torch.cat([z, feats], dim=1)).squeeze(1)

    return TinyVerifier()


def _load_shards(patch_dir):
    shards = sorted(glob.glob(os.path.join(patch_dir, 'shard_*.npz')))
    assert shards, 'no shards in %s' % patch_dir
    fs, ims, ms, ls, ids = [], [], [], [], []
    for s in shards:
        d = np.load(s)
        fs.append(d['feats'])
        ims.append(d['img'])
        ms.append(d['mask'])
        ls.append(d['label'])
        ids.append(d['img_id'])
    return (np.concatenate(fs), np.concatenate(ims), np.concatenate(ms),
            np.concatenate(ls), np.concatenate(ids))


def train_verifier(feats, imgs, masks, labels, img_ids, out_dir,
                   epochs=10, bs=256, lr=1e-3, val_frac=0.1, seed=0,
                   device=None, alive=None):
    """Fit the tiny verifier; returns (ckpt_path, val_auc, val_ap)."""
    import torch
    import torch.nn as nn
    from sklearn.metrics import average_precision_score, roc_auc_score

    dev = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    rng = np.random.RandomState(seed)
    uniq = np.unique(img_ids)
    val_ids = set(rng.choice(uniq, size=max(1, int(len(uniq) * val_frac)),
                             replace=False).tolist())
    is_val = np.isin(img_ids, list(val_ids))
    tr, va = np.nonzero(~is_val)[0], np.nonzero(is_val)[0]

    mu = feats[tr].mean(axis=0)
    sd = feats[tr].std(axis=0) + 1e-6
    fz = ((feats - mu) / sd).astype(np.float32)
    patches = np.stack([imgs.astype(np.float32) / 255.0,
                        masks.astype(np.float32)], axis=1)
    y = labels.astype(np.float32)

    model = build_verifier(feats.shape[1]).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params < 500_000, f'verifier too big: {n_params}'
    pos = max(1.0, float(y[tr].sum()))
    neg = float(len(tr)) - pos
    pos_weight = torch.tensor([min(20.0, neg / pos)], device=dev)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    torch.manual_seed(seed)

    def run_val():
        model.eval()
        outs = []
        with torch.no_grad():
            for b in range(0, len(va), 1024):
                sel = va[b:b + 1024]
                p = torch.from_numpy(patches[sel]).to(dev)
                f = torch.from_numpy(fz[sel]).to(dev)
                outs.append(torch.sigmoid(model(p, f)).cpu().numpy())
        pr = np.concatenate(outs) if outs else np.zeros(0)
        if len(va) and y[va].min() != y[va].max():
            return (roc_auc_score(y[va], pr),
                    average_precision_score(y[va], pr))
        return float('nan'), float('nan')

    best_auc, ckpt_path = -1.0, os.path.join(out_dir, 'verifier.pth')
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(tr)
        tot, nb = 0.0, 0
        for b in range(0, len(perm), bs):
            sel = perm[b:b + bs]
            p = torch.from_numpy(patches[sel]).to(dev)
            f = torch.from_numpy(fz[sel]).to(dev)
            t = torch.from_numpy(y[sel]).to(dev)
            loss = crit(model(p, f), t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
            nb += 1
            if alive and nb % 50 == 0:
                _touch(alive)
        auc, ap = run_val()
        print('SCIC_VERIFIER ep=%d loss=%.4f val_auc=%.4f val_ap=%.4f'
              % (ep, tot / max(nb, 1), auc, ap), flush=True)
        if not np.isnan(auc) and auc > best_auc:
            best_auc = auc
            torch.save(dict(model=model.state_dict(), feat_mu=mu, feat_sd=sd,
                            n_feats=feats.shape[1], epoch=ep, val_auc=auc,
                            val_ap=ap, feat_names=RESCUE_FEAT_NAMES),
                       ckpt_path)
    auc, ap = run_val()
    if best_auc < 0:  # degenerate val (single class) -> still save the model
        torch.save(dict(model=model.state_dict(), feat_mu=mu, feat_sd=sd,
                        n_feats=feats.shape[1], epoch=epochs - 1, val_auc=auc,
                        val_ap=ap, feat_names=RESCUE_FEAT_NAMES), ckpt_path)
    print('SCIC_VERIFIER_DONE params=%d best_val_auc=%.4f last_val_auc=%.4f '
          'last_val_ap=%.4f ckpt=%s' % (n_params, best_auc, auc, ap,
                                        ckpt_path), flush=True)
    return ckpt_path, best_auc, ap


def cmd_train(args):
    feats, imgs, masks, labels, img_ids = _load_shards(args.patches)
    print('SCIC_VERIFIER data n=%d pos=%d dim=%d'
          % (len(labels), int(labels.sum()), feats.shape[1]), flush=True)
    train_verifier(feats, imgs, masks, labels, img_ids, args.out,
                   epochs=args.epochs, bs=args.bs, lr=args.lr,
                   seed=args.seed, device=args.device, alive=args.alive)


# --------------------------------------------------------------------------- #
# apply
# --------------------------------------------------------------------------- #
def load_verifier(ckpt_path, device='cpu'):
    import torch
    ck = torch.load(ckpt_path, map_location='cpu')
    model = build_verifier(ck['n_feats'])
    model.load_state_dict(ck['model'])
    model.eval().to(device)
    return model, ck['feat_mu'], ck['feat_sd']


def rescue_image(img_gray, rles, scores, stats, base_kept_rles, model, mu, sd,
                 device, floor, keep_thr, iou_suppress, rescue_scale,
                 min_prob):
    """Rescued dets for one image: list of (rle, score', prob)."""
    import torch
    # R10 features frozen at the 6 consensus stats (see harvest_image).
    stats = np.asarray(stats)[:, :len(VOTE_STAT_NAMES)]
    scores = np.asarray(scores)
    pool_idx = np.nonzero((scores >= floor) & (scores < keep_thr))[0]
    if pool_idx.size == 0:
        return []
    if base_kept_rles:
        iou = np.atleast_2d(maskUtils.iou(
            [rles[i] for i in pool_idx], base_kept_rles,
            [0] * len(base_kept_rles)))
        free = iou.max(axis=1) < iou_suppress
        pool_idx = pool_idx[free]
    if pool_idx.size == 0:
        return []
    fs, ps = [], []
    for i in pool_idx:
        f, img_p, mask_p = candidate_features(
            img_gray, rles[i], float(scores[i]), stats[i])
        fs.append(f)
        ps.append(np.stack([img_p.astype(np.float32) / 255.0,
                            mask_p.astype(np.float32)]))
    fz = ((np.stack(fs) - mu) / sd).astype(np.float32)
    with torch.no_grad():
        p = torch.from_numpy(np.stack(ps)).to(device)
        f = torch.from_numpy(fz).to(device)
        prob = torch.sigmoid(model(p, f)).cpu().numpy()
    out = []
    for k, i in enumerate(pool_idx):
        if prob[k] < min_prob:
            continue
        out.append((rles[i], float(prob[k]) * rescue_scale, float(prob[k])))
    return out


def coco_eval_dets(coco_gt, dets, tag, variant, extra=''):
    if not dets:
        print('SCIC_RESULT tag=%s variant=%s EMPTY' % (tag, variant),
              flush=True)
        return None
    dt = coco_gt.loadRes([dict(d) for d in dets])
    ev = COCOeval(coco_gt, dt, 'segm')
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    s = ev.stats
    print('SCIC_RESULT tag=%s variant=%s AP=%.4f AP50=%.4f AP75=%.4f '
          'APs=%.4f APm=%.4f APl=%.4f%s'
          % (tag, variant, s[0], s[1], s[2], s[3], s[4], s[5], extra),
          flush=True)
    return s


def cmd_apply(args):
    import torch
    t0 = time.time()
    dev = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model, mu, sd = load_verifier(args.ckpt, dev)
    coco = COCO(args.ann)
    ids = set(coco.getImgIds())
    stack = load_stack_pkl(args.pred, need_stats=True)
    with open(args.base_json) as f:
        base = json.load(f)
    by_img = {}
    for d in base:
        by_img.setdefault(int(d['image_id']), []).append(d)

    out = list(base)
    n_res, n_img = 0, 0
    for n, img_id in enumerate(sorted(ids)):
        if img_id not in stack:
            continue
        info = coco.loadImgs([img_id])[0]
        img = cv2.imread(os.path.join(args.img_dir, info['file_name']),
                         cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        e = stack[img_id]
        kept = [d['segmentation'] for d in by_img.get(img_id, [])
                if d['score'] >= args.keep_thr]
        kept = [dict(size=s['size'], counts=s['counts'].encode()
                     if isinstance(s['counts'], str) else s['counts'])
                for s in kept]
        rescued = rescue_image(
            img, e['rles'], e['scores'], e['stats'], kept, model, mu, sd,
            dev, args.floor, args.keep_thr, args.iou_suppress,
            args.rescue_scale, args.min_prob)
        cat_id = sorted(coco.getCatIds())[0]
        for rle, sc, prob in rescued:
            counts = rle['counts']
            if isinstance(counts, bytes):
                counts = counts.decode()
            out.append(dict(image_id=int(img_id), category_id=int(cat_id),
                            segmentation=dict(size=[int(v) for v in
                                                    rle['size']],
                                              counts=counts),
                            score=float(sc)))
            n_res += 1
        n_img += 1
        if args.alive and n % 200 == 0:
            _touch(args.alive)
    with open(args.out, 'w') as f:
        json.dump(out, f)
    print('SCIC_RESCUE_APPLY tag=%s imgs=%d base=%d rescued=%d scale=%.2f '
          'secs=%.0f out=%s' % (args.tag, n_img, len(base), n_res,
                                args.rescue_scale, time.time() - t0,
                                args.out), flush=True)
    if not args.no_eval:
        coco_eval_dets(coco, base, args.tag, 'base')
        coco_eval_dets(coco, out, args.tag, args.variant,
                       extra=' n_base=%d n_rescued=%d' % (len(base), n_res))


def _touch(path):
    try:
        with open(path, 'w') as f:
            f.write(str(int(time.time())))
    except OSError:
        pass


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)

    h = sub.add_parser('harvest')
    h.add_argument('--pred', required=True)
    h.add_argument('--ann', required=True)
    h.add_argument('--img-dir', required=True)
    h.add_argument('--out', required=True)
    h.add_argument('--floor', type=float, default=0.01)
    h.add_argument('--keep-thr', type=float, default=0.35)
    h.add_argument('--iou-thr', type=float, default=0.5)
    h.add_argument('--neg-per-img', type=int, default=30)
    h.add_argument('--cap', type=int, default=0)
    h.add_argument('--shard-size', type=int, default=20000)
    h.add_argument('--seed', type=int, default=0)
    h.add_argument('--alive', default='')
    h.set_defaults(func=cmd_harvest)

    t = sub.add_parser('train')
    t.add_argument('--patches', required=True)
    t.add_argument('--out', required=True)
    t.add_argument('--epochs', type=int, default=10)
    t.add_argument('--bs', type=int, default=256)
    t.add_argument('--lr', type=float, default=1e-3)
    t.add_argument('--seed', type=int, default=0)
    t.add_argument('--device', default=None)
    t.add_argument('--alive', default='')
    t.set_defaults(func=cmd_train)

    a = sub.add_parser('apply')
    a.add_argument('--pred', required=True,
                   help='stack pkl WITH vote_stats (official superset ok)')
    a.add_argument('--base-json', required=True,
                   help='base COCO det json (e.g. runs/bpr/refined_*.json '
                        'or an R11 output for the R10+R11 stack)')
    a.add_argument('--ann', required=True)
    a.add_argument('--img-dir', required=True)
    a.add_argument('--ckpt', required=True)
    a.add_argument('--out', required=True)
    a.add_argument('--tag', default='split')
    a.add_argument('--variant', default='R10')
    a.add_argument('--floor', type=float, default=0.01)
    a.add_argument('--keep-thr', type=float, default=0.35)
    a.add_argument('--iou-suppress', type=float, default=0.5)
    a.add_argument('--rescue-scale', type=float, default=0.35,
                   help='score\'=sigmoid*scale; default=keep_thr so rescued '
                        'scores stay STRICTLY below the kept floor')
    a.add_argument('--min-prob', type=float, default=0.05)
    a.add_argument('--device', default=None)
    a.add_argument('--no-eval', action='store_true')
    a.add_argument('--alive', default='')
    a.set_defaults(func=cmd_apply)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()
