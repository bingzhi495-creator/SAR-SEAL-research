# -*- coding: utf-8 -*-
"""SCIC R11/R12: learned mask-quality re-ranker (interpretable GBDT, no pixels).

Pattern: BPR (tiny component on the frozen main model's OWN train-split
outputs). Axis attacked: SCORE-RANKING quality. Target = mask-IoU with the
best-matching GT (regression on ALL train candidates, kept + reject pool).
Features (14, scalar only -- NO image patch, instant + interpretable):
  vote-consensus (6: support, n_views, mean/min pairwise mask-IoU, seed_score,
  score_spread, exported by MaskTTAModel export_vote_stats=True) + det score
  + geometry (7: log_area, log_w, log_h, aspect, fill, cx, cy).

R12 (feature-count AUTODETECT): when the stack dump was ALSO run with
model.test_cfg.export_depth_stats=True, vote_stats carries 8 columns
(6 consensus + [seed_depth_iou_mean, seed_score_traj] = depth-consistency of
the seed instance across the last 3 decoder layers) and the feature vector
becomes 16-dim = stats(8) + score + geometry(7). Detection is purely by the
stats column count in the pkl/npz -- harvest/train/apply are otherwise
UNCHANGED and a 6-col (14-dim) stack keeps the exact R11 behaviour. At apply
time the model checkpoint's feature dim must match the stack's (asserted).

Model: sklearn HistGradientBoostingRegressor (the histogram variant of
GradientBoostingRegressor -- same GBDT family, ~50x faster on ~700k rows;
LS loss). Output blends into the det score:
    score' = alpha * clip(model_out, 0, 1) + (1 - alpha) * orig_score
alpha swept {0.3, 0.5, 1.0} at train time on HELD-OUT TRAIN images (20% group
split by img_id) via a train-side COCOeval segm AP proxy over the raw stack
candidates -- printed honestly as SCIC_RERANK_ALPHA lines; NO test peeking.
The final model is refit on 100% of the train rows with the chosen alpha
stored in the checkpoint.

Apply preserves the candidate set exactly: base detections (the +BPR refined
jsons, whose scores/order mirror the stack dump 1:1, see bpr_apply.py) keep
their masks; each base det is matched (greedy best mask-IoU >= match_iou) to
a stack-dump instance to fetch its consensus features; matched dets get the
blended score, unmatched keep the original (safety, counted).

Run (from E:/SAR/seal, PYTHONPATH=E:/SAR/seal):
  python carrier/scic_rerank.py harvest --pred runs/scic/train_stack.pkl \
    --ann data/hrsid/annotations/official_train.json \
    --out runs/scic/rerank_feats.npz
  python carrier/scic_rerank.py train --feats runs/scic/rerank_feats.npz \
    --ann data/hrsid/annotations/official_train.json \
    --pred runs/scic/train_stack.pkl --out runs/scic/reranker.pkl
  python carrier/scic_rerank.py apply --pred runs/scic/test_official_stack.pkl \
    --base-json runs/bpr/refined_inshore.json \
    --ann data/hrsid/annotations/test_inshore.json \
    --model runs/scic/reranker.pkl --out runs/scic/r11_inshore.json \
    --tag inshore
"""
from __future__ import annotations

import argparse
import json
import pickle
import time

import numpy as np
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO

from carrier.scic_rescue import (GEOM_NAMES, VOTE_STAT_NAMES, coco_eval_dets,
                                 geom_feats, load_stack_pkl, to_numpy, _touch)

DEPTH_STAT_NAMES = ['seed_depth_iou_mean', 'seed_score_traj']  # R12 extras
RERANK_FEAT_NAMES = VOTE_STAT_NAMES + ['score'] + GEOM_NAMES  # 14 dims (R11)
ALPHAS = (0.3, 0.5, 1.0)


def feat_names_for(stats_dim):
    """Feature names for a given vote_stats column count (6 -> 14 names,
    8 -> 16 names). Order ALWAYS = [stats cols..., score, geometry] because
    candidate_rows/rerank_dets concatenate in that order."""
    if stats_dim == 6:
        return RERANK_FEAT_NAMES
    if stats_dim == 8:
        return VOTE_STAT_NAMES + DEPTH_STAT_NAMES + ['score'] + GEOM_NAMES
    raise ValueError(f'unsupported vote_stats dim {stats_dim} (want 6 or 8)')


# --------------------------------------------------------------------------- #
# harvest
# --------------------------------------------------------------------------- #
def candidate_rows(rles, scores, stats, gt_rles, floor):
    """One image -> (feats (M, D), target (M,), idx (M,)) for score >= floor.

    D = stats_cols + 1 + 7 (14 for the 6-col R11 stats, 16 for the 8-col
    R12 depth-extended stats -- autodetected from `stats`).
    target = mask-IoU with the best-matching GT (0 if no GT / no overlap).
    """
    scores = np.asarray(scores)
    stats = np.asarray(stats)
    n_dim = stats.shape[1] + 1 + len(GEOM_NAMES) if stats.ndim == 2 \
        else len(RERANK_FEAT_NAMES)
    idx = np.nonzero(scores >= floor)[0]
    if idx.size == 0:
        return (np.zeros((0, n_dim), np.float32),
                np.zeros((0,), np.float32), idx)
    tgt = np.zeros(idx.size, dtype=np.float32)
    if gt_rles:
        iou = np.atleast_2d(maskUtils.iou(
            [rles[i] for i in idx], gt_rles, [0] * len(gt_rles)))
        tgt = iou.max(axis=1).astype(np.float32)
    feats = np.stack([
        np.concatenate([stats[i].astype(np.float32),
                        np.array([scores[i]], np.float32),
                        geom_feats(rles[i])])
        for i in idx])
    return feats, tgt, idx


def cmd_harvest(args):
    t0 = time.time()
    coco = COCO(args.ann)
    stack = load_stack_pkl(args.pred, need_stats=True)
    img_ids = sorted(set(coco.getImgIds()) & set(stack.keys()))
    all_f, all_t, all_id = [], [], []
    for n, img_id in enumerate(img_ids):
        e = stack[img_id]
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
        gt_rles = [coco.annToRLE(a) for a in anns]
        f, t, _ = candidate_rows(e['rles'], e['scores'], e['stats'], gt_rles,
                                 args.floor)
        if len(f):
            all_f.append(f)
            all_t.append(t)
            all_id.append(np.full(len(f), img_id, np.int64))
        if args.alive and n % 500 == 0:
            _touch(args.alive)
    feats = np.concatenate(all_f)
    target = np.concatenate(all_t)
    img_id_arr = np.concatenate(all_id)
    names = feat_names_for(feats.shape[1] - 1 - len(GEOM_NAMES))
    np.savez_compressed(args.out, feats=feats, target=target,
                        img_id=img_id_arr,
                        feat_names=np.array(names))
    print('SCIC_RERANK_HARVEST rows=%d imgs=%d dim=%d matched_frac=%.3f '
          'mean_target=%.4f secs=%.0f out=%s'
          % (len(target), len(img_ids), feats.shape[1],
             float((target >= 0.5).mean()),
             float(target.mean()), time.time() - t0, args.out), flush=True)


# --------------------------------------------------------------------------- #
# train (+ honest train-side alpha sweep)
# --------------------------------------------------------------------------- #
def fit_gbdt(X, y, seed=0, max_iter=300, lr=0.08):
    from sklearn.ensemble import HistGradientBoostingRegressor
    m = HistGradientBoostingRegressor(
        max_iter=max_iter, learning_rate=lr, max_depth=None,
        early_stopping=False, random_state=seed)
    m.fit(X, y)
    return m


def _eval_alpha_proxy(coco, stack, val_ids, model, alpha, floor, tag):
    """Train-side AP proxy: COCOeval segm on held-out TRAIN images over the
    raw stack candidates with blended scores."""
    cat_id = sorted(coco.getCatIds())[0]
    dets = []
    for img_id in val_ids:
        e = stack[img_id]
        f, _, idx = candidate_rows(e['rles'], e['scores'], e['stats'], [],
                                   floor)
        if idx.size == 0:
            continue
        pred = np.clip(model.predict(f), 0.0, 1.0) if alpha > 0 else None
        for k, i in enumerate(idx):
            s = float(e['scores'][i])
            if alpha > 0:
                s = alpha * float(pred[k]) + (1.0 - alpha) * s
            rle = e['rles'][i]
            counts = rle['counts']
            if isinstance(counts, bytes):
                counts = counts.decode()
            dets.append(dict(image_id=int(img_id), category_id=int(cat_id),
                             segmentation=dict(size=[int(v) for v in
                                                     rle['size']],
                                               counts=counts),
                             score=s))
    from pycocotools.cocoeval import COCOeval
    dt = coco.loadRes(dets)
    ev = COCOeval(coco, dt, 'segm')
    ev.params.imgIds = sorted(val_ids)
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    s = ev.stats
    print('SCIC_RERANK_ALPHA tag=%s alpha=%.1f trainval_AP=%.4f '
          'trainval_AP75=%.4f (train-side proxy on %d HELD-OUT train imgs, '
          'raw stack candidates -- NOT a test number)'
          % (tag, alpha, s[0], s[2], len(val_ids)), flush=True)
    return s[0]


def cmd_train(args):
    t0 = time.time()
    d = np.load(args.feats, allow_pickle=True)
    X, y, img_id = d['feats'], d['target'], d['img_id']
    # feature-count autodetect (14 = R11, 16 = R12 depth-extended)
    feat_names = ([str(n) for n in d['feat_names']] if 'feat_names' in d
                  else feat_names_for(X.shape[1] - 1 - len(GEOM_NAMES)))
    assert len(feat_names) == X.shape[1], \
        f'feat_names {len(feat_names)} != X dim {X.shape[1]}'
    rng = np.random.RandomState(args.seed)
    uniq = np.unique(img_id)
    val_ids = rng.choice(uniq, size=max(1, int(len(uniq) * args.val_frac)),
                         replace=False)
    is_val = np.isin(img_id, val_ids)
    tr, va = ~is_val, is_val
    print('SCIC_RERANK data rows=%d dim=%d fit_imgs=%d val_imgs=%d'
          % (len(y), X.shape[1], len(uniq) - len(val_ids), len(val_ids)),
          flush=True)

    m = fit_gbdt(X[tr], y[tr], seed=args.seed, max_iter=args.max_iter,
                 lr=args.lr)
    pv = np.clip(m.predict(X[va]), 0.0, 1.0)
    pearson = float(np.corrcoef(pv, y[va])[0, 1])
    from scipy.stats import spearmanr
    spear = float(spearmanr(pv, y[va]).correlation)
    mae = float(np.abs(pv - y[va]).mean())
    print('SCIC_RERANK val pearson=%.4f spearman=%.4f mae=%.4f'
          % (pearson, spear, mae), flush=True)

    # honest alpha sweep on the held-out train images (needs stack + ann)
    best_alpha, best_ap = None, -1.0
    if args.pred and args.ann:
        coco = COCO(args.ann)
        stack = load_stack_pkl(args.pred, need_stats=True)
        vids = sorted(int(v) for v in val_ids if int(v) in stack)
        ap0 = _eval_alpha_proxy(coco, stack, vids, m, 0.0, args.floor,
                                'alpha0-orig')
        for alpha in ALPHAS:
            ap = _eval_alpha_proxy(coco, stack, vids, m, alpha, args.floor,
                                   'sweep')
            if ap > best_ap:
                best_alpha, best_ap = alpha, ap
        print('SCIC_RERANK_SWEEP best_alpha=%.1f trainval_AP=%.4f '
              'orig_AP=%.4f delta=%+.4f'
              % (best_alpha, best_ap, ap0, best_ap - ap0), flush=True)
    else:
        best_alpha = args.alpha_default
        print('SCIC_RERANK_SWEEP skipped (no --pred/--ann); alpha=%.1f'
              % best_alpha, flush=True)

    # permutation importance (interpretability), on a val subsample
    try:
        from sklearn.inspection import permutation_importance
        sub = np.nonzero(va)[0]
        sub = sub[rng.permutation(len(sub))[:20000]]
        imp = permutation_importance(m, X[sub], y[sub], n_repeats=3,
                                     random_state=args.seed)
        order = np.argsort(-imp.importances_mean)
        top = ', '.join('%s=%.4f' % (feat_names[i],
                                     imp.importances_mean[i])
                        for i in order[:6])
        print('SCIC_RERANK importance(top6): %s' % top, flush=True)
    except Exception as exc:  # noqa: BLE001 - diagnostics only
        print('SCIC_RERANK importance skipped: %r' % exc, flush=True)

    # final model: refit on ALL rows (standard; alpha chosen on held-out)
    m_full = fit_gbdt(X, y, seed=args.seed, max_iter=args.max_iter, lr=args.lr)
    with open(args.out, 'wb') as f:
        pickle.dump(dict(model=m_full, alpha=float(best_alpha),
                         feat_names=feat_names, floor=args.floor,
                         val_pearson=pearson, val_spearman=spear,
                         val_mae=mae, trainval_ap=best_ap), f)
    print('SCIC_RERANK_DONE alpha=%.1f val_pearson=%.4f val_spearman=%.4f '
          'secs=%.0f out=%s' % (best_alpha, pearson, spear,
                                time.time() - t0, args.out), flush=True)


# --------------------------------------------------------------------------- #
# apply
# --------------------------------------------------------------------------- #
def rerank_dets(base_dets, rles, scores, stats, model, alpha, match_iou):
    """Blend model scores into one image's base dets. Returns
    (new_dets, n_matched). Candidate set / masks / order are PRESERVED --
    only 'score' changes on matched dets."""
    if not base_dets or not rles:
        return [dict(d) for d in base_dets], 0
    base_rles = []
    for d in base_dets:
        s = d['segmentation']
        c = s['counts']
        base_rles.append(dict(size=s['size'],
                              counts=c.encode() if isinstance(c, str) else c))
    iou = np.atleast_2d(maskUtils.iou(base_rles, rles, [0] * len(rles)))
    order = np.argsort(-np.asarray([d['score'] for d in base_dets]))
    used = np.zeros(len(rles), dtype=bool)
    match = np.full(len(base_dets), -1, dtype=np.int64)
    for bi in order:
        row = iou[bi].copy()
        row[used] = -1.0
        ci = int(np.argmax(row))
        if row[ci] >= match_iou:
            match[bi] = ci
            used[ci] = True
    feats_cache = {}
    out, n_matched = [], 0
    for bi, d in enumerate(base_dets):
        nd = dict(d)
        ci = int(match[bi])
        if ci >= 0:
            if ci not in feats_cache:
                feats_cache[ci] = np.concatenate([
                    stats[ci].astype(np.float32),
                    np.array([scores[ci]], np.float32),
                    geom_feats(rles[ci])])[None]
            pred = float(np.clip(model.predict(feats_cache[ci])[0], 0.0, 1.0))
            nd['score'] = alpha * pred + (1.0 - alpha) * float(d['score'])
            n_matched += 1
        out.append(nd)
    return out, n_matched


def cmd_apply(args):
    t0 = time.time()
    with open(args.model, 'rb') as f:
        ck = pickle.load(f)
    model = ck['model']
    alpha = args.alpha if args.alpha is not None else ck['alpha']
    coco = COCO(args.ann)
    ids = set(coco.getImgIds())
    stack = load_stack_pkl(args.pred, need_stats=True)
    # feature-count autodetect + hard guard: the ckpt's dim must match what
    # this stack's stats produce (14 vs 16), else predictions are garbage.
    stats_dim = next(e['stats'].shape[1] for e in stack.values())
    n_dim = stats_dim + 1 + len(GEOM_NAMES)
    ck_names = ck.get('feat_names', RERANK_FEAT_NAMES)
    assert len(ck_names) == n_dim, (
        f'reranker ckpt expects {len(ck_names)} features '
        f'({", ".join(ck_names[:3])}...) but stack {args.pred} yields '
        f'{n_dim} (vote_stats has {stats_dim} cols) -- re-dump or retrain')
    print('SCIC_RERANK_APPLY dim=%d (%s)' % (
        n_dim, 'R12 depth-extended' if stats_dim == 8 else 'R11'), flush=True)
    with open(args.base_json) as f:
        base = json.load(f)
    by_img = {}
    for d in base:
        by_img.setdefault(int(d['image_id']), []).append(d)

    out, n_m, n_tot = [], 0, 0
    for n, img_id in enumerate(sorted(ids)):
        dets = by_img.get(img_id, [])
        n_tot += len(dets)
        if img_id in stack and dets:
            e = stack[img_id]
            nd, nm = rerank_dets(dets, e['rles'], e['scores'], e['stats'],
                                 model, alpha, args.match_iou)
            out.extend(nd)
            n_m += nm
        else:
            out.extend(dict(d) for d in dets)
        if args.alive and n % 500 == 0:
            _touch(args.alive)
    with open(args.out, 'w') as f:
        json.dump(out, f)
    print('SCIC_RERANK_APPLY tag=%s alpha=%.1f dets=%d matched=%d '
          'unmatched=%d secs=%.0f out=%s'
          % (args.tag, alpha, n_tot, n_m, n_tot - n_m, time.time() - t0,
             args.out), flush=True)
    if not args.no_eval:
        coco_eval_dets(coco, base, args.tag, 'base')
        coco_eval_dets(coco, out, args.tag, args.variant,
                       extra=' alpha=%.1f matched=%d/%d' % (alpha, n_m,
                                                            n_tot))


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)

    h = sub.add_parser('harvest')
    h.add_argument('--pred', required=True)
    h.add_argument('--ann', required=True)
    h.add_argument('--out', required=True)
    h.add_argument('--floor', type=float, default=0.01)
    h.add_argument('--alive', default='')
    h.set_defaults(func=cmd_harvest)

    t = sub.add_parser('train')
    t.add_argument('--feats', required=True)
    t.add_argument('--out', required=True)
    t.add_argument('--pred', default='',
                   help='train stack pkl for the honest alpha sweep')
    t.add_argument('--ann', default='')
    t.add_argument('--floor', type=float, default=0.01)
    t.add_argument('--val-frac', type=float, default=0.2)
    t.add_argument('--max-iter', type=int, default=300)
    t.add_argument('--lr', type=float, default=0.08)
    t.add_argument('--alpha-default', type=float, default=0.3)
    t.add_argument('--seed', type=int, default=0)
    t.set_defaults(func=cmd_train)

    a = sub.add_parser('apply')
    a.add_argument('--pred', required=True)
    a.add_argument('--base-json', required=True)
    a.add_argument('--ann', required=True)
    a.add_argument('--model', required=True)
    a.add_argument('--out', required=True)
    a.add_argument('--tag', default='split')
    a.add_argument('--variant', default='R11')
    a.add_argument('--alpha', type=float, default=None,
                   help='override the alpha stored in the model ckpt')
    a.add_argument('--match-iou', type=float, default=0.5)
    a.add_argument('--no-eval', action='store_true')
    a.add_argument('--alive', default='')
    a.set_defaults(func=cmd_apply)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()
