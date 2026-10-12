# -*- coding: utf-8 -*-
"""E3: OFFLINE 3-seed prediction ensemble with vote-consensus fusion (CPU-only).

WHY: the model soup averages WEIGHTS; this fuses the PREDICTIONS of the 3 seed
checkpoints (each already 8-view wide-msTTA vote-consensus merged) - a different,
3x-cost estimator. Fusion logic is a faithful port of the 'vote' branch of
carrier/mask_tta.py MaskTTAModel._merge_single_sample (greedy per-class mask-IoU
clustering around score-desc seeds + consensus-band majority denoise + support-
scaled max score), with n_views = number of MODELS (3) instead of TTA views.

PKL FORMAT (source-verified against the installed mmdet 3.3.0 / mmengine):
  tools/test.py --out X.pkl appends mmdet's ``DumpDetResults`` metric
  (mmdet/evaluation/metrics/dump_det_results.py). mmengine ``Evaluator.process``
  converts every DetDataSample via ``.to_dict()`` BEFORE metrics see it, so the
  pkl is: ``list[dict]`` - one dict per image, dataset order - each carrying the
  sample's metainfo at TOP level ('img_id', 'ori_shape', 'img_path', ...) plus
  'pred_instances': a plain dict with
      'masks' : list[N] of pycocotools RLE dicts {'size': [H, W], 'counts': bytes}
                (DumpDetResults encodes the (N,H,W) bool tensor via
                encode_mask_results before dumping),
      'scores': torch float Tensor (N,) on CPU,
      'labels': torch int Tensor (N,) on CPU,
      'bboxes': torch float Tensor (N,4) on CPU.
  GT keys ('gt_instances', ...) are popped by DumpDetResults. With --tta the
  dumped predictions are the per-seed TTA-MERGED instances (the evaluator sees
  MaskTTAModel.test_step output).

MASK-IoU EQUIVALENCE: mask_tta.py computes pairwise IoU as inter/union on dense
bool masks; ``pycocotools.mask.iou`` with iscrowd=0 computes exactly the same
quantity on RLEs (no decode needed), so the clustering geometry is identical.
Only clusters entering majority fusion are decoded (2..max_support masks each).

IMG-ID SUBSETTING: HRSID test_inshore/test_offshore image ids EQUAL the ids of
the same files in official_test.json (verified 2026-07-19: both are id-preserving
subsets and partition official 369+1593=1962). Predictions do not depend on the
ann_file used at dump time, so ONE official-split pkl per seed serves all three
evals; --ann restricts fusion+eval to the ann's img ids.

Usage:
  python carrier/ensemble_fuse.py \
      --pkls runs/e3/s0_official.pkl runs/e3/s2_official.pkl runs/e3/s3_official.pkl \
      --ann data/hrsid/annotations/test_inshore.json \
      --out runs/e3/e3_inshore.json --tag inshore

Prints one machine-greppable line:  E3_RESULT tag=... segm_mAP: ... segm_mAP_75: ...
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys

import numpy as np
from pycocotools import mask as mask_util
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def _to_numpy(x):
    """torch.Tensor / ndarray / list -> ndarray (torch imported lazily)."""
    if isinstance(x, np.ndarray):
        return x
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


def load_pkl(path):
    """Load one seed's dump -> {img_id: {'rles': list, 'scores': f64, 'labels': i64}}."""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    assert isinstance(data, list) and len(data) > 0, f'{path}: not a non-empty list'
    per_img = {}
    for ds in data:
        assert isinstance(ds, dict) and 'pred_instances' in ds, (
            f'{path}: entry is not a to_dict()-ed DetDataSample: {type(ds)}')
        img_id = int(ds['img_id'])
        pred = ds['pred_instances']
        rles = list(pred.get('masks', []))
        scores = _to_numpy(pred['scores']).astype(np.float64).reshape(-1)
        labels = _to_numpy(pred['labels']).astype(np.int64).reshape(-1)
        assert len(rles) == len(scores) == len(labels), (
            f'{path} img {img_id}: N mismatch {len(rles)}/{len(scores)}/{len(labels)}')
        if rles:
            assert isinstance(rles[0], dict) and 'counts' in rles[0], (
                f'{path} img {img_id}: masks are not RLE dicts (got {type(rles[0])}); '
                'was the pkl dumped by mmdet DumpDetResults?')
        per_img[img_id] = dict(rles=rles, scores=scores, labels=labels)
    return per_img


# --------------------------------------------------------------------------- #
# fusion (faithful port of mask_tta.py vote branch)
# --------------------------------------------------------------------------- #
def _majority_rle(rles, member_idx, frac):
    """Pixelwise majority over the cluster members -> fused RLE."""
    ms = mask_util.decode([rles[i] for i in member_idx])          # (H, W, M)
    fused = (ms.astype(np.float64).mean(axis=2) >= frac)
    return mask_util.encode(np.asfortranarray(fused.astype(np.uint8)))


def fuse_image(rles, scores, labels, n_views, vote_iou=0.5, vote_frac=0.5,
               vote_gamma=0.0, vote_mask='consensus', vote_min_support=2,
               vote_max_support=6, max_per_img=100):
    """Fuse the pooled instances of ONE image. Returns (rles, scores, labels).

    Mirrors mask_tta.py merge='vote': greedy agglomeration around score-desc
    seeds WITHIN each label by TRUE mask-IoU >= vote_iou; fused mask per
    vote_mask ('consensus' = majority only for min<=size<=max agreement band,
    seed otherwise; 'seed' = always seed; 'majority' = always majority);
    score = cluster max * (size/n_views)**vote_gamma; sort + truncate.
    """
    K = len(rles)
    if K == 0:
        return [], np.zeros((0,), np.float64), np.zeros((0,), np.int64)

    # pairwise TRUE mask-IoU straight on RLEs (== dense inter/union of mask_tta)
    iou = np.nan_to_num(
        np.asarray(mask_util.iou(rles, rles, [0] * K), dtype=np.float64))

    order = np.argsort(-scores, kind='stable')
    used = np.zeros(K, dtype=bool)
    f_rles, f_scores, f_labels = [], [], []
    for idx in order.tolist():
        if used[idx]:
            continue
        same = labels == labels[idx]
        members = (~used) & same & (iou[idx] >= vote_iou)
        members[idx] = True                      # seed always joins its cluster
        used |= members
        member_idx = np.nonzero(members)[0]
        cluster_size = int(members.sum())
        if vote_mask == 'seed':
            fused = rles[idx]
        elif vote_mask == 'consensus':
            if vote_min_support <= cluster_size <= vote_max_support:
                fused = _majority_rle(rles, member_idx, vote_frac)
            else:
                fused = rles[idx]
        elif vote_mask == 'majority':
            fused = _majority_rle(rles, member_idx, vote_frac)
        else:
            raise ValueError(f'unsupported vote_mask: {vote_mask!r}')
        support = (cluster_size / n_views) ** vote_gamma
        f_rles.append(fused)
        f_scores.append(scores[members].max() * support)
        f_labels.append(labels[idx])

    f_scores = np.asarray(f_scores, dtype=np.float64)
    f_labels = np.asarray(f_labels, dtype=np.int64)
    keep = np.argsort(-f_scores, kind='stable')[:max_per_img]
    return [f_rles[i] for i in keep.tolist()], f_scores[keep], f_labels[keep]


def fuse_predictions(models, img_ids, cat_ids, **vote_kw):
    """Pool the models' instances per image, fuse, emit COCO det dicts.

    Args:
        models: list of per-seed dicts from load_pkl.
        img_ids: iterable of image ids to fuse (the eval split's ids).
        cat_ids: sorted COCO category ids; label L -> cat_ids[L].
    """
    n_views = len(models)
    dets = []
    n_pool = n_out = 0
    for img_id in img_ids:
        rles, scores, labels = [], [], []
        for m in models:
            assert img_id in m, f'img_id {img_id} missing from a seed pkl'
            e = m[img_id]
            rles.extend(e['rles'])
            scores.append(e['scores'])
            labels.append(e['labels'])
        scores = np.concatenate(scores) if scores else np.zeros((0,))
        labels = (np.concatenate(labels).astype(np.int64)
                  if labels else np.zeros((0,), np.int64))
        if rles:
            sz0 = tuple(rles[0]['size'])
            assert all(tuple(r['size']) == sz0 for r in rles), (
                f'img {img_id}: mixed RLE sizes across seeds')
        f_rles, f_scores, f_labels = fuse_image(
            rles, scores, labels, n_views=n_views, **vote_kw)
        n_pool += len(rles)
        n_out += len(f_rles)
        for r, s, lab in zip(f_rles, f_scores.tolist(), f_labels.tolist()):
            counts = r['counts']
            if isinstance(counts, bytes):
                counts = counts.decode('utf-8')
            dets.append(dict(
                image_id=int(img_id),
                category_id=int(cat_ids[int(lab)]),
                segmentation=dict(size=[int(v) for v in r['size']],
                                  counts=counts),
                score=float(s),
                bbox=[float(v) for v in mask_util.toBbox(r).tolist()]))
    print(f'[e3-fuse] pooled {n_pool} instances over {len(list(img_ids))} imgs '
          f'-> fused {n_out}')
    return dets


# --------------------------------------------------------------------------- #
# end-to-end
# --------------------------------------------------------------------------- #
def run(pkls, ann, out_json, tag, vote_iou=0.5, vote_frac=0.5, vote_gamma=0.0,
        vote_mask='consensus', vote_min_support=2, vote_max_support=6,
        max_per_img=100):
    """Load -> fuse -> dump COCO json -> COCOeval segm. Returns E.stats."""
    models = [load_pkl(p) for p in pkls]
    coco = COCO(ann)
    img_ids = sorted(coco.getImgIds())
    cat_ids = sorted(coco.getCatIds())
    missing = [i for i in img_ids if any(i not in m for m in models)]
    assert not missing, (
        f'{len(missing)} ann img ids missing from the pkls (e.g. {missing[:5]}); '
        'were the dumps run on a superset split like official_test?')
    dets = fuse_predictions(
        models, img_ids, cat_ids, vote_iou=vote_iou, vote_frac=vote_frac,
        vote_gamma=vote_gamma, vote_mask=vote_mask,
        vote_min_support=vote_min_support, vote_max_support=vote_max_support,
        max_per_img=max_per_img)
    with open(out_json, 'w') as f:
        json.dump(dets, f)
    print(f'[e3-fuse] wrote {len(dets)} dets -> {out_json}')

    coco_dt = coco.loadRes(out_json)
    E = COCOeval(coco, coco_dt, 'segm')
    E.evaluate()
    E.accumulate()
    E.summarize()
    s = E.stats
    print(f'E3_RESULT tag={tag} segm_mAP: {s[0]:.4f} segm_mAP_50: {s[1]:.4f} '
          f'segm_mAP_75: {s[2]:.4f} segm_mAP_s: {s[3]:.4f} '
          f'segm_mAP_m: {s[4]:.4f} segm_mAP_l: {s[5]:.4f} '
          f'AR100: {s[8]:.4f} n_models={len(pkls)} vote_mask={vote_mask} '
          f'vote_iou={vote_iou} band=[{vote_min_support},{vote_max_support}] '
          f'frac={vote_frac} gamma={vote_gamma}')
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--pkls', nargs='+', required=True,
                    help='per-seed DumpDetResults pkls (same or superset split)')
    ap.add_argument('--ann', required=True, help='COCO ann json of the eval split')
    ap.add_argument('--out', required=True, help='output COCO det json')
    ap.add_argument('--tag', default='e3', help='tag echoed in E3_RESULT line')
    ap.add_argument('--vote-iou', type=float, default=0.5)
    ap.add_argument('--vote-frac', type=float, default=0.5)
    ap.add_argument('--vote-gamma', type=float, default=0.0)
    ap.add_argument('--vote-mask', default='consensus',
                    choices=['consensus', 'seed', 'majority'])
    ap.add_argument('--vote-min-support', type=int, default=2)
    ap.add_argument('--vote-max-support', type=int, default=6,
                    help='consensus band upper edge; with 3 models any '
                         'size-2/3 cluster falls inside the default band')
    ap.add_argument('--max-per-img', type=int, default=100)
    a = ap.parse_args(argv)
    run(a.pkls, a.ann, a.out, a.tag, vote_iou=a.vote_iou, vote_frac=a.vote_frac,
        vote_gamma=a.vote_gamma, vote_mask=a.vote_mask,
        vote_min_support=a.vote_min_support, vote_max_support=a.vote_max_support,
        max_per_img=a.max_per_img)
    return 0


if __name__ == '__main__':
    sys.exit(main())
