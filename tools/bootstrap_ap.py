# -*- coding: utf-8 -*-
"""Paired percentile bootstrap of COCO segm AP over images (P0-6).

Two prediction sets A and B on the SAME ground-truth split are re-scored on
identical image resamples, so the difference dAP = AP(B) - AP(A) is paired.

Method
------
1. Each configuration is evaluated ONCE with pycocotools COCOeval ('segm',
   area='all', maxDets=100).  This yields, per image, the per-detection score,
   the per-IoU-threshold match flag and the ignore flags, plus the number of
   non-ignored ground-truth instances.  Those per-image quantities do not
   depend on which other images are present.
2. A bootstrap resample draws N image ids WITH replacement from the N images of
   the GT file (fixed seed).  A duplicated image contributes its detections and
   its ground truth once per draw, i.e. it is treated as a fresh image with the
   same content.  The COCO accumulate step (global score sort, cumulative TP/FP,
   101-point interpolated precision, mean over the 10 IoU thresholds) is then
   re-run on the resampled pool for A and B on the SAME resample.
   This is exactly equivalent to duplicating the image + its GT + its
   detections under new image ids and calling COCOeval again, but ~1000x
   faster because the per-image IoU matching is not recomputed.
3. Reported: N images, the full-data AP of A and B (identity resample, which
   reproduces the logged COCOeval numbers to 4 decimals), dAP mean over the
   resamples, its 2.5 / 97.5 percentiles (95% percentile CI), and the fraction
   of resamples with dAP > 0.  AP50 / AP75 are reported alongside.

Inputs: a COCO-style results json (list of {image_id, category_id,
segmentation(RLE), score}) as written by carrier/bpr_apply.py and
carrier/scic_rerank.py, or an mmdet test.py --out pkl (list of per-image dicts
with 'img_id' and 'pred_instances' {scores, masks[RLE]}), which is converted
exactly as carrier/bpr_apply.py converts it (kind=orig).

Usage (CPU only; from E:/SAR/seal):
  CUDA_VISIBLE_DEVICES=-1 python tools/bootstrap_ap.py --preset paper --n 1000 --skip-done
  CUDA_VISIBLE_DEVICES=-1 python tools/bootstrap_ap.py \
      --pair inshore_bpr_vs_r11 runs/bpr/refined_inshore.json \
             runs/scic/r11_inshore.json data/hrsid/annotations/test_inshore.json
Result lines (greppable, appended to --out, default runs/_bootstrap.out):
  BOOTSTRAP_RESULT pair=<name> ... dAP=<B-A> mean=<..> ci95=[lo,hi] frac_pos=<..>
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import pickle
import sys
import time

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# (name, A, B, GT) -- dAP is B minus A.  Paths are ROOT-relative.
PRESETS = {
    'paper': [
        # (i) HRSID, three splits: +BPR (A) vs +re-rank R11 on top of +BPR (B)
        ('hrsid_inshore_bpr_vs_rerank', 'runs/bpr/refined_inshore.json',
         'runs/scic/r11_inshore.json', 'data/hrsid/annotations/test_inshore.json'),
        ('hrsid_official_bpr_vs_rerank', 'runs/bpr/refined_official.json',
         'runs/scic/r11_official.json', 'data/hrsid/annotations/official_test.json'),
        ('hrsid_offshore_bpr_vs_rerank', 'runs/bpr/refined_offshore.json',
         'runs/scic/r11_offshore.json', 'data/hrsid/annotations/test_offshore.json'),
        # (ii) HRSID, three splits: stack (A, the --tta dump the refiner consumed,
        #      run_bpr_queue.sh step4a) vs +BPR (B, run_bpr_queue.sh step4b)
        ('hrsid_inshore_stack_vs_bpr', 'runs/bpr/test_inshore.pkl',
         'runs/bpr/refined_inshore.json', 'data/hrsid/annotations/test_inshore.json'),
        ('hrsid_official_stack_vs_bpr', 'runs/bpr/test_official.pkl',
         'runs/bpr/refined_official.json', 'data/hrsid/annotations/official_test.json'),
        ('hrsid_offshore_stack_vs_bpr', 'runs/bpr/test_offshore.pkl',
         'runs/bpr/refined_offshore.json', 'data/hrsid/annotations/test_offshore.json'),
        # (iii) PSeg-SSDD inshore / offshore: soup base no-TTA (A, run_ssdd2_queue.sh (a))
        #       vs stack (B, run_ssdd_queue.sh step5 --tta dump)
        ('ssdd_inshore_base_vs_stack', 'runs/ssdd/base_inshore.pkl',
         'runs/ssdd/stack_ins.pkl', 'data/pseg_ssdd/annotations/test_inshore.json'),
        ('ssdd_offshore_base_vs_stack', 'runs/ssdd/base_offshore.pkl',
         'runs/ssdd/stack_ofs.pkl', 'data/pseg_ssdd/annotations/test_offshore.json'),
        # (iv) PSeg-SSDD inshore / offshore: stack (A) vs +BPR (B, run_ssdd2_queue.sh (c))
        ('ssdd_inshore_stack_vs_bpr', 'runs/ssdd/stack_ins.pkl',
         'runs/ssdd/refined_inshore.json', 'data/pseg_ssdd/annotations/test_inshore.json'),
        ('ssdd_offshore_stack_vs_bpr', 'runs/ssdd/stack_ofs.pkl',
         'runs/ssdd/refined_offshore.json', 'data/pseg_ssdd/annotations/test_offshore.json'),
        # (v) extras for the Table II lower block (R2 5.7): SSDD +BPR vs +re-rank (r11fix)
        ('ssdd_inshore_bpr_vs_rerank', 'runs/ssdd/refined_inshore.json',
         'runs/scic/ssdd_r11fix_inshore.json', 'data/pseg_ssdd/annotations/test_inshore.json'),
        ('ssdd_offshore_bpr_vs_rerank', 'runs/ssdd/refined_offshore.json',
         'runs/scic/ssdd_r11fix_offshore.json', 'data/pseg_ssdd/annotations/test_offshore.json'),
    ],
}


def _abs(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def load_dets(path, cat_id):
    """json results list, or mmdet --out pkl converted like bpr_apply (orig)."""
    path = _abs(path)
    if path.endswith('.json'):
        with open(path) as f:
            return json.load(f)
    with open(path, 'rb') as f:
        preds = pickle.load(f)
    dets = []
    for entry in preds:
        img_id = int(entry['img_id'])
        pi = entry['pred_instances']
        scores = np.asarray(pi['scores']).tolist()
        for i, rle in enumerate(pi['masks']):
            counts = rle['counts']
            if isinstance(counts, bytes):
                counts = counts.decode()
            dets.append(dict(image_id=img_id, category_id=cat_id,
                             segmentation=dict(size=list(rle['size']),
                                               counts=counts),
                             score=float(scores[i])))
    return dets


class PerImageEval:
    """Run COCOeval.evaluate once; keep per-image quantities for re-accumulation."""

    def __init__(self, coco_gt, dets, max_det=100):
        self.T = 10
        self.R = 101
        self.max_det = max_det
        with contextlib.redirect_stdout(io.StringIO()):
            coco_dt = coco_gt.loadRes(dets) if dets else COCO()
            ev = COCOeval(coco_gt, coco_dt, 'segm')
            p = ev.params
            p.areaRng = [[0 ** 2, 1e5 ** 2]]      # 'all' only -> AP/AP50/AP75
            p.areaRngLbl = ['all']
            p.maxDets = [max_det]
            p.imgIds = sorted(coco_gt.getImgIds())
            p.catIds = sorted(coco_gt.getCatIds())
            ev.evaluate()
        self.img_ids = list(p.imgIds)
        self.cat_ids = list(p.catIds)
        self.rec_thrs = p.recThrs
        self.iou_thrs = p.iouThrs
        I, K = len(self.img_ids), len(self.cat_ids)
        # per category: concatenated det arrays + owner image index + npig/img
        self.cats = []
        for k in range(K):
            scores, dtm, dtig, owner = [], [], [], []
            npig = np.zeros(I, dtype=np.int64)
            for i in range(I):
                e = ev.evalImgs[k * I + i]          # A=1 -> offset k*I+i
                if e is None:
                    continue
                gt_ig = np.asarray(e['gtIgnore']).reshape(-1)
                npig[i] = int(np.count_nonzero(gt_ig == 0))
                s = np.asarray(e['dtScores'])[:max_det]
                if len(s):
                    scores.append(s)
                    dtm.append(np.asarray(e['dtMatches'])[:, :max_det] > 0)
                    dtig.append(np.asarray(e['dtIgnore'])[:, :max_det].astype(bool))
                    owner.append(np.full(len(s), i, dtype=np.int64))
            if scores:
                scores = np.concatenate(scores)
                dtm = np.concatenate(dtm, axis=1)
                dtig = np.concatenate(dtig, axis=1)
                owner = np.concatenate(owner)
            else:
                scores = np.zeros(0)
                dtm = np.zeros((self.T, 0), bool)
                dtig = np.zeros((self.T, 0), bool)
                owner = np.zeros(0, np.int64)
            self.cats.append(dict(scores=scores, tp=dtm & ~dtig,
                                  fp=(~dtm) & ~dtig, owner=owner, npig=npig))
        self.n_img = I
        self.n_det = int(sum(len(c['scores']) for c in self.cats))

    def precision(self, mult):
        """mult: (I,) int multiplicity of each image. -> precision [T,R,K]."""
        K = len(self.cats)
        prec = -np.ones((self.T, self.R, K))
        for k, c in enumerate(self.cats):
            npig = int(np.dot(mult, c['npig']))
            if npig == 0:
                continue
            m_det = mult[c['owner']]
            idx = np.repeat(np.arange(len(c['scores'])), m_det)
            if len(idx) == 0:
                prec[:, :, k] = 0.0
                continue
            sc = c['scores'][idx]
            order = np.argsort(-sc, kind='mergesort')
            idx = idx[order]
            tp = np.cumsum(c['tp'][:, idx], axis=1, dtype=np.float64)
            fp = np.cumsum(c['fp'][:, idx], axis=1, dtype=np.float64)
            rc = tp / npig
            pr = tp / (fp + tp + np.spacing(1))
            # monotone-decreasing envelope (COCOeval's backward max loop)
            pr = np.maximum.accumulate(pr[:, ::-1], axis=1)[:, ::-1]
            for t in range(self.T):
                inds = np.searchsorted(rc[t], self.rec_thrs, side='left')
                q = np.zeros(self.R)
                ok = inds < len(pr[t])
                q[ok] = pr[t][inds[ok]]
                prec[t, :, k] = q
        return prec

    def stats(self, mult):
        prec = self.precision(mult)

        def ap_of(x):
            v = x[x > -1]
            return float(v.mean()) if v.size else -1.0
        return ap_of(prec), ap_of(prec[0]), ap_of(prec[5])   # AP, AP50, AP75

    def summarize_reference(self, coco_gt, dets):
        """Stock COCOeval.summarize() on the full data (sanity check)."""
        with contextlib.redirect_stdout(io.StringIO()):
            coco_dt = coco_gt.loadRes(dets)
            ev = COCOeval(coco_gt, coco_dt, 'segm')
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
        return float(ev.stats[0]), float(ev.stats[1]), float(ev.stats[2])


def run_pair(name, pa, pb, gt, n_boot, seed, out_fh, label_a='A', label_b='B',
             check=True):
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(_abs(gt))
    cat_id = sorted(coco_gt.getCatIds())[0]
    dets_a = load_dets(pa, cat_id)
    dets_b = load_dets(pb, cat_id)
    ea = PerImageEval(coco_gt, dets_a)
    eb = PerImageEval(coco_gt, dets_b)
    assert ea.img_ids == eb.img_ids
    N = ea.n_img
    ones = np.ones(N, dtype=np.int64)
    full_a, full_b = ea.stats(ones), eb.stats(ones)
    ref_line = ''
    if check:
        ra = ea.summarize_reference(coco_gt, dets_a)
        rb = eb.summarize_reference(coco_gt, dets_b)
        ok = all(abs(x - y) < 5e-5 for x, y in zip(full_a, ra)) and \
            all(abs(x - y) < 5e-5 for x, y in zip(full_b, rb))
        ref_line = ('  check vs stock COCOeval.summarize: A=%.4f/%.4f/%.4f '
                    'B=%.4f/%.4f/%.4f -> %s' %
                    (tuple(ra) + tuple(rb) + ('MATCH' if ok else 'MISMATCH',)))
    rng = np.random.RandomState(seed)
    d = np.zeros((n_boot, 3))
    for b in range(n_boot):
        draw = rng.randint(0, N, size=N)
        mult = np.bincount(draw, minlength=N)
        sa, sb = ea.stats(mult), eb.stats(mult)
        d[b] = [100.0 * (y - x) for x, y in zip(sa, sb)]

    def summ(col):
        v = d[:, col]
        lo, hi = np.percentile(v, [2.5, 97.5])
        return (v.mean(), v.std(ddof=1), lo, hi, float((v > 0).mean()),
                float((v >= 0).mean()))
    sap, s50, s75 = summ(0), summ(1), summ(2)
    point = [100.0 * (y - x) for x, y in zip(full_a, full_b)]
    lines = [
        'BOOTSTRAP_RESULT pair=%s gt=%s n_img=%d n_det_A=%d n_det_B=%d n_boot=%d seed=%d '
        'A=%s B=%s AP_A=%.4f AP_B=%.4f dAP=%+.2f mean=%+.2f sd=%.2f ci95=[%+.2f,%+.2f] '
        'frac_pos=%.3f frac_nonneg=%.3f | AP50_A=%.4f AP50_B=%.4f dAP50=%+.2f '
        'ci95_50=[%+.2f,%+.2f] frac_pos_50=%.3f | AP75_A=%.4f AP75_B=%.4f dAP75=%+.2f '
        'ci95_75=[%+.2f,%+.2f] frac_pos_75=%.3f | secs=%.0f' % (
            name, gt, N, ea.n_det, eb.n_det, n_boot, seed, label_a, label_b,
            full_a[0], full_b[0], point[0], sap[0], sap[1], sap[2], sap[3], sap[4], sap[5],
            full_a[1], full_b[1], point[1], s50[2], s50[3], s50[4],
            full_a[2], full_b[2], point[2], s75[2], s75[3], s75[4],
            time.time() - t0),
        '  A=%s (%s)' % (label_a, pa),
        '  B=%s (%s)' % (label_b, pb),
    ]
    if ref_line:
        lines.append(ref_line)
    text = '\n'.join(lines)
    print(text, flush=True)
    if out_fh is not None:
        out_fh.write(text + '\n')
        out_fh.flush()
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pair', nargs=4, action='append', metavar=('NAME', 'A', 'B', 'GT'),
                    help='one paired comparison (dAP = AP(B) - AP(A)); repeatable')
    ap.add_argument('--preset', choices=sorted(PRESETS), default=None,
                    help='run the hard-coded pair list (paper = P0-6 + R2 5.7)')
    ap.add_argument('--n', type=int, default=1000, help='bootstrap resamples')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='runs/_bootstrap.out', help='append result lines')
    ap.add_argument('--no-check', action='store_true',
                    help='skip the stock-COCOeval cross-check of the full-data AP')
    ap.add_argument('--skip-done', action='store_true',
                    help='skip pairs that already have a BOOTSTRAP_RESULT line in --out')
    ap.add_argument('--label-a', default='A')
    ap.add_argument('--label-b', default='B')
    args = ap.parse_args(argv)

    pairs = list(PRESETS[args.preset]) if args.preset else []
    for p in (args.pair or []):
        pairs.append(tuple(p))
    if not pairs:
        ap.error('give --preset or at least one --pair')
    out_path = _abs(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    done = set()
    if args.skip_done and os.path.isfile(out_path):
        with open(out_path, encoding='utf-8', errors='replace') as fh:
            for line in fh:
                if line.startswith('BOOTSTRAP_RESULT pair='):
                    done.add(line.split()[1][len('pair='):])
    pairs = [p for p in pairs if p[0] not in done]
    if not pairs:
        print('nothing to do (all pairs already in %s)' % out_path)
        return 0
    with open(out_path, 'a', encoding='utf-8') as fh:
        fh.write('===== bootstrap_ap n=%d seed=%d %s =====\n' %
                 (args.n, args.seed, time.strftime('%Y-%m-%d %H:%M:%S')))
        for (name, pa, pb, gt) in pairs:
            for p in (pa, pb, gt):
                if not os.path.isfile(_abs(p)):
                    msg = 'BOOTSTRAP_SKIP pair=%s missing=%s' % (name, p)
                    print(msg, flush=True)
                    fh.write(msg + '\n')
                    break
            else:
                run_pair(name, pa, pb, gt, args.n, args.seed, fh,
                         args.label_a, args.label_b, check=not args.no_check)
    return 0


if __name__ == '__main__':
    sys.exit(main())
