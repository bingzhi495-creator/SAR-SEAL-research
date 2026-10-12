# -*- coding: utf-8 -*-
"""Fixed-seed image holdout of a COCO annotation file (P0-4 train-side re-scan).

Draws a fraction of the IMAGES of --ann without replacement (numpy RandomState
with --seed), keeps every annotation of the drawn images, and writes a COCO
file with the same info/licenses/categories.  Used to build
data/hrsid/annotations/train_holdout20.json = 20% of the HRSID official
TRAINING split (images the trained models have seen) on which the
training-free hyper-parameters are re-scanned by run_post_queue.sh.

Usage (CPU):
  python tools/make_train_holdout.py --ann data/hrsid/annotations/official_train.json \
      --out data/hrsid/annotations/train_holdout20.json --frac 0.2 --seed 0
Prints: HOLDOUT ann=<..> out=<..> frac=.. seed=.. n_img_total=.. n_img=.. n_ann=..
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def _abs(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--ann', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--frac', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args(argv)

    with open(_abs(args.ann), encoding='utf-8') as f:
        d = json.load(f)
    imgs = sorted(d['images'], key=lambda im: im['id'])
    ids = np.array([im['id'] for im in imgs])
    n = len(ids)
    n_keep = int(round(n * args.frac))
    rng = np.random.RandomState(args.seed)
    keep = set(ids[rng.permutation(n)[:n_keep]].tolist())
    out_imgs = [im for im in imgs if im['id'] in keep]
    out_anns = [a for a in d['annotations'] if a['image_id'] in keep]
    out = {k: v for k, v in d.items() if k not in ('images', 'annotations')}
    out['images'] = out_imgs
    out['annotations'] = out_anns
    out_path = _abs(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f)
    print('HOLDOUT ann=%s out=%s frac=%g seed=%d n_img_total=%d n_img=%d n_ann=%d '
          'n_ann_total=%d' % (args.ann, args.out, args.frac, args.seed, n,
                              len(out_imgs), len(out_anns), len(d['annotations'])),
          flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
