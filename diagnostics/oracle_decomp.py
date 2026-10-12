# -*- coding: utf-8 -*-
"""B2 fix: a REAL AP75-additive oracle decomposition of the inshore gap, replacing
the count-share heuristic the reviewer flagged.

Builds three prediction sets and runs COCOeval on each (inshore stratum):
  base       : the A0 predictions as-is                      -> AP75_base
  mask_oracle: every A0 pred that matches a GT (IoU>=0.5) has its mask REPLACED
               by that GT's mask (perfect masks for DETECTED instances; detection
               unchanged)                                    -> AP75_mask
  full_oracle: mask_oracle + every MISSED GT added as a perfect prediction
               (score 1, GT mask)                            -> AP75_full (~1.0)

Decomposition of the deficit (1 - AP75_base):
  mask-resolution recoverable  = AP75_mask - AP75_base
  detection-coverage recoverable = AP75_full - AP75_mask
This is AP-additive and commensurable (unlike the old count-share). Pure CPU.

Usage: python diagnostics\\oracle_decomp.py [--stratum inshore]
"""
from __future__ import annotations
import argparse, os, pickle, copy
import numpy as np
from pycocotools import mask as mask_util
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def to_rle(seg, h, w):
    if isinstance(seg, dict):
        c = seg["counts"]
        return {"size": seg["size"], "counts": c.decode() if isinstance(c, bytes) else c}
    r = mask_util.frPyObjects(seg, h, w)
    r = mask_util.merge(r) if isinstance(r, list) else r
    return {"size": r["size"], "counts": r["counts"].decode() if isinstance(r["counts"], bytes) else r["counts"]}


def ap75(coco, results, img_ids, by_size=False):
    if not results:
        return (0.0, {}) if by_size else 0.0
    for r in results:  # COCOeval needs bbox + area; derive from the RLE
        seg = r["segmentation"]
        rle = {"size": seg["size"], "counts": seg["counts"].encode() if isinstance(seg["counts"], str) else seg["counts"]}
        r["bbox"] = mask_util.toBbox(rle).tolist()
        r["area"] = float(mask_util.area(rle))
    dt = coco.loadRes(results)
    ev = COCOeval(coco, dt, "segm"); ev.params.imgIds = list(img_ids)
    ev.evaluate(); ev.accumulate(); ev.summarize()
    if not by_size:
        return ev.stats[2]
    # AP75 per area range from the precision array: [T,R,K,A,M], T idx 5 = IoU 0.75,
    # A = {0:all,1:small,2:medium,3:large}, M idx 2 = maxDets 100.
    pr = ev.eval["precision"]
    sizes = {}
    for nm, ai in (("all", 0), ("small", 1), ("medium", 2), ("large", 3)):
        p = pr[5, :, 0, ai, 2]
        sizes[nm] = float(p[p > -1].mean()) if (p > -1).any() else float("nan")
    return ev.stats[2], sizes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pseg", choices=["pseg", "hrsid"])
    ap.add_argument("--split", default="official", choices=["official", "scene"],
                    help="hrsid only: official (leaky) vs scene (leakage-free) test set")
    ap.add_argument("--stratum", default="inshore", choices=["inshore", "offshore", "all"])
    args = ap.parse_args()
    import torch  # noqa

    if args.dataset == "pseg":
        gt = os.path.join(BASE, "data", "pseg_ssdd", "annotations", "test.json")
        preds_path = os.path.join(BASE, "runs", "refine_gate", "A0_stock_preds.pkl")
        tag = "PSeg A0"
    else:  # hrsid: no local inshore/offshore strata
        if args.stratum != "all":
            raise SystemExit("HRSID has no inshore/offshore strata locally; use --stratum all")
        ann = "scene_test.json" if args.split == "scene" else "official_test.json"
        gt = os.path.join(BASE, "data", "hrsid", "annotations", ann)
        preds_path = os.path.join(BASE, "runs",
                                  "a0_hrsid_scene" if args.split == "scene" else "a0_hrsid_seed0",
                                  "preds.pkl")
        tag = f"HRSID A0 ({args.split})"

    coco = COCO(gt)
    if args.dataset == "pseg":
        from protocol.datasets import _subset_stems, PSEG_ROOT
        ins, off = _subset_stems(PSEG_ROOT)
        name2id = {os.path.splitext(os.path.basename(im["file_name"]))[0]: i for i, im in coco.imgs.items()}
        if args.stratum == "inshore":
            img_ids = [name2id[s] for s in ins if s in name2id]
        elif args.stratum == "offshore":
            img_ids = [name2id[s] for s in off if s in name2id]
        else:
            img_ids = list(coco.imgs.keys())
    else:
        img_ids = list(coco.imgs.keys())
    img_id_set = set(img_ids)

    preds = pickle.load(open(preds_path, "rb"))
    id_by_name = {os.path.splitext(os.path.basename(im["file_name"]))[0]: i for i, im in coco.imgs.items()}

    base, mask_oracle, full_oracle = [], [], []
    for p in preds:
        stem = os.path.splitext(os.path.basename(p["img_path"]))[0]
        iid = id_by_name.get(stem)
        if iid is None or iid not in img_id_set:
            continue
        im = coco.imgs[iid]; h, w = im["height"], im["width"]
        gts = coco.imgToAnns.get(iid, [])
        gt_rles = [to_rle(a["segmentation"], h, w) for a in gts]
        gt_dec = [mask_util.decode(r).astype(bool) for r in gt_rles]
        gt_used = [False] * len(gts)

        inst = p["pred_instances"]
        scores = inst["scores"].cpu().numpy() if hasattr(inst["scores"], "cpu") else np.asarray(inst["scores"])
        masks = inst["masks"]
        order = np.argsort(-scores)  # greedy match high score first
        for k in order:
            rle = masks[k]
            rle = {"size": rle["size"], "counts": rle["counts"].decode() if isinstance(rle["counts"], bytes) else rle["counts"]}
            pm = mask_util.decode(rle).astype(bool)
            rec = {"image_id": iid, "category_id": 1, "segmentation": rle, "score": float(scores[k])}
            base.append(rec)
            # match to best unused GT
            best, bj = 0.5, -1
            for j, g in enumerate(gt_dec):
                if gt_used[j]:
                    continue
                inter = np.logical_and(g, pm).sum()
                if inter == 0:
                    continue
                iou = inter / (g.sum() + pm.sum() - inter)
                if iou > best:
                    best, bj = iou, j
            mrec = dict(rec)
            if bj >= 0:
                gt_used[bj] = True
                mrec = {"image_id": iid, "category_id": 1, "segmentation": gt_rles[bj], "score": float(scores[k])}
            mask_oracle.append(mrec)
            full_oracle.append(mrec)
        # add missed GTs as perfect preds (detection oracle)
        for j, a in enumerate(gts):
            if not gt_used[j]:
                full_oracle.append({"image_id": iid, "category_id": 1, "segmentation": gt_rles[j], "score": 1.0})

    a_base, base_sz = ap75(coco, base, img_ids, by_size=True)
    a_mask, mask_sz = ap75(coco, mask_oracle, img_ids, by_size=True)
    a_full = ap75(coco, full_oracle, img_ids)
    d_mask = a_mask - a_base
    d_det = a_full - a_mask
    tot = d_mask + d_det
    print(f"\n===== oracle AP75 decomposition [{args.stratum}] ({tag}) =====")
    print(f"  AP75 base={a_base:.4f}  mask_oracle={a_mask:.4f}  full_oracle={a_full:.4f}")
    print(f"  deficit (full-base) = {a_full - a_base:.4f}")
    print(f"  mask-resolution recoverable  = {d_mask:.4f}  ({d_mask/tot*100:.0f}% of recoverable)")
    print(f"  detection-coverage recoverable = {d_det:.4f}  ({d_det/tot*100:.0f}% of recoverable)")
    print("  --- mask-recoverable AP75 BY SIZE (base -> perfect-mask oracle) ---")
    for nm in ("small", "medium", "large"):
        b, m = base_sz.get(nm, float("nan")), mask_sz.get(nm, float("nan"))
        print(f"    {nm:7s}: AP75 {b:.4f} -> {m:.4f}  (mask-recoverable {m-b:+.4f})")


if __name__ == "__main__":
    main()
