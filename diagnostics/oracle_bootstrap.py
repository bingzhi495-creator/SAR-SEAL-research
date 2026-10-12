# -*- coding: utf-8 -*-
"""PR-gate #3: image-level bootstrap CIs for the oracle decomposition (base AP75,
mask-share, coverage-share). Fixes the COCOeval np.unique(imgIds) dedup flaw of
bootstrap_ap75.py by RELABELLING every resampled image occurrence with a fresh
image id (cloning its GT image+annotations and the base/mask/full pred records),
so a with-replacement resample is counted with true multiplicity.

Builds base/mask_oracle/full_oracle exactly as diagnostics/oracle_decomp.py
(re-implemented here to avoid mutating that code-verified script). Pure CPU.

Usage: python diagnostics\\oracle_bootstrap.py --dataset pseg --stratum inshore --B 1000
"""
from __future__ import annotations
import argparse, os, pickle
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


def with_bbox_area(rec):
    seg = rec["segmentation"]
    rle = {"size": seg["size"], "counts": seg["counts"].encode() if isinstance(seg["counts"], str) else seg["counts"]}
    rec = dict(rec)
    rec["bbox"] = mask_util.toBbox(rle).tolist()
    rec["area"] = float(mask_util.area(rle))
    return rec


def build_sets(dataset, split, stratum):
    """Returns coco + per-image {iid: (img_dict, gt_anns, base_recs, mask_recs, full_recs)}."""
    if dataset == "pseg":
        gt = os.path.join(BASE, "data", "pseg_ssdd", "annotations", "test.json")
        preds_path = os.path.join(BASE, "runs", "refine_gate", "A0_stock_preds.pkl")
    else:
        ann = "scene_test.json" if split == "scene" else "official_test.json"
        gt = os.path.join(BASE, "data", "hrsid", "annotations", ann)
        preds_path = os.path.join(BASE, "runs",
                                  "a0_hrsid_scene" if split == "scene" else "a0_hrsid_seed0", "preds.pkl")
    coco = COCO(gt)
    if dataset == "pseg" and stratum != "all":
        from protocol.datasets import _subset_stems, PSEG_ROOT
        ins, off = _subset_stems(PSEG_ROOT)
        sel = set(ins if stratum == "inshore" else off)
        name2id = {os.path.splitext(os.path.basename(im["file_name"]))[0]: i for i, im in coco.imgs.items()}
        keep = {name2id[s] for s in sel if s in name2id}
    else:
        keep = set(coco.imgs.keys())

    preds = pickle.load(open(preds_path, "rb"))
    id_by_name = {os.path.splitext(os.path.basename(im["file_name"]))[0]: i for i, im in coco.imgs.items()}
    per = {}
    for p in preds:
        stem = os.path.splitext(os.path.basename(p["img_path"]))[0]
        iid = id_by_name.get(stem)
        if iid is None or iid not in keep:
            continue
        im = coco.imgs[iid]; h, w = im["height"], im["width"]
        gts = coco.imgToAnns.get(iid, [])
        gt_rles = [to_rle(a["segmentation"], h, w) for a in gts]
        gt_dec = [mask_util.decode(r).astype(bool) for r in gt_rles]
        gt_used = [False] * len(gts)
        inst = p["pred_instances"]
        scores = inst["scores"].cpu().numpy() if hasattr(inst["scores"], "cpu") else np.asarray(inst["scores"])
        masks = inst["masks"]
        base_r, mask_r, full_r = [], [], []
        for k in np.argsort(-scores):
            rle = masks[k]
            rle = {"size": rle["size"], "counts": rle["counts"].decode() if isinstance(rle["counts"], bytes) else rle["counts"]}
            pm = mask_util.decode(rle).astype(bool)
            sc = float(scores[k])
            base_r.append({"category_id": 1, "segmentation": rle, "score": sc})
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
            seg = rle if bj < 0 else gt_rles[bj]
            if bj >= 0:
                gt_used[bj] = True
            mrec = {"category_id": 1, "segmentation": seg, "score": sc}
            mask_r.append(mrec); full_r.append(mrec)
        for j, a in enumerate(gts):
            if not gt_used[j]:
                full_r.append({"category_id": 1, "segmentation": gt_rles[j], "score": 1.0})
        per[iid] = (im, gts, base_r, mask_r, full_r)
    return coco, per


def ap75_relabeled(sample_iids, per):
    """Build a relabeled GT + 3 result sets (base/mask/full) with fresh ids per
    occurrence, then COCOeval AP75 for each. True with-replacement multiplicity."""
    images, anns = [], []
    res = {"base": [], "mask": [], "full": []}
    nid = 0; nann = 0
    for iid in sample_iids:
        im, gts, base_r, mask_r, full_r = per[iid]
        nid += 1
        images.append({"id": nid, "height": im["height"], "width": im["width"], "file_name": f"{nid}.jpg"})
        for a in gts:
            nann += 1
            anns.append({"id": nann, "image_id": nid, "category_id": 1,
                         "segmentation": a["segmentation"], "area": a.get("area", 1.0),
                         "bbox": a.get("bbox", [0, 0, 1, 1]), "iscrowd": a.get("iscrowd", 0)})
        for key, recs in (("base", base_r), ("mask", mask_r), ("full", full_r)):
            for r in recs:
                rr = with_bbox_area(r); rr["image_id"] = nid
                res[key].append(rr)
    gt = COCO(); gt.dataset = {"images": images, "annotations": anns,
                               "categories": [{"id": 1, "name": "ship"}]}
    gt.createIndex()
    out = {}
    for key in ("base", "mask", "full"):
        if not res[key]:
            out[key] = 0.0; continue
        dt = gt.loadRes(res[key])
        ev = COCOeval(gt, dt, "segm"); ev.evaluate(); ev.accumulate(); ev.summarize()
        out[key] = ev.stats[2]
    return out["base"], out["mask"], out["full"]


def main():
    import sys, contextlib, io
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pseg", choices=["pseg", "hrsid"])
    ap.add_argument("--split", default="official", choices=["official", "scene"])
    ap.add_argument("--stratum", default="inshore", choices=["inshore", "offshore", "all"])
    ap.add_argument("--B", type=int, default=1000)
    args = ap.parse_args()
    import torch  # noqa

    coco, per = build_sets(args.dataset, args.split, args.stratum)
    iids = list(per.keys()); n = len(iids)
    print(f"[{args.dataset}/{args.split}/{args.stratum}] {n} images, B={args.B}", flush=True)

    def silent(f, *a):
        with contextlib.redirect_stdout(io.StringIO()):
            return f(*a)

    b0, m0, f0 = silent(ap75_relabeled, iids, per)
    tot0 = (f0 - b0) or 1e-9
    print(f"  point: base={b0:.4f} mask_oracle={m0:.4f} full={f0:.4f} | mask%={100*(m0-b0)/tot0:.1f} cov%={100*(f0-m0)/tot0:.1f}", flush=True)

    rng = np.random.default_rng(0)
    B, Mp, Cp = [], [], []
    for b in range(args.B):
        samp = rng.choice(iids, size=n, replace=True)
        bb, mm, ff = silent(ap75_relabeled, samp, per)
        tot = (ff - bb) or 1e-9
        B.append(bb); Mp.append(100 * (mm - bb) / tot); Cp.append(100 * (ff - mm) / tot)
        if (b + 1) % 100 == 0:
            print(f"  boot {b+1}/{args.B}  base CI~[{np.percentile(B,2.5):.3f},{np.percentile(B,97.5):.3f}]  mask%~{np.mean(Mp):.1f}", flush=True)
    B, Mp, Cp = map(np.array, (B, Mp, Cp))
    def ci(x): return f"{x.mean():.3f} [95% {np.percentile(x,2.5):.3f}, {np.percentile(x,97.5):.3f}]"
    print(f"\n=== oracle decomposition bootstrap ({args.dataset}/{args.split}/{args.stratum}, B={args.B}, relabeled) ===")
    print(f"  base AP75 : point {b0:.4f} | boot {ci(B)}")
    print(f"  mask-share%   : point {100*(m0-b0)/tot0:.1f} | boot {ci(Mp)}")
    print(f"  coverage-share%: point {100*(f0-m0)/tot0:.1f} | boot {ci(Cp)}")


if __name__ == "__main__":
    main()
