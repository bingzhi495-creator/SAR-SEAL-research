# -*- coding: utf-8 -*-
"""Where does the SeedDINO recall gain (+1.85 AR) come from? Per-GT box recall
(IoU>=0.5 with any pred, top-100 by score) for SeedDINO vs no-seed DINO, sliced
by size + how marginal the recall is. Pure CPU on dumped det preds.
"""
import os, pickle
import numpy as np
from pycocotools.coco import COCO

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def box_iou(g, P):
    if len(P) == 0:
        return np.zeros(0)
    gx1, gy1, gw, gh = g; gx2, gy2 = gx1 + gw, gy1 + gh
    px1, py1, pw, ph = P[:, 0], P[:, 1], P[:, 2], P[:, 3]; px2, py2 = px1 + pw, py1 + ph
    ix1 = np.maximum(gx1, px1); iy1 = np.maximum(gy1, py1)
    ix2 = np.minimum(gx2, px2); iy2 = np.minimum(gy2, py2)
    iw = np.clip(ix2 - ix1, 0, None); ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih; union = gw * gh + pw * ph - inter
    return inter / np.clip(union, 1e-9, None)


def load(path):
    preds = pickle.load(open(path, "rb"))
    out = {}
    for p in preds:
        stem = os.path.splitext(os.path.basename(p["img_path"]))[0]
        inst = p["pred_instances"]
        b = inst["bboxes"].cpu().numpy() if hasattr(inst["bboxes"], "cpu") else np.asarray(inst["bboxes"])
        s = inst["scores"].cpu().numpy() if hasattr(inst["scores"], "cpu") else np.asarray(inst["scores"])
        order = np.argsort(-s)[:100]
        b = b[order]
        # xyxy -> xywh
        if len(b):
            b = np.stack([b[:, 0], b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1)
        out[stem] = b
    return out


def recall_by_size(coco, preds, ins_stems):
    buckets = {"small": [0, 1024], "medium": [1024, 9216], "large": [9216, 1e18]}
    rec = {k: [0, 0] for k in buckets}  # [hit, total]
    rec_inshore = [0, 0]
    id_by_name = {os.path.splitext(im["file_name"])[0]: i for i, im in coco.imgs.items()}
    for stem, iid in id_by_name.items():
        if stem not in preds:
            continue
        P = preds[stem]
        for a in coco.imgToAnns.get(iid, []):
            ar = a["area"]; hit = (box_iou(a["bbox"], P).max() >= 0.5) if len(P) else False
            for k, (lo, hi) in buckets.items():
                if lo <= ar < hi:
                    rec[k][0] += hit; rec[k][1] += 1
            if stem in ins_stems:
                rec_inshore[0] += hit; rec_inshore[1] += 1
    return rec, rec_inshore


def main():
    import torch  # noqa
    coco = COCO(os.path.join(BASE, "data", "pseg_ssdd", "annotations", "test.json"))
    from protocol.datasets import _subset_stems, PSEG_ROOT
    ins, _ = _subset_stems(PSEG_ROOT)
    nocrop = load(os.path.join(BASE, "runs", "multiseed", "dino_nocrop_s1_preds.pkl"))
    seed = load(os.path.join(BASE, "runs", "multiseed", "seeddino_s1_preds.pkl"))
    rn, rni = recall_by_size(coco, nocrop, ins)
    rs, rsi = recall_by_size(coco, seed, ins)
    print("\n===== SeedDINO recall mechanism (PSeg seed1, recall@100 IoU>=0.5) =====")
    for k in ("small", "medium", "large"):
        a = rn[k][0] / max(rn[k][1], 1); b = rs[k][0] / max(rs[k][1], 1)
        print(f"  {k:7s} (n={rn[k][1]:3d}): no-seed {a:.4f} -> seed {b:.4f}  ({b-a:+.4f})")
    a = rni[0] / max(rni[1], 1); b = rsi[0] / max(rsi[1], 1)
    print(f"  inshore (n={rni[1]:3d}): no-seed {a:.4f} -> seed {b:.4f}  ({b-a:+.4f})")


if __name__ == "__main__":
    main()
