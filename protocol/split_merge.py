# -*- coding: utf-8 -*-
"""Split / merge rate and connected-component purity (verdict-required eval,
v2 §5 definitions). Quantifies the邻船粘连 failure mode directly, beyond AP.

Definitions (per matched GT-pred at a fixed score threshold):
  merge : one predicted instance overlaps (IoU_partial >= t) >= 2 distinct GTs
  split : one GT is covered (IoU_partial >= t) by >= 2 distinct predictions
  purity: for each predicted mask, fraction of its pixels belonging to the
          single best-matching GT (1.0 = clean, low = spans multiple objects)
"""
from __future__ import annotations

import numpy as np


def _partial_iou(pred, gt):
    """Intersection / GT-area (coverage of GT by pred)."""
    inter = np.logical_and(pred, gt).sum()
    return inter / gt.sum() if gt.sum() > 0 else 0.0


def split_merge_rates(pred_masks, gt_masks, cover_t: float = 0.3):
    """Return dict with merge_rate, split_rate, mean_purity.

    pred_masks, gt_masks: lists of (H,W) bool arrays for ONE image.
    """
    P, G = len(pred_masks), len(gt_masks)
    if P == 0 or G == 0:
        return {"merge_rate": 0.0, "split_rate": 0.0, "mean_purity": float("nan"),
                "n_pred": P, "n_gt": G}
    cover = np.zeros((P, G))
    for i, p in enumerate(pred_masks):
        for j, g in enumerate(gt_masks):
            cover[i, j] = _partial_iou(p, g)

    merges = sum(1 for i in range(P) if (cover[i] >= cover_t).sum() >= 2)
    splits = sum(1 for j in range(G) if (cover[:, j] >= cover_t).sum() >= 2)

    purities = []
    for i, p in enumerate(pred_masks):
        a = p.sum()
        if a == 0:
            continue
        best = max((np.logical_and(p, g).sum() for g in gt_masks), default=0)
        purities.append(best / a)

    return {"merge_rate": merges / P, "split_rate": splits / G,
            "mean_purity": float(np.mean(purities)) if purities else float("nan"),
            "n_pred": P, "n_gt": G, "n_merge": merges, "n_split": splits}


def aggregate(per_image):
    """Aggregate per-image dicts into dataset-level rates (instance-weighted)."""
    tp = sum(d["n_pred"] for d in per_image)
    tg = sum(d["n_gt"] for d in per_image)
    mg = sum(d.get("n_merge", 0) for d in per_image)
    sp = sum(d.get("n_split", 0) for d in per_image)
    pur = [d["mean_purity"] for d in per_image if d["mean_purity"] == d["mean_purity"]]
    return {"merge_rate": mg / tp if tp else 0.0, "split_rate": sp / tg if tg else 0.0,
            "mean_purity": float(np.mean(pur)) if pur else float("nan"),
            "n_pred": tp, "n_gt": tg}


if __name__ == "__main__":
    # self-test: two GTs, one merged prediction covering both
    g1 = np.zeros((40, 40), bool); g1[10:20, 5:15] = True
    g2 = np.zeros((40, 40), bool); g2[10:20, 20:30] = True
    merged = np.zeros((40, 40), bool); merged[10:20, 5:30] = True
    print("merged pred over 2 GTs:", split_merge_rates([merged], [g1, g2]))
    print("clean preds:", split_merge_rates([g1, g2], [g1, g2]))
