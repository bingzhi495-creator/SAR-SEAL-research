# -*- coding: utf-8 -*-
"""v5 CPU decision gates — verify hidden assumptions BEFORE spending GPU.
All zero-training, using existing A0 preds.pkl + GT + MNE evidence.

D2 (GEPL affinity separability): are ship-INTERIOR neighbor pairs more
   evidence-similar than CROSS-BOUNDARY pairs? AUROC>0.7 => evidence affinity
   is a valid connectivity signal => E1 (GEPL attacks split 0.558) justified.

D4 (decision-head evidence factor, the G1 instance-level variant): can the mean
   evidence inside a predicted mask separate TP masks from FP masks? AUROC>0.65
   => keep the beta evidence factor; else DROP it, decision head = pure maskIoU.

Honest expectation per v5 design: D2 likely PASS (connectivity, orthogonal to
G1), D4 likely FAIL (same root as G1 -- evidence can't classify ship vs clutter).

Usage: python diagnostics\\cpu_gates.py --gate D2|D4 [--limit N]
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mne.engine import mne_field  # noqa: E402
from protocol.datasets import load_one, test_stems  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _auroc(scores, labels):
    s = np.asarray(scores); y = np.asarray(labels).astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    # tie-correct
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    # simple mid-rank via average per unique value
    sums = np.zeros(len(cnt)); np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def gate_D2(limit=None):
    """ship-interior vs cross-boundary neighbor-pair evidence affinity."""
    sig = "exp(-|E(u)-E(v)|_1/sigma)"
    inter, cross = [], []
    work = test_stems(limit=limit)
    for stem, sub in work:
        sm = load_one(stem, sub)
        if sm is None or not sm.instances:
            continue
        E = mne_field(sm.image)["E"]  # (S,H,W)
        ship = sm.ship_mask
        # 4-neighbor pairs at dilation 2
        for dy, dx in [(0, 2), (2, 0)]:
            a = E[:, :-dy or None, :-dx or None] if (dy or dx) else E
            b = np.roll(E, shift=(-dy, -dx), axis=(1, 2))
            aff = np.exp(-np.abs(E - b).sum(0))  # (H,W)
            s1 = np.roll(ship, shift=(-dy, -dx), axis=(0, 1))
            both_in = ship & s1
            cross_bd = ship ^ s1  # one in ship, one out -> boundary pair
            inter.extend(aff[both_in].tolist())
            cross.extend(aff[cross_bd].tolist())
    inter = np.array(inter); cross = np.array(cross)
    sc = np.concatenate([inter, cross]); lb = np.concatenate([np.ones(len(inter)), np.zeros(len(cross))])
    auroc = _auroc(sc, lb)
    print(f"D2 GEPL affinity ({sig}):")
    print(f"  ship-interior pairs n={len(inter)} mean_aff={inter.mean():.3f}")
    print(f"  cross-boundary pairs n={len(cross)} mean_aff={cross.mean():.3f}")
    print(f"  separability AUROC = {auroc:.4f}  (pass >0.7 -> E1 GEPL justified)")
    print(f"  VERDICT: {'PASS' if auroc > 0.7 else 'WEAK -> tune sigma/dilation or downweight affinity, rely on connectivity'}")


def gate_D4(limit=None, score_t=0.3):
    """TP-mask vs FP-mask separability by mean evidence inside the mask."""
    from pycocotools import mask as mask_util
    preds = pickle.load(open(os.path.join(BASE, "runs", "a0_pseg_seed0", "preds.pkl"), "rb"))
    pred_by_stem = {os.path.splitext(os.path.basename(p["img_path"]))[0]: p for p in preds}
    work = test_stems(limit=limit)
    tp_ev, fp_ev = [], []
    for stem, sub in work:
        sm = load_one(stem, sub)
        if sm is None:
            continue
        p = pred_by_stem.get(stem)
        if p is None:
            continue
        ev = mne_field(sm.image)["E"].max(0)  # ev_zmax (H,W)
        inst = p["pred_instances"]
        scores = inst["scores"].cpu().numpy() if hasattr(inst["scores"], "cpu") else np.asarray(inst["scores"])
        masks = inst["masks"]
        gts = sm.instances
        for k in range(len(scores)):
            if scores[k] < score_t:
                continue
            pm = mask_util.decode(masks[k]).astype(bool) if isinstance(masks[k], dict) else np.asarray(masks[k]).astype(bool)
            if pm.sum() == 0:
                continue
            best = max((np.logical_and(pm, g).sum() / np.logical_or(pm, g).sum() for g in gts), default=0.0) if gts else 0.0
            mean_ev = float(ev[pm].mean())
            (tp_ev if best >= 0.5 else fp_ev).append(mean_ev)
    sc = np.array(tp_ev + fp_ev); lb = np.concatenate([np.ones(len(tp_ev)), np.zeros(len(fp_ev))])
    auroc = _auroc(sc, lb)
    print(f"D4 decision-head evidence factor (G1 instance-level variant):")
    print(f"  TP masks n={len(tp_ev)} mean_ev={np.mean(tp_ev):.3f}")
    print(f"  FP masks n={len(fp_ev)} mean_ev={np.mean(fp_ev) if fp_ev else float('nan'):.3f}")
    print(f"  TP-vs-FP separability AUROC = {auroc:.4f}  (pass >0.65 -> keep beta factor)")
    print(f"  VERDICT: {'PASS -> keep beta evidence factor' if auroc > 0.65 else 'FAIL -> drop beta, decision head = pure maskIoU (still sound)'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", required=True, choices=["D2", "D4"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    {"D2": gate_D2, "D4": gate_D4}[args.gate](limit=args.limit)


if __name__ == "__main__":
    main()
