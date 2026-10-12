# -*- coding: utf-8 -*-
"""G1 diagnostic: training-free pixel AUROC of MNE vs the classical control
family on PSeg-SSDD test (inshore / offshore stratified).

Pre-registered protocol (verdict-1 required degrees of freedom, frozen here):
  * pixel population: per image, positives = union of GT instance masks,
    negatives = all remaining pixels (PSeg-SSDD has no land mask; land
    stratification is carried by the dataset's own inshore/offshore split)
  * per-image AUROC, then averaged per stratum (macro); pooled AUROC reported
    as secondary
  * multi-scale merge rule: max over per-scale z-standardized maps
  * MNE-E uses ONLY the rank-evidence channels (so the comparison against
    CFAR controls is statistic-vs-statistic); MNE-full (max(E) + T fusion) is
    reported as a separate row, explicitly labelled
  * pass rule (裁汰规则): MNE-E must beat box_z by >= 0.03 macro AUROC, else
    the input-injection pillar is CUT. Beating box_z does NOT license any
    "not a CFAR re-skin" claim - that requires beating ca/ts/wilcoxon too.

Usage:  python -m diagnostics.g1_auroc [--limit N] [--out runs/g1]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mne.engine import mne_field, empirical_null, background_sample_mask, SCALES  # noqa: E402
from mne import baselines  # noqa: E402
from protocol.datasets import test_stems, load_one, hrsid_test_stems, load_one_hrsid  # noqa: E402



def _auroc(score: np.ndarray, pos_mask: np.ndarray) -> float:
    """Exact AUROC via rank statistic (handles ties by mid-rank)."""
    s = score.ravel()
    y = pos_mask.ravel()
    n_pos = int(y.sum())
    n_neg = y.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_s = s[order]
    # mid-ranks for ties
    i = 0
    r = np.arange(1, s.size + 1, dtype=np.float64)
    boundaries = np.flatnonzero(np.diff(sorted_s)) + 1
    start = 0
    for end in list(boundaries) + [s.size]:
        r[start:end] = 0.5 * (start + 1 + end)
        start = end
    ranks[order] = r
    rank_sum_pos = ranks[y].sum()
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def _zmax(stack: np.ndarray, bg_mask: np.ndarray, rng) -> np.ndarray:
    """Per-scale empirical-null standardization, then max over scales."""
    zs = []
    for k in range(stack.shape[0]):
        med, sig = empirical_null(stack[k], bg_mask, rng=rng)
        zs.append((stack[k] - med) / sig)
    return np.max(np.stack(zs), axis=0)


def score_maps(img: np.ndarray) -> dict:
    """All method score maps for one image, identical merge protocol.

    Single-pass shared-histogram implementation: per scale the 256-bin box
    sums (test window + background annulus) are computed once and consumed by
    every histogram-based method (truncated rank / untruncated rank /
    truncated-moment contrast a.k.a. TS-CFAR). Exactness identical to the
    reference implementation in mne.engine / mne.baselines.
    """
    from mne.engine import (_onehot_hist, _box_sum, _truncate_hist,
                            annulus_geometry, NBINS)
    rng = np.random.default_rng(0)
    bg = background_sample_mask(img)
    img_f = img.astype(np.float64)
    oh = _onehot_hist(img)
    vals = np.arange(NBINS, dtype=np.float64)[:, None, None]

    A_tr, A_un, Z_ts, Z_ca = [], [], [], []
    wilcoxon7 = None
    for s in SCALES:
        gb, ob = annulus_geometry(s)
        hist_t = _box_sum(oh, s)
        hist_bg = _box_sum(oh, ob) - _box_sum(oh, gb)
        np.clip(hist_bg, 0.0, None, out=hist_bg)
        mean_t = _box_sum(img_f, s) / (s * s)
        m = hist_t.sum(axis=0)

        # untruncated rank (also the Wilcoxon-CFAR control at s=7)
        n_bg = hist_bg.sum(axis=0)
        midrank = np.cumsum(hist_bg, axis=0) - 0.5 * hist_bg
        with np.errstate(invalid="ignore", divide="ignore"):
            a_un = np.nan_to_num((hist_t * midrank).sum(axis=0) / (m * n_bg), nan=0.5)
        if s == 7:
            wilcoxon7 = a_un
        A_un.append(a_un)

        # truncated background once -> rank + TS moments
        hist_bp = _truncate_hist(hist_bg)
        del hist_bg, midrank
        n_bp = hist_bp.sum(axis=0)
        midrank_p = np.cumsum(hist_bp, axis=0) - 0.5 * hist_bp
        with np.errstate(invalid="ignore", divide="ignore"):
            a_tr = np.nan_to_num((hist_t * midrank_p).sum(axis=0) / (m * n_bp), nan=0.5)
            mu_p = (hist_bp * vals).sum(axis=0) / n_bp
            var_p = (hist_bp * vals * vals).sum(axis=0) / n_bp - mu_p * mu_p
        A_tr.append(a_tr)
        sd_p = np.sqrt(np.clip(np.nan_to_num(var_p), 0.0, None))
        Z_ts.append(np.nan_to_num((mean_t - np.nan_to_num(mu_p)) / (sd_p + 1e-3)))
        del hist_bp, midrank_p, hist_t

        # CA-CFAR via cheap image-domain annulus moments (no histograms)
        mu_c, sd_c = baselines._annulus_moments(img, s, truncate=False)
        Z_ca.append((mean_t - mu_c) / (sd_c + 1e-3))

    A_tr, A_un = np.stack(A_tr), np.stack(A_un)
    Z_ts, Z_ca = np.stack(Z_ts), np.stack(Z_ca)
    box_raw = np.stack([_box_sum(img_f, s) / (s * s) for s in SCALES])

    maps = {
        "mne_E": _zmax(A_tr, bg, rng),
        "mne_full": _zmax(A_tr, bg, rng) + 0.5 * _zmax(Z_ts, bg, rng),
        "rank_untrunc": _zmax(A_un, bg, rng),      # isolates truncation's contribution
        "box_z": _zmax(baselines.box_z(img), bg, rng),
        "box_raw": _zmax(box_raw, bg, rng),        # arithmetic-mean control (box_z is geometric via log)
        "ca_cfar": _zmax(Z_ca, bg, rng),
        "ts_cfar": _zmax(Z_ts, bg, rng),
        "wilcoxon_s7": wilcoxon7,
    }
    return maps


def _process_one(stem_subset) -> dict | None:
    stem, subset, dataset = stem_subset
    try:
        sample = load_one_hrsid(stem, subset) if dataset == "hrsid" else load_one(stem, subset)
        if sample is None:
            return None
        pos = sample.ship_mask
        if pos.sum() == 0:
            return None
        maps = score_maps(sample.image)
        row = {"stem": sample.stem, "subset": sample.subset,
               "n_pos": int(pos.sum()), "hw": list(sample.image.shape)}
        for name, m in maps.items():
            row[name] = _auroc(m, pos)
        return row
    except Exception as e:  # one bad/OOM image must not kill the whole run
        print(f"ERROR {stem}: {type(e).__name__}: {e}", flush=True)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--dataset", choices=["pseg", "hrsid"], default="pseg")
    ap.add_argument("--out", type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "g1"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from concurrent.futures import ProcessPoolExecutor

    jsonl = os.path.join(args.out, "per_image.jsonl")
    done = set()
    rows = []
    if os.path.exists(jsonl):  # resume: skip stems already processed
        with open(jsonl, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    done.add(r["stem"])
                    rows.append(r)
                except json.JSONDecodeError:
                    pass
        print(f"resume: {len(done)} images already done", flush=True)

    stems = hrsid_test_stems(limit=args.limit) if args.dataset == "hrsid" else test_stems(limit=args.limit)
    work = [(s, sub, args.dataset) for s, sub in stems if s not in done]
    t0 = time.time()
    n_done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex, \
            open(jsonl, "a", encoding="utf-8") as ck:
        for row in ex.map(_process_one, work, chunksize=2):
            if row is None:
                continue
            rows.append(row)
            ck.write(json.dumps(row) + "\n")
            ck.flush()
            n_done += 1
            if n_done % 20 == 0:
                el = time.time() - t0
                print(f"[{n_done}/{len(work)}] {el:.0f}s elapsed", flush=True)

    with open(os.path.join(args.out, "per_image.json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=1)

    methods = ["mne_E", "mne_full", "rank_untrunc", "box_z", "box_raw", "ca_cfar", "ts_cfar", "wilcoxon_s7"]
    print("\n=== G1: per-image macro AUROC (mean ± std) ===")
    header = f"{'method':<14}" + "".join(f"{s:>22}" for s in ["all", "inshore", "offshore"])
    print(header)
    summary = {}
    for m in methods:
        line = f"{m:<14}"
        summary[m] = {}
        for sub in ["all", "inshore", "offshore"]:
            vals = [r[m] for r in rows if (sub == "all" or r["subset"] == sub) and not np.isnan(r[m])]
            mu = float(np.mean(vals)) if vals else float("nan")
            sd = float(np.std(vals)) if vals else float("nan")
            summary[m][sub] = {"mean": mu, "std": sd, "n": len(vals)}
            line += f"{mu:>14.4f}±{sd:.4f}"
        print(line)

    gate = summary["mne_E"]["all"]["mean"] - summary["box_z"]["all"]["mean"]
    strongest = max(summary[m]["all"]["mean"] for m in ["box_z", "box_raw", "ca_cfar", "ts_cfar", "wilcoxon_s7"])
    print(f"\nG1 gate (mne_E - box_z)         : {gate:+.4f}  (pass requires >= +0.03)")
    print(f"mne_E - strongest classical ctrl: {summary['mne_E']['all']['mean'] - strongest:+.4f}  (needed for the 'not-a-CFAR-reskin' claim)")
    summary["_gate"] = {"mne_E_minus_box_z": gate,
                        "mne_E_minus_strongest": summary["mne_E"]["all"]["mean"] - strongest}
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
