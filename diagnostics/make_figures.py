# -*- coding: utf-8 -*-
"""Publication figures for the SEAL paper, from confirmed numbers (no GPU).
Outputs PDF (vector) + PNG into paper/figs/.

fig_mechanism : net instances crossing IoU>=0.75 by size, PSeg vs HRSID
                (the diagnostic centerpiece: clean +15 small on PSeg vs -58 on
                 dense HRSID -> the resolution lever does not transfer).
fig_ap75      : A0 vs A0R-refine AP75 across settings (PSeg overall/inshore, HRSID).
fig_oracle    : inshore AP75 gap decomposition (60% detection / 40% resolution).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(BASE, "paper", "figs")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.spilers" if False else "axes.grid": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "axes.linewidth": 0.7,
})
C_A0, C_A0R, C_DET, C_RES = "#7f8c9b", "#2c6fbf", "#c0654a", "#3f8f5e"


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"))
    plt.close(fig)
    print("wrote", name)


# ---- fig_mechanism: net cross-0.75 by size, PSeg vs HRSID ----
def fig_mechanism():
    sizes = ["small", "medium", "large"]
    pseg = [15, 0, -1]
    hrsid = [-58, -20, 2]
    x = range(len(sizes)); w = 0.36
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.bar([i - w/2 for i in x], pseg, w, label="PSeg-SSDD (sparse)", color=C_A0R)
    ax.bar([i + w/2 for i in x], hrsid, w, label="HRSID (dense)", color=C_DET)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xticks(list(x)); ax.set_xticklabels(sizes)
    ax.set_ylabel(r"net $\Delta$ instances crossing IoU$\geq$0.75")
    ax.set_title("Refine head: where mask IoU crosses the AP$_{75}$ bar", fontsize=8.5)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    for i, v in enumerate(pseg):
        ax.annotate(f"{v:+d}", (i - w/2, v), ha="center", va="bottom" if v >= 0 else "top", fontsize=7)
    for i, v in enumerate(hrsid):
        ax.annotate(f"{v:+d}", (i + w/2, v), ha="center", va="bottom" if v >= 0 else "top", fontsize=7)
    save(fig, "fig_mechanism")


# ---- fig_ap75: A0 vs A0R AP75 across settings ----
def fig_ap75():
    labels = ["PSeg\noverall", "PSeg\ninshore", "HRSID\noverall"]
    a0 = [0.619, 0.341, 0.378]
    a0r = [0.645, 0.387, 0.383]
    # delta CIs (as +/- on the A0R bar for inshore only); overall +2.6 robust, HRSID +0.5 noise
    yerr = [[0, 0, 0], [0.004, 0.050, 0.004]]  # rough seed/bootstrap spread on A0R
    x = range(len(labels)); w = 0.36
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.bar([i - w/2 for i in x], a0, w, label="A0 (Mask2Former-R50)", color=C_A0)
    ax.bar([i + w/2 for i in x], a0r, w, yerr=yerr, capsize=2, label="A0R (+refine head)", color=C_A0R,
           error_kw=dict(lw=0.7))
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel(r"mask AP$_{75}$"); ax.set_ylim(0.30, 0.70)
    ax.set_title("Refine head AP$_{75}$ (PSeg seed-robust; HRSID neutral)", fontsize=8.5)
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    deltas = ["+2.6", "+4.6*", "+0.5†"]
    for i, (b, d) in enumerate(zip(a0r, deltas)):
        ax.annotate(d, (i + w/2, b + (yerr[1][i] if i else 0) + 0.01), ha="center", fontsize=7, color=C_A0R)
    ax.annotate("* suggestive, 95% CI [-0.5,+9.5]\n† within single-seed noise",
                (0.02, 0.02), xycoords="axes fraction", fontsize=5.8, va="bottom", color="#555")
    save(fig, "fig_ap75")


# ---- fig_oracle: inshore AP75 gap decomposition ----
def fig_oracle():
    fig, ax = plt.subplots(figsize=(3.4, 1.3))
    ax.barh([0], [60], color=C_DET, label="detection coverage (~60%)")
    ax.barh([0], [40], left=[60], color=C_RES, label="mask resolution (~40%)")
    ax.set_xlim(0, 100); ax.set_yticks([])
    ax.set_xlabel("share of inshore AP$_{75}$ deficit (oracle-box analysis)")
    ax.annotate("~60%\n(query recall)", (30, 0), ha="center", va="center", color="white", fontsize=7.5)
    ax.annotate("~40%\n(mask resolution\n= our target)", (80, 0), ha="center", va="center", color="white", fontsize=7)
    ax.set_title("Oracle-box decomposition of the inshore gap", fontsize=8.5)
    save(fig, "fig_oracle")


# ---- fig_iou_cdf: matched mask-IoU CDF for PSeg small objects, A0 vs A0R ----
def fig_iou_cdf():
    import numpy as np, os as _os, pickle
    from pycocotools import mask as mask_util
    from pycocotools.coco import COCO
    import torch  # noqa
    from refine_mechanism import load_preds_by_stem, best_iou_per_gt
    gt = _os.path.join(BASE, "data", "pseg_ssdd", "annotations", "test.json")
    coco = COCO(gt)
    a0 = load_preds_by_stem(_os.path.join(BASE, "runs", "refine_gate", "A0_stock_preds.pkl"))
    a0r = load_preds_by_stem(_os.path.join(BASE, "runs", "refine_gate", "A0R_refine_preds.pkl"))
    n2i = {_os.path.splitext(im["file_name"])[0]: i for i, im in coco.imgs.items()}
    io0, ior = [], []
    for stem, iid in n2i.items():
        if stem not in a0 or stem not in a0r:
            continue
        anns = coco.imgToAnns.get(iid, [])
        small = [a for a in anns if a["area"] < 1024]
        if not small:
            continue
        im = coco.imgs[iid]; h, w = im["height"], im["width"]
        gm = []
        for a in small:
            seg = a["segmentation"]
            r = seg if isinstance(seg, dict) else mask_util.merge(mask_util.frPyObjects(seg, h, w))
            gm.append(mask_util.decode(r).astype(bool))
        b0 = best_iou_per_gt(gm, a0[stem], 0.3); br = best_iou_per_gt(gm, a0r[stem], 0.3)
        m = b0 >= 0.5  # A0-detected (mask-quality lever)
        io0.extend(b0[m]); ior.extend(br[m])
    io0 = np.sort(np.array(io0)); ior = np.sort(np.array(ior))
    y0 = np.arange(1, len(io0)+1)/len(io0); yr = np.arange(1, len(ior)+1)/len(ior)
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.plot(io0, y0, color=C_A0, lw=1.4, label=f"A0 (n={len(io0)})")
    ax.plot(ior, yr, color=C_A0R, lw=1.4, label="A0R (+refine)")
    ax.axvline(0.75, color=C_RES, ls="--", lw=0.9)
    ax.annotate("AP$_{75}$ bar", (0.75, 0.08), fontsize=6.5, color=C_RES, rotation=90, va="bottom", ha="right")
    ax.set_xlabel("matched mask-IoU (PSeg small, A0-detected)"); ax.set_ylabel("CDF")
    ax.set_xlim(0.5, 1.0)
    ax.set_title("Refine head shifts IoU mass across 0.75", fontsize=8.5)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    save(fig, "fig_iou_cdf")


# ---- fig_capacity: the NEW centerpiece -- capacity not resolution (B1) ----
def fig_capacity():
    labels = ["A0\n(stock)", "bilinear\n(resolution)", "caps\n(conv @1/4)", "refine\n(conv @1/2)"]
    ap75 = [0.612, 0.628, 0.643, 0.642]      # stock 3-seed mean; bilinear/caps/refine
    colors = [C_A0, C_RES, C_A0R, C_A0R]
    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    bars = ax.bar(range(4), ap75, color=colors, width=0.62)
    ax.set_ylim(0.59, 0.66); ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel(r"mask AP$_{75}$ (PSeg-SSDD)")
    ax.set_title("Gain is added capacity, not resolution", fontsize=8.8)
    deltas = ["", "+1.3", "+3.1", "+3.0"]
    for i, (b, d) in enumerate(zip(ap75, deltas)):
        if d:
            ax.annotate(d, (i, b + 0.002), ha="center", fontsize=7.5,
                        color=(C_RES if i == 1 else C_A0R))
    ax.annotate("same conv module as refine,\nbut at the ORIGINAL 1/4 stride\n-> resolution lift is redundant",
                (0.97, 0.04), xycoords="axes fraction", fontsize=5.8, ha="right", va="bottom", color="#555")
    save(fig, "fig_capacity")


# ---- fig_det_localization: SeedDINO improves high-IoU localization, not coverage ----
def fig_det_localization():
    thr = [0.5, 0.7, 0.75, 0.9]
    delta_all = [-0.013, +0.016, +0.004, +0.062]
    delta_ins = [-0.035, +0.006, -0.023, +0.017]
    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    ax.plot(thr, [d*100 for d in delta_all], "o-", color=C_A0R, lw=1.5, ms=4, label="all")
    ax.plot(thr, [d*100 for d in delta_ins], "s--", color=C_DET, lw=1.3, ms=4, label="inshore")
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xlabel("IoU threshold"); ax.set_ylabel(r"$\Delta$ recall@100 (pp), seed $-$ no-seed")
    ax.set_title("Evidence seeding: localization, not coverage", fontsize=8.8)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.annotate("loose (coverage):\nslightly worse", (0.5, -2.6), fontsize=5.8, color="#777", ha="left")
    ax.annotate("strict (localization):\n+6.2pp", (0.9, 4.6), fontsize=5.8, color=C_A0R, ha="right")
    save(fig, "fig_det_localization")


if __name__ == "__main__":
    fig_mechanism(); fig_ap75(); fig_oracle(); fig_capacity(); fig_det_localization()
    try:
        fig_iou_cdf()
    except Exception as e:
        print("fig_iou_cdf skipped:", e)
    print("figures in", FIG)
