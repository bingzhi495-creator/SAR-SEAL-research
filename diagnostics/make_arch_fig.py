# -*- coding: utf-8 -*-
"""Publication system-overview figure (Fig. 1) for the SEAL paper, in matplotlib
(vector PDF, no external SVG tooling). Diagnose-before-fix + dual delivery over a
shared evidence prior. Mirrors the rendered concept diagram."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(BASE, "paper", "figs")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 8, "savefig.bbox": "tight"})

C = {"gray": "#e7e5dd", "grayE": "#888780", "blue": "#cfe2f7", "blueE": "#185fa5",
     "amber": "#f7e1bd", "amberE": "#ba7517", "green": "#dcebc4", "greenE": "#3b6d11", "tx": "#1d1d1b"}


def box(ax, x, y, w, h, face, edge, lw=0.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                 fc=face, ec=edge, lw=lw, mutation_aspect=1))


def txt(ax, x, y, s, fs=8, w="normal", c=None, style="normal"):
    ax.text(x, y, s, ha="center", va="center", fontsize=fs, fontweight=w,
            color=c or C["tx"], style=style)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8,
                 lw=1.0, color=C["grayE"], shrinkA=0, shrinkB=0))


def main():
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    # diagnosis band
    box(ax, 4, 88, 92, 9, C["gray"], C["grayE"])
    txt(ax, 50, 94.5, "Diagnosis — oracle decomposition of the inshore AP$_{75}$ deficit", 8.5, "bold")
    txt(ax, 50, 90.5, "51% mask quality   ·   49% detection coverage   (real AP-additive, detection held fixed)", 7.5)

    # input
    box(ax, 38, 78, 24, 6.5, C["gray"], C["grayE"])
    txt(ax, 50, 81.2, "SAR inshore image", 8)
    arrow(ax, 50, 88, 50, 84.5)

    # MNE shared prior
    box(ax, 30, 66, 40, 9, C["amber"], C["amberE"], 1.1)
    txt(ax, 50, 72, "MNE evidence field  $E$", 8.5, "bold")
    txt(ax, 50, 68.2, "coherent-speckle rank statistics $\\rightarrow$ high-recall seeds", 7.3)
    arrow(ax, 50, 78, 50, 75)
    arrow(ax, 33, 66, 22, 57)
    arrow(ax, 67, 66, 78, 57)

    txt(ax, 23, 60, "Segmentation branch", 7.6, "bold")
    txt(ax, 77, 60, "Detection branch", 7.6, "bold")

    # seg branch
    box(ax, 6, 47, 34, 8, C["gray"], C["grayE"])
    txt(ax, 23, 52.5, "Mask2Former-R50", 8); txt(ax, 23, 49, "ResNet-50 + 9$\\times$ decoder (baseline)", 7)
    arrow(ax, 23, 47, 23, 43.5)
    box(ax, 6, 35, 34, 8.5, C["blue"], C["blueE"], 1.6)
    txt(ax, 23, 40.6, "SealM2FHead refine", 8.5, "bold"); txt(ax, 23, 37, "R$\\theta$ conv on final-layer mask", 7)
    arrow(ax, 23, 35, 23, 31.5)
    box(ax, 11, 25, 24, 6, C["gray"], C["grayE"])
    txt(ax, 23, 28, "instance masks", 8)
    box(ax, 6, 12, 34, 8, C["blue"], C["blueE"])
    txt(ax, 23, 17.4, "+3.0 AP$_{75}$  (3 seeds)", 8.5, "bold", C["blueE"])
    txt(ax, 23, 14, "added capacity, not resolution", 7)

    # det branch
    box(ax, 60, 47, 34, 8, C["gray"], C["grayE"])
    txt(ax, 77, 52.5, "DINO detector", 8); txt(ax, 77, 49, "DETR query reference points", 7)
    arrow(ax, 77, 47, 77, 43.5)
    box(ax, 60, 35, 34, 8.5, C["amber"], C["amberE"], 1.6)
    txt(ax, 77, 40.6, "SeedDINO seeding", 8.5, "bold"); txt(ax, 77, 37, "$E$-seeds $\\rightarrow$ reference points", 7)
    arrow(ax, 77, 35, 77, 31.5)
    box(ax, 65, 25, 24, 6, C["gray"], C["grayE"])
    txt(ax, 77, 28, "ship boxes", 8)
    box(ax, 60, 12, 34, 8, C["amber"], C["amberE"])
    txt(ax, 77, 17.4, "+1.85 AR@100  (2 seeds)", 8.5, "bold", C["amberE"])
    txt(ax, 77, 14, "high-IoU localization", 7)

    txt(ax, 50, 6, "dual delivery over one shared prior — no aggregate-AP SOTA claim; each branch plays to its strength", 7, c=C["grayE"], style="italic")

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, "fig_overview." + ext), dpi=200)
    plt.close(fig)
    print("wrote fig_overview")


if __name__ == "__main__":
    main()
