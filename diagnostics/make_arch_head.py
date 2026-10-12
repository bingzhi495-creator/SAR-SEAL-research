# -*- coding: utf-8 -*-
"""TGRS-style detailed SealM2FHead architecture figure (vector PDF) for the paper.
Backbone + pixel decoder -> mask feature F; transformer decoder -> queries q;
the Mask2Former dot product gives the coarse mask; SealM2FHead inserts a residual
R-theta conv module (Conv-GN-ReLU-Conv) that refines F before the dot product."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Ellipse

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(BASE, "paper", "figs")
plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "savefig.bbox": "tight"})
GR, GRE = "#ece9e1", "#7a796f"
BL, BLE = "#cfe2f7", "#185fa5"
AM, AME = "#f7e1bd", "#9c6113"
TX, SEC, INFO = "#1d1d1b", "#6a6a64", "#185fa5"


def box(ax, x, y, w, h, fc, ec, lw=0.8, r=0.1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw))


def t(ax, x, y, s, fs=7.5, w="normal", c=TX, st="normal"):
    ax.text(x, y, s, ha="center", va="center", fontsize=fs, fontweight=w, color=c, style=st)


def ar(ax, x1, y1, x2, y2, c=SEC, lw=0.9, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=7, lw=lw, color=c, ls=ls, shrinkA=0, shrinkB=0))


def chip(ax, x, y, s, dots, ec=GRE):
    ax.add_patch(Rectangle((x, y), s, s, fc="#0d2030", ec=ec, lw=0.7))
    for (dx, dy, r, col) in dots:
        ax.add_patch(Ellipse((x + dx, y + dy), r, r * 0.62, fc=col))


def opnode(ax, x, y, sym, c=SEC):
    ax.add_patch(Circle((x, y), 1.7, fc="white", ec=c, lw=0.9))
    t(ax, x, y, sym, 11, c=c)


def main():
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.axis("off")

    # ---- top backbone pipeline ----
    chip(ax, 3, 47, 8, [(2.5, 5.5, 1.4, "#dfe8ef"), (5.5, 2.8, 1.1, "#cdd8e0"), (4, 1.6, 0.8, "#b9c6d0")])
    t(ax, 7, 45, "SAR chip", 6.8, c=SEC)
    box(ax, 15, 48, 11, 7, GR, GRE); t(ax, 20.5, 53, "ResNet-50", 7); t(ax, 20.5, 49.6, "backbone", 6.5, c=SEC)
    for i, (xx, hh) in enumerate([(29, 9), (32.4, 7.4), (35.6, 5.8)]):
        box(ax, xx, 47 + (9 - hh) / 2, 2.8, hh, GR, GRE, 0.6, 0.05)
    t(ax, 32, 44.5, "C3–C5", 6.5, c=SEC)
    box(ax, 41, 48, 12, 7, GR, GRE); t(ax, 47, 53, "pixel", 7); t(ax, 47, 49.6, "decoder", 7)
    box(ax, 57, 47.5, 8.5, 8.5, BL, BLE, 1.1); t(ax, 61.2, 51.7, "F", 9.5, "bold")
    t(ax, 61.2, 44.5, "256×H/4×W/4", 6.5, c=SEC)
    for a, b in [(11, 22), (26, 28.5), (38.6, 40.5), (53.2, 56.5)]:
        ar(ax, a, 51.5, b, 51.5)

    # ---- transformer decoder + queries ----
    box(ax, 15, 33, 38, 7, AM, AME); t(ax, 34, 38, "transformer decoder  ×9", 7)
    t(ax, 34, 34.6, "100 object queries, masked cross-attn", 6.3, c=SEC)
    box(ax, 57, 33, 8.5, 7, AM, AME); t(ax, 61.2, 37.6, "q", 8.5, "bold"); t(ax, 61.2, 34.4, "100×256", 6.3, c=SEC)
    ar(ax, 40, 47.5, 36, 40.2)
    ar(ax, 53, 36.5, 56.5, 36.5)

    # ---- standard dot product -> coarse mask ----
    opnode(ax, 72, 45, "⊗")
    t(ax, 72, 41.4, "dot product", 6.3, c=SEC)
    ar(ax, 65.5, 51.7, 70.4, 46.4); ar(ax, 65.5, 36.7, 70.4, 43.4)
    chip(ax, 78, 41, 8, [(2.5, 5, 1.6, "#9fb0bd"), (5.6, 2.6, 1.3, "#8fa3b1")])
    t(ax, 82, 38.5, "coarse mask 1/4", 6.5, c=SEC)
    ar(ax, 73.7, 45, 77.5, 45)
    t(ax, 92, 47, "standard", 6.5, c=SEC); t(ax, 92, 44.8, "Mask2Former", 6.5, c=SEC)

    # ---- SealM2FHead callout ----
    ax.add_patch(FancyBboxPatch((3, 3), 94, 26, boxstyle="round,pad=0,rounding_size=0.6", fc="#f3f8fd", ec=BLE, lw=1.4))
    t(ax, 32, 26.5, "SealM2FHead — our final-layer refinement (before the dot product)", 8, "bold", INFO)

    box(ax, 6, 14, 7, 8, BL, BLE, 1.0); t(ax, 9.5, 18, "F", 9, "bold")
    ar(ax, 13, 18, 16, 18)
    t(ax, 31, 23.2, "Rθ  (residual conv — the only added parameters)", 6.8, "bold", INFO)
    mods = [("Conv 3×3", 16, 8), ("GN", 25.5, 6), ("ReLU", 32.7, 7), ("Conv 3×3", 40.7, 8)]
    for nm, mx, mw in mods:
        box(ax, mx, 14.5, mw, 7, BL, BLE, 0.8); t(ax, mx + mw / 2, 18, nm, 6.5)
    ar(ax, 24, 18, 25.3, 18); ar(ax, 31.5, 18, 32.5, 18); ar(ax, 39.7, 18, 40.5, 18)
    opnode(ax, 53, 18, "+", INFO); ar(ax, 48.7, 18, 51.1, 18)
    # residual skip routed BELOW the conv row to stay clear of the title
    ax.add_patch(FancyArrowPatch((9.5, 14), (51.4, 16.4), connectionstyle="arc3,rad=0.13", arrowstyle="-|>", mutation_scale=6, lw=0.7, color=SEC, ls=(0, (3, 2))))
    t(ax, 30, 11.0, "residual skip", 6, c=SEC)
    box(ax, 58, 14, 7.5, 8, BL, BLE, 1.1); t(ax, 61.7, 18, "F′", 9, "bold")
    ar(ax, 54.9, 18, 57.7, 18)
    opnode(ax, 72, 18, "⊗", INFO); ar(ax, 65.5, 18, 70.2, 18)
    ax.add_patch(FancyArrowPatch((61.2, 33), (72, 20), connectionstyle="arc3,rad=0.25", arrowstyle="-|>", mutation_scale=6, lw=0.7, color=SEC))
    t(ax, 69.5, 28, "q", 6.5, c=SEC)
    chip(ax, 78, 13.5, 8, [(2.4, 4.8, 1.5, "#9fc4e8"), (5.6, 2.6, 1.2, "#9fc4e8")], ec=INFO)
    t(ax, 82, 11, "refined mask", 6.5, c=INFO)
    ar(ax, 73.8, 18, 77.6, 18)
    box(ax, 89, 14.5, 7.5, 7, GR, GRE, 0.7); t(ax, 92.7, 19, "caps:1/4", 6); t(ax, 92.7, 16, "refine:↑1/2", 6)
    ar(ax, 86, 18, 88.7, 18)

    t(ax, 50, 8.2, "Equal-param control: Rθ at the original 1/4 stride (caps) gives +3.1 AP$_{75}$ = refine's +3.0; bilinear alone +1.3", 7, c=TX)
    t(ax, 50, 5.2, "→ the gain is the Rθ capacity, not the 1/4→1/2 resolution. Only Rθ is added; everything else is unchanged.", 6.6, c=SEC, st="italic")

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, "fig_arch_head." + ext), dpi=200)
    plt.close(fig)
    print("wrote fig_arch_head")


if __name__ == "__main__":
    main()
