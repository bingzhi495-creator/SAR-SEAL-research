"""Draw the conceptual native-grid refinement figure as editable vector artwork."""
from pathlib import Path
from xml.etree import ElementTree as ET

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

OUT = Path(__file__).resolve().parent
mpl.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none",
                     "pdf.fonttype": 42, "svg.hashsalt": "native-grid-v1"})
W, H = 516, 183
fig = plt.figure(figsize=(W / 72, H / 72), facecolor="white")
ax = fig.add_axes((0, 0, 1, 1), xlim=(0, W), ylim=(H, 0), aspect="equal")
ax.set_axis_off()
NAVY, TEAL, RUST = "#233C56", "#007D80", "#A74329"
INK, MUTED, GRID = "#1D2C38", "#485863", "#CFD5D9"
LIGHT, GRAY = "#B7BFC3", "#69757C"


def label(key, x, y, s, size=7.6, align="left", bold=False, color=INK):
    ax.text(x, y, s, fontsize=size, color=color, ha=align, va="center",
            weight="bold" if bold else "normal", gid=key, zorder=5)


def line(key, x1, y1, x2, y2, color, width, style="-", order=2):
    ax.plot((x1, x2), (y1, y2), color=color, lw=width, linestyle=style,
            solid_capstyle="butt", gid=key, zorder=order)


def rect(key, x, y, w, h, fill, edge="none", lw=0, dash="-", order=1):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge,
                           lw=lw, linestyle=dash, gid=key, zorder=order))


def endpoint(key, x, y, add, active=None, radius=4.4):
    # Active edits have a filled center, so the policy can also be read in gray.
    color = (TEAL if add else RUST) if active is not False else LIGHT
    if add:
        ax.add_patch(Circle((x, y), radius, facecolor="white", edgecolor=color,
                            lw=1.4 if active is not False else 0.8, gid=key, zorder=4))
    else:
        rect(key, x-radius, y-radius, 2*radius, 2*radius, "white", color,
             1.4 if active is not False else 0.8, order=4)
    if active:
        ax.add_patch(Circle((x, y), radius*0.47, facecolor=color,
                            edgecolor="none", gid=key+"-center", zorder=5))


# Columns are in native image coordinates. The enlarged panel reads the same
# mask and crop coordinates as the small panel.
top = [8,7,6,5,5,4,4,4,4,4,4,4,4,4,4,4,5,5,6,7,8,9,9,10]
bottom = [10,11,12,13,13,14,14,14,14,14,14,14,14,14,14,14,14,13,13,12,12,11,11,10]
crop_col, crop_row, crop_n = 16, 4, 6


def grid(prefix, x, y, cell, cols, rows, col0=0, row0=0):
    rect(prefix+"-paper", x, y, cols*cell, rows*cell, "white")
    for col in range(cols):
        for row in range(rows):
            orig_col = col + col0
            if 0 <= orig_col < len(top) and top[orig_col] <= row+row0 <= bottom[orig_col]:
                rect(f"{prefix}-mask-c{col}-r{row}", x+col*cell, y+row*cell,
                     cell, cell, NAVY)
    for col in range(cols+1):
        line(f"{prefix}-col-{col}", x+col*cell, y, x+col*cell, y+rows*cell,
             GRID, 0.28)
    for row in range(rows+1):
        line(f"{prefix}-row-{row}", x, y+row*cell, x+cols*cell, y+row*cell,
             GRID, 0.28)
    rect(prefix+"-border", x, y, cols*cell, rows*cell, "none", "#AAB4BA", .5, order=3)


label("a-title", 4, 10, "(a)  Frozen instance mask", 9.25, bold=True)
label("b-title", 170, 10, "(b)  Boundary support", 9.25, bold=True)
label("c-title", 309, 10, "(c)  Equal-count selection", 9.25, bold=True)

ax0, ay0, s = 5, 43, 5.55
grid("a", ax0, ay0, s, 26, 18, -1)
cx, cy = ax0 + (crop_col+1)*s, ay0 + crop_row*s
for key, color, thick, dash in [("halo", "white", 2.0, "-"),
                                  ("box", INK, 1.05, (0,(2.3,2.3)))]:
    rect("a-crop-"+key, cx, cy, crop_n*s, crop_n*s, "none", color, thick, dash, 4)
label("a-note", 5, 161, "Dashed box marks the enlarged patch.", 7.15, color=MUTED)

bx, by, bs = 170, 35, 19
grid("b", bx, by, bs, crop_n, crop_n, crop_col, crop_row)
for i in range(1,5):
    x = bx + (i+.5)*bs
    y = by + (top[crop_col+i] - crop_row + .5)*bs
    endpoint(f"b-add-{i}", x, y-bs, True)
    endpoint(f"b-remove-{i}", x, y, False)
endpoint("b-key-add", bx+4, 161, True, radius=3.5)
label("b-label-add", bx+12, 161, "Add", 7.5)
endpoint("b-key-drop", bx+48, 161, False, radius=3.5)
label("b-label-drop", bx+56, 161, "Remove", 7.5)
line("zoom-upper", cx+crop_n*s+1.5, cy, bx-2, by, LIGHT, .55, (0,(2,2)))
line("zoom-lower", cx+crop_n*s+1.5, cy+crop_n*s, bx-2, by+bs*crop_n, LIGHT, .55, (0,(2,2)))

columns = [395, 428, 461, 494]
label("c-index-title", 310, 29, "Position", 7.35, color=MUTED)
for i, x in enumerate(columns, 1):
    label(f"c-index-{i}", x, 29, str(i), 7.6, "center", color=MUTED)
line("c-header-rule", 310, 37, 510, 37, GRID, .55)

# Numbers are positions along the illustrated edge, never score ranks.
choices = [("fixed", "Fixed pairs", 49, 63, 83, {1,3}, {1,3}),
           ("independent", "Independent", 112, 126, 146, {1,2}, {3,4})]
for tag, name, heading, add_y, drop_y, adds, drops in choices:
    assert len(adds) == len(drops) == 2
    label(f"c-{tag}-title", 310, heading, name, 8.2, bold=True)
    label(f"c-{tag}-add-label", 310, add_y, "Add", 7.6)
    label(f"c-{tag}-remove-label", 310, drop_y, "Remove", 7.6)
    for i, x in enumerate(columns, 1):
        if tag == "fixed":
            line(f"c-fixed-pair-{i}", x, add_y, x, drop_y,
                 GRAY if i in adds else LIGHT, 1.55 if i in adds else .65)
        endpoint(f"c-{tag}-add-{i}", x, add_y, True, i in adds)
        endpoint(f"c-{tag}-remove-{i}", x, drop_y, False, i in drops)
line("c-divider", 310, 96, 510, 96, GRID, .55)
label("c-selection-key", 310, 170, "Filled centers mark the two edits on each side.", 7.15, color=MUTED)

stem = OUT / "native_grid_selection"
fig.savefig(stem.with_suffix(".pdf"), metadata={"Title":"Equal-count refinement on the native mask grid",
                                              "CreationDate":None, "ModDate":None})
fig.savefig(stem.with_suffix(".png"), dpi=216)
fig.savefig(stem.with_suffix(".svg"), metadata={"Date":None, "Title":"Equal-count refinement on the native mask grid"})
plt.close(fig)

# Group the individual grid cells, labels, and marks by panel for vector editors.
svg = "http://www.w3.org/2000/svg"
ET.register_namespace("", svg)
ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
tree = ET.parse(stem.with_suffix(".svg"))
axes = tree.find(".//*[@id='axes_1']")
groups = {p: ET.Element(f"{{{svg}}}g", {"id":"panel-"+p}) for p in "abc"}
groups["zoom"] = ET.Element(f"{{{svg}}}g", {"id":"crop-connectors"})
for child in list(axes):
    prefix = child.get("id", "").partition("-")[0]
    if prefix in groups:
        axes.remove(child)
        groups[prefix].append(child)
for prefix in ("a","zoom","b","c"):
    axes.append(groups[prefix])
tree.write(stem.with_suffix(".svg"), encoding="utf-8", xml_declaration=True)
