"""Recompose measured case plates from previously rendered figure artwork.

The public renderer uses only the released pooled SVG panels and their numeric
labels. The optional local renderer reuses images and overlays from existing
native author-review PDFs through clipped PDF forms, without raw source inputs.
"""
from argparse import ArgumentParser
from colorsys import rgb_to_hsv
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import json
import re
import runpy

import fitz
from lxml import etree
import matplotlib


N = "{http://www.w3.org/2000/svg}"
XLINK = "{http://www.w3.org/1999/xlink}"
INK = "#473830"
ACCENT = "#A95B36"
ADD = "#477163"
REMOVE = "#A14A44"
LOCATOR = "#B07B3E"
GRAY = "#756C65"
RULE = "#DED3C8"
LIGHT = "#F5EEE4"
WHITE = "#FFFCF8"
W = 516
FONT_SIZE = 8.2
DATA = {
    "cap": {
        "title": "Fixed-count endpoint selection",
        "context": "Instance-RSDD reserve · D-FINE",
        "public": "rsdd74_common_cap_cases",
        "local": "rsdd74_common_cap_cases",
        "labels": ("GT hull", "N  Native", "F  Distance-first", "L  Bounded",
                   "I  Independent", "I − L changes"),
        "cases": (
            ("+0.18", "4 add / 4 remove", ("", "IoU 51.71", "IoU 51.52",
                                          "IoU 51.73", "IoU 51.92", "")),
            ("−0.15", "8 add / 8 remove", ("", "IoU 67.67", "IoU 67.73",
                                          "IoU 68.00", "IoU 67.85", "")),
        ),
        "footer": ((219, 239), (219, 239)),
    },
    "gf3": {
        "title": "ROH ranking at the dynamic-attraction count",
        # Closed visual lineage: detector 5203, occupancy head 5217.
        "context": "Gaofen-3 · frozen RF-DETR · scalar ROH",
        "public": "gf3_matched_cases_binned",
        "local": "gf3_cases_exact",
        "labels": ("Native mask", "Dynamic attraction", "ROH ranking", "Intersection change"),
        "cases": (
            ("+0.69", "8 swaps", ("IoU 78.61", "IoU 78.13", "IoU 78.83", "Area +2.93 px²")),
            ("+0.70", "8 swaps", ("IoU 72.13", "IoU 71.86", "IoU 72.56", "Area +3.71 px²")),
            ("−0.57", "8 swaps", ("IoU 86.89", "IoU 87.09", "IoU 86.52", "Area −1.30 px²")),
            ("−0.55", "8 swaps", ("IoU 67.79", "IoU 68.14", "IoU 67.58", "Area −3.00 px²")),
        ),
        "footer": ((478, 495), (478, 495)),
    },
    "pseg": {
        "title": "Whole-grid threshold and fixed-count writing",
        "context": "PSeg-SSDD · 293 fixed hulls · 209 gains · 76 losses · 8 ties",
        "public": "pseg_budget_cases_binned",
        "local": "pseg_budget_cases_exact",
        "labels": ("GT coverage", "Whole threshold (WT)", "Fixed count (WC)", "Area error change"),
        "local_labels": ("Native + GT", "Whole threshold (WT)", "Fixed count (WC)", "Boundary detail"),
        "cases": (
            ("+5.87", "", ("8×8 bins · 5×5 pixels/bin", "38 flips · IoU 56.40\nFP 0.00% · FN 43.60%",
                           "16 flips · IoU 62.26\nFP 0.04% · FN 37.71%", "WT − WC · FP + FN")),
            ("−2.12", "", ("8×8 bins · 7×7 pixels/bin", "75 flips · IoU 58.75\nFP 1.26% · FN 40.51%",
                           "16 flips · IoU 56.64\nFP 0.22% · FN 43.24%", "WT − WC · FP + FN")),
        ),
        "native_base": ("Native IoU 66.88", "Native IoU 56.04"),
        "footer": ((351, 381), (400, 416)),
        "note": "FP and FN (% of GT hull area)",
    },
    "seg": {
        "title": "Whole-crop refinement and local edits",
        "context": "Instance-RSDD reserve · native mask, ROH and SegRefiner-LR",
        "public": "rsdd74_segrefiner_cases_binned",
        "local": "rsdd74_segrefiner_cases_SAR_PRIVATE",
        "labels": ("Native mask", "Local ROH", "SegRefiner-LR", "Intersection change"),
        "cases": (
            ("+2.28", "", ("IoU 58.68", "IoU 58.89", "IoU 60.96", "")),
            ("−4.50", "", ("IoU 71.93", "IoU 71.95", "IoU 67.43", "")),
        ),
        "footer": ((249, 264), (249, 264.23)),
    },
    "sarssp": {
        "title": "Scene-to-mask comparison at an equal edit count",
        "context": "SAR-SSP · frozen transfer",
        "local": "sarssp_private_cases",
        "labels": ("SAR context", "GT mask", "Hull detail", "Native mask",
                   "Distance-paired", "Independent"),
        "cases": (
            ("+0.61", "8 add / 8 remove", ("", "", "", "IoU 63.58%", "IoU 63.18%", "IoU 63.78%")),
            ("−0.44", "8 add / 8 remove", ("", "", "", "IoU 75.78%", "IoU 76.01%", "IoU 75.56%")),
        ),
        "footer": (None, (258, 271)),
    },
}


def require(test, reason):
    if not test:
        raise ValueError(reason)


def rgb(s):
    return tuple(int(s[i:i+2], 16) / 255 for i in (1, 3, 5))



def warm_vector_ink(color, kind):
    """Recolor existing pooled SVGs while retaining their ordered shapes and data."""
    color = color.lower()
    fixed = {
        "#ffffff": WHITE, "#000000": INK,
        "#dce5ee": RULE, "#ecf3fa": LIGHT,
        "#e9f5f4": "#E7EFE8", "#fbf0ea": "#FAEBE7",
        "#243247": INK, "#5c6675": GRAY, "#2d5e91": ACCENT,
        "#007f83": ADD, "#c4552d": REMOVE,
    }
    if color in fixed:
        return fixed[color]
    source = rgb(color)
    hue, sat, _ = rgb_to_hsv(*source)
    brightness = 0.2126*source[0] + 0.7152*source[1] + 0.0722*source[2]
    if sat < 0.035 and brightness < 0.45:
        return INK
    if brightness >= 0.93 and sat <= 0.16:
        backing = WHITE if brightness > 0.975 else LIGHT
        f = min(1.0, max(0.0, (0.975-brightness)/0.09))
        return mix(WHITE, backing, f)
    if hue < 0.13 or hue > 0.90:
        endpoint = REMOVE if sat > 0.07 else INK
    elif hue < 0.53 or kind == "cap":
        endpoint = ADD
    else:
        endpoint = "#77513F"
    mix_level = min(1.0, max(0.0, (0.97-brightness)/0.63))
    return mix(WHITE, endpoint, mix_level)


def mix(left, right, weight):
    values = [round(255*(u*(1-weight)+v*weight)) for u,v in zip(rgb(left), rgb(right))]
    return "#" + "".join(f"{ch:02x}" for ch in values)


def recolor(svg, kind):
    paths_before = [p.get("d") for p in svg.iter(N + "path")]
    for node in svg.iter():
        for key in ("style", "fill", "stroke", "stop-color"):
            value = node.attrib.get(key)
            if value is not None:
                node.attrib[key] = re.sub(r"#[0-9A-Fa-f]{6}(?![0-9A-Fa-f])",
                                          lambda m: warm_vector_ink(m[0], kind), value)
    require(paths_before == [p.get("d") for p in svg.iter(N + "path")],
            "Pooled footprints, edits, and line paths remain exact")


def number(s):
    return re.findall(r"\d+(?:\.\d+)?", s)


def original_svg_boxes(root, count):
    axes = [x for x in root.iter(N + "g") if (x.get("id") or "").startswith("axes_")]
    require(len(axes) >= count, "Original vector axes available")
    boxes = []
    for ax in axes[:count]:
        path = next(ax.iter(N + "path"))
        values = [float(v) for v in re.findall(r"[-+]?\d+(?:\.\d+)?", path.get("d"))]
        require(len(values) == 8, "Original panel rectangle")
        xs, ys = values[::2], values[1::2]
        boxes.append(fitz.Rect(min(xs), min(ys), max(xs), max(ys)))
    return boxes


def original_pdf_boxes(page, count):
    boxes = []
    for info in page.get_image_info():
        box = fitz.Rect(info["bbox"])
        if any(abs(box.x0-b.x0)+abs(box.y0-b.y0)+abs(box.x1-b.x1)+abs(box.y1-b.y1) < 1e-3
               for b in boxes):
            continue
        boxes.append(box)
    require(len(boxes) == count, f"Expected {count} rendered native panels, found {len(boxes)}")
    return boxes


def image_bytes(page):
    return {
        (sha256(page.parent.xref_stream_raw(img[0])).hexdigest(),
         sha256(page.parent.xref_stream_raw(img[1])).hexdigest() if img[1] else None)
        for img in page.get_images(full=True)
    }


class VectorCanvas:
    def __init__(self, origin, h):
        require(not origin.findall(".//" + N + "image"), "No public raster images")
        self.root = etree.Element(N + "svg", nsmap={None: N[1:-1], "xlink": XLINK[1:-1]},
                                  width=f"{W}pt", height=f"{h}pt", viewBox=f"0 0 {W} {h}")
        defs = etree.SubElement(self.root, N + "defs")
        self.content = etree.SubElement(defs, N + "g", id="source-case-art")
        old_art = next(g for g in origin.iter(N + "g") if g.get("id") == "figure_1")
        foreign_defs = [child for child in origin if child.tag == N + "defs"]
        for child in foreign_defs:
            defs.append(child)
        self.content.append(old_art)

    def rect(self, box, fill=WHITE, stroke=None):
        style = {"x": str(box.x0), "y": str(box.y0), "width": str(box.width),
                 "height": str(box.height), "fill": fill or "none"}
        if stroke:
            style.update({"stroke": stroke, "stroke-width": "0.6"})
        etree.SubElement(self.root, N + "rect", **style)

    def text(self, x, y, value, color=INK, size=FONT_SIZE, bold=False, centered=False):
        node = etree.SubElement(self.root, N + "text", x=str(x), y=str(y), fill=color,
                                **{"font-family": "DejaVu Sans", "font-size": str(size),
                                   "font-weight": "700" if bold else "400",
                                   "text-anchor": "middle" if centered else "start"})
        node.text = value

    def panel(self, source_box, dest):
        box = etree.SubElement(self.root, N + "svg", x=str(dest.x0), y=str(dest.y0),
                               width=str(dest.width), height=str(dest.height),
                               viewBox=" ".join(str(v) for v in (source_box.x0, source_box.y0,
                                                                source_box.width, source_box.height)),
                               preserveAspectRatio="xMidYMid meet", overflow="hidden")
        etree.SubElement(box, N + "use", {XLINK + "href": "#source-case-art"})


class LocalCanvas:
    def __init__(self, doc, height):
        self.source = doc
        self.pdf = fitz.open()
        self.pg = self.pdf.new_page(width=W, height=height)
        resources = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
        self.regular_path = resources / "DejaVuSans.ttf"
        self.bold_path = resources / "DejaVuSans-Bold.ttf"
        self.regular = fitz.Font(fontfile=str(self.regular_path))
        self.bold = fitz.Font(fontfile=str(self.bold_path))
        self.pg.insert_font(fontname="DVS", fontfile=str(self.regular_path))
        self.pg.insert_font(fontname="DVSB", fontfile=str(self.bold_path))

    def rect(self, box, fill=WHITE, stroke=None):
        edge = rgb(stroke) if stroke else None
        self.pg.draw_rect(box, color=edge, fill=rgb(fill) if fill else None, width=0.6)

    def text(self, x, y, content, color=INK, size=FONT_SIZE, bold=False, centered=False):
        if centered:
            font = self.bold if bold else self.regular
            x -= font.text_length(content, fontsize=size) / 2
        self.pg.insert_text((x, y), content, fontname="DVSB" if bold else "DVS",
                            fontsize=size, color=rgb(color))

    def panel(self, source_box, dest):
        require(self.source[0].rect.contains(source_box), "Clipped panel remains in its source page")
        self.pg.show_pdf_page(dest, self.source, 0, clip=source_box, keep_proportion=True)


@dataclass(frozen=True)
class Layout:
    width: float
    tile: float
    block_height: float
    y: float
    height: float
    shape: str


def layout(key, records, variants):
    # The original rows become two adjacent cases. All the images for an
    # individual hull retain a common crop scale within its card.
    tile = 96 if variants == 4 else 67
    if key == "gf3":
        tile = 78
    width = (W - 32) / 2
    block_height = 88 + 2 * tile
    if key == "pseg":
        block_height += 10
    rows = (len(records) + 1) // 2
    top = 36
    body_end = top + rows * block_height + (rows - 1) * 10
    legend_heights = [r[1] - r[0] for r in DATA[key]["footer"] if r is not None]
    footer = max(legend_heights) + (29 if key == "pseg" else 13)
    return Layout(width, tile, block_height, top, body_end + footer,
                  "2×2" if variants == 4 else "2×3")


def multiline(canvas, x, y, value, *, color=INK, size=7.2, step=11):
    for line_no, line in enumerate(value.splitlines()):
        canvas.text(x, y + step*line_no, line, color=color, size=size,
                    centered=True)


def paint(canvas, key, spec, boxes, size, local, old_width):
    canvas.rect(fitz.Rect(0, 0, W, size.height))
    canvas.text(11, 14, spec["title"], size=11.1, bold=True)
    canvas.text(11, 28, spec["context"], color=GRAY, size=7.5)
    cols = 3 if size.shape == "2×3" else 2
    starts = 11
    case_count = len(spec["cases"])

    for case_no, (effect, count, labels) in enumerate(spec["cases"]):
        i, j = case_no % 2, case_no // 2
        x = starts + i * (size.width + 10)
        y = size.y + j * (size.block_height + 10)
        tint = ADD if effect.startswith("+") else REMOVE
        heading = ("Gain" if effect.startswith("+") else "Loss")
        canvas.rect(fitz.Rect(x, y, x + size.width, y+size.block_height), WHITE, RULE)
        canvas.rect(fitz.Rect(x, y, x + size.width, y+17), LIGHT)
        canvas.rect(fitz.Rect(x, y, x+2.3, y+17), tint)
        case_id = f"({chr(ord('A' if key == 'pseg' else 'a')+case_no)})"
        canvas.text(x+8, y+11.9, f"{case_id}  {heading}", color=tint, size=8.9, bold=True)
        canvas.text(x+68, y+11.9, f"ΔIoU {effect} pt", color=tint, size=8.7, bold=True)
        if count:
            canvas.text(x + 144, y+11.9, count, color=GRAY, size=7.2)
        inner = (size.width - cols*size.tile - (cols-1)*9) / 2
        local_labels = list(spec.get("local_labels", spec["labels"])) if local else list(spec["labels"])
        metrics = list(labels)
        if key == "gf3":
            # The original signed area measurement already appears within each change map.
            metrics[-1] = ""
        indices = list(range(len(local_labels)))
        if key == "pseg" and local:
            # Keep the source reference beside the concrete threshold/budget
            # contrast. Threshold and fixed-count masks below share the same crop.
            indices = [0, None, 1, 2]
            local_labels = [local_labels[j] for j in (0, 3, 1, 2)]
            metrics = [spec["native_base"][case_no], "", metrics[1], metrics[2]]
        if key == "sarssp":
            indices = [0, 1, None, 2, 3, 4]
        zoom_parent = 0 if key == "pseg" and local else 1 if key == "sarssp" else None
        if zoom_parent is not None:
            original = boxes[case_no][zoom_parent]
            # These dimensionless coordinates select display-only insets
            # from existing artwork. They do not enter any mask evaluation.
            cx, cy = ((0.46, 0.48), (0.52, 0.36))[case_no] if key == "pseg" else (0.51, 0.50)
            zoom_size = 0.47
            zoom = fitz.Rect(original.x0 + (cx-zoom_size/2)*original.width,
                             original.y0 + (cy-zoom_size/2)*original.height,
                             original.x0 + (cx+zoom_size/2)*original.width,
                             original.y0 + (cy+zoom_size/2)*original.height)
        for tile_index, label in enumerate(local_labels):
            row, col = divmod(tile_index, cols)
            bx = x + inner + col * (size.tile + 9)
            by = y + 40 + row * (size.tile + 32)
            center = bx + 0.5 * size.tile
            highlighted = label in {"I  Independent", "ROH ranking", "Fixed count (WC)",
                                    "Local ROH", "Independent"}
            color = ACCENT if highlighted else LOCATOR if label in {"GT hull", "GT mask", "Native + GT",
                                                               "GT coverage"} else INK
            canvas.text(center, by-4, label, color=color, centered=True,
                        size=7.9 if cols == 2 else 7.0, bold=True)
            dest = fitz.Rect(bx, by, bx+size.tile, by+size.tile)
            old = indices[tile_index]
            canvas.rect(dest, WHITE, RULE)
            if old is None:
                canvas.panel(zoom, dest)
            else:
                canvas.panel(boxes[case_no][old], dest)
                if old == zoom_parent:
                    crop_scale = min(dest.width/original.width, dest.height/original.height)
                    x0 = dest.x0 + (dest.width-crop_scale*original.width)/2
                    y0 = dest.y0 + (dest.height-crop_scale*original.height)/2
                    marker = fitz.Rect(x0+(zoom.x0-original.x0)*crop_scale,
                                       y0+(zoom.y0-original.y0)*crop_scale,
                                       x0+(zoom.x1-original.x0)*crop_scale,
                                       y0+(zoom.y1-original.y0)*crop_scale)
                    canvas.rect(marker, None, LOCATOR)
            if metrics[tile_index]:
                multiline(canvas, center, by+size.tile+10, metrics[tile_index], size=7.2, step=9)

    bottom = size.y + ((case_count+1)//2)*size.block_height + (((case_count+1)//2)-1)*10
    region = spec["footer"][int(local)]
    # Keep the original data legend and its graphic scales, including both
    # PSeg colorbars and the exact native overlay legend line styles.
    y0, y1 = region
    legend_clip = fitz.Rect(0, y0, old_width, y1)
    max_legend_height = y1-y0
    canvas.panel(legend_clip, fitz.Rect(6, bottom+5, W-6, bottom+5+max_legend_height))
    if spec.get("note"):
        canvas.text(11, bottom+max_legend_height+19, spec["note"], size=7.3)


def check_values(data, kind, source_page, output, local):
    source = source_page.get_text()
    with fitz.open(output) as doc:
        p = doc[0]
        updated = p.get_text()
        # Reflowed labels carry exact scores, edit counts, and original signs.
        for delta, count, metrics in data["cases"]:
            require(delta.replace("−", "-") in source.replace("−", "-")
                    and delta in updated.replace("-", "−"), "Original signed effect " + delta)
            for detail in metrics[1:] if kind == "pseg" and local else metrics:
                for value in number(detail):
                    require(value in source and value in updated, "Original case measurement " + value)
                    if kind == "sarssp":
                        require("10 px" in updated, "Original native scale bars")
        if kind == "pseg" and local:
            require(all(v in updated for v in data["native_base"]), "Native starting IoUs")
        no_new_raw = image_bytes(p) == image_bytes(source_page)
        require(no_new_raw, "Original raster and alpha stream identities")
        images = p.get_images(full=True)
        require(bool(images) == local, "Rasters confined to local manuscript version")
        require(len(doc) == 1 and p.rect.width > 510, "A valid, full-width one-page plate")
        return {"filename": output.name, "source_cases": len(data["cases"]),
                "original_image_resources": len(image_bytes(source_page)),
                "visible_raster_images": bool(images)}


def main():
    cli = ArgumentParser(description=__doc__)
    cli.add_argument("--public-source", type=Path, required=True)
    cli.add_argument("--native-source", type=Path)
    cli.add_argument("--work", type=Path, required=True)
    cli.add_argument("--edge", type=Path, default=Path(
        "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"))
    args = cli.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    public = args.work / "public"
    local_dir = args.work / "local"
    public.mkdir(exist_ok=True)
    helpers = runpy.run_path(str(args.public_source / "style_case_panels.py"))
    report = {"public": [], "local": []}
    for local, directory in ((False, args.public_source), (True, args.native_source)):
        if directory is None:
            continue
        output = local_dir if local else public
        output.mkdir(exist_ok=True)
        for key, d in DATA.items():
            mode = "local" if local else "public"
            if mode not in d:
                continue
            name = d[mode]
            fig = output / (name + ".pdf")
            svg = output / (name + ".svg")
            require(not fig.exists() and not svg.exists(), "Fresh output for " + name)
            with fitz.open(directory / (name + ".pdf")) as source:
                old_width = source[0].rect.width
                num_cases = len(d["cases"])
                count_in = ((3 if key == "pseg" else 5 if key == "sarssp"
                             else len(d["labels"])) if local else len(d["labels"]))
                size = layout(key, d["cases"], len(d["labels"]))
                if local:
                    boxes = original_pdf_boxes(source[0], num_cases * count_in)
                    canvas = LocalCanvas(source, size.height)
                else:
                    origin = etree.parse(str(directory / (name + ".svg"))).getroot()
                    boxes = original_svg_boxes(origin, num_cases * count_in)
                    recolor(origin, key)
                    canvas = VectorCanvas(origin, size.height)
                    old_width = float(origin.get("viewBox").split()[2])
                boxes = [boxes[j:j+count_in] for j in range(0, len(boxes), count_in)]
                paint(canvas, key, d, boxes, size, local, old_width)
                if local:
                    canvas.pdf.save(fig, garbage=4, deflate=False)
                    svg.write_text(canvas.pg.get_svg_image(text_as_path=False), encoding="utf-8")
                    canvas.pdf.close()
                else:
                    data = etree.tostring(canvas.root, encoding="utf-8", xml_declaration=True)
                    svg.write_bytes(data)
                    helpers["chromium_pdf"](data, canvas.root, fig, args.edge)
                    require(not canvas.root.findall(".//" + N + "image"), "Vector-only SVG")
                report[mode].append(check_values(d, key, source[0], fig, local))
    print(json.dumps(report, indent=2))
    report_file = (local_dir if args.native_source is not None else public) / "CASE_REDRAW_VALIDATION.json"
    report_file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

