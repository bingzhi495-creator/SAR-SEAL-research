"""Restyle completed SAR case art without reopening prediction, label, or image inputs."""
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
import json
import re

import fitz
from lxml import etree


INK = "#243247"
CONTROL = "#5C6675"
TEAL = "#007F83"
ORANGE = "#C4552D"
BLUE = "#2D5E91"
RULE = "#DCE5EE"
PALE_BLUE = "#ECF3FA"
PALE_TEAL = "#E9F5F4"
PALE_ORANGE = "#FBF0EA"
PUBLIC = (
    "rsdd74_common_cap_cases",
    "dev705_overlap_cases",
    "gf3_matched_cases_binned",
    "pseg_budget_cases_binned",
    "rsdd74_segrefiner_cases_binned",
)
PRIVATE = (
    "gf3_cases_exact",
    "pseg_budget_cases_exact",
    "rsdd74_segrefiner_cases_SAR_PRIVATE",
    "sarssp_private_cases",
    "rsdd74_common_cap_cases",
)
CONTROLS = {
    "Native (N)", "Distance-first (F)", "Bounded (L)", "Native mask",
    "Dynamic attraction", "Whole threshold (WT)", "Native", "SegRefiner-LR",
    "Distance-paired",
}
ESTIMATORS = {"Independent (I)", "Learned occupancy", "Fixed count (WC)",
              "Local occupancy", "Independent"}
REFERENCES = {"Author hull", "Author footprint", "Author mask", "SAR + native mask", "SAR chip"}
DIFFERENCES = {"I minus L", "Overlap change", "Area error change", "Intersection change"}
HEADINGS = ESTIMATORS | CONTROLS | REFERENCES | DIFFERENCES
PALETTE = {
    "#183040": INK, "#24313a": INK, "#1b2735": INK, "#243347": INK,
    "#111d2b": INK, "#25384b": INK, "#697780": CONTROL,
    "#087b83": TEAL, "#087960": TEAL, "#087e8b": TEAL, "#007e8b": TEAL,
    "#c14b32": ORANGE, "#ab4232": ORANGE, "#a94e37": ORANGE, "#ac452c": ORANGE,
    "#254f70": BLUE, "#315981": BLUE, "#264e69": BLUE,
    "#dce2e5": RULE, "#e4e5e3": RULE, "#e6ecee": RULE, "#9caab3": RULE,
    "#a4b1ba": RULE, "#a6afb8": RULE, "#a8b0b5": RULE,
}
SVG = "{http://www.w3.org/2000/svg}"


def label(element):
    return "".join(element.itertext()).strip()


def style(element, **updates):
    old = element.get("style", "")
    parsed = dict(piece.strip().split(": ", 1) for piece in old.split("; ") if ": " in piece)
    parsed.update({k.replace("_", "-"): str(v) for k, v in updates.items()})
    element.set("style", "; ".join(f"{k}: {v}" for k, v in parsed.items()))


def sign(value):
    if re.search(r"gain|improved|positive case", value, re.I) or re.search(r"\+\d", value):
        return 1
    if re.search(r"loss|degraded|negative case", value, re.I) or re.search(r"[−-]\d", value):
        return -1
    return 0


def signed_value(value):
    return sign(value) if ("points" in value or "Δ" in value) and "IoU is reported" not in value else 0


def panel_label(value):
    return (bool(re.match(r"^\([a-d]\)", value)) and sign(value) != 0
            or bool(re.match(r"^[AB]\s+Gain|^[AB]\s+Loss", value))
            or value in {"Gain", "Loss"})


def rgb(hex_color):
    return tuple(int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))


def region(group, x, y, width, height, fill, border=None):
    params = {"x": f"{x:.4f}", "y": f"{y:.4f}", "width": f"{width:.4f}",
              "height": f"{height:.4f}", "fill": fill}
    if border:
        params.update(stroke=border, **{"stroke-width": "0.55"})
    etree.SubElement(group, SVG + "rect", **params)


def svg_headings(root, back):
    for node in root.iter(SVG + "text"):
        value = label(node)
        x, y = node.get("x"), node.get("y")
        if value not in HEADINGS or x is None or y is None or "text-anchor: start" in node.get("style", ""):
            continue
        x, y = float(x), float(y)
        chosen = value in ESTIMATORS
        control = value in CONTROLS
        style(node, fill=BLUE if chosen else CONTROL if control else INK,
              font_weight=700 if chosen else 600)
        halfspan = min(52.0, max(26.0, 2.1 * len(value) + 8.0))
        if chosen:
            back_at = etree.Element(SVG + "g")
            region(back_at, x - halfspan, y - 8.9, 2 * halfspan, 11.6, PALE_BLUE)
            # Place the light heading background immediately behind the label itself.
            node.addprevious(back_at)
        region(back, x - halfspan, y + 2.8, 2 * halfspan, 0.8 if chosen else 0.55,
               BLUE if chosen else RULE)


def svg_case_signs(root, accents):
    for node in root.iter(SVG + "text"):
        value = label(node)
        kind = sign(value) if panel_label(value) else signed_value(value)
        if not kind or not value:
            continue
        color, pale = (TEAL, PALE_TEAL) if kind > 0 else (ORANGE, PALE_ORANGE)
        style(node, fill=color, font_weight=700)
        if node.get("x") is None or node.get("y") is None or node.get("transform"):
            continue
        x, y = float(node.get("x")), float(node.get("y"))
        if panel_label(value):
            # The rail is wholly beside the signed text. Keep the supplied sign words.
            region(accents, max(2, x - 5), y - 6.7, 1.4, 7.7, color)
        elif re.match(r"^[+−-]\d|^Δ", value):
            # Restrict emphasis to score labels centered directly under their column.
            width = min(94, max(50, 3.9 * len(value) + 12))
            pad = etree.Element(SVG + "g")
            region(pad, x - width / 2, y - 8.2, width, 10.9, pale)
            node.addprevious(pad)



def chromium_pdf(data, root, path, edge):
    from base64 import b64encode
    import subprocess
    from tempfile import TemporaryDirectory
    from time import monotonic, sleep

    import matplotlib
    width, height = root.get("width"), root.get("height")
    font_dir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    fonts = []
    for name, variant in (
        ("DejaVuSans.ttf", "font-weight:400"),
        ("DejaVuSans-Bold.ttf", "font-weight:600 900"),
        ("DejaVuSans-Oblique.ttf", "font-style:italic; font-weight:400"),
    ):
        encoded = b64encode((font_dir / name).read_bytes()).decode("ascii")
        fonts.append(
            f"@font-face{{font-family:'DejaVu Sans';"
            f"src:url(data:font/ttf;base64,{encoded}) format('truetype');{variant}}}")
    doc = (
        '<!doctype html><meta charset="utf-8"><style>' + "".join(fonts)
        + f"@page{{size:{width} {height};margin:0}}"
        + "html,body{margin:0;padding:0}svg{display:block;position:absolute;inset:0}</style>"
        + data.decode("utf-8").split("?>", 1)[-1])
    with TemporaryDirectory(prefix="case_print_", dir=path.parent.parent, ignore_cleanup_errors=True) as tmp:
        work = Path(tmp)
        html = work / "figure.html"
        html.write_text(doc, encoding="utf-8")
        work_pdf = work / "render.pdf"
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        subprocess.run(
            [str(edge), "--headless=new", "--disable-gpu", "--no-first-run",
             "--disable-background-networking", f"--user-data-dir={work / 'profile'}",
             "--no-pdf-header-footer", f"--print-to-pdf={work_pdf}", html.as_uri()],
            capture_output=True, check=True, timeout=15, creationflags=flags)
        deadline = monotonic() + 10
        while monotonic() < deadline:
            try:
                payload = work_pdf.read_bytes()
            except FileNotFoundError:
                payload = b""
            if payload.rstrip().endswith(b"%%EOF"):
                path.write_bytes(payload)
                return
            sleep(0.05)
    raise RuntimeError("Local browser did not finish vector PDF printing")



def apply_public(stem, incoming, dest, edge):
    source = incoming / (stem + ".svg")
    root = etree.parse(str(source)).getroot()
    original_labels = [label(node) for node in root.iter(SVG + "text")]
    for node in root.iter():
        for key in ("style", "fill", "stroke", "stop-color"):
            if key in node.attrib:
                node.attrib[key] = re.sub(
                    r"#[0-9a-fA-F]{6}(?![0-9a-fA-F])",
                    lambda value: PALETTE.get(value[0].lower(), value[0]), node.attrib[key])
    art = next(node for node in root.iter(SVG + "g") if node.get("id") == "figure_1")
    accents = etree.SubElement(art, SVG + "g", id="case-style-rules")
    svg_headings(root, accents)
    svg_case_signs(root, accents)
    updated_labels = [label(node) for node in root.iter(SVG + "text")]
    if original_labels != updated_labels or len(updated_labels) < 12:
        raise ValueError(f"Public labels differ for {stem}")
    if root.findall(".//" + SVG + "image"):
        raise ValueError(f"Unexpected raster in public figure {stem}")
    public_svg = dest / (stem + ".svg")
    data = etree.tostring(root, encoding="utf-8", xml_declaration=True)
    public_svg.write_bytes(data)
    public_pdf = dest / (stem + ".pdf")
    chromium_pdf(data, root, public_pdf, edge)
    with fitz.open(public_pdf) as pdf:
        page = pdf[0]
        svg_all_numbers = Counter(number for text in original_labels
                                  for number in re.findall(r"[+\-]?\d+\.\d+", text.replace("−", "-")))
        all_lines = page.get_text().replace("−", "-")
        pdf_all_numbers = Counter(re.findall(r"[+\-]?\d+\.\d+", all_lines))
        if svg_all_numbers - pdf_all_numbers:
            raise ValueError(f"Lost public PDF score text in {stem}: {svg_all_numbers - pdf_all_numbers}")
        if len(pdf) != 1 or page.get_images(full=True):
            raise ValueError(f"Unexpected pages or raster in public PDF {stem}")
    return {"figure": stem, "public_svg_labels": len(updated_labels), "native_images": 0,
            "decimal_labels_verified": sum(svg_all_numbers.values())}


def pdf_spans(page):
    data = page.get_text("dict", flags=fitz.TEXTFLAGS_DICT ^ fitz.TEXT_PRESERVE_IMAGES)
    return [span for block in data["blocks"] for line in block.get("lines", [])
            for span in line["spans"] if span["text"].strip()]


def apply_private(stem, incoming, dest):
    with fitz.open(incoming / (stem + ".pdf")) as original:
        page = original[0]
        spans = pdf_spans(page)
        images = page.get_image_info(hashes=True)
        expected_text = Counter(span["text"] for span in spans)
        # Retain the already-qualified native page. The added rules occupy its
        # existing label gutters. No redraw or recoloring of the radar crops.
        with fitz.open() as pdf:
            pdf.insert_pdf(original)
            final = pdf[0]
            image_rects = [fitz.Rect(row["bbox"]) for row in images]
            decorations = 0
            for span in spans:
                value = span["text"].strip()
                bounds = fitz.Rect(span["bbox"])
                if value in HEADINGS:
                    chosen = value in ESTIMATORS
                    y = bounds.y1 + 0.45
                    center = (bounds.x0 + bounds.x1) / 2
                    halfspan = max(23, bounds.width / 2 + 6)
                    stroke = fitz.Rect(center - halfspan, y, center + halfspan, y + 0.9)
                    border = BLUE if chosen else RULE
                elif panel_label(value):
                    kind = sign(value)
                    x = max(2, bounds.x0 - 4.25)
                    stroke = fitz.Rect(x, bounds.y0 + .2, x + 1.15, bounds.y1 - .2)
                    border = TEAL if kind > 0 else ORANGE
                elif signed_value(value) and value.startswith(("Δ", "+", "-", "−")):
                    kind = signed_value(value)
                    y = bounds.y1 + 0.25
                    stroke = fitz.Rect(bounds.x0 - 1.5, y, bounds.x1 + 1.5, y + 0.85)
                    border = TEAL if kind > 0 else ORANGE
                else:
                    continue
                if any(stroke.intersects(box) for box in image_rects):
                    continue
                final.draw_rect(stroke, color=None, fill=rgb(border), overlay=True)
                decorations += 1
            # Mark the selected method in each row with a fine blue rule
            # outside the crop, so even the authors' original color scales
            # and overlaid outlines retain their exact appearance.
            centers = [(span["bbox"][0] + span["bbox"][2]) / 2 for span in spans
                       if span["text"].strip() in ESTIMATORS]
            for box in image_rects:
                if not any(box.x0 <= center <= box.x1 for center in centers):
                    continue
                marker = fitz.Rect(box.x0 - 2.0, box.y0, box.x0 - 1.25, box.y1)
                if not any(marker.intersects(other) for other in image_rects):
                    final.draw_rect(marker, color=None, fill=rgb(BLUE), overlay=True)
                    decorations += 1
            if not decorations:
                raise ValueError(f"No private gutter rules added for {stem}")
            if Counter(span["text"] for span in pdf_spans(final)) != expected_text:
                raise ValueError(f"Private figure labels changed in {stem}")
            if [(i["digest"], i["bbox"]) for i in final.get_image_info(hashes=True)] != [
                    (i["digest"], i["bbox"]) for i in images]:
                raise ValueError(f"Private figure image bytes or placement changed in {stem}")
            pdf.save(dest / (stem + ".pdf"), garbage=4, deflate=True)
    return {"figure": stem, "native_images": len(images), "unchanged_labels": len(spans),
            "gutter_styles": decorations}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--private", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--edge", type=Path,
                        default=Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"))
    args = parser.parse_args()
    publish = args.out / "public"
    private = args.out / "private"
    publish.mkdir(parents=True)
    public_results = [apply_public(stem, args.public, publish, args.edge) for stem in PUBLIC]
    results = {"public": public_results}
    if args.private is not None:
        private.mkdir()
        results["private"] = [apply_private(stem, args.private, private) for stem in PRIVATE]
    (args.out / "CASE_STYLE_VALIDATION.json").write_text(json.dumps(results, indent=2) + "\n",
                                                         encoding="utf-8")
    print(json.dumps(results))


if __name__ == "__main__":
    main()

