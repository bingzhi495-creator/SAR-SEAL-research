"""Rename two case-figure columns with unchanged masks, scores, and crop geometry."""

from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
from pathlib import Path
import json
import runpy

import fitz
from lxml import etree
import matplotlib
import numpy as np


CHANGES = (
    ("gf3_matched_cases_binned", "gf3_cases_exact", "Learned occupancy", "ROH ranking"),
    ("rsdd74_segrefiner_cases_binned", "rsdd74_segrefiner_cases_SAR_PRIVATE",
     "Local occupancy", "Local ROH"),
)
SVG = "{http://www.w3.org/2000/svg}"
FONT_NAME = "DejaVuSans-Bold.ttf"


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def spans(page):
    text = page.get_text("dict", flags=fitz.TEXTFLAGS_DICT ^ fitz.TEXT_PRESERVE_IMAGES)
    return [span for block in text["blocks"] for line in block.get("lines", [])
            for span in line["spans"] if span["text"].strip()]


def unique_label(page, label):
    hits = [row for row in spans(page) if row["text"] == label]
    require(len(hits) == 1, f"Expected exactly one label {label!r}")
    return hits[0]


def source_images(page):
    placed = tuple((img["digest"].hex(), img["bbox"], img["transform"])
                   for img in page.get_image_info(hashes=True))
    raw = tuple(tuple(sha256(page.parent.xref_stream_raw(xref)).hexdigest() if xref else None
                      for xref in row[:2]) for row in page.get_images(full=True))
    return placed, raw


def check_render(before, after, label, replacement, native):
    with fitz.open(before) as src, fitz.open(after) as dst:
        require(len(src) == len(dst) == 1, "Both figures are one page")
        baseline, revised = src[0], dst[0]
        require(baseline.rect == revised.rect, "Original page geometry")
        old = unique_label(baseline, label)
        new = unique_label(revised, replacement)
        expected = Counter(span["text"] for span in spans(baseline))
        expected[label] -= 1
        expected[replacement] += 1
        require(+expected == Counter(span["text"] for span in spans(revised)),
                "All other labels and metrics remain exact")
        require(abs(old["origin"][1] - new["origin"][1]) < 0.03
                and abs(old["size"] - new["size"]) < 0.03
                and old["color"] == new["color"], "Same baseline, type size and ink")
        left, right = fitz.Rect(old["bbox"]), fitz.Rect(new["bbox"])
        require(abs((left.x0 + left.x1) - (right.x0 + right.x1)) < 0.55,
                "Unshifted column center")
        before_images = source_images(baseline)
        after_images = source_images(revised)
        require(before_images == after_images, "Exact encoded images and native placement")
        require(bool(before_images[0]) == native, "Only native figures embed rasters")

        scale = 3
        transform = fitz.Matrix(scale, scale)
        pictures = [page.get_pixmap(matrix=transform, alpha=False)
                    for page in (baseline, revised)]
        grids = [np.frombuffer(pic.samples, dtype=np.uint8)
                 .reshape(pic.height, pic.width, pic.n) for pic in pictures]
        changed = np.any(grids[0] != grids[1], axis=-1)
        y, x = np.where(changed)
        require(x.size, "Rendered heading changed")
        touched = fitz.Rect(x.min() / scale, y.min() / scale,
                            (x.max() + 1) / scale, (y.max() + 1) / scale)
        envelope = (left | right) + (-1, -1, 1, 1)
        require(envelope.contains(touched),
                f"Unchanged pixels outside the heading at 216 dpi: {touched}")
        return {"file": after.name, "label": replacement,
                "native_placements_preserved": len(before_images[0]),
                "unchanged_other_text_spans": sum(expected.values()) - 1,
                "changed_pixels_outside_heading": 0}


def revise_public(directory, out, name, old, new, print_svg, browser):
    source = (directory / (name + ".svg")).read_bytes()
    original, replacement = f">{old}<".encode(), f">{new}<".encode()
    require(source.count(original) == 1 and replacement not in source,
            "Replace exactly one editable SVG heading")
    data = source.replace(original, replacement)
    require(data.replace(replacement, original) == source,
            "Byte-exact non-heading SVG content")
    root = etree.fromstring(data)
    require(len(root.findall(".//" + SVG + "image")) == 0, "Public SVG is vector-only")
    (out / (name + ".svg")).write_bytes(data)
    pdf = out / (name + ".pdf")
    print_svg(data, root, pdf, browser)
    return check_render(directory / (name + ".pdf"), pdf, old, new, native=False)


def revise_native(directory, out, name, old, new):
    before = directory / (name + ".pdf")
    after = out / (name + ".pdf")
    font_dir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    font = fitz.Font(fontfile=str(font_dir / FONT_NAME))
    with fitz.open(before) as pdf:
        page = pdf[0]
        old_span = unique_label(page, old)
        box = fitz.Rect(old_span["bbox"])
        baseline = old_span["origin"][1]
        size = old_span["size"]
        width = font.text_length(new, fontsize=size)
        x = (box.x0 + box.x1 - width) / 2
        ink = tuple(((old_span["color"] >> shift) & 255) / 255 for shift in (16, 8, 0))
        page.add_redact_annot(box, fill=False, cross_out=False)
        page.apply_redactions(images=0, graphics=0, text=0)
        name_in_pdf = "NoraRohCaseHeading"
        page.insert_font(fontname=name_in_pdf, fontfile=str(font_dir / FONT_NAME))
        page.insert_text((x, baseline), new, fontname=name_in_pdf, fontsize=size, color=ink)
        pdf.save(after, garbage=0, deflate=False)
    return check_render(before, after, old, new, native=True)


def main():
    cli = ArgumentParser(description=__doc__)
    cli.add_argument("--public", type=Path, required=True)
    cli.add_argument("--native", type=Path)
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--edge", type=Path, default=Path(
        "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"))
    args = cli.parse_args()
    helpers = runpy.run_path(str(args.public / "style_case_panels.py"))
    out_public = args.out / "public"
    out_public.mkdir(parents=True)
    report = {"public": [revise_public(args.public, out_public, item[0], *item[2:],
                                      helpers["chromium_pdf"], args.edge)
                         for item in CHANGES]}
    if args.native is not None:
        out_native = args.out / "private"
        out_native.mkdir()
        report["private"] = [revise_native(args.native, out_native, *item[1:])
                             for item in CHANGES]
    (args.out / "ROH_CASE_LABEL_VALIDATION.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()

