"""Render matched Instance-RSDD D-FINE cases from the closed common-cap result.

Use --safe-input to redraw the pooled vector from the released display data.
Use --closed --code --capture --intake --archive to reconstruct cases with the
same verified scoring and image lineage for a private native overlay.
"""
from argparse import ArgumentParser
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.collections import PatchCollection
from matplotlib.patches import Patch, Rectangle
import numpy as np


RESULT_SHA = "9176c4ba506549025ea981396cbd71633333d62e3e063d468c17817ef6b8d354"
TERMINAL_SHA = "f99f55fa2de368c782ba6d01421f3c0d6c5145d7d921175a7bda8b949cf5431e"
SCORER_SHA = "df8494fa2275076b6eb8d3c054e641074400f4bea510c6263bac1f8db1eb0d5a"
TEAL = "#087B83"
ORANGE = "#C14B32"
INK = "#183040"
COLS = ("Author hull", "Native (N)", "Distance-first (F)",
        "Bounded (L)", "Independent (I)", "I minus L")


def check(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def get_cases(args):
    import shapely

    check(shapely.__version__ == "2.0.2" and shapely.geos_version_string == "3.11.2",
          "Match the closed polygon-scoring versions")
    source = args.code / "DFINE_RSDD74_COMMON_CAP_20261126"
    scorer_file = source / "run_dfine.py"
    check(digest(scorer_file) == SCORER_SHA, "Exact accepted scorer source")
    import runpy
    scorer = SimpleNamespace(**runpy.run_path(str(scorer_file)))
    scorer.records_for_score.__globals__["np"] = np
    result = scorer.closed(args.closed / "score/RESULT_SAFE.json",
                           args.closed / "observed_score/TERMINAL.json",
                           RESULT_SHA, TERMINAL_SHA, "score", scorer.SCORE_STATUS)
    protocol = source / "PRELOCK.md"
    scorer.pin(protocol, result["protocol_sha256"])
    work = SimpleNamespace(
        code=args.code, work_root=args.closed, capture_root=args.capture,
        intake_root=args.intake, protocol=protocol,
        protocol_sha=result["protocol_sha256"], prelock_commit=result["prelock_commit"],
        capture_protocol_sha=result["protocol_sha256"], capture_commit=result["prelock_commit"],
        capture_sha=result["capture_result_sha256"],
        capture_terminal_sha=result["capture_terminal_sha256"],
        capture_readback_sha=result["capture_readback_sha256"],
        capture_readback_terminal_sha=result["capture_readback_terminal_sha256"],
        render_sha=result["render_result_sha256"],
        render_terminal_sha=result["render_terminal_sha256"],
        verify_sha=result["verify_result_sha256"],
        verify_terminal_sha=result["verify_terminal_sha256"])
    blocks, counts = scorer.records_for_score(work)
    check(counts == result["counts"], "Exact accepted native and edited counts")
    linked = scorer.module(args.code / "LEARNED_BOUNDED_PAIRING_20261126/rsdd/score.py",
                           scorer.HISTORICAL_SCORE, "_linked_score")
    original = linked.module(
        args.code / "INSTANCE_RSDD_FROZEN_HRSID_20261122/score_rsdd.py",
        linked.ORIGINAL_SCORE_SHA)
    evaluate = original.module(
        args.code / "PSEG_RFDETR_FROZEN_SOURCE_HEAD_20261122/score.py", original.SCORE_SOURCE)
    adapter = original.module(
        args.code / "UEIKAP_RFDETR_CARRIER_20261122/score_rf.py", original.GEOMETRY_SOURCE)
    geometry = adapter.load_geometry(args.code / "UEIKAP_RESERVED_HULL_FROZEN_20261122/score.py")
    reserve = linked.module(
        args.code / "INSTANCE_RSDD_ACQUISITION_RESERVE_20261123/score_reserve.py",
        linked.RESERVE_SCORE_SHA)
    settings = json.loads(scorer.pin(args.archive / "AUTHORITY_SCORE_PRIVATE.json",
                                   linked.RESERVE_AUTHORITY_SHA))
    label_bytes = scorer.pin(Path(settings["author_train"]), reserve.TRAIN)
    gt, _ = original.coco_annotation_ids(reserve.select_labels(label_bytes, blocks))
    ap, anchors = evaluate.score_one(gt, blocks, "raw", anchors=True)
    expected = result["views"]["full"]
    check(len(anchors) == expected["original_native_matches"] == 78
          and len(gt["annotations"]) == expected["author_hulls"] == 81
          and ap == expected["mask_metrics_fraction"]["raw"], "Accepted original matching and AP")

    eligible = []
    group_values = defaultdict(list)
    for ann in gt["annotations"]:
        match = anchors.get(ann["id"])
        if match is None:
            continue
        image_id, rank, original_mask = match
        check(image_id == ann["image_id"], "Original candidate/reference image alignment")
        stock = original_mask["segmentation"]
        polygon = shapely.Polygon(np.asarray(ann["segmentation"][0]).reshape(-1, 2))
        if not polygon.is_valid or polygon.area <= 0 or not 32 <= int(geometry.codec.area(stock)) < 1024:
            continue
        changes = blocks[image_id]["changes"].get(rank, {})
        rles = [changes.get(arm, {}).get("rle", stock) for arm in scorer.ARMS]
        ious = [100 * geometry.geometric_iou(rle, polygon, shapely) for rle in rles]
        delta = ious[3] - ious[2]
        group_values[blocks[image_id]["group"]].append(delta)
        if changes:
            counts_by_arm = [changes[arm]["swaps"] for arm in scorer.ARMS[1:]]
            check(len(set(counts_by_arm)) == 1, "Same action budget within each candidate")
            eligible.append(SimpleNamespace(image=image_id, ann=ann["id"], rank=rank,
                                            polygon=polygon, rles=rles,
                                            ious=ious, delta=delta, k=counts_by_arm[0]))

    score = expected["contrasts"]["I_minus_L"]["continuous_and_boundary"]["continuous_iou_points"]
    reproduced = float(np.mean([np.mean(group_values[key]) for key in sorted(group_values)]))
    check(sum(map(len, group_values.values())) == expected["eligible_matches"] == 44
          and np.isclose(reproduced, score["delta"], rtol=0, atol=1e-12),
          "Same eligibility and independently reproduced fixed-geometry mean")

    m, roster, _ = scorer.load_capture(work)
    roster = {item["image_id"]: item for item in roster}
    selected = []
    by_sign_count = []
    for sign in (1, -1):
        pool = [row for row in eligible if sign * row.delta > 1e-12]
        median = float(np.median([row.delta for row in pool]))
        by_sign_count.append(len(pool))
        chosen = min((row for row in pool if row.image not in {c.image for c in selected}),
                     key=lambda row: (abs(row.delta - median), row.image, row.ann, row.rank))
        chosen.masks = [geometry.codec.decode(rle).astype(bool) for rle in chosen.rles]
        for mask in chosen.masks[1:]:
            check(int((mask & ~chosen.masks[0]).sum()) == chosen.k and
                  int((chosen.masks[0] & ~mask).sum()) == chosen.k,
                  "Independent native mask count readback")
        item = roster[chosen.image]
        rgb = m.legacy.rgb(work, item)
        check(sha256(rgb.tobytes(order="C")).hexdigest() == item["rgb_sha256"],
              "Exact original decoded SAR image")
        chosen.gray = np.asarray(rgb, dtype=np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        selected.append(chosen)
    return selected, by_sign_count


def crop_bounds(row):
    minx, miny, maxx, maxy = row.polygon.bounds
    y, x = np.where(np.logical_or.reduce(row.masks))
    h, w = row.gray.shape
    lo_x = min(int(np.floor(minx)), int(x.min()))
    hi_x = max(int(np.ceil(maxx)), int(x.max()) + 1)
    lo_y = min(int(np.floor(miny)), int(y.min()))
    hi_y = max(int(np.ceil(maxy)), int(y.max()) + 1)
    span = min(h, w, max(hi_x - lo_x, hi_y - lo_y, 12) + 8)
    x0 = max(0, min(w - span, (lo_x + hi_x - span) // 2))
    y0 = max(0, min(h - span, (lo_y + hi_y - span) // 2))
    check(x0 <= lo_x < hi_x <= x0 + span and y0 <= lo_y < hi_y <= y0 + span,
          "A shared native crop encloses all displayed contours and masks")
    return x0, y0, x0 + span, y0 + span


def pooled(row, bounds):
    from shapely import box, area, intersection

    x0, y0, x1, y1 = bounds
    x = np.linspace(x0, x1, 9).astype(int)
    y = np.linspace(y0, y1, 9).astype(int)
    left, top = np.meshgrid(x[:-1], y[:-1])
    right, bottom = np.meshgrid(x[1:], y[1:])
    cells = box(left, top, right, bottom)
    author = area(intersection(cells, row.polygon)) / area(cells)
    result = [author]
    for matrix in (*row.masks, row.masks[3].astype(float) - row.masks[2]):
        values = [[float(matrix[ys:ye, xs:xe].mean())
                   for xs, xe in zip(x[:-1], x[1:], strict=True)]
                  for ys, ye in zip(y[:-1], y[1:], strict=True)]
        result.append(values)
    return [np.round(grid, 5).tolist() for grid in result]


def frame(ax, data):
    ax.set_xlim(data[0], data[2])
    ax.set_ylim(data[3], data[1])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for border in ax.spines.values():
        border.set_color("#9CAAB3")
        border.set_linewidth(0.45)


def draw_native(ax, row, bounds, column):
    x0, y0, x1, y1 = bounds
    gray = row.gray[y0:y1, x0:x1]
    low, high = np.percentile(gray, (2, 98))
    ax.imshow(gray, cmap="gray", vmin=low, vmax=max(high, low + 1),
              extent=(x0, x1, y1, y0), interpolation="nearest")
    x, y = row.polygon.exterior.xy
    if 1 <= column <= 4:
        yy, xx = np.where(row.masks[column - 1])
        squares = PatchCollection([Rectangle((int(i), int(j)), 1, 1)
                                   for i, j in zip(xx, yy, strict=True)],
                                  facecolor="#51D1C4", edgecolor="none", alpha=0.42)
        ax.add_collection(squares)
    elif column == 5:
        for changes, color in ((row.masks[3] & ~row.masks[2], "#51D1C4"),
                               (row.masks[2] & ~row.masks[3], "#FF805C")):
            yy, xx = np.where(changes)
            squares = PatchCollection([Rectangle((int(i), int(j)), 1, 1)
                                       for i, j in zip(xx, yy, strict=True)],
                                      facecolor=color, edgecolor="none", alpha=0.88)
            ax.add_collection(squares)
    ax.plot(x, y, color="#FFD566", linestyle=(0, (2, 1.4)), linewidth=0.85)
    frame(ax, bounds)


def draw_public(ax, case, column):
    grids = np.asarray(case["pooled_fraction"])
    scalar = LinearSegmentedColormap.from_list("coverage", ["#F0F4F6", "#43888C", "#14545B"])
    signed = LinearSegmentedColormap.from_list("edits", [ORANGE, "#F0F4F6", TEAL])
    if column == 5:
        limit = max(float(np.max(np.abs(grids[column]))), 1e-6)
        ax.pcolormesh(grids[column], cmap=signed, norm=TwoSlopeNorm(0., -limit, limit),
                  shading="flat")
    else:
        ax.pcolormesh(grids[column], cmap=scalar, vmin=0, vmax=1,
                  shading="flat")
    ax.vlines(range(1, 8), 0, 8, linewidth=0.22, color="white", alpha=0.48)
    ax.hlines(range(1, 8), 0, 8, linewidth=0.22, color="white", alpha=0.48)
    frame(ax, (0, 0, 8, 8))


def render(cases, output, native=None, save_preview=False):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7,
                        "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 6, figsize=(7.16, 3.35))
    fig.subplots_adjust(left=0.027, right=0.977, top=0.80, bottom=0.17,
                        hspace=0.73, wspace=0.13)
    for ax, title in zip(axes[0], COLS, strict=True):
        ax.set_title(title, fontsize=7.1, fontweight="semibold", color=INK, pad=6)
    for row_number, (case, panels) in enumerate(zip(cases, axes, strict=True)):
        delta = case["independent_minus_bounded_iou_points"]
        for column, ax in enumerate(panels):
            if native is None:
                draw_public(ax, case, column)
            else:
                draw_native(ax, native[row_number], crop_bounds(native[row_number]), column)
            if 1 <= column <= 4:
                annotation = f'IoU {case["continuous_iou_points"][COLS[column].split("(")[-1][0]]:.2f}'
            elif column == 5:
                annotation = f"{delta:+.2f} points"
            else:
                annotation = f'{case["swaps"]} add / {case["swaps"]} remove'
            ax.text(0.5, -0.11, annotation, transform=ax.transAxes,
                    va="top", ha="center", fontsize=6.8, color=INK,
                    fontweight="semibold" if column == 5 else "normal")
        row_caption = "(a) Median gain" if row_number == 0 else "(b) Median loss"
        first = panels[0].get_position()
        if row_number == 0:
            fig.text(first.x0, 0.955, row_caption, ha="left", va="center",
                     fontsize=7.6, color=TEAL, fontweight="bold")
        else:
            fig.text(first.x0, first.y1 + 0.04, row_caption, ha="left", va="bottom",
                     fontsize=7.6, color=ORANGE, fontweight="bold")
    if native is None:
        handles = [Patch(color="#43888C", label="Pooled occupancy"),
                   Patch(color=TEAL, label="Net I-only foreground"),
                   Patch(color=ORANGE, label="Net L-only foreground")]
    else:
        handles = [Patch(color="#FFD566", label="Author outline"),
                   Patch(color="#51D1C4", label="Foreground in I only"),
                   Patch(color="#FF805C", label="Foreground in L only")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=6.9, handlelength=1.2, columnspacing=1.65,
               bbox_to_anchor=(0.5, 0.016))
    stem = output / "rsdd74_common_cap_cases"
    fig.savefig(stem.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(stem.with_suffix(".svg"), metadata={"Date": None})
    if save_preview:
        fig.savefig(stem.with_suffix(".png"), dpi=220)
    plt.close(fig)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    for name in ("safe-input", "closed", "code", "capture", "intake", "archive"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    if args.safe_input:
        payload = json.loads(args.safe_input.read_text(encoding="utf-8"))
        args.out.mkdir(parents=True, exist_ok=False)
        render(payload["cases"], args.out, save_preview=True)
        return
    check(all((args.closed, args.code, args.capture, args.intake, args.archive)),
          "All closed evidence roots are required")
    check(not args.out.exists(), "Use a fresh figure-only directory")
    selected, counts = get_cases(args)
    cases = [dict(swaps=row.k, continuous_iou_points=dict(zip("NFLI", row.ious, strict=True)),
                  independent_minus_bounded_iou_points=row.delta,
                  pooled_fraction=pooled(row, crop_bounds(row))) for row in selected]
    payload = dict(source_result_sha256=RESULT_SHA, eligible_matches=44,
                   support_nonzero_change_above_1e_12=dict(gain=counts[0], loss=counts[1]),
                   selection="Nearest sign-specific individual continuous-IoU median; distinct images",
                   pooling="Mean mask occupancy on an 8 by 8 display grid; author polygon area fraction",
                   cases=cases)
    public = args.out / "public"
    private = args.out / "private"
    public.mkdir(parents=True)
    private.mkdir()
    (public / "rsdd74_common_cap_cases_safe.json").write_text(
        json.dumps(payload, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    (private / "CASE_TRACE_PRIVATE.json").write_text(json.dumps([
        dict(panel=case, image_id=row.image, reference_id=row.ann, native_rank=row.rank,
             crop_xyxy=crop_bounds(row)) for case, row in zip(("a", "b"), selected, strict=True)
    ], indent=2) + "\n", encoding="utf-8")
    render(cases, public, save_preview=True)
    render(cases, private, native=selected, save_preview=True)
    print(json.dumps(dict(status="matched_case_figures_complete", cases=len(cases),
                          eligible=44, gain=counts[0], loss=counts[1],
                          iou_points=[r["continuous_iou_points"] for r in cases])))


if __name__ == "__main__":
    main()

