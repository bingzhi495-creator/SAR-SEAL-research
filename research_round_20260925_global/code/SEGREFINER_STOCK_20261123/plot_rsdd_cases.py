"""Render fixed median positive/negative SegRefiner cases without publishing SAR chips."""
import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from types import ModuleType

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from PIL import Image
import shapely


SCORE = "ca904c131af51f2fb49daa6b5f30808917a37635c11de12815782f65c6c06eff"
SCORE_WORKER = "ccbcc1d253b10f5cbd1320c7eff29662b1d2f525b04de17443732d687df22734"
PLOT_SOURCE = "6713f46f0eabec91f05271885bac4e3376f49203027e19583ac39da4b3fd6f63"
EPS = 1e-10
N_TILES = 8
BLUE, TEAL, RUST, INK = "#315981", "#087960", "#ab4232", "#1b2735"
PUBLIC = "rsdd74_segrefiner_cases_binned"
PRIVATE = "rsdd74_segrefiner_cases_SAR_PRIVATE"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def pin(path, expected):
    raw = path.read_bytes()
    require(sha256(raw).hexdigest() == expected, "Digest differs for " + path.name)
    return raw


def module(path, expected):
    obj = ModuleType(path.stem)
    obj.__file__ = str(path)
    exec(compile(pin(path, expected), str(path), "exec"), obj.__dict__)
    return obj


def qualified_cases(a, scorer):
    exposure_source = scorer.exposure_source()
    whole, deps = exposure_source.sources(a)
    _, api, cfa, history, reserve, old, evaluator, adapter, core = deps
    whole.ARMS, whole.MATCHED, whole.PAIRS = scorer.ARMS, ("raw",), scorer.PAIRS
    report, _, _ = api.observed(a.run / "score", a.score_terminal, "score", scorer.STATUS, SCORE, worker=SCORE_WORKER)
    require(report["protocol_sha256"] == scorer.LOCK and report["ledgers_verified_before_labels"] is True
            and report["original_coco_and_local_geometry_exact"] is True
            and report["counts"]["eligible_masks_changed"] == report["counts"]["eligible_candidates"] == 3074,
            "Closed SegRefiner experiment differs")
    infer, _, _ = api.observed(a.run / "inference", a.inference_terminal, "inference", scorer.INFER_STATUS,
                              report["inference_result_sha256"], report["inference_terminal_sha256"],
                              report["inference_worker_sha256"])
    require(infer["refined_masks_sha256"] == report["refined_masks_sha256"]
            and infer["target_labels_opened"] is False and infer["scientific_replay"] is True
            and infer["exact_original_bypass"] is True and infer["candidate_ordinals_preserved"] is True,
            "Verified no-label author replay required")
    a.cohort, a.support_root = 74, None
    loader = cfa.source(a.code / "POINTREND_SIGNED_QUOTA_SWAP_20261123/inputs.py", cfa.INPUTS_SHA)
    rows, lineage = loader.load_population(a)
    require(lineage == report["input_lineage"] == infer["input_lineage"], "Original candidate lineage differs")
    blocks, areas, point_positions = exposure_source.native_blocks(rows, adapter, core)
    stream = api.pin(a.run / "inference" / scorer.LEDGER, report["refined_masks_sha256"])
    totals = scorer.attach_verified_masks(rows, blocks, areas, point_positions, stream, infer["counts"])
    require(totals == report["counts"], "Readback stock/local mask census differs")

    gt, _, polygons, _, dates, label_sha = history.views_and_labels(a, old, reserve, blocks)
    require(label_sha == report["author_labels_sha256"], "Original author labels differ")
    published = report["views"]["full"]
    metric, anchors = evaluator.score_one(gt, blocks, "raw", anchors=True)
    require(len(anchors) == 81 and {k: history.available(v) for k, v in metric.items()}
            == published["metrics_fraction"]["raw"], "Native matching or metric parity differs")
    gate = {gid: (i, rank, det) for gid, (i, rank, det) in anchors.items()
            if scorer.GATE <= areas[i][rank] < 1024 and polygons[gid].is_valid and polygons[gid].area > 0}
    annotations = {ann["id"]: ann for ann in gt["annotations"]}
    geometry = whole.fixed_geometry(gate, blocks, polygons, annotations, dates, cfa, core, reserve)
    target = published["gate_exposed_geometry"]
    require(len(gate) == 39 and geometry["counts"] == target["counts"]
            and geometry["represented_groups"] == 12 and geometry["sensor_date_blocks"] == 11
            and all(abs(value-target["group_equal_arms"][arm][key]) <= EPS
                    for arm, fields in geometry["group_equal_arms"].items() for key, value in fields.items()),
            "All 39 original gate-positive fixed hulls and geometry must reproduce")
    items = {row[0]["image_id"]: row[0] for row in rows}
    cases = []
    for gid, (image, slot, det) in gate.items():
        raw = det["segmentation"]
        opts = blocks[image]["changes"][slot]
        masks = [opts.get(arm, {"rle": raw})["rle"] for arm in scorer.ARMS]
        poly = polygons[gid]
        iou = np.array([core.geometric_iou(mask, poly, shapely) for mask in masks])
        require(masks[2] != raw, "Published gate-positive hull must have an actual edit")
        cases.append(dict(image_id=image, slot=slot, annotation_id=gid, item=items[image], polygon=poly,
                          masks=masks, iou_points=100*iou, delta_points=100*(iou[2]-iou[0])))
    return cases, core


def choose_cases(cases):
    pools = {name: [row for row in cases if direction*row["delta_points"] > EPS]
             for name, direction in (("gain", 1), ("loss", -1))}
    require(all(pools.values()), "Both observed signs must exist to illustrate the stated selection")
    selected = {}
    counts = {name: len(rows) for name, rows in pools.items()}
    counts["ties"] = len(cases)-sum(counts.values())
    decisions = {}
    for sign, pool in pools.items():
        median = float(np.median([row["delta_points"] for row in pool]))
        previous = selected.get("gain")
        distinct = [row for row in pool if previous is None or row["image_id"] != previous["image_id"]]
        eligible = distinct or pool
        nearest = min(abs(row["delta_points"]-median) for row in eligible)
        ties = [row for row in eligible if abs(row["delta_points"]-median) - nearest <= EPS]
        chosen = min(ties, key=lambda row: (row["image_id"], row["slot"], row["annotation_id"]))
        selected[sign] = chosen
        decisions[sign] = dict(within_sign_median_points=median, median_distance_points=abs(chosen["delta_points"]-median),
                               tied_nearest_choices=len(ties),
                               other_image_available=bool(distinct) if previous is not None else None)
    return selected, counts, decisions


def cropped_case(row, intake, core):
    item = row["item"]
    ref = item["image_ref"]
    path = (intake / ref["path"]).resolve(strict=True)
    require(path.parent == (intake / "images").resolve(strict=True), "Only original closed intake JPEG allowed")
    compressed = pin(path, ref["sha256"])
    require(len(compressed) == ref["bytes"], "Original image byte count differs")
    with Image.open(BytesIO(compressed)) as image:
        require(image.format == "JPEG" and image.size == (512, 512), "Original JPEG expected")
        rgb = np.array(image.convert("RGB"), dtype=np.uint8, order="C")
    require(sha256(rgb.tobytes()).hexdigest() == item["rgb_sha256"], "Original decoded image differs")
    masks = np.stack([core.codec.decode(mask).astype(bool) for mask in row["masks"]])
    y, x = np.nonzero(np.logical_or.reduce(masks))
    poly = row["polygon"]
    x0, y0, x1, y1 = poly.bounds
    x0, y0, x1, y1 = min(x0, x.min()), min(y0, y.min()), max(x1, x.max()+1), max(y1, y.max()+1)
    side = min(512, N_TILES*int(np.ceil(max(40, x1-x0+12, y1-y0+12)/N_TILES)))
    left = int(np.clip(np.floor((x0+x1-side)/2), 0, 512-side))
    top = int(np.clip(np.floor((y0+y1-side)/2), 0, 512-side))
    ys, xs = slice(top, top+side), slice(left, left+side)
    gy, gx = np.mgrid[ys, xs]
    fraction = shapely.area(shapely.intersection(shapely.box(gx, gy, gx+1, gy+1), poly))
    crop_masks = masks[:, ys, xs]
    require(abs(float(fraction.sum())-poly.area) < 1e-7
            and np.array_equal(crop_masks.sum(axis=(1, 2)), masks.sum(axis=(1, 2))), "Case crop loses masks or truth")
    inter = (crop_masks*fraction).sum(axis=(1, 2))
    iou = 100*inter/(crop_masks.sum(axis=(1, 2))+float(fraction.sum())-inter)
    require(np.allclose(iou, row["iou_points"], rtol=0, atol=EPS), "Case crop and continuous scorer disagree")
    gray = rgb[ys, xs] @ np.array([.2126, .7152, .0722])
    extent = np.percentile(gray, [2, 98]).tolist()
    return dict(gray=gray, gray_range=extent, masks=crop_masks, truth=fraction,
                contour=np.asarray(poly.exterior.coords)-[left, top], iou_points=iou, side=side,
                delta_points=float(iou[2]-iou[0]))


def pooled(array):
    block = array.shape[0]//N_TILES
    return array.reshape(N_TILES, block, N_TILES, block).mean(axis=(1, 3))


def render(cases, edges, public):
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "svg.hashsalt": PUBLIC,
                         "pdf.fonttype": 42, "figure.dpi": 120, "axes.unicode_minus": True})
    fig, grid = plt.subplots(len(cases), 4, figsize=(7.15, 4.65), squeeze=False)
    fig.subplots_adjust(left=.129, right=.987, top=.894, bottom=.194, wspace=.14, hspace=.28)
    headings = ("Native mask", "Local occupancy", "SegRefiner-LR", "Intersection change")
    for ax, title in zip(grid[0], headings, strict=True):
        ax.set_title(title, fontsize=9.0, fontweight="semibold", color=INK, pad=6)
    shades = ListedColormap(["#f8f8f4", "#dce7ee", "#acc4d6", "#7499b4", BLUE])
    signed = matplotlib.colors.LinearSegmentedColormap.from_list("intersection", [RUST, "#fcfcf7", TEAL])
    edge_int = np.arange(N_TILES+1)
    tile_centers = edge_int[:-1]+.5
    if public:
        pooled_changes = {sign: pooled((case["masks"][2].astype(float)-case["masks"][0])*case["truth"])
                          for sign, case in cases.items()}
        scale = max(np.abs(value).max() for value in pooled_changes.values()) or 1.
    for (sign, case), panels in zip(cases.items(), grid, strict=True):
        side = case["side"]
        masks = case["masks"]
        delta = (masks[2].astype(float)-masks[0])*case["truth"]
        exact_extent = (0, side, side, 0)
        for col, ax in enumerate(panels):
            if public:
                geom = pooled(case["truth"])
                if col < 3:
                    ax.pcolormesh(edge_int, edge_int, np.rint(4*pooled(masks[col])), cmap=shades,
                                  vmin=0, vmax=4, edgecolors="#eef1ee", linewidth=.25)
                else:
                    pooled_delta = pooled_changes[sign]
                    binned = np.sign(pooled_delta)*np.maximum(1, np.rint(4*np.abs(pooled_delta)/scale))
                    ax.pcolormesh(edge_int, edge_int, binned, cmap=signed, norm=TwoSlopeNorm(0, -4, 4),
                                  edgecolors="#eef1ee", linewidth=.25)
                if np.min(geom) < .25 < np.max(geom):
                    ax.contour(tile_centers, tile_centers, geom, levels=[.25], colors=INK,
                               linestyles="--", linewidths=.85)
                ax.set(xlim=(0, N_TILES), ylim=(N_TILES, 0))
            else:
                if col < 3:
                    lo, hi = case["gray_range"]
                    ax.imshow(case["gray"], cmap="gray", vmin=lo, vmax=max(hi, lo+1),
                              extent=exact_extent, interpolation="nearest")
                    ax.add_collection(LineCollection(edges(masks[col]), colors="#47d4ff", linewidths=.78))
                    truth_color = "#f4cd60"
                else:
                    ax.imshow(delta, cmap=signed, norm=TwoSlopeNorm(0, -1, 1),
                              extent=exact_extent, interpolation="nearest")
                    truth_color = INK
                ax.plot(*case["contour"].T, color=truth_color, lw=.9, ls=(0, (2.4, 1.5)))
                ax.set(xlim=(0, side), ylim=(side, 0))
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#afbbc3")
                spine.set_linewidth(.55)
            label = (f"IoU {case['iou_points'][col]:.2f}" if col < 3
                     else f"ΔIoU {case['delta_points']:+.2f} pt")
            ax.set_xlabel(label, fontsize=8, color=INK if col < 3 else TEAL if sign == "gain" else RUST, labelpad=3)
        bound = panels[0].get_position()
        fig.text(.024, bound.y0+bound.height*.54, f"({'a' if sign == 'gain' else 'b'})\n{sign.capitalize()}",
                 ha="left", va="center", fontsize=8.4, fontweight="semibold", color=INK)
    legend = ([Patch(facecolor=BLUE, label="Pooled mask fraction"),
               Line2D([], [], color=INK, ls="--", label="25% reference contour")] if public else
              [Line2D([], [], color="#007ba9", label="Predicted boundary"),
               Line2D([], [], color="#b08319", ls="--", label="Author reference")])
    legend += [Patch(facecolor=TEAL, label="Gained intersection"), Patch(facecolor=RUST, label="Lost intersection")]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.53, .083), frameon=False, ncols=4,
               fontsize=7.4, handlelength=1.45, columnspacing=1.0)
    note = ("Each map pools an 8×8 grid. No SAR intensity is shown." if public
            else "Both rows use the original SAR crop, shown in grayscale with a shared stretch within each row.")
    fig.text(.53, .062, note, fontsize=7.0, color=INK, ha="center")
    fig.text(.53, .036, "Intersection is SegRefiner-LR minus native. IoU uses the original continuous polygon.",
             fontsize=7.0, color=INK, ha="center")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("code", "run", "stock-root", "point-root", "author-labels", "protocol", "public-output", "private-output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("score-terminal", "inference-terminal"):
        parser.add_argument("--" + name, type=Path)
    a = parser.parse_args()
    for name, path in vars(a).items():
        if isinstance(path, Path):
            setattr(a, name, path.resolve(strict=name not in {"author_labels", "public_output", "private_output"}))
    a.score_terminal = a.score_terminal or a.run / "observed_score" / "TERMINAL.json"
    a.inference_terminal = a.inference_terminal or a.run / "observed_inference" / "TERMINAL.json"
    public, private = a.public_output, a.private_output
    require(public != private and public not in private.parents and private not in public.parents
            and not public.exists() and not private.exists(), "Distinct fresh public and local-only output directories required")
    scorer = module(a.code / "SEGREFINER_STOCK_20261123/score_rsdd.py", SCORE_WORKER)
    require(a.protocol.name == scorer.LOCK_NAME, "Published stock comparison prelock required")
    pin(a.protocol, scorer.LOCK)
    style = module(a.code.parent / "paper_boundary_refinement_20261120/figures/plot_gf3_matched_cases.py", PLOT_SOURCE)
    rows, core = qualified_cases(a, scorer)
    selected, sign_counts, decisions = choose_cases(rows)
    cases = {k: cropped_case(row, a.stock_root / "conversion", core) for k, row in selected.items()}
    public.mkdir(parents=True)
    private.mkdir(parents=True)
    full = render(cases, style.edges, False)
    full.savefig(private / (PRIVATE + ".pdf"), metadata={"CreationDate": None, "ModDate": None})
    plt.close(full)
    binned = render(cases, style.edges, True)
    binned.savefig(public / (PUBLIC + ".svg"), metadata={"Date": None})
    binned.savefig(public / (PUBLIC + ".pdf"), metadata={"CreationDate": None, "ModDate": None})
    plt.close(binned)
    private_rows = {key: {field: row[field] for field in ("image_id", "slot", "annotation_id")}
                    for key, row in selected.items()}
    (private / "rsdd74_segrefiner_case_selection_PRIVATE.json").write_text(
        json.dumps(private_rows, sort_keys=True, indent=2)+"\n", encoding="utf-8")
    summary = dict(
        closed_score_sha256=SCORE, gate_exposed_hulls=len(rows), native_gate_pixels=scorer.GATE,
        positive_negative_zero_tolerance_points=EPS, sign_counts=sign_counts, selection_details=decisions,
        arm_order=list(scorer.ARMS), public_grid=[N_TILES, N_TILES], sar_pixels_public=False,
        different_selected_images=selected["gain"]["image_id"] != selected["loss"]["image_id"],
        cases={key: dict(iou_points=case["iou_points"].tolist(), delta_points=case["delta_points"])
               for key, case in cases.items()},
    )
    encoded = json.dumps(summary, sort_keys=True, indent=2) + "\n"
    (public / "rsdd74_segrefiner_case_summary_SAFE.json").write_text(encoded, encoding="utf-8")
    print(encoded, end="", flush=True)


if __name__ == "__main__":
    main()
