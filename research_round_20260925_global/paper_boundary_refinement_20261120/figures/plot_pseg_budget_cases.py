"""Render signed-median examples from the closed official PSeg-SSDD TEST232 factorial.

Give --code the public release's code directory, --run the closed score/inference
directory, --request the closed input request JSON, --owner the original image-only
package, --capture the D-FINE capture directory, and --author-labels the author
COCO JSON. Only --public-dir receives the pooled vector artwork. --private-dir
receives the original-image artwork. Use --selection-only to stop before reading
either of the two selected image files.
"""
import argparse
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
from types import ModuleType

import numpy as np
import shapely
from PIL import Image
from pycocotools import mask as codec


SCORE = "e96c7fe8dc857fa85e0fd6051cf267068ae4a027ea8299a19678f5facfcfbcff"
GEOMETRY = "107f31864bbb7e807a4194e552c7a1778d410f0e564f80437c5717fe6b47098d"
ARMS = ("raw", "wt", "wc")
FIELDS = ("continuous_iou_points", "fp_pixels_per_gt_area_pct", "fn_pixels_per_gt_area_pct")
EPS = 1e-9  # points; roundoff-level equality


def check(condition, message):
    if not condition:
        raise ValueError(message)


def pinned(path, expected):
    blob = path.read_bytes()
    check(hashlib.sha256(blob).hexdigest() == expected, "Digest mismatch: " + path.name)
    return blob


def module(path, digest):
    mod = ModuleType(path.stem)
    mod.__file__ = str(path)
    exec(compile(pinned(path, digest), str(path), "exec"), mod.__dict__)
    return mod


def closed(directory, phase, expected):
    data = json.loads(pinned(directory / phase / "RESULT_SAFE.json", expected))
    receipt = json.loads((directory / ("observed_" + phase) / "TERMINAL.json").read_bytes())
    check(receipt["phase"] == data["phase"] == phase and receipt["actual_exit"] == 0
          and receipt["output_footer_verified"] and receipt["result_sha256"] == expected
          and receipt["worker_sha256"] == data["worker_sha256"], "Incomplete closure")
    return data


def jsonlines(path, digest):
    return [json.loads(line) for line in pinned(path, digest).splitlines()]


def cases(args):
    check(shapely.__version__ == "2.0.2" and shapely.geos_version_string == "3.11.2",
          "Original continuous geometry engine required")
    score = closed(args.run, "score", SCORE)
    infer = closed(args.run, "inference", score["inference_result_sha256"])
    check(infer["masks_sha256"] == score["masks_sha256"], "Mask receipt mismatch")
    orig = score["input_lineage"]
    cohort = json.loads(pinned(args.owner / "TEST232_IMAGE_ONLY_MANIFEST_PRIVATE.json",
                               orig["manifest_sha256"]))["items"]
    gt = json.loads(pinned(args.author_labels, score["author_labels_sha256"]))
    pred = jsonlines(args.capture / "DEV_MASK_PREDICTIONS_PRIVATE.jsonl", orig["predictions_sha256"])
    edits = jsonlines(args.run / "inference/FACTORIAL_EDITS_PRIVATE.jsonl", score["masks_sha256"])
    request = json.loads(pinned(args.request, score["request_sha256"]))["images"]
    images = {im["file_name"]: im for im in gt["images"]}
    records = {}
    for j, (item, p, e, req) in enumerate(zip(cohort, pred, edits, request, strict=True), 1):
        check(item["image_id"] == p["image_id"] == e["image_id"] == req["image_id"]
              and j == e["ordinal"], "Native image order mismatch")
        image = images[item["image_ref"]["path"]]
        check([image["height"], image["width"]] == item["original_hw"] == req["hw"],
              "Canvas mismatch")
        candidates = {c["position"]: c for c in req["candidates"]}
        check(set(map(int, e["changes"])) == set(candidates), "Edit/request ranks mismatch")
        records[image["id"]] = dict(native=p["coco"][:100], item=item, edits=e["changes"],
                                    request=candidates, rgb=req["decoded_rgb_sha256"], ordinal=j)
    geom = module(args.code / "PSEG_TEST232_PAIRED_GEOMETRY_20261121/paired_geometry.py", GEOMETRY)
    shared = module(args.code / "MASK_RTDETR_DEV705_CONTINUOUS_AP_20261112/continuous_endpoint.py", geom.GEOM)
    evaluator = module(args.code / "DFINE_SEG_EXTERNAL_BASELINE_FIT32_E24_20261112/e24/dev705_free_endpoint/score_endpoint.py", geom.OLD)
    mods = shared.geometry_modules(args.code)
    polygons, neutral, _ = mods.geometry.polygon_support(gt, mods.vector)
    check(len(polygons) == 546 and not neutral and len(records) == 232, "Official scope mismatch")
    official = score["views"]["official"]
    pairs, _ = geom.evaluate_raw(gt, records, sorted(records), official["metrics_fraction"]["raw"])
    rows = []
    groups = defaultdict(list)
    for gid, (iid, rank) in pairs.items():
        rec = records[iid]
        raw = rec["native"][rank - 1]["segmentation"]
        if not 32 <= int(codec.area(raw)) < 1024:
            continue
        req = rec["request"].get(rank)
        if req:
            check(req["segmentation"] == raw, "Matched rank/native mask mismatch")
        edit = rec["edits"].get(str(rank), {})
        rles = [edit.get(arm, {}).get("rle", raw) for arm in ARMS]
        hull = polygons[gid][0]
        xyz = []
        for rle in rles:
            area = int(codec.area(rle))
            xyz.append(evaluator.continuous(rle, area, hull, mods) if area else (0., 0., 100.))
        xyz = np.asarray(xyz)
        delta = float(xyz[2, 0] - xyz[1, 0])
        rows.append(dict(ordinal=rec["ordinal"], rank=rank, gid=gid, item=rec["item"],
                         rgb=rec["rgb"], delta=delta, metrics=xyz, rles=rles, hull=hull,
                         k=req["k"] if req else 0))
        groups[rec["rgb"]].append(xyz)
    check((len(pairs), len(rows), len(groups)) == (402, 293, 143), "Fixed hull count mismatch")
    means = np.mean([np.mean(groups[g], axis=0) for g in sorted(groups)], axis=0)
    expected = [[official["geometry"]["arms"][a][f] for f in FIELDS] for a in ARMS]
    check(np.allclose(means, expected, rtol=0, atol=1e-9), "Original fixed geometry mismatch")
    sign = lambda value: 1 if value > EPS else -1 if value < -EPS else 0
    counts = Counter(sign(r["delta"]) for r in rows)
    medians = {}
    chosen = []
    for direction in (1, -1):
        pool = [r for r in rows if sign(r["delta"]) == direction]
        medians[direction] = float(np.median([r["delta"] for r in pool]))
        ordered = sorted(pool, key=lambda r: (abs(r["delta"]-medians[direction]),
                                               r["ordinal"], r["rank"], r["gid"]))
        chosen.append(next(r for r in ordered if all(r["rgb"] != other["rgb"] for other in chosen)))
    print(json.dumps(dict(population=len(rows), improved=counts[1], worsened=counts[-1],
                          unchanged=counts[0], medians=medians,
                          examples=[dict(delta=r["delta"], wt=r["metrics"][1].tolist(),
                                         wc=r["metrics"][2].tolist(), k=r["k"]) for r in chosen])),
          flush=True)
    return chosen, counts


def prepare_case(row, owner):
    item, ref = row["item"], row["item"]["image_ref"]
    filename = Path(ref["path"])
    check(filename.name == ref["path"], "Expected original basename")
    source = pinned(owner / "IMAGE_ONLY" / filename, ref["sha256"])
    with Image.open(io.BytesIO(source)) as image:
        rgb = image.convert("RGB")
        check(hashlib.sha256(rgb.tobytes()).hexdigest() == row["rgb"], "Source RGB mismatch")
        grey = np.asarray(rgb.convert("L"))
    masks = np.stack([codec.decode(r).astype(bool) for r in row["rles"]])
    check(list(grey.shape) == item["original_hw"] and masks.shape[1:] == grey.shape, "Canvas mismatch")
    hull = row["hull"]
    yy, xx = np.nonzero(np.any(masks, axis=0))
    l, t, r, b = hull.bounds
    l, r, t, b = min(l, xx.min()), max(r, xx.max()+1), min(t, yy.min()), max(b, yy.max()+1)
    block = max(4, int(np.ceil((max(r-l, b-t)+8) / 8)))
    side = 8*block
    h, w = grey.shape
    check(side <= min(h, w), "Window exceeds source extent")
    left = int(np.clip(np.floor((l+r-side)/2), 0, w-side))
    top = int(np.clip(np.floor((t+b-side)/2), 0, h-side))
    ys, xs = slice(top, top+side), slice(left, left+side)
    gy, gx = np.mgrid[ys, xs]
    ref_fraction = shapely.area(shapely.intersection(shapely.box(gx, gy, gx+1, gy+1), hull))
    crop = masks[:, ys, xs]
    areas = crop.sum(axis=(1, 2))
    check(np.array_equal(areas, masks.sum(axis=(1, 2)))
          and abs(ref_fraction.sum() - hull.area) < 1e-7, "Displayed geometry clipped")
    inter = np.sum(crop*ref_fraction, axis=(1, 2))
    truth = hull.area
    measures = 100*np.stack((inter/(areas+truth-inter), (areas-inter)/truth, 1-inter/truth), axis=-1)
    check(np.allclose(measures, row["metrics"], rtol=0, atol=1e-8), "Display and score disagree")
    flips = np.count_nonzero(crop != crop[0], axis=(1, 2))
    check(flips[2] == 2*row["k"], "Fixed count mismatch")
    pieces = [hull] if hull.geom_type == "Polygon" else hull.geoms
    rings = [np.asarray(ring.coords)-[left, top] for piece in pieces
             for ring in (piece.exterior, *piece.interiors)]
    return dict(masks=crop, grey=grey[ys, xs], truth=ref_fraction, rings=rings, block=block,
                side=side, flips=flips, metrics=measures, delta=row["delta"])


def pool(field, block):
    return field.reshape(8, block, 8, block).sum(axis=(1, 3))


def boundaries(mask):
    # Cell-corner coordinates preserve every horizontal/vertical unit edge.
    padded = np.pad(mask.astype(np.int8), 1)
    y, x = np.nonzero(np.diff(padded[1:-1, :], axis=1))
    vertical = np.stack((np.stack((x,y), axis=-1), np.stack((x,y+1), axis=-1)), axis=1)
    y, x = np.nonzero(np.diff(padded[:, 1:-1], axis=0))
    horizontal = np.stack((np.stack((x,y), axis=-1), np.stack((x+1,y), axis=-1)), axis=1)
    return np.concatenate((vertical, horizontal), axis=0)


def render(args, selected, counts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.collections import LineCollection
    from matplotlib.colors import BoundaryNorm, ListedColormap, LinearSegmentedColormap, Normalize
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":8.5,
                         "svg.fonttype":"none", "svg.hashsalt":"pseg-budget-v1",
                         "pdf.fonttype":42, "axes.unicode_minus":True})
    data = [prepare_case(row, args.owner) for row in selected]
    ink, teal, rust = "#243347", "#087667", "#ac452c"
    shades = ListedColormap(["#f5f6f5","#d3e0e6","#9eb9c7","#60899f","#264e69"])
    levels = BoundaryNorm(np.arange(-.5, 5.5, 1), shades.N)
    signed = LinearSegmentedColormap.from_list("area_error_change", [rust,"#f8f8f6",teal])
    residuals = []
    for case in data:
        raw, wt, wc = case["masks"].astype(float)
        truth = case["truth"]
        # |prediction - fraction| is exactly cell FP + FN area for binary predictions.
        residuals.append(pool(np.abs(wt-truth)-np.abs(wc-truth), case["block"]))
    limit = max(float(np.abs(v).max()) for v in residuals)
    norm = Normalize(-limit, limit)
    tagline = (f"{sum(counts.values())} fixed ships    ·    {counts[1]} improved"
               f"    ·    {counts[-1]} worsened    ·    {counts[0]} unchanged")
    for public in (True, False):
        fig = plt.figure(figsize=(7.16, 5.52 if public else 6.08))
        head = "Whole-grid edit budgets on PSeg-SSDD"
        fig.text(.028, .975, head, va="top", fontsize=13, weight="bold", color=ink)
        fig.text(.029, .927, tagline, va="top", fontsize=9, color=ink)
        configs = ((.59, .879), (.174, .463)) if public else ((.56, .88), (.143, .463))
        titles = (["Author footprint", "Whole threshold (WT)", "Fixed count (WC)", "Area error change"]
                  if public else ["SAR + native mask", "Whole threshold (WT)", "Fixed count (WC)"])
        centers = [.141, .380, .619, .858] if public else [.169, .500, .831]
        width, height = (.171, .171*7.16/5.52) if public else (.225, .225*7.16/6.08)
        for i, (case, (bottom, band)) in enumerate(zip(data, configs, strict=True)):
            pos = case["delta"] > 0
            delta = case["delta"]
            fig.text(.029, band, ("A   Gain" if pos else "B   Loss") + f"     ΔIoU {delta:+.2f} points",
                     va="top", fontsize=10, weight="bold", color=teal if pos else rust)
            block = case["block"]
            for j, center in enumerate(centers):
                ax = fig.add_axes([center-width/2, bottom, width, height])
                ax.set_title(titles[j], color=ink, fontsize=8.6, pad=3.0)
                for spine in ax.spines.values():
                    spine.set(color="#a8b0b5", linewidth=.5)
                ax.set(xticks=[], yticks=[], aspect="equal")
                if public:
                    if j < 3:
                        support = case["truth"] if j == 0 else case["masks"][j]
                        vals = np.rint(4*pool(support, block)/(block*block))
                        ax.pcolormesh(np.arange(9), np.arange(9), vals, cmap=shades, norm=levels)
                    else:
                        ax.pcolormesh(np.arange(9), np.arange(9), residuals[i], cmap=signed, norm=norm)
                    ax.set(xlim=(0,8), ylim=(8,0))
                    if j == 0:
                        sub = f"8 × 8 bins\n{block} × {block} pixels per bin"
                    elif j == 3:
                        sub = "WT − WC\nFP + FN area per bin"
                else:
                    side = case["side"]
                    ax.imshow(case["grey"], cmap="gray", vmin=0, vmax=255, extent=(0,side,side,0),
                              interpolation="nearest")
                    edge = LineCollection(boundaries(case["masks"][j]), colors="#6eeaff",
                                          linewidths=.9, zorder=4)
                    edge.set_path_effects([pe.Stroke(linewidth=1.75,foreground="#0c1a23"), pe.Normal()])
                    ax.add_collection(edge)
                    for ring in case["rings"]:
                        line, = ax.plot(ring[:,0], ring[:,1], color="#ffcc49", linewidth=1.13,
                                        linestyle=(0, (3,1.65)), zorder=5)
                        line.set_path_effects([pe.Stroke(linewidth=1.78,foreground="#24201b"), pe.Normal()])
                    ax.set(xlim=(0,side), ylim=(side,0))
                    if j == 0:
                        sub = f"Native IoU {case['metrics'][0,0]:.2f}\nOriginal grayscale"
                if j in (1, 2):
                    iou, fp, fn = case["metrics"][j]
                    sub = f"{case['flips'][j]} flips  ·  IoU {iou:.2f}\nFP {fp:.2f}%  ·  FN {fn:.2f}%"
                ax.set_xlabel(sub, fontsize=8, color=ink, labelpad=3, linespacing=1.35)
        if public:
            fig.text(.028,.080,"Footprint fraction",va="center",color=ink,fontsize=8.2)
            cb = fig.colorbar(plt.cm.ScalarMappable(norm=levels,cmap=shades),
                              cax=fig.add_axes([.179,.072,.195,.017]),orientation="horizontal",
                              ticks=[0,2,4])
            cb.set_ticklabels(["0","½","1"])
            fig.text(.431,.080,"Area error change",va="center",fontsize=8.2,color=ink)
            cb2 = fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=signed),
                               cax=fig.add_axes([.594,.072,.281,.017]),orientation="horizontal",
                               ticks=[-limit,0,limit])
            cb2.set_ticklabels([f"−{limit:.1f}","0",f"+{limit:.1f} px²"])
            for bar in (cb,cb2):
                bar.solids.set_rasterized(False)
                bar.outline.set_linewidth(.45)
                bar.ax.tick_params(labelsize=8,length=2,pad=1)
            fig.text(.028,.008,"Teal marks less error under the fixed count. FP and FN use author hull area as denominator.",
                     color=ink,fontsize=8.2,va="bottom")
        else:
            a = Line2D([],[],color="#007b98",linewidth=1.7,label="Native / edited pixel boundary")
            b = Line2D([],[],color="#ac7300",linewidth=1.7,linestyle="--",label="Author polygon")
            fig.legend(handles=[a,b], loc="lower center", bbox_to_anchor=(.50,.039),
                       ncol=2, frameon=False, fontsize=8.8)
            fig.text(.5,.018,"Flips count native pixels edited. FP and FN are percentages of author hull area.",
                     ha="center",fontsize=8.2,color=ink)
        dest = args.public_dir if public else args.private_dir
        dest.mkdir(parents=True, exist_ok=True)
        stem = "pseg_budget_cases_binned" if public else "pseg_budget_cases_native"
        for ext in (("pdf", "svg") if public else ("pdf", "svg", "png")):
            kw = {"metadata":{"Date":None}} if ext == "svg" else {}
            if ext == "pdf":
                kw = {"metadata":{"CreationDate":None,"ModDate":None}}
            fig.savefig(dest / (stem+"."+ext), dpi=300, **kw)
        plt.close(fig)
    print(json.dumps({"rendered":[dict(native_iou=c["metrics"][0,0], wt_flips=int(c["flips"][1]),
                        wc_flips=int(c["flips"][2]), pixels_per_bin=c["block"],
                        hull_area=float(c["truth"].sum()),
                        intersection_change_px2=float(np.sum((c["masks"][2].astype(float)-
                                                             c["masks"][1])*c["truth"])),
                        intersection_change_pct_gt=float(c["metrics"][1,2]-c["metrics"][2,2]))
                        for c in data]}))


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("code", "run", "request", "owner", "capture", "author-labels", "public-dir", "private-dir"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--selection-only", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    a = arguments()
    selected, tally = cases(a)
    if not a.selection_only:
        render(a, selected, tally)
