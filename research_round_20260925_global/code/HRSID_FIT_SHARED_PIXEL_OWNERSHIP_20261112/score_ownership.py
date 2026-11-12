"""Test duplicate-pixel ownership on fixed accepted HRSID FIT predictions."""
import argparse
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
from statistics import mean
import sys
from time import perf_counter
from types import ModuleType

import numpy as np

PROTOCOL = "57133e919347bec5f10f72b66f6b986590f867b711e6ec97b9cd053699217003"
CASE = "8cf7e7b138ee86105aee17cfe6239a355597cd15179708414df5dd288bda7b3c"
MATCH = "12fd1684b262a3c9b6732b90a0c1e624034c523a7d130919a079e17728f80075"
MODES = ("raw", "oracle", "depth", "centroid")


def checked_module(path, digest, name):
    data = path.read_bytes()
    if sha256(data).hexdigest() != digest:
        raise ValueError("Pinned source mismatch: " + path.name)
    module = ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(data, str(path), "exec"), module.__dict__)
    return module


def isolated_pairs(size, edges):
    adjacency = [set() for _ in range(size)]
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    pairs = []
    for a, neighbors in enumerate(adjacency):
        if len(neighbors) == 1:
            b = next(iter(neighbors))
            if b > a and len(adjacency[b]) == 1:
                pairs.append((a, b))
    return pairs


def end_removals(ids, support, count_a, count_b):
    if count_a < 0 or count_b < 0 or count_a + count_b > len(ids):
        raise ValueError("Deletion budgets must fit the shared support")
    ordered = ids[np.lexsort((ids, support))]
    return ordered[:count_a], ordered[len(ids) - count_b:]


def measures(pred_area, truth_area, inter):
    upper = min(pred_area, truth_area)
    if not -1e-8 <= inter <= upper + 1e-8:
        raise ValueError("Exact continuous intersection outside bounds")
    inter = min(upper, max(0.0, inter))
    return dict(iou=100 * inter / (pred_area + truth_area - inter),
                fp=100 * (pred_area - inter) / truth_area,
                fn=100 * (truth_area - inter) / truth_area)


def rectangles(rle, geom):
    return geom.geometry.boxes_from_counts(geom.shift.runs(rle["counts"]), *rle["size"])


def edits_for_pair(a, b, width, geom, codec):
    from scipy.ndimage import distance_transform_edt

    masks = [codec.decode(r["pred"]["rle"]).astype(bool) for r in (a, b)]
    overlap = masks[0] & masks[1]
    shared = np.flatnonzero(overlap)
    truth = [codec.decode(r["raster"]).ravel()[shared].astype(bool) for r in (a, b)]
    if np.any(truth[0] & truth[1]):
        raise ValueError("Raster-disjoint GT pair violated")
    per_mask_budget = [min(16, int(r["pred"]["area"] // 10)) for r in (a, b)]
    oracle = (shared[truth[1]][:per_mask_budget[0]], shared[truth[0]][:per_mask_budget[1]])
    counts = [len(part) for part in oracle]
    outputs = {"raw": (shared[:0], shared[:0]), "oracle": oracle}
    if sum(counts):
        ys, xs = np.divmod(shared, width)
        y_slice, x_slice = slice(ys.min(), ys.max()+1), slice(xs.min(), xs.max()+1)
        # Retain every foreground pixel for the exact Euclidean complement distance.
        depth = [distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1][y_slice, x_slice]
                 [ys - ys.min(), xs - xs.min()] for mask in masks]
        centroid = []
        for mask in masks:
            yy, xx = np.nonzero(mask)
            centroid.append((ys - yy.mean()) ** 2 + (xs - xx.mean()) ** 2)
        outputs["depth"] = end_removals(shared, depth[0] - depth[1], *counts)
        outputs["centroid"] = end_removals(shared, centroid[1] - centroid[0], *counts)
    else:
        outputs.update(depth=outputs["raw"], centroid=outputs["raw"])

    results = {}
    for mode, parts in outputs.items():
        all_edits = np.concatenate(parts)
        if len(np.unique(all_edits)) != len(all_edits) or not np.isin(all_edits, shared).all():
            raise ValueError("Editing must conserve pair union")
        if mode != "raw" and list(map(len, parts)) != counts:
            raise ValueError("Controls require identical per-mask edit counts")
        results[mode] = []
        for record, removed in zip((a, b), parts, strict=True):
            ys, xs = np.divmod(removed, width)
            pixels = np.column_stack((xs, ys, xs + 1, ys + 1))
            lost = geom.geometry.overlap_area(pixels, record["truth"]) if len(pixels) else 0.0
            results[mode].append(measures(record["pred"]["area"] - len(removed),
                                          record["truth"].area, record["inter"] - lost))
    return dict(modes=results, changed_masks=sum(k > 0 for k in counts), deleted_pixels=sum(counts),
                shared=len(shared), raster_a_only=int(truth[0].sum()),
                raster_b_only=int(truth[1].sum()), raster_neither=int((~(truth[0] | truth[1])).sum()),
                gt_distance_px=float(a["truth"].distance(b["truth"])))


def score_image(image, annotations, pool, targets, neutral, geom, codec, stock, census):
    pred = {p["rank"]: p for p in pool}
    with redirect_stdout(StringIO()):
        fixed, gt_rle, _ = stock.freeze_coco(image, annotations, pool)
    census["stock_matches"] += len(fixed)
    anchors = []
    for identity, rank in sorted(fixed.items(), key=lambda item: item[1]):
        if identity in neutral or identity not in targets:
            continue
        raster, candidate, truth = gt_rle[identity], pred[rank], targets[identity][0]
        inter = geom.geometry.overlap_area(rectangles(candidate["rle"], geom), truth)
        iou = measures(candidate["area"], truth.area, inter)["iou"]
        anchors.append(dict(pred=candidate, raster=raster, truth=truth, inter=inter))
        census["valid_matches"] += 1
        census["near_strict_matches"] += 75 <= iou < 90
        census["raw_strict_matches"] += iou >= 90

    shared_counts = {}
    for i, a in enumerate(anchors):
        for j in range(i + 1, len(anchors)):
            pair = (a["pred"]["rle"], anchors[j]["pred"]["rle"])
            shared = int(codec.area(codec.merge(pair, intersect=True)))
            if shared:
                shared_counts[i, j] = shared
    census["overlap_edges"] += len(shared_counts)
    isolated = isolated_pairs(len(anchors), shared_counts)
    census["isolated_pair_components"] += len(isolated)
    results = []
    for i, j in isolated:
        if shared_counts[i, j] < 4:
            census["fewer_than_four_shared_pixels"] += 1
            continue
        a, b = anchors[i], anchors[j]
        if codec.area(codec.merge((a["raster"], b["raster"]), intersect=True)) \
                or a["truth"].intersection(b["truth"]).area > 1e-9:
            census["non_disjoint_truth"] += 1
            continue
        result = edits_for_pair(a, b, image["width"], geom, codec)
        if result["shared"] != shared_counts[i, j]:
            raise ValueError("Independent RLE and raster shared counts disagree")
        results.append(result)
    return results


def summarize(rows, census):
    by_group = defaultdict(list)
    for group, row in rows:
        by_group[group].append(row)
    groups = sorted(by_group)
    modes = {}
    for name in MODES:
        flattened = [(base, changed) for _, row in rows
                     for base, changed in zip(row["modes"]["raw"], row["modes"][name], strict=True)]
        strict = sum(m["iou"] >= 90 for _, m in flattened)
        gain = sum(base["iou"] < 90 <= m["iou"] for base, m in flattened)
        loss = sum(m["iou"] < 90 <= base["iou"] for base, m in flattened)
        near = sum(75 <= base["iou"] < 90 <= m["iou"] for base, m in flattened)
        modes[name] = dict(
            **{key + "_points": mean(mean(mean(v[key] for v in row["modes"][name])
                                         for row in by_group[g]) for g in groups) if groups else None
               for key in ("iou", "fp", "fn")},
            strict_hits=strict, strict_gains=gain, strict_losses=loss, near_strict_recovered=near,
            worse_instances=sum(m["iou"] + 1e-9 < base["iou"] for base, m in flattened))
    comparisons = {}
    if groups:
        draws = np.random.Generator(np.random.PCG64(5221)).integers(len(groups), size=(4096, len(groups)))
        for control in ("raw", "depth", "centroid"):
            changes = np.asarray([mean(mean(v["iou"] - c["iou"] for v, c in
                                             zip(row["modes"]["oracle"], row["modes"][control], strict=True))
                                       for row in by_group[g]) for g in groups])
            comparisons[control] = dict(mean_points=float(changes.mean()),
                                        ci95_points=np.percentile(changes[draws].mean(axis=1), [2.5, 97.5]).tolist())
    edited = [(g, r) for g, r in rows if r["deleted_pixels"]]
    edited_groups = len({g for g, _ in edited})
    gates = dict(coverage=len(edited) >= 20 and edited_groups >= 10,
                 mean_vs_raw=bool(comparisons and comparisons["raw"]["mean_points"] >= .50),
                 excess_over_controls=bool(comparisons and all(comparisons[c]["mean_points"] >= .25
                     and comparisons[c]["ci95_points"][0] > 0 for c in ("depth", "centroid"))),
                 net_strict=modes["oracle"]["strict_hits"] - modes["raw"]["strict_hits"] >= 3)
    return dict(pairs=len(rows), contributing_groups=len(groups), edited_pairs=len(edited),
                edited_groups=edited_groups, modes=modes, oracle_contrasts=comparisons, gates=gates,
                advance=all(gates.values()), pair_union_violations=0,
                total_edits=sum(r["deleted_pixels"] for _, r in rows),
                shared_pixel_partition={key: sum(r[key] for _, r in rows) for key in
                                        ("shared", "raster_a_only", "raster_b_only", "raster_neither")},
                median_gt_distance_px=float(np.median([r["gt_distance_px"] for _, r in rows])) if rows else None,
                recoveries_among_all_near_strict=modes["oracle"]["near_strict_recovered"],
                census=dict(census))


def run(args):
    started = perf_counter()
    protocol = args.protocol.resolve(strict=True)
    if sha256(protocol.read_bytes()).hexdigest() != PROTOCOL:
        raise ValueError("Pinned protocol")
    folder = args.code / "HRSID_FIT_HULL_CDF_20261112"
    case = checked_module(folder / "score_hull_cdf.py", CASE, "_ownership_case")
    stock = checked_module(folder / "stock_coco_match.py", MATCH, "_ownership_stock")
    roster = case.select_blocks(args.capture, args.witness)
    sys.path.append(str(args.shapely_site.resolve(strict=True)))
    import shapely
    from pycocotools import mask as codec
    case.need(shapely.__version__ == "2.0.2" and shapely.geos_version_string == "3.11.2", "Exact geometry")
    admission = case.load_source(args.code / "MASK_RTDETR_DEV705_CONTINUOUS_AP_20261112/continuous_endpoint.py",
                                 case.GEOM, "_ownership_geometry")
    geom = admission.geometry_modules(args.code)
    roster_sha = sha256(json.dumps([b["basename"] for b in roster], ensure_ascii=False).encode()).hexdigest()
    case.need(roster_sha == args.roster_sha, "Must reproduce the prelocked CDF roster")
    receipt = dict(status="HRSID_FIT_SHARED_PIXEL_OWNERSHIP_PRELABEL", FIT=True, DEV=False, TEST=False,
                   GT_opened=False, roster_sha256=roster_sha, protocol_sha256=PROTOCOL, selected_images=40)
    args.output.mkdir()
    case.write_json(args.output / "PRELABEL_SAFE.json", receipt)
    index = json.loads(case.exact(args.fit_index, case.FIT))
    case.need(len(index["images"]) == 2937 and len(index["annotations"]) == 9260, "Exact FIT population")
    names = {r["basename"] for r in roster}
    images = {im["file_name"]: im for im in index["images"] if im["file_name"] in names}
    ids = {im["id"] for im in images.values()}
    case.need(set(images) == names and len(ids) == 40, "Exact FIT join")
    anns = defaultdict(list)
    for ann in index["annotations"]:
        if ann["image_id"] in ids:
            anns[ann["image_id"]].append(ann)
    subset = dict(images=list(images.values()), categories=index["categories"],
                  annotations=[ann for group in anns.values() for ann in group])
    targets, neutral, _ = geom.geometry.polygon_support(subset, geom.vector)
    rows = []
    census = Counter(dict(non_disjoint_truth=0, fewer_than_four_shared_pixels=0))
    for group, block in enumerate(roster):
        image = images[block["basename"]]
        rows.extend((group, row) for row in score_image(image, anns[image["id"]], block["candidates"],
                                                        targets, neutral, geom, codec, stock, census))
    result = dict(receipt, status="HRSID_FIT_SHARED_PIXEL_OWNERSHIP_ACTUAL", GT_opened=True,
                  capture_sha256=case.CAPTURE, accepted_carrier_sha256=case.WITNESS, fit_sha256=case.FIT,
                  source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                  AP=False, model_forwards=0, optimizer_updates=0, worker_seconds=perf_counter() - started,
                  **summarize(rows, census))
    case.write_json(args.output / "RESULT_SAFE.json", result)
    print(json.dumps({k: result[k] for k in ("status", "pairs", "edited_pairs", "edited_groups", "advance")},
                     sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("protocol", "code", "capture", "witness", "fit-index", "shapely-site", "output"):
        parser.add_argument("--" + key, required=True, type=Path)
    parser.add_argument("--roster-sha", required=True)
    def offline(event, _args):
        if event.startswith(("socket.", "subprocess.")):
            raise RuntimeError("Offline CPU diagnostic")
    sys.addaudithook(offline)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
