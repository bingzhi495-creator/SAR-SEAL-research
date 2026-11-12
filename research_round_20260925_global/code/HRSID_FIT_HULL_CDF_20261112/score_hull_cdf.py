"""Read back the pinned HRSID FIT projected foreground-mass diagnostic."""
import argparse
from collections import defaultdict
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

PROTOCOL = "eec8994c30c453ce255d126a8f42d1fe9ea6d50c7eea4fd92a35d7f1bac60287"
CAPTURE = "aa3ef487ec1788591608bb0bd387eab9ec37904795af7a8e3084d3c22976cc62"
WITNESS = "f830fcd2806a484edd246b55551144733a553586467c7409a9a84caf32b82b0e"
PRED = "bcce34282370920dc338fc211116abf9e6a48c31abb05c73af6ffc63cd69c1e0"
FIT = "e95cf26cc60a2c6fb651e907541380a256f2df7b1e153244a6c0f803b0b19f8f"
MATCH = "8741bebb448120e37e0c10b58467a83d724f2294d20fddbc93720301285084aa"
GEOM = "7cad8b51a9230101e2233db8455f84c28163abfb2ee8ddd8cbe8718a2e719bd0"
HULL = "dedd47d5453b2442ff02585c9b4d216917e5ed175e01eae07a354a22710f3a94"
HELPER = "e46adbc736af12db37ef609ab7a5ec6abec665b6e67d62120c8593e760c3c5b7"
SALT = "HRSID-FIT-CDF-v1|"
SHORT_CONTROLS = ("centroid", "median", "q3_mean", "q3_median")
CONTROLS = SHORT_CONTROLS + ("long_w1",)
MASK_METRICS = ("iou_points", "fp_per_truth_points", "fn_per_truth_points")


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def exact(path, digest, limit=64 * 2**20):
    need(path.resolve(strict=True) == path and 0 < path.stat().st_size <= limit,
         "Bounded exact path required")
    data = path.read_bytes()
    need(sha256(data).hexdigest() == digest, "Pinned file mismatch: " + path.name)
    return data


def load_source(path, digest, name):
    module = ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(exact(path, digest, 2**20), str(path), "exec"), module.__dict__)
    return module


def write_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as file:
        json.dump(value, file, sort_keys=True, allow_nan=False)
        file.write("\n")


def salted(key):
    return sha256((SALT + key).encode()).hexdigest()


def select_blocks(capture, witness):
    report = json.loads(exact(capture / "RESULT_SAFE.json", CAPTURE))
    check = json.loads(exact(witness, WITNESS))
    need(report["status"] == "HRSID_FIT_SEAM_CONTEXT_CAPTURE_ACTUAL"
         and report["FIT"] and not any(report[k] for k in ("DEV", "TEST", "GT_opened", "AP"))
         and report["images"] == report["model_forwards"] == 144 and report["source_groups"] == 40
         and report["optimizer_updates"] == 0 and report["prediction_sha256"] == PRED,
         "Expected closed FIT prediction carrier")
    need(check["status"] == "HRSID_FIT_SEAM_CONTEXT_CAPTURE_INDEPENDENT_ACCEPTED"
         and check["result_sha256"] == CAPTURE and check["prediction_sha256"] == PRED
         and check["predictions_verified"] is True and check["FIT"]
         and not any(check[k] for k in ("DEV", "TEST", "GT_opened")),
         "Expected independently accepted prediction carrier")
    blocks = [json.loads(line) for line in exact(capture / "CANDIDATES_PRIVATE.jsonl", PRED).splitlines()]
    need(len({b["basename"] for b in blocks}) == 144
         and [b["ordinal"] for b in blocks] == list(range(1, 145)), "Expected 144 distinct views")
    by_group = defaultdict(list)
    for block in blocks:
        by_group[block["group"]].append(block)
    need(len(by_group) == 40, "Expected 40 original source groups")
    return [min(by_group[g], key=lambda b: (salted(b["basename"]), b["basename"]))
            for g in sorted(by_group, key=lambda g: (salted(g), g))]


def paired_effect(rows, test, control, draws):
    by_group = defaultdict(list)
    for group, metrics in rows:
        by_group[group].append(metrics[test] - metrics[control])
    groups = sorted(by_group)
    if not groups:
        return dict(group_mean_points=None, interval95_points=None,
                    positive_partitions=0, partition_points=[None] * 3, contributing_groups=0)
    effects = np.asarray([mean(by_group[g]) for g in groups])
    partitions = [[effects[i] for i, g in enumerate(groups) if g % 3 == fold] for fold in range(3)]
    partition_means = [float(mean(part)) if part else None for part in partitions]
    return dict(group_mean_points=float(effects.mean()),
                interval95_points=np.percentile(effects[draws].mean(axis=1), [2.5, 97.5]).tolist(),
                positive_partitions=sum(x is not None and x > 0 for x in partition_means),
                partition_points=partition_means, contributing_groups=len(groups))


def inside(bounds, axes, menu, width, height):
    x0, y0, x1, y1 = bounds
    for axis in axes.values():
        for shift in menu:
            dx, dy = axis * shift
            if x0 + dx < -1e-9 or y0 + dy < -1e-9 or x1 + dx > width + 1e-9 or y1 + dy > height + 1e-9:
                return False
    return True


def score_pair(boxes, prediction, truth, axes, geom, helper):
    def measures(displacement):
        intersection = geom.geometry.overlap_area(boxes + np.tile(displacement, 2), truth)
        upper = min(prediction.area, truth.area)
        need(-1e-8 <= intersection <= upper + 1e-8, "Continuous intersection bounds")
        overlap = max(0.0, min(upper, intersection))
        iou = overlap / (prediction.area + truth.area - overlap)
        return (100 * iou, 100 * (prediction.area - overlap) / truth.area,
                100 * (truth.area - overlap) / truth.area)

    actions = {key: {shift: measures(shift * vector) for shift in helper.MENU}
               for key, vector in axes.items()}
    best_score = max(v[0] for v in actions["short"].values())
    best = min((s for s, v in actions["short"].items() if abs(v[0] - best_score) <= 1e-10),
               key=lambda s: (abs(s), s))
    shifts = {"raw": ("short", 0.0), "short_true_best": ("short", best)}
    diagnostics = {}
    for step, suffix in ((.125, ""), (.0625, "_fine")):
        short = helper.projected_choices(truth, prediction, axes["short"], step)
        long = helper.projected_choices(truth, prediction, axes["long"], step)
        shifts["w1" + suffix] = ("short", short["w1_shift"])
        shifts["long_w1" + suffix] = ("long", long["w1_shift"])
        for name, key in zip(SHORT_CONTROLS, ("mean", "median", "q3_mean", "q3_median"), strict=True):
            shifts[name + suffix] = ("short", short[key + "_shift"])
        if not suffix:
            diagnostics = dict(short_mean_delta=short["mean_delta"], long_mean_delta=long["mean_delta"],
                               short_mean_minus_median=short["mean_delta"] - short["median_delta"])
    return {k: actions[axis][shift] for k, (axis, shift) in shifts.items()}, shifts, diagnostics


def summarize(rows):
    modes = ("raw", "short_true_best", "w1", *CONTROLS, "w1_fine",
             *(name + "_fine" for name in CONTROLS))
    active = sorted({r[0] for r in rows})
    draws = np.random.Generator(np.random.PCG64(5217)).integers(
        len(active), size=(4096, len(active))) if active else np.empty((0, 0), dtype=int)
    group_values = defaultdict(list)
    for group, metrics, _, _ in rows:
        group_values[group].append(metrics)
    summary = {}
    for mode in modes:
        summary[mode] = dict(
            **{key: mean(mean(r[mode][i] for r in rr) for rr in group_values.values()) if rows else None
               for i, key in enumerate(MASK_METRICS)},
            strict_hits=sum(row[1][mode][0] >= 90 for row in rows),
            nonzero_shifts=sum(row[2][mode][1] != 0 for row in rows))
    comparison_rows = [(g, {m: v[0] for m, v in metrics.items()}) for g, metrics, _, _ in rows]
    comparisons = {}
    checks = {}
    for suffix in ("", "_fine"):
        mode = "w1" + suffix
        controls = [k + suffix for k in CONTROLS]
        comparisons[mode] = {control: paired_effect(comparison_rows, mode, control, draws)
                             for control in [*controls, "raw"]}
        selected = [comparisons[mode][control] for control in controls]
        checks[mode] = dict(
            margin=all(e["group_mean_points"] is not None and e["group_mean_points"] >= .25 for e in selected),
            interval=all(e["interval95_points"] is not None and e["interval95_points"][0] > 0 for e in selected),
            partitions=all(e["positive_partitions"] >= 2 for e in selected),
            strict=all(summary[mode]["strict_hits"] >= summary[c]["strict_hits"] for c in [*controls, "raw"]))
    primary = len(rows) >= 30 and len(active) >= 15
    passed = primary and all(checks["w1"].values()) and checks["w1_fine"]["margin"] and checks["w1_fine"]["strict"]
    practical_raw_gain = all(comparisons[mode]['raw']['group_mean_points'] is not None
                             and comparisons[mode]['raw']['group_mean_points'] > 0
                             for mode in ('w1', 'w1_fine'))
    associations = dict(
        mean_absolute_delta={k: mean(abs(r[3][k]) for r in rows) if rows else None
                             for k in ("short_mean_delta", "long_mean_delta", "short_mean_minus_median")},
        w1_vs_centroid_different=sum(r[2]["w1"] != r[2]["centroid"] for r in rows),
        w1_grid_different=sum(r[2]["w1"] != r[2]["w1_fine"] for r in rows),
        long_w1_grid_different=sum(r[2]["long_w1"] != r[2]["long_w1_fine"] for r in rows))
    return dict(retained_pairs=len(rows), contributing_groups=len(active), sample_gate=primary,
                branch_passed=passed, positive_gain_vs_unchanged=practical_raw_gain,
                training_followup_eligible=passed and practical_raw_gain,
                per_mode=summary, comparisons=comparisons,
                gates=checks, associations=associations)


def run(args):
    start = perf_counter()
    exact(args.protocol, PROTOCOL, 65536)
    roster = select_blocks(args.capture, args.witness)
    sys.path.append(str(args.shapely_site.resolve(strict=True)))
    import shapely
    from pycocotools import mask as codec
    need(shapely.__version__ == "2.0.2" and shapely.geos_version_string == "3.11.2",
         "Expected pinned geometry version")
    helper = load_source(args.protocol.parent / "hull_cdf.py", HELPER, "_fit_cdf_helper")
    match = load_source(args.match_source, MATCH, "_fit_cdf_match")
    code = args.code
    admission = load_source(code / "MASK_RTDETR_DEV705_CONTINUOUS_AP_20261112/continuous_endpoint.py",
                            GEOM, "_fit_cdf_geometry")
    hull = load_source(code / "MASK_RTDETR_DEV705_FIXED_MATCH_GEOMETRY_20261112/fixed_match_geometry.py",
                       HULL, "_fit_cdf_hull")
    geom = admission.geometry_modules(code)
    roster_sha = sha256(json.dumps([b["basename"] for b in roster], ensure_ascii=False).encode()).hexdigest()
    receipt = dict(status="HRSID_FIT_HULL_CDF_PRELABEL", FIT=True, DEV=False, TEST=False,
                   GT_opened=False, protocol_sha256=PROTOCOL, capture_sha256=CAPTURE,
                   independent_sha256=WITNESS, roster_sha256=roster_sha, selected_images=40)
    args.output.mkdir()
    write_json(args.output / "PRELABEL_SAFE.json", receipt)

    index = json.loads(exact(args.fit_index, FIT))
    need(len(index["images"]) == 2937 and len(index["annotations"]) == 9260, "Expected FIT population")
    names = {b["basename"] for b in roster}
    images = {im["file_name"]: im for im in index["images"] if im["file_name"] in names}
    ids = {im["id"] for im in images.values()}
    need(set(images) == names and len(ids) == 40, "Expected exact FIT join")
    annotations = defaultdict(list)
    for a in index["annotations"]:
        if a["image_id"] in ids:
            annotations[a["image_id"]].append(a)
    subset = dict(images=list(images.values()), categories=index["categories"],
                  annotations=[a for group in annotations.values() for a in group])
    targets, neutral, _ = geom.geometry.polygon_support(subset, geom.vector)

    rows = []
    stock_pairs = eligible = boundary = 0
    for group, block in enumerate(roster):
        image = images[block["basename"]]
        truth_by_id = {a["id"]: a for a in annotations[image["id"]]}
        pool = block["candidates"]
        predicted_by_rank = {p["rank"]: p for p in pool}
        with redirect_stdout(StringIO()):
            matches, _, _ = match.freeze_coco(image, list(truth_by_id.values()), pool)
        stock_pairs += len(matches)
        for identity, rank in matches.items():
            candidate = predicted_by_rank[rank]
            if (identity in neutral or identity not in targets or not 32 <= candidate["area"] < 1024
                    or geom.geometry.stratum(truth_by_id[identity]["area"]) != "small"):
                continue
            truth = targets[identity][0]
            aspect, _, tangent, normal = hull.hull(truth)
            if aspect < 2:
                continue
            eligible += 1
            rle = candidate["rle"]
            boxes = geom.geometry.boxes_from_counts(geom.shift.runs(rle["counts"]), *rle["size"])
            prediction = shapely.union_all(shapely.box(*boxes.T))
            need(prediction.area == candidate["area"] == float(codec.area(rle))
                 and rle["size"] == [image["height"], image["width"]], "Expected native mask footprint")
            axes = dict(short=np.asarray(normal, dtype=float), long=np.asarray(tangent, dtype=float))
            if not inside(prediction.bounds, axes, helper.MENU, image["width"], image["height"]):
                boundary += 1
                continue
            rows.append((group, *score_pair(boxes, prediction, truth, axes, geom, helper)))
    need(eligible == len(rows) + boundary, "Expected exhaustive eligible support")
    result = dict(receipt, status="HRSID_FIT_HULL_CDF_ACTUAL", GT_opened=True,
                  FIT_sha256=FIT, source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                  model_forwards=0, optimizer_updates=0, AP=False, stock_pairs=stock_pairs,
                  eligible_pairs=eligible, boundary_excluded=boundary, worker_seconds=perf_counter() - start,
                  **summarize(rows))
    write_json(args.output / "RESULT_SAFE.json", result)
    print(json.dumps({k: result[k] for k in ("status", "stock_pairs", "eligible_pairs", "retained_pairs",
                                             "contributing_groups", "branch_passed", "worker_seconds")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for path in ("protocol", "capture", "witness", "fit-index", "match-source", "code", "shapely-site", "output"):
        parser.add_argument("--" + path, required=True, type=Path)
    def offline(event, _args):
        if event.startswith(("socket.", "subprocess.")):
            raise RuntimeError("Offline CPU diagnostic")
    sys.addaudithook(offline)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
