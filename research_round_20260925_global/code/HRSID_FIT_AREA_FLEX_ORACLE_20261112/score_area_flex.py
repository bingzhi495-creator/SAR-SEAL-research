"""Audit area flexibility on the unchanged closed FIT36 boundary support."""

from argparse import ArgumentParser
from collections import defaultdict
from contextlib import redirect_stdout
from hashlib import sha256
from io import StringIO
from pathlib import Path
from statistics import mean
import json
import os
import sys
from time import perf_counter
from types import ModuleType

import numpy as np
from pycocotools import mask as coco
from scipy.ndimage import label

PROTOCOL = "b4259b1621fc66d36a04fc2dbb6e3fa04e5e7488aed294684ec37bd2c7ff2e0d"
PRIOR_SOURCE = "d4c5c81d3bd73197fdc17cc0db26e38f1b3b663ec507518fab1cccbb9d27e044"
PRIOR_REQUEST = "ccac34ef91dc1aa49873d957c3355da5f691c87b522b974e62f3ad72a226f5d0"
PRIOR_RESULT = "768e833dba8f37c9db86e8b8d9cafbda189a00a8c758784b0ebdfe28f6677075"
MODES = ("zero", "balanced", "flex_equal", "flex_cap")


def read(path, digest, limit=32 * 2**20):
    if path.resolve(strict=True) != path or not 0 < path.stat().st_size <= limit:
        raise ValueError("Bounded canonical path required: " + path.name)
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != digest:
        raise ValueError("Unexpected source digest: " + path.name)
    return raw


def module(path, digest, name):
    imported = ModuleType(name)
    imported.__file__ = str(path)
    sys.modules[name] = imported
    exec(compile(read(path, digest), str(path), "exec"), imported.__dict__)
    return imported


def fixed_examples(args, prior, readout, phase_math, geometry):
    matching = prior.load(args.match_source, prior.MATCH_SHA, "_flex_phase_match")
    req = json.loads(read(args.request, PRIOR_REQUEST))
    prior.require(req["FIT"] is True and req["DEV"] is req["TEST"] is False
                  and req["original_capture_accepted"] and req["forward_calls"] == 180
                  and req["original_support_sha"] == prior.SUPPORT_SHA, "Closed FIT36 only")
    views = req["images"]
    blocks = [json.loads(line) for line in read(Path(req["prediction_path"]), prior.PREDICTIONS_SHA).splitlines()]
    ids = [entry["image_id"] for entry in views]
    prior.require(len(blocks) == len(set(ids)) == len({v["group"] for v in views}) == 36
                  and [v["ordinal"] for v in views] == list(range(1, 37)), "Exact distinct source roster")
    routed = prior.reconstruct_support(views, blocks, matching)
    prior.require(sum(x[2] for x in routed) == 267782 and sum(x[3] for x in routed) == 1, "Exact captured point census")
    labels = json.loads(read(args.fit_index, prior.LABEL_SHA))
    prior.require(len(labels["images"]) == 2937 and len(labels["annotations"]) == 9260, "Exact original FIT labels")
    images = {im["id"]: im for im in labels["images"] if im["id"] in ids}
    annotations = defaultdict(list)
    for annotation in labels["annotations"]:
        if annotation["image_id"] in images:
            annotations[annotation["image_id"]].append(annotation)
    subset = dict(images=list(images.values()), annotations=[a for aa in annotations.values() for a in aa],
                  categories=labels["categories"])
    targets, neutral, _ = geometry.geometry.polygon_support(subset, geometry.vector)

    examples = []
    matches_total = 0
    matched_discordance = 0
    with redirect_stdout(StringIO()):
        for entry, (pool, points, _, _) in zip(views, routed, strict=True):
            image_id = entry["image_id"]
            matches, _, _ = matching.freeze_coco(images[image_id], annotations[image_id], pool)
            matches_total += len(matches)
            rank_to_gt = {rank: key for key, rank in matches.items()}
            for index, candidate in enumerate(pool):
                rank = candidate["rank"]
                truth_id = rank_to_gt.get(rank)
                if rank not in points or truth_id is None or truth_id in neutral:
                    continue
                target, _ = targets[truth_id]
                rle = candidate["rle"]
                prediction = coco.decode(rle).astype(bool)
                boxes = geometry.geometry.boxes_from_counts(geometry.shift.runs(rle["counts"]), *rle["size"])
                intersection = geometry.geometry.overlap_area(boxes, target)
                xy, inside, logits = points[rank]
                matched_discordance += int(np.count_nonzero((logits[:, 0] > 0) != inside))
                neighbors = [p["rle"] for p in pool[:index]]
                examples.append(readout.example(entry["ordinal"], prediction, xy, inside, logits,
                                                target, intersection, neighbors, phase_math))
    prior.require(matches_total == 190 and len(examples) == 166 and matched_discordance == 0,
                  "Exact previous matched scoring cohort")
    return examples


def measure(example, choice, area, initial_components, connectivity):
    mask = example.shape
    flips = 0 if choice is None else choice.flips
    if flips:
        mask = mask.copy()
        xy = example.points_xy
        add = xy[list(choice.adds)]
        delete = xy[list(choice.deletes)]
        mask[add[:, 1], add[:, 0]] = True
        mask[delete[:, 1], delete[:, 0]] = False
        if np.count_nonzero(mask ^ example.shape) != flips or int(mask.sum()) != choice.pred_area:
            raise ValueError("Witness differs from optimal edit counts")
    inter = example.intersection if choice is None else choice.intersection
    final_area = area if choice is None else choice.pred_area
    truth = example.gt_area
    inter = max(0., min(inter, truth, final_area))
    iou = 100 * inter / (final_area + truth - inter)
    components_changed = bool(flips and label(mask, connectivity)[1] != initial_components)
    crossing = False
    if flips and example.neighbor_masks:
        rle = coco.encode(np.asfortranarray(mask, dtype=np.uint8))
        crossing = bool(np.any(coco.iou([rle], example.neighbor_masks, [0] * len(example.neighbor_masks)) >= .20))
    net = final_area - area
    return dict(iou=iou, fp=100 * (final_area - inter) / truth, fn=100 * (truth - inter) / truth,
                area_error=100 * abs(final_area - truth) / truth,
                adds=0 if choice is None else len(choice.adds),
                deletes=0 if choice is None else len(choice.deletes), flips=flips,
                component_change=components_changed, new_neighbor_crossing=crossing,
                area_change=bool(net), area_change_toward_truth=bool(net and net * (truth - area) > 0))


def evaluate(examples, numeric):
    connectivity = np.ones((3, 3), dtype=bool)
    rows = []
    for example in examples:
        area = int(example.shape.sum())
        choices = numeric.solve_edit_oracles(area, example.gt_area, example.intersection,
                                             example.target_occupancy, example.inside, example.budget)
        balanced, flex_cap, flex_equal = choices
        if (flex_equal.flips != balanced.flips or flex_cap.flips > 2 * example.budget
                or len(balanced.adds) != len(balanced.deletes)
                or balanced.iou > flex_equal.iou + 1e-12 or flex_equal.iou > flex_cap.iou + 1e-12):
            raise ValueError("Nested oracle envelope or realized count mismatch")
        components = label(example.shape, connectivity)[1]
        metrics = {mode: measure(example, choice, area, components, connectivity)
                   for mode, choice in zip(MODES, (None, balanced, flex_equal, flex_cap), strict=True)}
        rows.append((example.ordinal, example.budget, metrics))
    return rows


def summarize(rows):
    out = {}
    for mode in MODES:
        values = [metrics[mode] for _, _, metrics in rows]
        out[mode] = {key: mean(v[key] for v in values) for key in ("iou", "fp", "fn", "area_error")}
        for key in ("adds", "deletes", "flips", "component_change", "new_neighbor_crossing",
                    "area_change", "area_change_toward_truth"):
            out[mode][key] = sum(v[key] for v in values)
        out[mode]["strict"] = sum(v["iou"] >= 90. for v in values)
        out[mode]["strict_gain_from_balanced"] = sum(m["balanced"]["iou"] < 90. <= m[mode]["iou"] for _, _, m in rows)
        out[mode]["strict_loss_from_balanced"] = sum(m[mode]["iou"] < 90. <= m["balanced"]["iou"] for _, _, m in rows)
    return out


def main(args):
    started = perf_counter()
    source = Path(__file__).resolve().parent
    read(source / "PROTOCOL.md", PROTOCOL)
    previous = args.code_root / "HRSID_FIT_FIXED_QUERY_SHIFT_PHASE_SCOUT_20261112"
    prior = module(previous / "score_shift_phase.py", PRIOR_SOURCE, "_area_prior_score")
    numeric = module(source / "area_flex_math.py", args.math_sha, "_area_numeric")
    readout = prior.load(previous / "phase_readout.py", prior.SOURCE_DIGESTS["phase_readout.py"], "_area_prior_readout")
    phase_math = prior.load(previous / "phase_probe_math.py", prior.SOURCE_DIGESTS["phase_probe_math.py"], "_area_prior_math")
    admission = prior.load(args.code_root / "MASK_RTDETR_DEV705_CONTINUOUS_AP_20261112/continuous_endpoint.py",
                           prior.GEOMETRY_SHA, "_area_geometry")
    sys.path.append(str(args.shapely_site.resolve(strict=True)))
    import shapely
    prior.require(shapely.__version__ == "2.0.2" and shapely.geos_version_string == "3.11.2", "Exact CPU geometry")
    geometry = admission.geometry_modules(args.code_root)
    previous_result = json.loads(read(args.prior_result, PRIOR_RESULT))
    rows = evaluate(fixed_examples(args, prior, readout, phase_math, geometry), numeric)
    metrics = summarize(rows)
    for field in ("iou", "fp", "fn", "strict"):
        if abs(metrics["zero"][field] - previous_result["metrics"]["zero"][field]) > 1e-10:
            raise ValueError("Original baseline failed independent arithmetic check")
    partitions = []
    for fold in range(3):
        part = [row for row in rows if row[0] % 3 == fold]
        partitions.append(dict(rows=len(part), metrics=summarize(part)))
    gain = metrics["flex_equal"]["iou"] - metrics["balanced"]["iou"]
    positive = sum(p["metrics"]["flex_equal"]["iou"] > p["metrics"]["balanced"]["iou"] + 1e-12
                   for p in partitions)
    gate = gain >= .30 and positive >= 2 and metrics["flex_equal"]["strict"] >= metrics["balanced"]["strict"]
    result = dict(status="HRSID_FIT_AREA_FLEX_ORACLE_ACTUAL", FIT=True, DEV=False, TEST=False, AP=False,
                  forwards=0, optimizer_updates=0, truth_selected=True, matched_rows=len(rows), original_matches=190,
                  image_groups=36, zero_budgets=sum(budget == 0 for _, budget, _ in rows),
                  total_flip_cap=sum(2 * budget for _, budget, _ in rows),
                  gate="AREA_COMPOSITION_LEAD" if gate else "NO_AREA_COMPOSITION_LEAD",
                  exact_realized_flex_minus_balanced_iou=gain, positive_partitions=positive,
                  metrics=metrics, partitions=partitions, protocol_sha256=PROTOCOL,
                  math_sha256=args.math_sha, prior_result_sha256=PRIOR_RESULT,
                  worker_seconds=perf_counter() - started)
    encoded = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False, default=prior.json_scalar) + "\n").encode()
    with args.output.open("xb") as file:
        file.write(encoded)
    print(json.dumps(dict(status=result["status"], gate=result["gate"], result_sha256=sha256(encoded).hexdigest())))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("request", "fit-index", "match-source", "code-root", "shapely-site", "prior-result", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--math-sha", required=True)
    if not sys.flags.isolated or os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("Use an isolated CPU process")
    def offline(event, _args):
        if event.startswith("socket."):
            raise PermissionError("Closed input only")
    sys.addaudithook(offline)
    main(parser.parse_args())
