"""Count SegRefiner-LR area exposure on the frozen Instance-RSDD RF-DETR predictions."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter
from types import ModuleType

import numpy as np
from pycocotools import mask as codec
import shapely


STATUS = "RSDD74_SEGREFINER_STOCK_EXPOSURE_COMPLETE"
LOCK_NAME = "SEGREFINER_STOCK_PRELOCK_20261123.md"
LOCK_SHA256 = "30007c832d47c6e086445d474d4f319642854de20da7844706ebac8e40a7df9c"
PAIR_SOURCE_SHA256 = "3cbdbf62076e52ab65425228302241c975a35a48d99151b76a8157794f7d5c06"
REFERENCE_NAME = "RSDD74_PHOENIX_BOUNDED_MATCH_ACTUAL_SAFE_20261123.json"
REFERENCE_SHA256 = "0d7acb89575725479a15405c4bc49649a9cd996aa43fc99f92c618583de68d6c"
VIEWS = {"full": 74, "date_separated": 42}
MINIMUM_AREA = 512
AREA_CLASSES = ("zero", "1_to_31", "32_to_511", "512_to_1023", "1024_or_more")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return sha256(raw).hexdigest()


def area_class(n):
    require(n >= 0, "Native mask area cannot be negative")
    index = sum(n >= limit for limit in (1, 32, 512, 1024))
    return AREA_CLASSES[index]


def count_strata(entries, dates):
    bins = dict.fromkeys(AREA_CLASSES, 0)
    sensor_bins = defaultdict(Counter)
    all_parents, eligible_parents = set(), set()
    eligible = 0
    for area, parent in entries:
        label = area_class(area)
        bins[label] += 1
        sensor, _ = dates[parent]
        sensor_bins[sensor][label] += 1
        all_parents.add(parent)
        if area >= MINIMUM_AREA:
            eligible += 1
            eligible_parents.add(parent)
    require(eligible == bins["512_to_1023"] + bins["1024_or_more"], "Area bins disagree with LR exposure")
    return dict(
        total=sum(bins.values()), eligible=eligible, native_area_counts=bins,
        parent_groups=len(all_parents), sensor_date_blocks=len({dates[p] for p in all_parents}),
        eligible_parent_groups=len(eligible_parents), eligible_sensor_date_blocks=len({dates[p] for p in eligible_parents}),
        by_sensor={sensor: {key: counts[key] for key in AREA_CLASSES} for sensor, counts in sorted(sensor_bins.items())},
    )


def sources(a):
    path = a.code / "PHOENIX_PAIR_OPERATOR_20261123/score_rsdd.py"
    raw = path.read_bytes()
    require(digest(raw) == PAIR_SOURCE_SHA256, "Frozen Instance-RSDD source loader differs")
    pair = ModuleType("frozen_rsdd_pair_source")
    pair.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), pair.__dict__)
    whole, _, deps = pair.load_area(a.code).sources(a)
    whole.ARMS, whole.MATCHED, whole.PAIRS = ("raw",), ("raw",), ()
    return whole, deps


def native_blocks(rows, adapter, core):
    blocks, areas, point_positions = {}, {}, {}
    for item, audit, pred, point, _ in rows:
        image = item["image_id"]
        require(image not in blocks, "Duplicate native image")
        masks = adapter.native_masks(core, pred, audit, image)
        ranks = [candidate["position"] for candidate in audit["candidates"]]
        measured = [int(codec.area(mask)) for mask in masks]
        require(measured == [candidate["mask_area"] for candidate in audit["candidates"]],
                "Native mask pixels and recorded areas differ")
        areas[image] = dict(zip(ranks, measured, strict=True))
        point_positions[image] = {int(pos) for pos, edits in point["changes"].items()
                                  if edits.get("point_equal", {}).get("swaps", 0) > 0}
        require(point_positions[image] <= areas[image].keys(), "PointRend positions must remain native candidates")
        blocks[image] = dict(stock=pred["coco"], ranks=ranks, changes={}, group=audit["acquisition_group"],
                             sensor=audit["sensor"], file=audit["source_image_name"], rgb=audit["rgb_sha256"])
    return blocks, areas, point_positions


def score_view(ids, gt, all_blocks, areas, point_positions, polygons, dates, reference, whole, deps):
    phx, _, cfa, history, reserve, _, evaluator, _, core = deps
    blocks = {image: data for image, data in all_blocks.items() if image in ids}
    part = dict(gt, images=[im for im in gt["images"] if im["id"] in ids],
                annotations=[ann for ann in gt["annotations"] if ann["image_id"] in ids])
    annotations = {ann["id"]: ann for ann in part["annotations"]}
    metrics, anchors = evaluator.score_one(part, blocks, "raw", anchors=True)
    geometry = whole.fixed_geometry(anchors, blocks, polygons, annotations, dates, cfa, core, reserve)
    native = dict(images=len(blocks), annotations=len(annotations), source_groups=len({b["group"] for b in blocks.values()}),
                  metrics_fraction={"raw": {key: history.available(value) for key, value in metrics.items()}},
                  geometry=geometry)
    phx.parity(native, reference, ("raw",))
    require(all(geometry[key] == reference["geometry"][key]
                for key in ("counts", "represented_groups", "sensor_date_blocks"))
            and geometry["native_anchor_retained_50"]["raw"] == reference["geometry"]["native_anchor_retained_50"]["raw"],
            "Historical native anchors or original geometry strata differ")

    candidates = [(area, block["group"]) for image, block in blocks.items() for area in areas[image].values()]
    matching, fixed_geometry = [], []
    for gt_id, (image, position, det) in anchors.items():
        area = areas[image][position]
        require(area == int(codec.area(det["segmentation"])), "Historical anchor candidate area differs")
        entry = (area, blocks[image]["group"])
        matching.append(entry)
        polygon = polygons[gt_id]
        if polygon.is_valid and polygon.area > 0 and 32 <= area < 1024:
            fixed_geometry.append(entry)

    exposure = dict(native_candidates=count_strata(candidates, dates),
                    original_matched_hulls=count_strata(matching, dates),
                    fixed_geometry_matches=count_strata(fixed_geometry, dates))
    require(exposure["original_matched_hulls"]["total"] == geometry["counts"]["native_matches"]
            and exposure["fixed_geometry_matches"]["total"] == geometry["counts"]["eligible_matches"],
            "Exposure census differs from fixed original cohorts")
    eligible_in_point = sum(areas[image][pos] >= MINIMUM_AREA for image in blocks for pos in point_positions[image])
    exposure["native_candidates"]["point_edit_stream_total"] = sum(len(point_positions[image]) for image in blocks)
    exposure["native_candidates"]["eligible_in_point_edit_stream"] = eligible_in_point
    exposure["native_candidates"]["eligible_outside_point_edit_stream"] = exposure["native_candidates"]["eligible"] - eligible_in_point
    return dict(images=len(blocks), annotations=len(annotations), raw_metrics_fraction=native["metrics_fraction"]["raw"],
                native_geometry_counts=geometry["counts"], **exposure)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("code", "stock-root", "point-root", "author-labels", "protocol", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--prior-score", type=Path)
    parser.add_argument("--protocol-sha256", required=True)
    a = parser.parse_args()
    for key, value in vars(a).items():
        if isinstance(value, Path):
            setattr(a, key, value.resolve(strict=key not in {"author_labels", "output"}))
    require(a.protocol.name == LOCK_NAME and a.protocol_sha256 == digest(a.protocol.read_bytes()) == LOCK_SHA256
            and not a.output.exists(), "Published exposure prelock and fresh output required")
    require((np.__version__, version("pycocotools"), shapely.__version__, shapely.geos_version_string)
            == ("1.26.4", "2.0.11", "2.0.2", "3.11.2"), "Historical mask and geometry runtime required")
    a.cohort, a.support_root = 74, None
    started = perf_counter()
    whole, deps = sources(a)
    _, api, cfa, history, reserve, old, _, adapter, core = deps
    reference = json.loads(api.pin(a.prior_score or a.code.parent / "results" / REFERENCE_NAME, REFERENCE_SHA256))
    loader = cfa.source(a.code / "POINTREND_SIGNED_QUOTA_SWAP_20261123/inputs.py", cfa.INPUTS_SHA)
    rows, lineage = loader.load_population(a)
    require(lineage == reference["input_lineage"] and lineage["mapped_candidates"] == 7346,
            "Require the exact original reserve capture and candidate population")
    blocks, areas, point_positions = native_blocks(rows, adapter, core)
    require(len(blocks) == 74 and sum(map(len, areas.values())) == 7346
            and sum(map(len, point_positions.values())) == 4114, "Full native and old point edit censuses differ")
    print(json.dumps(dict(phase="native_candidates_verified", images=74, native_candidates=7346,
                          target_labels_opened=False)), flush=True)

    a.author_labels = a.author_labels.resolve(strict=True)
    gt, rekeyed, polys, sets, dates, label_sha = history.views_and_labels(a, old, reserve, blocks)
    require(label_sha == reference["author_labels_sha256"], "Historical label input differs")
    views = {}
    for name, images in VIEWS.items():
        require(len(sets[name]) == images, "Historical admitted view size differs")
        views[name] = score_view(sets[name], gt, blocks, areas, point_positions, polys, dates,
                                 reference["views"][name], whole, deps)
        print(json.dumps(dict(phase="scored_view", view=name, images=images)), flush=True)

    result = dict(
        status=STATUS, phase="score", worker_sha256=digest(Path(__file__).read_bytes()),
        protocol_sha256=LOCK_SHA256, prior_score_sha256=REFERENCE_SHA256, input_lineage=lineage, author_labels_sha256=label_sha,
        author_ids_rekeyed=rekeyed, target_labels_opened=True, minimum_native_mask_area_pixels=MINIMUM_AREA,
        native_mask_area_bins=list(AREA_CLASSES), all_native_candidates_verified_before_labels=True,
        fixed_geometry_rule="original .50 matches; valid reference polygons and native area in [32,1024)",
        raw_coco_and_fixed_geometry_history_exact=True, detector_forwards=0, segrefiner_forwards=0, optimizer_updates=0,
        primary_view="full", sensitivity_view="date_separated", views=views, score_seconds=perf_counter() - started,
    )
    a.output.mkdir()
    stamp = api.write(a.output / "RESULT_SAFE.json", api.encode(result))
    print(json.dumps(dict(status=STATUS, result_sha256=stamp)), flush=True)


if __name__ == "__main__":
    main()
