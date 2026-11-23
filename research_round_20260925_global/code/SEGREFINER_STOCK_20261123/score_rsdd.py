"""Score the observed stock SegRefiner-LR masks on the fixed Instance-RSDD reserve."""
import argparse
from collections import Counter
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter
from types import ModuleType

import numpy as np
from pycocotools import mask as codec
import shapely


LOCK_NAME = "SEGREFINER_STOCK_PRELOCK_20261123.md"
LOCK = "30007c832d47c6e086445d474d4f319642854de20da7844706ebac8e40a7df9c"
EXPOSURE_SOURCE = "26a217cccc7a5814a8033cdfef66e8cdc27b9f70a812b37f7721b8a42a011f44"
EXPOSURE = "78e2dd5de0901e996087b319dbefe78549f702333fd87a542b2d70a69534867c"
EXPOSURE_NAME = "RSDD74_SEGREFINER_STOCK_EXPOSURE_SAFE_20261123.json"
POINT_REFERENCE = "INSTANCE_RSDD_RESERVE_POINTREND_ACTUAL_SAFE_20261123.json"
POINT_REFERENCE_SHA = "c54b9a53a5bfd3f6a54b6d31d93a0789eb176b07bd2cb52c58c6f2ceb4e77b27"
AUTHOR_COMMIT = "53419a2d38ea3da0b6e2be77e5b45e139195a0b3"
CHECKPOINT_SHA = "d5806a41a4bff6f1e313d1a9dcf18d11dab28e3a8e1497871a156214cd8fe07a"
INFER_STATUS = "RSDD74_SEGREFINER_STOCK_INFERENCE_COMPLETE"
STATUS = "RSDD74_SEGREFINER_STOCK_SCORE_COMPLETE"
LEDGER = "REFINED_MASKS_PRIVATE.jsonl"
GATE = 512
LOCAL = "learned_common"
ARMS = ("raw", LOCAL, "stock_lr")
PAIRS = (("stock_lr", "raw"), ("stock_lr", LOCAL))
VIEWS = ("full", "date_separated")


def require(test, message):
    if not test:
        raise ValueError(message)


def digest(raw):
    return sha256(raw).hexdigest()


def exposure_source():
    path = Path(__file__).with_name("exposure_rsdd.py")
    data = path.read_bytes()
    require(digest(data) == EXPOSURE_SOURCE, "Archived pre-inference exposure verifier differs")
    module = ModuleType(path.stem)
    module.__file__ = str(path)
    exec(compile(data, str(path), "exec"), module.__dict__)
    return module


def checked_rle(value, shape):
    require(set(value) == {"size", "counts"} and value["size"] == shape and isinstance(value["counts"], str),
            "Unexpected native COCO mask shape or encoding")
    binary = codec.decode(value)
    require(binary.shape == tuple(shape) and np.all(binary <= 1), "Non-binary or non-native decoded output")
    encoded = codec.encode(np.asfortranarray(binary))
    require(value == dict(size=encoded["size"], counts=encoded["counts"].decode("ascii")),
            "Edited RLE must be canonical")
    return int(np.count_nonzero(binary))


def attach_verified_masks(rows, blocks, areas, point_positions, stream, reference):
    totals = Counter()
    for row, line in zip(rows, stream.splitlines(), strict=True):
        item, audit, pred, point, _ = row
        image = item["image_id"]
        parsed = json.loads(line)
        require(set(parsed) == {"image_id", "changes"} and parsed["image_id"] == image,
                "Inferred image or original ordering differs")
        edited = parsed["changes"]
        expected = {str(position) for position, area in areas[image].items() if area >= GATE}
        require(set(edited) == expected, "Inferred edit slots must cover all and only original 512-pixel candidates")
        totals.update(images=1, native_candidates=len(pred["coco"]), eligible_candidates=len(edited),
                      bypassed_candidates=len(pred["coco"])-len(edited))
        for candidate, native in zip(audit["candidates"], pred["coco"], strict=True):
            position = candidate["position"]
            slot = str(position)
            shape = item["original_hw"]
            area = areas[image][position]
            rle = native["segmentation"]
            updates = {}
            if slot in edited:
                entry = edited[slot]
                require(set(entry) == {"rle"}, "Unexpected stock output fields")
                refined = entry["rle"]
                new_area = checked_rle(refined, shape)
                updates["stock_lr"] = entry
                totals.update(eligible_masks_changed=int(refined != rle), eligible_native_pixels=area,
                              eligible_refined_pixels=new_area)
            if position in point_positions[image]:
                original_entry = point["changes"][slot][LOCAL]
                local_mask = original_entry["rle"]
                local_area = checked_rle(local_mask, shape)
                intersection = int(codec.area(codec.merge([rle, local_mask], intersect=True)))
                require(local_area == area and area-intersection == original_entry["swaps"],
                        "Original learned independent edit no longer preserves its exact local quota")
                updates[LOCAL] = {"rle": local_mask}
                totals["local_edited_candidates"] += 1
            if updates:
                blocks[image]["changes"][position] = updates
    require(all(totals[k] == v for k, v in reference.items()) and totals["local_edited_candidates"] == 4114,
            "Re-read stock mask or accepted comparator census differs")
    return dict(totals)


def gate_summary(anchors, blocks, dates, comparator):
    totals = dict(matches=0, parents=set(), sensor_date_blocks=set())
    for _, (image, position, detection) in anchors.items():
        block = blocks[image]
        native = detection["segmentation"]
        edits = block["changes"][position]
        candidate = edits["stock_lr"]["rle"]
        baseline = edits.get(comparator, {"rle": native})["rle"]
        if candidate != baseline:
            parent = block["group"]
            totals["matches"] += 1
            totals["parents"].add(parent)
            totals["sensor_date_blocks"].add(dates[parent])
    return dict(matches=totals["matches"], parents=len(totals["parents"]),
                sensor_date_blocks=len(totals["sensor_date_blocks"]))


def score_view(ids, gt, blocks, polygons, dates, raw_reference, point_reference, exposure_view, whole, deps):
    _, _, cfa, _, reserve, _, evaluator, _, core = deps
    view = whole.score_view(ids, gt, blocks, polygons, dates, raw_reference, deps)
    require(view["metrics_fraction"][LOCAL] == point_reference["metrics_fraction"][LOCAL],
            "Archived local independent full COCO metric vector differs")
    geom = view["geometry"]["group_equal_arms"][LOCAL]
    point_geom = point_reference["geometry"]["group_equal_arms"][LOCAL]
    require(set(point_geom) <= set(geom) and all(abs(geom[k]-v) < 1e-9 for k, v in point_geom.items()),
            "Archived local independent polygon geometry differs")
    selected = {image: block for image, block in blocks.items() if image in ids}
    sub = dict(gt, images=[im for im in gt["images"] if im["id"] in ids],
               annotations=[ann for ann in gt["annotations"] if ann["image_id"] in ids])
    _, anchors = evaluator.score_one(sub, selected, "raw", anchors=True)
    gate = {}
    for gid, row in anchors.items():
        polygon = polygons[gid]
        area = int(codec.area(row[2]["segmentation"]))
        if polygon.is_valid and polygon.area > 0 and GATE <= area < 1024:
            gate[gid] = row
    gate_geometry = whole.fixed_geometry(gate, selected, polygons, {ann["id"]: ann for ann in sub["annotations"]},
                                         dates, cfa, core, reserve)
    exposed = exposure_view["fixed_geometry_matches"]
    require(gate_geometry["counts"]["eligible_matches"] == exposed["eligible"]
            and gate_geometry["represented_groups"] == exposed["eligible_parent_groups"]
            and gate_geometry["sensor_date_blocks"] == exposed["eligible_sensor_date_blocks"],
            "Previously measured gate-exposed fixed geometry differs")
    stock_altered = gate_summary(gate, blocks, dates, "raw")
    enough_altered = stock_altered["matches"] >= 20 and stock_altered["sensor_date_blocks"] >= 8
    for arm, comparator in PAIRS:
        key = arm + "_minus_" + comparator
        contrast = view["contrasts"][key]
        ap = contrast["COCO_points"]["AP"]
        interval = view["geometry"]["paired"][key]["continuous_iou_points"]["interval_95"]
        contrast["criteria"] = dict(
            stock_altered_gate_exposure_sufficient=enough_altered,
            practical_positive_transfer=bool(enough_altered and ap is not None and ap >= .30 and interval[0] > 0),
            supported_adverse_transfer=bool(enough_altered and ap is not None and ap <= -.30 and interval[1] < 0),
        )
        contrast["gate_exposed_differences"] = gate_summary(gate, blocks, dates, comparator)
    view["gate_exposed_geometry"] = gate_geometry
    view["stock_altered_gate_exposure"] = stock_altered
    return view


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("code", "stock-root", "point-root", "inference", "inference-terminal",
                 "author-labels", "protocol", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--exposure", type=Path)
    parser.add_argument("--protocol-sha256", required=True)
    a = parser.parse_args()
    for key, value in vars(a).items():
        if isinstance(value, Path):
            setattr(a, key, value.resolve(strict=key not in {"author_labels", "output"}))
    require(a.protocol.name == LOCK_NAME and a.protocol_sha256 == digest(a.protocol.read_bytes()) == LOCK
            and not a.output.exists(), "Published stock-refiner prelock and fresh output required")
    require((np.__version__, version("pycocotools"), shapely.__version__, shapely.geos_version_string)
            == ("1.26.4", "2.0.11", "2.0.2", "3.11.2"), "Historical mask and polygon runtime required")
    start = perf_counter()
    a.cohort, a.support_root = 74, None
    exposure_worker = exposure_source()
    whole, deps = exposure_worker.sources(a)
    _, api, cfa, history, reserve, old, _, adapter, core = deps
    reference_root = a.code.parent / "results"
    exposure = json.loads(api.pin(a.exposure or reference_root / EXPOSURE_NAME, EXPOSURE))
    prior = json.loads(api.pin(reference_root / exposure_worker.REFERENCE_NAME, exposure_worker.REFERENCE_SHA256))
    local = json.loads(api.pin(reference_root / POINT_REFERENCE, POINT_REFERENCE_SHA))
    infer_worker_sha = digest(Path(__file__).with_name("infer_rsdd.py").read_bytes())
    runtime_sha = digest(Path(__file__).with_name("author_runtime.py").read_bytes())
    infer, infer_sha, terminal_sha = api.observed(a.inference, a.inference_terminal, "inference", INFER_STATUS,
                                                  worker=infer_worker_sha)
    require(infer["phase"] == "inference" and infer["protocol_sha256"] == LOCK and infer["author_commit"] == AUTHOR_COMMIT
            and infer["checkpoint_sha256"] == CHECKPOINT_SHA and infer["runtime_sha256"] == runtime_sha
            and infer["exposure_sha256"] == EXPOSURE and infer["target_labels_opened"] is infer["smoke"] is False
            and infer["scientific_replay"] is infer["exact_original_bypass"] is infer["candidate_ordinals_preserved"] is True
            and infer["optimizer_updates"] == infer["detector_forwards"] == 0
            and infer["random_seed"] == 5217 and infer["minimum_native_mask_area_pixels"] == GATE
            and infer["environment"]["device"].startswith("cuda") and infer["segrefiner_image_calls"] == 74
            and infer["full_population_counts"] == dict(images=74, native_candidates=7346, eligible_candidates=3074)
            and infer["refined_masks_filename"] == LEDGER, "Approved actual observed author inference required")
    loader = cfa.source(a.code / "POINTREND_SIGNED_QUOTA_SWAP_20261123/inputs.py", cfa.INPUTS_SHA)
    rows, lineage = loader.load_population(a)
    require(lineage == infer["input_lineage"] == exposure["input_lineage"] == prior["input_lineage"]
            and lineage["point_edits_sha256"] == local["edits_sha256"]
            and lineage["predictions_sha256"] == local["predictions_sha256"],
            "Exact complete stock and original local independent lineages required")
    blocks, areas, point_positions = exposure_worker.native_blocks(rows, adapter, core)
    stream = api.pin(a.inference / LEDGER, infer["refined_masks_sha256"])
    totals = attach_verified_masks(rows, blocks, areas, point_positions, stream, infer["counts"])
    require(all(totals[k] == v for k, v in infer["full_population_counts"].items()),
            "All original candidates and 3074 author-gated masks must be present")
    print(json.dumps(dict(phase="observed_masks_and_local_controls_verified", images=74,
                          eligible=3074, target_labels_opened=False)), flush=True)

    a.author_labels = a.author_labels.resolve(strict=True)
    gt, rekeyed, polygons, sets, dates, label_sha = history.views_and_labels(a, old, reserve, blocks)
    require(label_sha == prior["author_labels_sha256"] == exposure["author_labels_sha256"] == local["author_train_sha256"],
            "Original author reference label identity differs")
    whole.ARMS, whole.MATCHED, whole.PAIRS = ARMS, ("raw",), PAIRS
    views = {}
    for name in VIEWS:
        views[name] = score_view(sets[name], gt, blocks, polygons, dates, prior["views"][name], local["views"][name],
                                 exposure["views"][name], whole, deps)
        print(json.dumps(dict(phase="scored_view", view=name, images=views[name]["images"])), flush=True)
    result = dict(
        status=STATUS, phase="score", worker_sha256=digest(Path(__file__).read_bytes()), protocol_sha256=LOCK,
        input_lineage=lineage, inference_worker_sha256=infer_worker_sha, inference_result_sha256=infer_sha,
        inference_terminal_sha256=terminal_sha, runtime_sha256=runtime_sha, exposure_sha256=EXPOSURE,
        refined_masks_sha256=infer["refined_masks_sha256"], native_reference_sha256=exposure_worker.REFERENCE_SHA256,
        point_reference_sha256=POINT_REFERENCE_SHA, author_labels_sha256=label_sha, author_ids_rekeyed=rekeyed,
        counts=totals, original_coco_and_local_geometry_exact=True, ledgers_verified_before_labels=True,
        detector_forwards=0, segrefiner_forwards=0, optimizer_updates=0, target_labels_opened=True,
        bootstrap_draws=4096, bootstrap_seed=5217, cluster_unit="sensor/date; equal parent weight", arms=list(ARMS),
        primary_view="full", sensitivity_view="date_separated", views=views, score_seconds=perf_counter()-start,
    )
    a.output.mkdir()
    stamp = api.write(a.output / "RESULT_SAFE.json", api.encode(result))
    print(json.dumps(dict(status=STATUS, result_sha256=stamp)), flush=True)


if __name__ == "__main__":
    main()
