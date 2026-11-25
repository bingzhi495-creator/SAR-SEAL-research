"""Frozen prediction comparison under two ship references on the same decoded images."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import re
import subprocess
from time import perf_counter
from types import ModuleType
import zipfile

import numpy as np
from PIL import Image
from pycocotools import mask
from scipy.optimize import linear_sum_assignment

LOCK = "d522781b8ac937a465e27a810bdd40735e39b3a4869f486f69ba0aa7eda70362"
ERRATUM = "d5da0cdb38f80f1129bb88ece2c77c23c4636d6b242d0d3c8defed64433561b2"
SLGSEA = "23505cd2de8faa3afa9b4cb329397c486724d02cc24482fd1883ba70a01f77a5"
READBACK = "e99d0f2bbd9e1839e260a88075f6af187e05ffc2b86abaf6cc33b788372b7b71"
REPLAY = "531b979c316c58f608e48205a2f86e66e666eae61c1029de59c8229a941c937e"
CLASSICAL_WORKER = "633da1df76185bb32177422eba0c3b51f2c0b1a01931ce432b993641c10aa6c7"
SCORER = "dd8e435976d2331146c8f83d70f368ee4e0954c05520c54fa9a3b1205838d40f"
GEOMETRY = "107f31864bbb7e807a4194e552c7a1778d410f0e564f80437c5717fe6b47098d"
PS = "Official-SSDD-OPEN/BBox_RBox_PSeg_SSDD/coco_style"
SPSD = "datasets/SPSD"
EXTRA = ("attraction_dynamic", "attraction_static_at_dynamic_count", "learned_at_attraction_count")
PAIRS = (
    ("free_matched_count", "fixed"),
    ("free_natural", "raw"),
    ("free_natural", "native_matched_count"),
    (EXTRA[2], EXTRA[0]),
    (EXTRA[2], EXTRA[1]),
    (EXTRA[1], EXTRA[0]),
)
METRICS = ("AP", "AP50", "AP75", "AP90", "AR100")
STATUS = "SPSD_PSEG232_FROZEN_REFERENCE_SENSITIVITY_COMPLETE"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return sha256(data).hexdigest()


def pinned(path, expected):
    data = path.read_bytes()
    require(digest(data) == expected, "Closed file checksum: " + path.name)
    return data


def file_digest(path):
    h = sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def module(path, expected):
    m = ModuleType(path.stem)
    m.__file__ = str(path)
    exec(compile(pinned(path, expected), str(path), "exec"), m.__dict__)
    return m


def observed(root, phase, status):
    done = json.loads((root / f"observed_{phase}" / "TERMINAL.json").read_bytes())
    require(done["actual_exit"] == 0 and done["output_footer_verified"] is True and done["phase"] == phase
            and done["worker_sha256"] == CLASSICAL_WORKER and bool(done["ended_utc"]), "Closed " + phase)
    result = json.loads(pinned(root / phase / "RESULT_SAFE.json", done["result_sha256"]))
    require(result["status"] == status and bool(result["GT_opened"]) == (phase == "score"), "Accepted " + phase)
    return result, done["result_sha256"]


def originals(args, score):
    readback = json.loads(pinned(args.old_readback, READBACK))
    score.check_acceptance(readback)
    terminal = json.loads(pinned(args.owner / "REPLAY_TERMINAL_PRIVATE.json", readback["original_tool_wait_sha256"]))
    require(terminal["original_process_terminal"] is True and terminal["original_exit"] == 0 and terminal["result_sha256"] == REPLAY, "Closed original replay")
    previous = json.loads(pinned(args.original_replay / "RESULT_SAFE.json", REPLAY))
    require(previous["edits_sha256"] == readback["edits_sha256"], "Original frozen edit stream")
    index = score.validate_manifest(json.loads(pinned(args.owner / "TEST232_IMAGE_ONLY_MANIFEST_PRIVATE.json", readback["manifest_sha256"])))
    prepared, prep_sha = observed(args.classical_root, "prepare-edits", "PSEG_CLASSICAL_EDITS_PREPARED")
    verified, verify_sha = observed(args.classical_root, "verify-edits", "PSEG_CLASSICAL_EDITS_VERIFIED")
    scored, score_sha = observed(args.classical_root, "score", "PSEG_CLASSICAL_MATCHED_FROZEN_SCORE_COMPLETE")
    require(prepared["counts"] == verified["counts"] and prepared["edits_sha256"] == verified["edits_sha256"] == scored["edits_sha256"]
            and verified["original_readback_sha256"] == READBACK, "Independent closed classical extension binding")
    path = args.classical_root / "prepare-edits" / "EDITS_PRIVATE.jsonl"
    edits = [json.loads(line) for line in pinned(path, verified["edits_sha256"]).splitlines()]
    old_edits = [json.loads(line) for line in pinned(args.original_replay / "EDITS_PRIVATE.jsonl", readback["edits_sha256"]).splitlines()]
    require(len(edits) == len(old_edits) == 232, "Original and extended edit rows")
    sums = Counter()
    candidates = Counter()
    for extended, original in zip(edits, old_edits, strict=True):
        require(extended["rgb_sha256"] == original["rgb_sha256"] and extended["image_id"] == original["image_id"],
                "Original same decoded-image sequence")
        projected = {}
        for rank, actions in extended["changes"].items():
            swaps = [actions.get(arm, {}).get("swaps", 0) for arm in EXTRA]
            require(swaps[0] == swaps[1] == swaps[2], "Exact classical matched budget")
            kept = {arm: change for arm, change in actions.items() if arm not in EXTRA}
            if kept:
                projected[rank] = kept
            for arm, change in actions.items():
                sums[arm] += change["swaps"]
                candidates[arm] += 1
        require(projected == original["changes"], "Unchanged original treatment choices and RLEs")
    for arm, n in sums.items():
        require(verified["checked"][arm + "_swaps"] == n and verified["checked"][arm + "_edited"] == candidates[arm],
                "Closed realized budgets")
    native = [json.loads(line) for line in pinned(args.capture / "DEV_MASK_PREDICTIONS_PRIVATE.jsonl", readback["input_predictions_sha256"]).splitlines()]
    audit = [json.loads(line) for line in pinned(args.capture / "DEV_CANDIDATE_AUDIT_PRIVATE.jsonl", readback["candidate_audit_sha256"]).splitlines()]
    score.ARMS += EXTRA
    blocks = score.join_blocks(index, native, audit, edits)
    require(sum(len(value[0]) for value in blocks.values()) == 23200, "Complete native rank population")
    rgbs = {item["image_ref"]["path"]: row["rgb_sha256"] for item, row in zip(index, edits, strict=True)}
    require(len(set(rgbs.values())) == len(rgbs) == 232, "Unique decoded-image groups")
    identity = {"source_readback_sha256": READBACK, "source_replay_sha256": REPLAY,
                "classical_prepare_sha256": prep_sha, "classical_verify_sha256": verify_sha,
                "classical_score_sha256": score_sha, "extended_edits_sha256": verified["edits_sha256"],
                "native_sha256": readback["input_predictions_sha256"], "manifest_sha256": readback["manifest_sha256"]}
    return index, blocks, rgbs, scored["views"]["official"]["metrics_fraction"], identity, dict(sums), dict(candidates)


def tar_bytes(archive, option, *members):
    command = ["tar", option, str(archive), *members]
    return subprocess.run(command, capture_output=True, check=True).stdout


def read_references(args, score, roster):
    require(file_digest(args.slgsea) == SLGSEA, "Exact already accepted author archive")
    polygon = tar_bytes(args.official, "-xOf", PS + "/annotations/test.json")
    require(digest(polygon) == score.GT_SHA, "Exact original frozen PSeg polygons")
    pseg = json.loads(polygon)
    index = {row["image_ref"]["path"]: row for row in roster}
    pattern = re.compile(rb"^\S+\s+\d+\s+\S+\s+\S+\s+(\d+)\s+.* " + PS.encode() + rb"/images/test/(\d+\.jpg)$")
    lengths = []
    for line in tar_bytes(args.official, "-tvf").splitlines():
        match = pattern.match(line)
        if match:
            length, name = match.groups()
            lengths.append((int(length), name.decode("ascii")))
    require(len(lengths) == 232 and {name for _, name in lengths} == set(index), "Official author JPEG roster")
    payload = tar_bytes(args.official, "-xOf", "--include", PS + "/images/test/*.jpg")
    require(sum(n for n, _ in lengths) == len(payload), "Complete original archive JPEG extraction")
    with zipfile.ZipFile(args.slgsea) as archive:
        cursor = 0
        for length, name in lengths:
            expected = index[name]["image_ref"]
            source = payload[cursor:cursor + length]
            new = archive.read(f"{SPSD}/val2017/{name}")
            require(length == expected["bytes"] and digest(source) == expected["sha256"] and source == new,
                    "Identical original PSeg, admitted model-input and SPSD JPEGs")
            cursor += length
        pan_bytes = archive.read(f"{SPSD}/annotations/panoptic_val2017.json")
        pan = json.loads(pan_bytes)
        require([c["id"] for c in pan["categories"] if c["isthing"]] == [0], "Only ship thing category")
        rgb_record = {a["image_id"]: a for a in pan["annotations"]}
        native_images = {a["file_name"]: a for a in pan["images"]}
        require(set(native_images) == set(index) and len(rgb_record) == len(native_images) == 232, "Same complete SPSD roster")
        instances = []
        metadata_disagreement = Counter()
        for image in pseg["images"]:
            source = native_images[image["file_name"]]
            require(all(image[key] == source[key] for key in ("width", "height")), "Reference alignment")
            record = rgb_record[source["id"]]
            png = archive.read(f"{SPSD}/annotations/panoptic_val2017/{record['file_name']}")
            with Image.open(BytesIO(png)) as handle:
                require(handle.mode == "RGB" and handle.size == (image["width"], image["height"]), "Original RGB instance raster")
                rgb = np.asarray(handle,dtype=np.uint32)
            id_map = rgb[:, :, 0] + rgb[:, :, 1] * 256 + rgb[:, :, 2] * 65536
            seen = set()
            for segment in record["segments_info"]:
                require(segment["id"] not in seen, "Unique segment in original panoptic raster metadata")
                seen.add(segment["id"])
                if segment["category_id"] != 0:
                    continue
                rle = mask.encode(np.asfortranarray(id_map == segment["id"],dtype=np.uint8))
                area = int(mask.area(rle))
                bbox = mask.toBbox(rle).tolist()
                require(area > 0 and segment["area"] > 0 and len(segment["bbox"]) == 4 and not segment["iscrowd"],
                        "Author PNG ship ID with positive area and declared metadata")
                metadata_disagreement["area"] += area != segment["area"]
                metadata_disagreement["bounding_box"] += bbox != segment["bbox"]
                rle["counts"] = rle["counts"].decode("ascii")
                instances.append(dict(id=len(instances) + 1, category_id=0, image_id=image["id"], area=area, bbox=bbox,
                                      iscrowd=0, segmentation=rle))
    require(len(pseg["images"]) == 232 and len(pseg["annotations"]) == 546 and len(instances) == 547,
            "Complete two author populations")
    require({a["image_id"] for a in pseg["annotations"]} == {a["image_id"] for a in instances}, "Positive hull coverage for all roster images")
    require(dict(metadata_disagreement) == {"area": 547, "bounding_box": 547},
            "Input metadata discrepancy census from the prescore clarification")
    return pseg, dict(pseg, annotations=instances), digest(pan_bytes), dict(metadata_disagreement)


def labels_only(pseg, spsd):
    old = defaultdict(list)
    new = defaultdict(list)
    images = {image["id"]: image for image in pseg["images"]}
    for a in pseg["annotations"]:
        image = images[a["image_id"]]
        old[a["image_id"]].append(mask.merge(mask.frPyObjects(a["segmentation"], image["height"], image["width"])))
    for a in spsd["annotations"]:
        new[a["image_id"]].append(a["segmentation"])
    values = []
    area_ratios = []
    equal_unions = 0
    equal_counts = 0
    for iid in images:
        lhs, rhs = old[iid], new[iid]
        overlap = mask.iou(lhs, rhs, [0] * len(rhs))
        i, j = linear_sum_assignment(-overlap)
        values.extend(overlap[i, j])
        area_ratios.extend(np.asarray(mask.area(rhs))[j] / np.asarray(mask.area(lhs))[i])
        equal_unions += mask.iou([mask.merge(lhs)], [mask.merge(rhs)], [0]).item() == 1.0
        equal_counts += len(lhs) == len(rhs)
    v = np.asarray(values)
    return dict(pseg_hulls=sum(map(len, old.values())), spsd_hulls=sum(map(len, new.values())),
                equal_instance_count_images=int(equal_counts), exact_union_images=int(equal_unions),
                assigned_hulls=len(values), overlap_at_least_half=int((v >= .5).sum()),
                overlap_at_least_nine_tenths=int((v >= .9).sum()), exact_assigned_masks=int((v == 1).sum()),
                assigned_iou_mean=float(v.mean()), assigned_iou_median=float(np.median(v)),
                assigned_spsd_to_pseg_area_ratio_median=float(np.median(area_ratios)))


def rows_for(gt, blocks, arm):
    rows = []
    for image in gt["images"]:
        native, edits, _ = blocks[image["file_name"]]
        for rank, output in enumerate(native, 1):
            edit = edits.get(str(rank), {}).get(arm)
            rows.append(dict(image_id=image["id"], category_id=0, score=output["score"],
                             segmentation=edit["rle"] if edit else output["segmentation"]))
    return rows


def interval(delta, draws):
    return {"points": float(delta.mean()), "percentile95_points": np.percentile(delta[draws].mean(axis=1), [2.5, 97.5]).tolist()}


def score_reference(gt, blocks, rgbs, scorer, geometry, draws, expected=None):
    metrics = {arm: scorer.evaluate(gt, rows_for(gt, blocks, arm)) for arm in scorer.ARMS}
    if expected is not None:
        require(metrics == expected, "Exact unrounded earlier PSeg COCO metrics across all eight arms")
    ids = sorted(i["id"] for i in gt["images"])
    source = {i["id"]: blocks[i["file_name"]] for i in gt["images"]}
    rgb = {i["id"]: rgbs[i["file_name"]] for i in gt["images"]}
    assignment, _ = geometry.evaluate_raw(gt, {i: {"native": source[i][0]} for i in ids}, ids, metrics["raw"])
    hull_value = defaultdict(list)
    changed = defaultdict(set)
    edited = defaultdict(set)
    matched_groups = set()
    matched_rows = 0
    for a in gt["annotations"]:
        i = a["image_id"]
        group = rgb[i]
        if a["id"] not in assignment:
            hull_value[group].append(np.zeros(len(scorer.ARMS)))
            continue
        image_id, rank = assignment[a["id"]]
        require(i == image_id, "COCO match stays in image")
        matched_rows += 1
        matched_groups.add(group)
        native, changes, canvas = source[i]
        edit = changes.get(str(rank), {})
        masks = {arm: edit.get(arm, {}).get("rle", native[rank - 1]["segmentation"]) for arm in scorer.ARMS}
        ref = a["segmentation"]
        if isinstance(ref, list):
            ref = mask.merge(mask.frPyObjects(ref, *canvas))
        values = mask.iou(list(masks.values()), [ref], [0]).flatten()
        require(values[0] >= .50 - 1e-12, "Native match raster IoU reproduces")
        hull_value[group].append(100 * values)
        for arm in edit:
            edited[arm].add((group, a["id"]))
        for p, q in PAIRS:
            if masks[p] != masks[q]:
                changed[p, q].add((group, a["id"]))
    require(matched_rows == len(assignment) and set(hull_value) == set(rgbs.values()), "Complete all-hull image-group denominator")
    matrix = np.stack([np.mean(hull_value[group], axis=0) for group in sorted(hull_value)], axis=1)
    deltas = {}
    output = dict(hulls=len(gt["annotations"]), matched_hulls=matched_rows, unmatched_hulls=len(gt["annotations"]) - matched_rows,
                  image_groups=len(hull_value), matched_image_groups=len(matched_groups), prior_score_parity=expected is not None,
                  metrics={}, pairwise={})
    for arm, v in zip(scorer.ARMS, matrix, strict=True):
        output["metrics"][arm] = dict(**{k: 100 * metrics[arm][k] for k in METRICS}, all_hull_image_equal_iou=float(v.mean()),
                                      edited_matched_hulls=len(edited[arm]), edited_matched_groups=len({r[0] for r in edited[arm]}))
    for arms in PAIRS:
        arm, base = arms
        key = f"{arm}_minus_{base}"
        deltas[key] = matrix[scorer.ARMS.index(arm)] - matrix[scorer.ARMS.index(base)]
        output["pairwise"][key] = dict(
            **{k: 100 * (metrics[arm][k] - metrics[base][k]) for k in METRICS},
            geometry=interval(deltas[key], draws), changed_matched_hulls=len(changed[arms]),
            changed_matched_groups=len({r[0] for r in changed[arms]}),
        )
    return output, deltas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "official", "slgsea", "owner", "capture", "old-readback", "original-replay",
                 "classical-root", "code", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args=parser.parse_args()
    start=perf_counter()
    pinned(args.protocol, LOCK)
    pinned(args.protocol.with_name("ERRATUM.md"), ERRATUM)
    scorer = module(args.code / "PSEG_TEST232_DFINE_FROZEN_SCORE_20261121/score_test232.py", SCORER)
    geometry = module(args.code / "PSEG_TEST232_PAIRED_GEOMETRY_20261121/paired_geometry.py", GEOMETRY)
    roster, blocks, rgbs, expected, identity, swaps, candidates = originals(args, scorer)
    pseg, spsd, pan_sha, metadata_disagreement = read_references(args, scorer, roster)
    label_geometry = labels_only(pseg, spsd)
    draws = np.random.Generator(np.random.PCG64(5297)).integers(232, size=(4096, 232))
    a, a_delta = score_reference(pseg, blocks, rgbs, scorer, geometry, draws, expected)
    b, b_delta = score_reference(spsd, blocks, rgbs, scorer, geometry, draws)
    result = dict(status=STATUS, protocol_sha256=LOCK, worker_sha256=digest(Path(__file__).read_bytes()),
                  slgsea_sha256=SLGSEA, pseg_annotation_sha256=scorer.GT_SHA, spsd_annotation_json_sha256=pan_sha,
                  identity=identity, references=dict(PSeg_SSDD=a, SPSD=b), exact_image_byte_matches=232,
                  erratum_sha256=ERRATUM, spsd_metadata_disagreement=metadata_disagreement, label_geometry=label_geometry,
                  source_predictions=23200, realized_swaps=swaps, edited_candidates=candidates,
                  paired_reference_shifts={
                      k: dict(**{metric: b["pairwise"][k][metric] - a["pairwise"][k][metric] for metric in METRICS},
                              geometry=interval(b_delta[k] - a_delta[k], draws)) for k in a_delta},
                  bootstrap_unit="unique decoded input JPEG; capture-scene independence unverified",
                  bootstrap_replicates=4096, bootstrap_seed=5297, model_forwards=0, optimizer_updates=0,
                  elapsed_worker_seconds=perf_counter() - start)
    data = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    with args.output.open("xb") as target:
        target.write(data)
        target.flush()
        os.fsync(target.fileno())
    print(json.dumps(dict(status=STATUS,result_sha256=digest(data))), flush=True)


if __name__ == "__main__":
    main()

