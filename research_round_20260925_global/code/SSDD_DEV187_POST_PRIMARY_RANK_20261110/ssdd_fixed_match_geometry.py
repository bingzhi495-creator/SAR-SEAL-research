"""Fixed original-grid mask geometry for the accepted frozen SSDD DEV187 pair diagnostic."""
import argparse
from builtins import __import__ as standard_import
from collections import Counter
from contextlib import redirect_stdout
from copy import deepcopy
from hashlib import sha256
from io import StringIO
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
from pycocotools import mask
from pycocotools.coco import COCO


RANK_PATH = "SSDD_DEV187_POST_PRIMARY_RANK_20261110/p250_rank_diagnostic.py"
RANK_SOURCE_SHA = "162eb595d9a3356c868ab4c94229622faf3f85dfc08d0d9a7e6da4ff82ffd252"
GEOMETRY_SOURCE_SHA = "37b6fb7eb7485e3c3d204da2689e0089285b1009c6e8e5d828eedd2a798f1002"
RANK_RESULT_SHA = "f51f68cf67ce36e4105999c40f842bdf9c0b8a7deebb291f4af6d55ecb6f488e"
ARMS = ("native", "center")


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def load_module(path, digest, name, modules):
    require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file(), "Canonical ordinary source required")
    with path.open("rb") as stream:
        source = stream.read(64 * 1024 + 1)
    require(0 < len(source) <= 64 * 1024 and sha256(source).hexdigest() == digest, "Pinned source changed")

    def pinned_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and name in modules:
            return modules[name]
        return standard_import(name, globals, locals, fromlist, level)

    module = ModuleType(name)
    module.__file__ = str(path)
    module.__dict__["__builtins__"] = dict(vars(standard_import("builtins")), __import__=pinned_import)
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def load_helpers(code, geometry_source):
    rank = load_module(code / RANK_PATH, RANK_SOURCE_SHA, "_closed_p255", {})
    p250, template, coverage, scorer, official = rank.load_support(code)
    # The geometry core only imports the adapter loader at module level. Pass
    # the identical already byte-pinned COCO adapter when measuring.
    adapter_loader = ModuleType("dev705_group_sensitivity")
    adapter_loader.load_official_adapter = lambda: official
    originals = {"dev705_group_sensitivity": adapter_loader, "dev705_paired_coverage": coverage,
                 "dev705_score_template": template}
    geometry = load_module(geometry_source, GEOMETRY_SOURCE_SHA, "_original_h705_geometry", originals)
    return rank, p250, template, scorer, official, geometry


def evaluate_pair(gt, native_rows, center_rows, helpers):
    _, _, template, _, official, geometry = helpers
    images = gt["images"]
    image_ids = [item["id"] for item in images]
    annotations = gt["annotations"]
    annotation_images = {item["id"]: item["image_id"] for item in annotations}
    require(len(annotation_images) == len(annotations), "Unique source annotation identities required")
    selected, counts = template.prefix_rows(template.nonempty_by_image(native_rows, images),
                                            template.nonempty_by_image(center_rows, images))
    matches = {}
    for mode in ARMS:
        _, records, _ = template.evaluate_cell(gt, selected[mode], image_ids, official)
        matches[mode] = geometry.match_official(records, selected[mode], annotation_images)
        del records

    coco = COCO()
    coco.dataset = deepcopy(gt)
    with redirect_stdout(StringIO()):
        coco.createIndex()
    exclusions = Counter(empty=0, crowd=0, ignore=0)
    census = {area: Counter() for area in geometry.AREAS}
    measured = {area: [] for area in geometry.AREAS}
    for row in annotations:
        annotation_id, area = row["id"], row["area"]
        target = coco.annToRLE(coco.anns[annotation_id])
        if mask.area(target) == 0:
            exclusions["empty"] += 1
            continue
        if row.get("iscrowd", 0):
            exclusions["crowd"] += 1
            continue
        if row.get("ignore", 0):
            exclusions["ignore"] += 1
            continue
        require(type(area) in (int, float) and math.isfinite(area) and area >= 0, "Exact COCO area required")
        strata = geometry.AREAS if area < 1024 else ("all",)
        native, center = (matches[mode].get(annotation_id) for mode in ARMS)
        state = "both" if native is not None and center is not None else (
            "native" if native is not None else "center" if center is not None else "neither")
        for stratum in strata:
            census[stratum][state] += 1
        if state == "both":
            record = geometry.measure_masks([target, native, center])
            for stratum in strata:
                measured[stratum].append(record)
    require(census["all"].total() + exclusions.total() == len(annotations), "All original annotations must be accounted")
    return dict(prefix_rows_per_arm=sum(counts), prefix_images=sum(count > 0 for count in counts),
                exclusions=dict(exclusions),
                strata={key: geometry.summarize(census[key], measured[key]) for key in geometry.AREAS})


def direction_gate(summary):
    paired = summary["paired"]
    direction = summary["directional"]
    a, b = (paired[arm]["mean_signed_dy_pixels"] for arm in ARMS)
    skew = {arm: direction[arm]["up"]["mean_delta_iou"] - direction[arm]["down"]["mean_delta_iou"] for arm in ARMS}
    return dict(native_downward=a > 0, center_absolute_signed_y_smaller=abs(b) < abs(a),
                native_up_gain_greater=skew["native"] > 0,
                center_absolute_direction_asymmetry_smaller=abs(skew["center"]) < abs(skew["native"]),
                up_minus_down_mean_iou_change=skew)


def deny_network(event, args):
    if event.startswith("socket."):
        raise RuntimeError("Offline diagnostic required")


def diagnose(args):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and sys.flags.isolated and sys.flags.utf8_mode
            and sys.dont_write_bytecode, "Exact isolated CPU runtime required")
    packages = {key: version(key) for key in ("numpy", "pycocotools")}
    require(packages == {"numpy": "1.26.4", "pycocotools": "2.0.11"}, "Original COCO runtime required")
    sys.addaudithook(deny_network)
    helpers = load_helpers(args.code, args.geometry_source)
    rank, p250, _, scorer, _, _ = helpers
    plan = p250.document(p250.stable_file(args.plan, rank.PLAN_SHA, p250.MIB))
    accepted = p250.document(p250.stable_file(args.accepted_score, rank.ACCEPTED_SHA, p250.MIB))
    ranking = p250.document(p250.stable_file(args.rank_score, RANK_RESULT_SHA, p250.MIB))
    require(plan["status"] == p250.PLAN_STATUS and plan["evaluation_protocol"] == rank.PROTOCOL
            and ranking["accepted_primary_score_sha256"] == rank.ACCEPTED_SHA and ranking["closed_plan_sha256"] == rank.PLAN_SHA,
            "Exact independently closed primary and equal-prefix scope required")
    refs, arms = rank.bound_prediction_refs(p250, plan, accepted)
    image_ref = plan["image_manifest"]
    _, ids, images = scorer.fixed_images(scorer.Artifact(image_ref, p250.read_ref(image_ref, p250.MIB)))
    gt = p250.document(p250.read_ref(plan["integer_gt"], p250.MIB))
    scorer.validate_gt(gt, ids, images)
    by_seed = {}
    for seed in p250.TRAIN_PROTOCOLS:
        rows = {}
        for mode in ARMS:
            key = seed, mode
            arm = arms[key]
            report = dict(mapped_rows=arm["mapped_rows"], unmapped_slot1_rows=18700-arm["mapped_rows"],
                          empty_image_predictions=arm["empty_image_predictions"])
            rows[mode] = p250.score_rows(p250.read_ref(refs[key], 64*p250.MIB), images, report)
        result = evaluate_pair(gt, rows["native"], rows["center"], helpers)
        prior = ranking["by_seed"][str(seed)]["prefix_counts"]
        require(result["prefix_rows_per_arm"] == prior["retained_per_arm"] and result["prefix_images"] == 187
                and result["exclusions"] == dict(empty=0, crowd=0, ignore=0)
                and result["strata"]["all"]["eligible_gt"] == 380, "Same accepted SSDD equal-prefix population required")
        result["direction"] = direction_gate(result["strata"]["all"])
        by_seed[str(seed)] = result
    names = ("native_downward", "center_absolute_signed_y_smaller", "native_up_gain_greater",
             "center_absolute_direction_asymmetry_smaller")
    return dict(status="P259_SSDD_EQUAL_NONEMPTY_FIXED_MATCH_GEOMETRY_COMPLETE_V1", evaluation_protocol=rank.PROTOCOL,
                images=187, annotations=380, by_seed=by_seed, packages=packages, DEV=True, official_TEST_opened=False,
                AP=False, training=False, model_forward_calls=0,
                all_seeds_support_direction=all(item["direction"][key] for item in by_seed.values() for key in names),
                primary_score_sha256=rank.ACCEPTED_SHA, closed_plan_sha256=rank.PLAN_SHA, equal_prefix_result_sha256=RANK_RESULT_SHA)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for flag in ("code", "geometry-source", "plan", "accepted-score", "rank-score", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    require(output.is_absolute() and output.parent.resolve(strict=True) == output.parent and not output.exists(),
            "Fresh canonical safe output required")
    report = diagnose(args)
    payload = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with output.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(status=report["status"], bytes=len(payload), sha256=sha256(payload).hexdigest())), flush=True)


if __name__ == "__main__":
    main()
