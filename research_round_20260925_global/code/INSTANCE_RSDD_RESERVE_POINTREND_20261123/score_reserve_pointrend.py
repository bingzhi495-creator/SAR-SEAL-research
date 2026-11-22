"""Evaluate the five frozen refinement policies on the 74-image Instance-RSDD reserve."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
from time import perf_counter
from types import ModuleType

import numpy as np
import shapely


PORTABLE = "45512c959eb914ae7713264ecff48eb437a2a52729810b34308abc52079148df"
TRANSFER = "45971f735e9e186ce20a38afb319a9a3b7dec815f3778ebe59099e0b5517ad88"
PROTOCOL = "4cdad2535417233eca5a959201ee8fabc0cc37a10d3e81baa03f594d0646fc82"
CAPTURE = "f43fb04035045b1e30b7ae06ced778e66fcc4cd3e34077f83ebff6096dacc71d"
HISTORIC_EDITS = "353362e0efabc0374eaaf0b9c27fdabbcad195402a1090230c4ef56ca2a10398"
POINTREND = "7401de08367ef31b909ea9c92daa27be02d4efb04460acbf96eab33648fdd7cc"
ACCEPTED = "INSTANCE_RSDD_RESERVE74_POINTREND_REPLAY_ACCEPTED_V1"
STATUS = "INSTANCE_RSDD_RESERVE74_POINTREND_SCORE_COMPLETE_V1"
N, C, P, B, L = ARMS = ("raw", "coarse_paste", "point_paste", "point_equal", "learned_common")
PAIRS = ((L, B), (P, L), (P, C), (P, N), (L, N), (B, N))
PRIMARY = L + " minus " + B
FIELDS = ("continuous_iou_points", "fp_per_truth_area_points", "fn_per_truth_area_points")


def require(test, message):
    if not test:
        raise ValueError(message)


def load_portable(code):
    path = code / "INSTANCE_RSDD_B5_REPRO_20261123/portable_eval.py"
    data = path.read_bytes()
    require(sha256(data).hexdigest() == PORTABLE, "Published B5 evaluator required")
    mod = ModuleType("reserve_portable")
    mod.__file__ = str(path)
    exec(compile(data, str(path), "exec"), mod.__dict__)
    return mod


def checked_blocks(args, portable, transfer, adapter, core, items, manifest_hash):
    portable.pin(args.capture / portable.RESULT, CAPTURE)
    capture, captures = portable.read_capture(args, items, manifest_hash)
    raw_receipt = (args.replay / portable.RESULT).read_bytes()
    receipt = json.loads(raw_receipt)
    common = ("manifest_sha256", "roster_sha256", "checkpoint_sha256", "predictions_sha256", "audit_sha256", "images")
    require(receipt["status"] == ACCEPTED and receipt["GT_opened"] is False and receipt["exact_independent_render"] is True
            and receipt["optimizer_updates"] == receipt["model_forwards"] == 0 and receipt["fit_weight_sha256"] == POINTREND
            and receipt["occupancy_edits_sha256"] == HISTORIC_EDITS and receipt["capture_result_sha256"] == CAPTURE
            and re.fullmatch("[a-f0-9]{64}", receipt["basis_result_sha256"])
            and all(receipt[key] == capture[key] for key in common), "Two matching label-free renders on the locked reserve required")
    edit_lines = portable.pin(args.replay / portable.EDITS, receipt["edits_sha256"]).splitlines()
    blocks, totals = {}, Counter()
    for item, (audit, pred), line in zip(items, captures, edit_lines, strict=True):
        row = json.loads(line)
        i = item["image_id"]
        require(set(row) == {"image_id", "changes"} and i == row["image_id"] and i not in blocks, "Replay image identity")
        adapter.native_masks(core, pred, audit, i)
        # Replay certified the archived L masks. The scorer independently checks actual L and B pixel exchanges.
        natural = {}
        for key, updates in row["changes"].items():
            require(updates and ((L in updates) == (B in updates)), "Both matched arms required at every positive budget")
            if L in updates:
                k = updates[L]["swaps"]
                area = audit["candidates"][int(key)-1]["mask_area"]
                require(type(k) is int and 1 <= k <= min(8, area // 10), "Original positive natural quota")
                natural[int(key)] = {"free_natural": updates[L]}
        changes, counts = transfer.output_changes(core.codec, core.checked_rle, audit, pred["coco"], natural, row["changes"])
        totals.update(counts)
        blocks[i] = dict(stock=pred["coco"], ranks=[c["position"] for c in audit["candidates"]], changes=changes,
                         rgb=item["rgb_sha256"], group=item["acquisition_group"], sensor=item["sensor"], file=item["source_image_name"])
    counts = Counter(receipt["counts"])
    require(len(blocks) == counts["images"] == 74 and sum(len(r["stock"]) for r in blocks.values()) == counts["mapped_candidates"]
            and totals["area_eligible"] == counts["area_eligible"] and totals["valid"] == counts["rendered"]
            and counts["point_evaluations"] == 3136 * counts["rendered"], "Complete candidate and render census")
    for arm in (C, P, B, L):
        require(totals[arm+"_edited_candidates"] == counts[arm+"_candidates"], "Edited candidate count differs")
        if arm in (B, L):
            require(totals[arm+"_adds"] == totals[arm+"_drops"] == counts[arm+"_swaps"], "Balanced edit totals differ")
        else:
            require(all(totals[arm+"_"+key] == counts[arm+"_"+key] for key in ("adds", "drops"))
                    and totals[arm+"_emptied_candidates"] == counts[arm+"_empty"], "Paste edit totals differ")
    return blocks, receipt, sha256(raw_receipt).hexdigest(), dict(totals)


def geometry(anchors, blocks, polygons, dates, core, reserve):
    groups = defaultdict(list)
    exposed = set()
    counts = dict(original_matches=len(anchors), eligible_matches=0, positive_budget_matches=0, invalid_polygons=0, area_excluded=0)
    for gid, (image, rank, prediction) in anchors.items():
        poly = polygons[gid]
        if not poly.is_valid or poly.area <= 0:
            counts["invalid_polygons"] += 1
            continue
        original = prediction["segmentation"]
        if not 32 <= int(core.codec.area(original)) < 1024:
            counts["area_excluded"] += 1
            continue
        counts["eligible_matches"] += 1
        block = blocks[image]
        updates = block["changes"].get(rank, {})
        if updates.get(L, {}).get("swaps", 0):
            counts["positive_budget_matches"] += 1
            exposed.add(block["group"])
        measures, cache = [], {}
        for arm in ARMS:
            rle = updates.get(arm, {}).get("rle", original)
            key = rle["counts"]
            if key not in cache:
                area = int(core.codec.area(rle))
                iou = core.geometric_iou(rle, poly, shapely)
                intersection = iou * (area + poly.area) / (1 + iou)
                cache[key] = [100*iou, 100*(area-intersection)/poly.area, 100*(poly.area-intersection)/poly.area]
            measures.append(cache[key])
        groups[block["group"]].append(measures)
    parents = sorted(groups)
    result = dict(counts=counts, represented_parents=len(parents), exposed_parents=len(exposed),
                  represented_sensor_date_blocks=len({dates[p] for p in parents}),
                  exposed_sensor_date_blocks=len({dates[p] for p in exposed}), group_equal_arms={}, paired_contrasts_points={})
    if parents:
        means = np.asarray([np.mean(groups[p], axis=0) for p in parents])
        result["group_equal_arms"] = {arm: dict(zip(FIELDS, average.tolist(), strict=True))
                                      for arm, average in zip(ARMS, means.mean(axis=0), strict=True)}
        for a, b in PAIRS:
            difference = means[:, ARMS.index(a)] - means[:, ARMS.index(b)]
            ci = reserve.bootstrap(difference, [dates[p] for p in parents])
            result["paired_contrasts_points"][a+" minus "+b] = {
                field: dict(delta=float(effect), sensor_date_interval_95=bounds)
                for field, effect, bounds in zip(FIELDS, difference.mean(axis=0), ci, strict=True)}
    return result


def score_view(ids, gt, blocks, dates, polygons, evaluator, core, reserve):
    subset = dict(gt, images=[r for r in gt["images"] if r["id"] in ids],
                  annotations=[r for r in gt["annotations"] if r["image_id"] in ids])
    scoped = {i: bloc for i, bloc in blocks.items() if i in ids}
    metrics = {}
    for arm in ARMS:
        values, matches = evaluator.score_one(subset, scoped, arm, anchors=arm == N)
        metrics[arm] = {k: float(v) if np.isfinite(v) and v >= 0 else None for k, v in values.items()}
        if arm == N:
            anchors = matches
    contrasts = {a+" minus "+b: {k: 100*(v-metrics[b][k]) if v is not None and metrics[b][k] is not None else None
                                 for k, v in metrics[a].items()} for a, b in PAIRS}
    geo = geometry(anchors, scoped, polygons, dates, core, reserve)
    ap = contrasts[PRIMARY]["AP"]
    iou = geo["paired_contrasts_points"].get(PRIMARY, {}).get(FIELDS[0])
    gates = dict(positive_mask_ap=ap is not None and ap > 0,
                 positive_sensor_date_iou_lower=iou is not None and iou["sensor_date_interval_95"][0] > 0,
                 positive_budget_20_matches_8_parents=geo["counts"]["positive_budget_matches"] >= 20 and geo["exposed_parents"] >= 8)
    return dict(images=len(ids), annotations=len(subset["annotations"]), parents=len({r["group"] for r in scoped.values()}),
                metrics_fraction=metrics, paired_contrasts_points=contrasts, geometry=geo, criteria=gates,
                within_recipe_ranking_criterion=all(gates.values()))


def run(args):
    portable = load_portable(args.code)
    reserve = portable.module(args.code / portable.RESERVE / "score_reserve.py", portable.SCORE_SOURCE)
    transfer = portable.module(args.code / "INSTANCE_RSDD_POINTREND_TRANSFER_20261122/score_transfer.py", TRANSFER)
    require(tuple(transfer.ARMS) == ARMS and shapely.__version__ == "2.0.2" and shapely.geos_version_string == "3.11.2",
            "Published arms and continuous geometry runtime required")
    old = transfer.predecessor(args.code)
    evaluator = old.module(args.code / "PSEG_RFDETR_FROZEN_SOURCE_HEAD_20261122/score.py", old.SCORE_SOURCE)
    adapter = old.module(args.code / portable.UEIKAP / "score_rf.py", old.GEOMETRY_SOURCE)
    core = adapter.load_geometry(args.code / "UEIKAP_RESERVED_HULL_FROZEN_20261122/score.py")
    native = portable.module(args.code / portable.NATIVE / "capture_instance_rsdd.py", portable.CAPTURE_SOURCE)
    items, manifest_hash = portable.load_intake(args, native)
    blocks, receipt, receipt_sha, edit_totals = checked_blocks(args, portable, transfer, adapter, core, items, manifest_hash)
    census = json.loads(portable.pin(args.code.parent / "results/INSTANCE_RSDD_PARENT_RESERVE_CENSUS_SAFE_20261123.json", reserve.CENSUS))
    require(Counter((v["group"], v["sensor"]) for v in blocks.values()) ==
            {(p["paper_scene"], p["sensor"]): p["images"] for p in census["per_parent"]}, "Census differs")
    # Author polygons are first opened after the externally closed replay and all independent RLE checks.
    gt, rekeyed = old.coco_annotation_ids(reserve.select_labels(portable.pin(args.author_train, reserve.TRAIN), blocks))
    require(len(gt["images"]) == 74 and len(gt["annotations"]) == 81, "Selected author annotation census")
    polys = {r["id"]: shapely.Polygon(np.asarray(r["segmentation"][0]).reshape(-1, 2)) for r in gt["annotations"]}
    dates = {p["paper_scene"]: (p["sensor"], p["calendar_date"]) for p in census["per_parent"]}
    dated = set(census["date_separated_sensitivity"]["parent_scene_ids"])
    full = set(blocks)
    gf = {i for i, b in blocks.items() if b["sensor"] == "GF-3"}
    sets = dict(full=full, date_separated={i for i, b in blocks.items() if b["group"] in dated},
                gf3=gf, terrasar_x=full-gf, full_deduplicated=old.representatives(blocks, full))
    require([len(ids) for ids in sets.values()] == [74, 42, 42, 32, 74], "Five locked views")
    cache, views = {}, {}
    for name, ids in sets.items():
        key = frozenset(ids)
        if key not in cache:
            cache[key] = score_view(ids, gt, blocks, dates, polys, evaluator, core, reserve)
        views[name] = cache[key]
    return dict(status=STATUS, protocol_sha256=PROTOCOL, views=views, author_train_sha256=reserve.TRAIN,
                census_sha256=reserve.CENSUS, roster_sha256=portable.ROSTER, manifest_sha256=manifest_hash,
                capture_result_sha256=CAPTURE, basis_result_sha256=receipt["basis_result_sha256"], replay_result_sha256=receipt_sha,
                edits_sha256=receipt["edits_sha256"], occupancy_edits_sha256=HISTORIC_EDITS, fit_weight_sha256=POINTREND,
                checkpoint_sha256=portable.CHECKPOINT, occupancy_head_sha256=portable.POINT_HEAD,
                predictions_sha256=receipt["predictions_sha256"], audit_sha256=receipt["audit_sha256"],
                evaluator_annotation_ids_rekeyed=rekeyed, non_simple_polygons_retained_in_ap=sum(not p.is_valid for p in polys.values()),
                detector_forwards=0, optimizer_updates=0, counts=receipt["counts"], realized_edits=edit_totals,
                full_primary_criterion=views["full"]["within_recipe_ranking_criterion"],
                date_separated_secondary_criterion=views["date_separated"]["within_recipe_ranking_criterion"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("code", "intake", "roster", "capture", "replay", "author-train", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "New score output required")
    start = perf_counter()
    result = run(args)
    result.update(elapsed_seconds=perf_counter()-start, worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    raw = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
    args.output.mkdir()
    (args.output / "RESULT_SAFE.json").write_bytes(raw)
    print(json.dumps(dict(status=STATUS, result_sha256=sha256(raw).hexdigest())), flush=True)


if __name__ == "__main__":
    main()
