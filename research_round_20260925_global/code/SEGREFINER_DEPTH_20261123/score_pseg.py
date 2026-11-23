"""Score same-count SegRefiner deletion relocation after the closed PSeg studies."""
import argparse
from collections import Counter
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import shapely


SIGNED_WORKER = "a658416573b591d657e49e7deb5db1aaeb754aeb471d567c4077fcad2188d16c"
PROTOCOL_NAME = "SEGREFINER_DEPTH_PRELOCK_20261123.md"
PROTOCOL_SHA256 = "2c7a106bd8c70de20902a76b11d98ae0136b8708441cd150cf1829600d2163af"
STATUS = "PSEG232_SEGREFINER_DEPTH_SCORE_COMPLETE"
RAW, STOCK = "raw", "segrefiner_stock"
PERIPHERY, PROXIMITY = "segrefiner_depth_periphery", "segrefiner_depth_proximity"
ARMS = (RAW, STOCK, PERIPHERY, PROXIMITY)
PAIRS = ((PERIPHERY, STOCK), (PERIPHERY, PROXIMITY), (PROXIMITY, STOCK), (STOCK, RAW))
GEOMETRY_FIELDS = ("continuous_iou_points", "fp_pixels_per_gt_area_pct", "fn_pixels_per_gt_area_pct", "intersection_per_gt_area_pct")


def load_module(path, expected):
    content = path.read_bytes()
    if sha256(content).hexdigest() != expected:
        raise ValueError(f"Pinned source mismatch for {path.name}")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(content, str(path), "exec"), module.__dict__)
    return module


def extended_mask_invariants(source, raw, stock, periphery, proximity):
    source.require(all(isinstance(mask, np.ndarray) and mask.dtype == np.bool_ and mask.shape == raw.shape
                       for mask in (periphery, proximity)), "Expected bool variant masks on the original canvas")
    added, deleted = stock & ~raw, raw & ~stock
    source.require(all(np.array_equal(mask & ~raw, added) for mask in (periphery, proximity)),
                   "All variants must preserve exact stock addition sites")
    peripheral_delete, proximal_delete = raw & ~periphery, raw & ~proximity
    source.require(int(peripheral_delete.sum()) == int(proximal_delete.sum()) == int(deleted.sum())
                   and int(periphery.sum()) == int(proximity.sum()) == int(stock.sum()), "Observed signed count or final mask area changed")
    old_peripheral, old_proximal = peripheral_delete & deleted, proximal_delete & deleted
    source.require(np.array_equal(old_peripheral, old_proximal), "Proximity control must retain the same old deletion sites")
    return dict(adds=int(added.sum()), deletes=int(deleted.sum()), retained_original_deletes=int(old_peripheral.sum()),
                changed_deletes=int(deleted.sum()-old_peripheral.sum()),
                periphery_differs_from_stock=int(not np.array_equal(periphery, stock)),
                proximity_differs_from_stock=int(not np.array_equal(proximity, stock)),
                periphery_differs_from_proximity=int(not np.array_equal(periphery, proximity)))


def extend_variants(source, depth, records):
    by_image, by_image_detail = {}, {}
    digest = sha256()
    for name, record in records.items():
        counts, detail = Counter(), Counter()
        for key, arms in record["edits"].items():
            if STOCK not in arms:
                continue
            i = int(key) - 1
            raw = source.decoded(record["native"][i]["segmentation"], record["size"])
            stock = source.decoded(arms[STOCK]["rle"], record["size"])
            periphery, proximity, stats = depth.deletion_location_masks(raw, stock)
            observed = extended_mask_invariants(source, raw, stock, periphery, proximity)
            counts.update(gate_candidates=1, **observed)
            require_stats = dict(native_pixels=int(raw.sum()), stock_pixels=int(stock.sum()), added_pixels=observed["adds"],
                                 deleted_pixels=observed["deletes"], relocated_deletions=observed["changed_deletes"])
            source.require(stats == require_stats and all(isinstance(v, (int, np.integer)) for v in stats.values()),
                           "Shared variant diagnostics must reproduce independent pixel counts")
            detail.update({key: int(value) for key, value in stats.items()})
            p_rle, q_rle = source.encode(periphery), source.encode(proximity)
            source.require(np.array_equal(source.decoded(p_rle, record["size"]), periphery)
                           and np.array_equal(source.decoded(q_rle, record["size"]), proximity), "Canonical signed writeback replay differs")
            arms[PERIPHERY], arms[PROXIMITY] = {"rle": p_rle}, {"rle": q_rle}
            digest.update((json.dumps([name, key, p_rle, q_rle], separators=(",", ":"), sort_keys=True) + "\n").encode("ascii"))
        by_image[name], by_image_detail[name] = counts, detail
    return by_image, by_image_detail, digest.hexdigest()


def geometry_views(stock, source, geom, base, modules, truth, records, anchors, cache):
    full, exposed = [], []
    differences = {pair: Counter() for pair in PAIRS}
    for aid, (iid, rank) in anchors.items():
        record = records[iid]
        area = record["areas"][rank-1]
        if not 32 <= area < 1024:
            continue
        group = record["rgb"]
        masks = {arm: stock.mask_at(record, rank, arm) for arm in ARMS}
        if aid not in cache:
            cache[aid] = stock.measure_candidate(source, base, modules, truth[aid][0], masks)
        row = (group, cache[aid])
        full.append(row)
        if area >= stock.GATE:
            exposed.append(row)
            for left, right in PAIRS:
                if masks[left] != masks[right]:
                    differences[left, right][group] += 1
    geom.ARMS, geom.PAIRS, geom.FIELDS = ARMS, PAIRS, GEOMETRY_FIELDS
    counts = {left + "_minus_" + right: dict(different_hulls=sum(groups.values()), different_decoded_groups=len(groups))
              for (left, right), groups in differences.items()}
    return geom.summarize(full), geom.summarize(exposed), counts


def primary_gate(ap, exposed_iou, exposure):
    lower, upper = exposed_iou["percentile95"]
    enough = exposure["different_hulls"] >= 20 and exposure["different_decoded_groups"] >= 8
    return dict(minimum_different_hulls=20, minimum_different_decoded_groups=8, minimum_AP_points=.30,
                exposure_supported=enough, AP_advantage_supported=ap >= .30, geometry_advantage_supported=lower > 0,
                supported_advantage=bool(enough and ap >= .30 and lower > 0),
                supported_loss=bool(enough and ap <= -.30 and upper < 0))


def evaluate(args):
    start = perf_counter()
    signed = load_module(args.signed_source, SIGNED_WORKER)
    stock = signed.load_stock(args.stock_source)
    exposure = stock.exposure_module()
    source = exposure.load_source(args.factorial_source)
    protocol = source.digest(args.protocol.read_bytes())
    source.require(args.protocol.name == PROTOCOL_NAME and protocol == args.protocol_sha256 == PROTOCOL_SHA256 and not args.output.exists()
                   and shapely.__version__ == "2.0.2" and shapely.geos_version_string == "3.11.2",
                   "Expected original geometry runtime, published protocol and fresh output")
    depth = load_module(Path(__file__).with_name("depth_priority.py"), args.depth_priority_sha256)
    expected_area = json.loads(source.pin(args.exposure_safe, stock.EXPOSURE_SAFE))
    source.require(expected_area["worker_sha256"] == stock.EXPOSURE_WORKER and expected_area["status"] == exposure.STATUS
                   and expected_area["protocol_sha256"] == exposure.PROTOCOL_SHA256, "Exact historical area exposure required")
    previous, prior_sha, prior_terminal = source.observed(args.stock_score, args.stock_score_terminal, "score", stock.STATUS, signed.STOCK_WORKER)
    signed_score, signed_sha, signed_terminal = source.observed(args.signed_score, args.signed_score_terminal, "score", signed.STATUS, SIGNED_WORKER)
    source.require(signed_score["protocol_sha256"] == signed.PROTOCOL_SHA256 and signed_score["stock_score_sha256"] == prior_sha
                   and signed_score["stock_score_terminal_sha256"] == prior_terminal and signed_score["stock_native_parity"] is True
                   and signed_score["verified_signed_writes_before_labels"] is True, "Closed signed study must bind exact stock score")
    _, scorer, geom, _, capture, roster, audits, original, blocks = source.historical(args)
    records_by_name = exposure.native_records(source, blocks, roster, audits, original)
    infer, infer_sha, infer_terminal = stock.verify_receipt(source, args, capture, exposure.PROTOCOL_SHA256)
    source.require(all(value["inference_result_sha256"] == infer_sha and value["inference_terminal_sha256"] == infer_terminal
                       and value["edits_sha256"] == infer["edits_sha256"] and value["author_labels_sha256"] == scorer.GT_SHA
                       for value in (previous, signed_score)) and previous["protocol_sha256"] == exposure.PROTOCOL_SHA256
                   and previous["historical_COCO_and_geometry_parity"] is True and previous["inference_verified_before_labels"] is True,
                   "Both previous studies must bind the same inference, native ledger and author labels")
    data = source.pin(args.inference / "EDITS_PRIVATE.jsonl", infer["edits_sha256"])
    totals, _ = stock.qualify_edits(source, records_by_name, roster, blocks, map(json.loads, data.splitlines()))
    source.require(totals == infer["counts"] == previous["counts"] == signed_score["counts"]
                   and (totals["images"], totals["native_candidates"], totals["area_admitted"]) == (232, 23200, 5067), "Original stock ledger census differs")
    masks, diagnostics, digest = extend_variants(source, depth, records_by_name)
    print(json.dumps(dict(phase="verified_depth_writeback_before_labels", **totals)), flush=True)

    gt = json.loads(source.pin(args.author_labels, scorer.GT_SHA))
    source.require(len(gt["images"]) == 232 and len(gt["annotations"]) == 546
                   and {im["file_name"] for im in gt["images"]} == set(records_by_name)
                   and source.digest(("\n".join(scorer.EXCLUDED) + "\n").encode("ascii")) == scorer.EXCLUSION_SHA,
                   "Published author polygons and exact sensitivity roster required")
    records = {im["id"]: records_by_name[im["file_name"]] for im in gt["images"]}
    source.require(len(records) == 232 and all(records[im["id"]]["size"] == [im["height"], im["width"]] for im in gt["images"]),
                   "Original native canvas and author image identities required")
    base = geom.load(args.code / "DFINE_SEG_EXTERNAL_BASELINE_FIT32_E24_20261112/e24/dev705_free_endpoint/score_endpoint.py", geom.OLD)
    continuous = geom.load(args.code / "MASK_RTDETR_DEV705_CONTINUOUS_AP_20261112/continuous_endpoint.py", geom.GEOM)
    modules = continuous.geometry_modules(args.code)
    truth, neutral, _ = modules.geometry.polygon_support(gt, modules.vector)
    source.require(len(truth) == 546 and not neutral, "Same original continuous author support required")
    stock.ARMS = ARMS
    cache, views, full_anchors = {}, {}, None
    for view, expected in exposure.VIEWS.items():
        names = set(records_by_name) if view == "official" else set(records_by_name) - set(scorer.EXCLUDED)
        ids = sorted(iid for iid, row in records.items() if row["name"] in names)
        id_set = set(ids)
        subset = dict(gt, images=[im for im in gt["images"] if im["id"] in id_set],
                      annotations=[ann for ann in gt["annotations"] if ann["image_id"] in id_set])
        original_order = [im["id"] for im in subset["images"]]
        metrics = {arm: scorer.evaluate(subset, [
            dict(image_id=iid, category_id=0, score=det["score"], segmentation=stock.mask_at(records[iid], rank, arm))
            for iid in original_order for rank, det in enumerate(records[iid]["native"], 1)]) for arm in ARMS}
        anchors, _ = geom.evaluate_raw(gt, records, ids, metrics[RAW])
        if full_anchors is None:
            full_anchors = anchors
        else:
            source.require(anchors == {aid: pair for aid, pair in full_anchors.items() if pair[0] in id_set}, "Original anchors must survive the exclusion view")
        geometry, exposed, diff = geometry_views(stock, source, geom, base, modules, truth, records, anchors, cache)
        old_exposure = expected_area["views"][view]["native_fixed_matches_iou50"]
        observed = (len(ids), len(subset["annotations"]), len(anchors), geometry["eligible_pairs"], geometry["decoded_image_groups_with_pairs"])
        source.require(observed == expected and exposed["eligible_pairs"] == old_exposure["stock_admitted_within_research_area"]
                       and exposed["decoded_image_groups_with_pairs"] == old_exposure["stock_admitted_within_research_area_decoded_groups"],
                       "Original native and gate-exposed geometry sets must reproduce")
        coverage = {arm: stock.calculate_coverage(source, geom, gt, ids, records, anchors, metrics, arm) for arm in ARMS}
        for old in (previous["views"][view], signed_score["views"][view]):
            source.require(all(metrics[arm] == old["metrics_fraction"][arm] and coverage[arm] == old["coverage"][arm] for arm in (RAW, STOCK)),
                           "Original and stock COCO or .50 coverage parity failed")
            for g, key in ((geometry, "geometry"), (exposed, "gate_exposed_geometry")):
                pair = STOCK + " minus " + RAW
                source.require(all(g["arms"][arm] == old[key]["arms"][arm] for arm in (RAW, STOCK))
                               and g["pairwise"][pair] == old[key]["pairwise"][pair], "Original or stock fixed polygon parity failed")
        contrasts = {}
        for left, right in PAIRS:
            key, label = left + "_minus_" + right, left + " minus " + right
            coco = source.difference(metrics[left], metrics[right])
            exposure_counts, exposed_effect = diff[key], exposed["pairwise"][label]
            contrasts[key] = dict(COCO_points=coco, full_fixed_geometry=geometry["pairwise"][label],
                                  gate_exposed_geometry=exposed_effect, gate_exposed_changes=exposure_counts,
                                  criteria=primary_gate(coco["AP"], exposed_effect["continuous_iou_points"], exposure_counts))
        counts = sum((masks[name] for name in names), Counter())
        detail = sum((diagnostics[name] for name in names), Counter())
        old_signed = signed_score["views"][view]["counts"]
        source.require(counts["adds"] == old_signed.get("added_pixels", 0) and counts["deletes"] == old_signed.get("removed_pixels", 0)
                       and counts["gate_candidates"] == expected_area["views"][view]["original_candidates"]["stock_area_admitted"],
                       "Realized signed counts must reproduce the closed signed-write study")
        views[view] = dict(images=len(ids), annotations=len(subset["annotations"]), candidates=100*len(ids), metrics_fraction=metrics,
                           geometry=geometry, gate_exposed_geometry=exposed, coverage=coverage, counts=dict(counts),
                           depth_diagnostics=dict(detail), contrasts=contrasts,
                           both_primary_gates_passed=all(contrasts[PERIPHERY + "_minus_" + right]["criteria"]["supported_advantage"]
                                                        for right in (STOCK, PROXIMITY)))
        print(json.dumps(dict(phase="scored_view", view=view, images=len(ids))), flush=True)
    return dict(status=STATUS, phase="score", worker_sha256=source.digest(Path(__file__).read_bytes()), protocol_sha256=protocol,
                depth_priority_sha256=args.depth_priority_sha256, signed_score_sha256=signed_sha, signed_success_terminal_sha256=signed_terminal,
                stock_score_sha256=prior_sha, stock_score_terminal_sha256=prior_terminal, inference_result_sha256=infer_sha,
                inference_terminal_sha256=infer_terminal, stock_native_parity=True, verified_relocated_writes_before_labels=True,
                native_gate_pixels=stock.GATE, source_edits_sha256=infer["edits_sha256"], counterfactual_mask_digest=digest,
                detector_forwards=0, segrefiner_forwards=0, optimizer_updates=0, bootstrap_draws=4096, bootstrap_seed=5217,
                primary_view="official", sensitivity_view="exclude_exact_fit_overlap", views=views, score_seconds=perf_counter()-start)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("signed-source", "signed-score", "signed-score-terminal", "stock-source", "stock-score", "stock-score-terminal",
                 "factorial-source", "code", "owner", "capture", "classical-root", "exposure-safe", "inference", "inference-terminal",
                 "author-labels", "protocol", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("inference-worker-sha256", "runtime-sha256", "checkpoint-sha256", "protocol-sha256", "depth-priority-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--original-readback", type=Path)
    args = parser.parse_args()
    result = evaluate(args)
    raw = (json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    args.output.mkdir()
    with (args.output / "RESULT_SAFE.json").open("xb") as stream:
        if stream.write(raw) != len(raw):
            raise OSError("Incomplete safe score receipt")
    print(json.dumps(dict(status=STATUS, result_sha256=sha256(raw).hexdigest())), flush=True)


if __name__ == "__main__":
    main()
