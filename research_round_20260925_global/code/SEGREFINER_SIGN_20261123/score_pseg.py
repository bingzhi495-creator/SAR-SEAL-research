"""Decompose closed stock SegRefiner binary writes on the original PSeg views."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from types import ModuleType

import numpy as np
import shapely


STOCK_WORKER = "c4fcc7aa4035a34d4e42fd8578781a754a08f5687792fcaecb82fce06d564129"
PROTOCOL_NAME = "SEGREFINER_SIGN_PRELOCK_20261123.md"
PROTOCOL_SHA256 = "3e5918d16dad9f6cf0642e4c3222d16673f53a56835045de3b2687ab9776a7b1"
STATUS = "PSEG232_SEGREFINER_SIGN_SCORE_COMPLETE"
RAW, ADD, DROP, BOTH = "raw", "segrefiner_add_only", "segrefiner_delete_only", "segrefiner_stock"
ARMS = (RAW, ADD, DROP, BOTH)
PAIRS = ((ADD, RAW), (DROP, RAW), (BOTH, ADD), (BOTH, DROP), (BOTH, RAW))
ENERGIES = ("add_truth_pct", "add_background_pct", "delete_truth_pct", "delete_background_pct",
            "add_truth_minus_background_pct", "delete_truth_minus_background_pct")


def load_stock(path):
    content = path.read_bytes()
    if sha256(content).hexdigest() != STOCK_WORKER:
        raise ValueError("Exact stock PSeg worker required")
    stock = ModuleType(path.stem)
    stock.__file__ = str(path)
    exec(compile(content, str(path), "exec"), stock.__dict__)
    return stock


def extend_masks(source, records):
    counters = {}
    for name, record in records.items():
        counts = Counter(added_pixels=0, removed_pixels=0, add_arm_changed=0, delete_arm_changed=0)
        for key, arms in record["edits"].items():
            if BOTH not in arms:
                continue
            ordinal = int(key) - 1
            original = source.decoded(record["native"][ordinal]["segmentation"], record["size"])
            stock = source.decoded(arms[BOTH]["rle"], record["size"])
            add = stock & ~original
            drop = original & ~stock
            union, intersection = original | add, original & ~drop
            source.require(np.array_equal(union & ~drop, stock) and not (add & drop).any(), "Invalid signed decomposition")
            arms[ADD], arms[DROP] = {"rle": source.encode(union)}, {"rle": source.encode(intersection)}
            counts.update(added_pixels=int(add.sum()), removed_pixels=int(drop.sum()),
                          add_arm_changed=int(add.any()), delete_arm_changed=int(drop.any()))
        counters[name] = counts
    return counters


def energies_for_masks(source, base, modules, truth, masks, measure, size):
    original = source.decoded(masks[RAW], size)
    stock = source.decoded(masks[BOTH], size)
    values = []
    for changed in (stock & ~original, original & ~stock):
        area = int(changed.sum())
        if area:
            _, false, missing = base.continuous(source.encode(changed), area, truth, modules)
            values.extend((100. - missing, false))
        else:
            values.extend((0., 0.))
    at, af, dt, df = values
    raw, add, drop, both = measure
    source.require(np.allclose((add[3]-raw[3], add[1]-raw[1], raw[3]-drop[3], raw[1]-drop[1]), values, atol=1e-8, rtol=0.)
                   and np.allclose((both[3]-raw[3], both[1]-raw[1]), (at-dt, af-df), atol=1e-8, rtol=0.),
                   "Signed overlap/false area must conserve the continuous fixed-hull measures")
    return np.asarray([at, af, dt, df, at-af, dt-df])


def summarize_cluster_fields(rows, fields):
    groups = defaultdict(list)
    for group, energies in rows:
        groups[group].append(energies)
    means = np.asarray([np.mean(groups[group], axis=0) for group in sorted(groups)])
    sample = np.random.Generator(np.random.PCG64(5217)).integers(len(means), size=(4096, len(means)))
    interval = np.percentile(means[sample].mean(axis=1), [2.5, 97.5], axis=0)
    return {key: dict(value=float(means[:, i].mean()), percentile95=interval[:, i].tolist()) for i, key in enumerate(fields)}


def geometry_for_view(source, stock, geom, base, modules, truth, records, anchors, cache):
    full, exposed, changed = [], [], Counter()
    sign_counts = {"added": Counter(), "deleted": Counter()}
    for aid, (iid, rank) in anchors.items():
        row = records[iid]
        area = row["areas"][rank-1]
        if not 32 <= area < 1024:
            continue
        masks = {arm: stock.mask_at(row, rank, arm) for arm in ARMS}
        if aid not in cache:
            measured = stock.measure_candidate(source, base, modules, truth[aid][0], masks)
            energy = energies_for_masks(source, base, modules, truth[aid][0], masks, measured, row["size"])
            cache[aid] = (measured, energy)
        measured, energy = cache[aid]
        pair = (row["rgb"], measured, energy)
        full.append(pair)
        if area >= stock.GATE:
            exposed.append(pair)
            if masks[ADD] != masks[RAW]:
                sign_counts["added"][row["rgb"]] += 1
            deletion_applied = masks[DROP] != masks[RAW]
            source.require(deletion_applied == (masks[BOTH] != masks[ADD]), "Both deletion contrasts must use the same changes")
            if deletion_applied:
                sign_counts["deleted"][row["rgb"]] += 1
            if masks[BOTH] != masks[RAW]:
                changed[row["rgb"]] += 1
    geom.ARMS, geom.PAIRS = ARMS, PAIRS
    geom.FIELDS = ("continuous_iou_points", "fp_pixels_per_gt_area_pct", "fn_pixels_per_gt_area_pct", "intersection_per_gt_area_pct")
    all_geometry, exposed_geometry, energy, interaction = {}, {}, {}, {}
    for name, rows in (("full", full), ("gate_exposed", exposed)):
        g = geom.summarize([(group, measured) for group, measured, _ in rows])
        energy[name] = summarize_cluster_fields([(group, values) for group, _, values in rows], ENERGIES)
        interaction[name] = summarize_cluster_fields(
            [(group, (m[3]-m[1])-(m[2]-m[0])) for group, m, _ in rows], geom.FIELDS)
        if name == "full":
            all_geometry = g
        else:
            exposed_geometry = g
    counts = {name: dict(different_hulls=sum(group.values()), different_decoded_groups=len(group))
              for name, group in {**sign_counts, "any_stock_change": changed}.items()}
    return all_geometry, exposed_geometry, energy, counts, interaction


def primary_deletion(exposed, differing):
    counts = differing["deleted"]
    enough = counts["different_hulls"] >= 20 and counts["different_decoded_groups"] >= 8
    contexts = {}
    for left, right in ((DROP, RAW), (BOTH, ADD)):
        metric = exposed["pairwise"][left + " minus " + right]["continuous_iou_points"]
        negative = metric["percentile95"][1] < 0
        contexts[left + "_minus_" + right] = dict(continuous_iou_points=metric, **counts, exposure_supported=enough,
                                                    strictly_negative_upper_bound=negative, supported_negative=bool(enough and negative))
    return dict(minimum_differing_hulls=20, minimum_differing_decoded_groups=8, contexts=contexts,
                both_contexts_supported=all(c["supported_negative"] for c in contexts.values()))


def evaluate(args):
    start = perf_counter()
    stock = load_stock(args.stock_source)
    exposure = stock.exposure_module()
    source = exposure.load_source(args.factorial_source)
    protocol = source.digest(args.protocol.read_bytes())
    source.require(args.protocol.name == PROTOCOL_NAME and protocol == args.protocol_sha256 == PROTOCOL_SHA256 and not args.output.exists()
                   and shapely.__version__ == "2.0.2" and shapely.geos_version_string == "3.11.2",
                   "Exact published protocol, historical geometry engine and fresh output required")
    established = json.loads(source.pin(args.exposure_safe, stock.EXPOSURE_SAFE))
    source.require(established["worker_sha256"] == stock.EXPOSURE_WORKER and established["status"] == exposure.STATUS
                   and established["protocol_sha256"] == exposure.PROTOCOL_SHA256, "Original exact area exposure required")
    old, old_result, old_terminal = source.observed(args.stock_score, args.stock_score_terminal, "score", stock.STATUS, STOCK_WORKER)
    _, scorer, geom, _, capture, roster, audits, original, blocks = source.historical(args)
    records_by_name = exposure.native_records(source, blocks, roster, audits, original)
    infer, result, terminal = stock.verify_receipt(source, args, capture, exposure.PROTOCOL_SHA256)
    source.require(old["inference_result_sha256"] == result and old["inference_terminal_sha256"] == terminal
                   and old["protocol_sha256"] == exposure.PROTOCOL_SHA256 and old["inference_verified_before_labels"] is True
                   and old["historical_COCO_and_geometry_parity"] is True and old["edits_sha256"] == infer["edits_sha256"]
                   and old["exposure_score_sha256"] == stock.EXPOSURE_SAFE and old["author_labels_sha256"] == scorer.GT_SHA,
                   "Stock score must bind the same observed inference, input census and author reference")
    data = source.pin(args.inference / "EDITS_PRIVATE.jsonl", infer["edits_sha256"])
    totals, _ = stock.qualify_edits(source, records_by_name, roster, blocks, map(json.loads, data.splitlines()))
    source.require(totals == infer["counts"] == old["counts"] and (totals["images"], totals["native_candidates"], totals["area_admitted"])
                   == (232, 23200, 5067), "Previous stock write count and immutable gate census required")
    signed_counts = extend_masks(source, records_by_name)
    print(json.dumps(dict(phase="verified_stock_and_signed_writeback_before_labels", **totals)), flush=True)

    gt = json.loads(source.pin(args.author_labels, scorer.GT_SHA))
    source.require(len(gt["images"]) == 232 and len(gt["annotations"]) == 546
                   and {im["file_name"] for im in gt["images"]} == set(records_by_name)
                   and source.digest(("\n".join(scorer.EXCLUDED) + "\n").encode("ascii")) == scorer.EXCLUSION_SHA,
                   "Same author polygons and prior split required")
    records = {im["id"]: records_by_name[im["file_name"]] for im in gt["images"]}
    source.require(len(records) == 232 and all(records[im["id"]]["size"] == [im["height"], im["width"]] for im in gt["images"]),
                   "Original native canvas and author identity required")
    base = geom.load(args.code / "DFINE_SEG_EXTERNAL_BASELINE_FIT32_E24_20261112/e24/dev705_free_endpoint/score_endpoint.py", geom.OLD)
    continuous = geom.load(args.code / "MASK_RTDETR_DEV705_CONTINUOUS_AP_20261112/continuous_endpoint.py", geom.GEOM)
    modules = continuous.geometry_modules(args.code)
    truth, neutral, _ = modules.geometry.polygon_support(gt, modules.vector)
    source.require(len(truth) == 546 and not neutral, "Floating author polygons must reproduce prior eligibility")
    stock.ARMS = ARMS
    cache, views, full_anchors = {}, {}, None
    for view, expected in exposure.VIEWS.items():
        native_previous = old["views"][view]
        names = set(records_by_name) if view == "official" else set(records_by_name) - set(scorer.EXCLUDED)
        ids = sorted(iid for iid, row in records.items() if row["name"] in names)
        included = set(ids)
        subset = dict(gt, images=[im for im in gt["images"] if im["id"] in included],
                      annotations=[ann for ann in gt["annotations"] if ann["image_id"] in included])
        original_order = [im["id"] for im in subset["images"]]
        metrics = {arm: scorer.evaluate(subset, [
            dict(image_id=iid, category_id=0, score=det["score"], segmentation=stock.mask_at(records[iid], rank, arm))
            for iid in original_order for rank, det in enumerate(records[iid]["native"], 1)]) for arm in ARMS}
        source.require(all(metrics[arm] == native_previous["metrics_fraction"][arm] for arm in (RAW, BOTH)),
                       "Published original and stock COCO vectors differ")
        anchors, _ = geom.evaluate_raw(gt, records, ids, metrics[RAW])
        if full_anchors is None:
            full_anchors = anchors
        else:
            source.require(anchors == {aid: value for aid, value in full_anchors.items() if value[0] in included},
                           "Sensitivity must retain original native anchors")
        geometry, exposed, energies, differing, geom_interaction = geometry_for_view(source, stock, geom, base, modules, truth, records, anchors, cache)
        observed = (len(ids), len(subset["annotations"]), len(anchors), geometry["eligible_pairs"], geometry["decoded_image_groups_with_pairs"])
        expected_exposure = established["views"][view]["native_fixed_matches_iou50"]
        source.require(observed == expected and exposed["eligible_pairs"] == expected_exposure["stock_admitted_within_research_area"]
                       and exposed["decoded_image_groups_with_pairs"] == expected_exposure["stock_admitted_within_research_area_decoded_groups"],
                       "Prior full and gate-exposed anchor populations differ")
        pair_name = BOTH + " minus " + RAW
        for result_geometry, key in ((geometry, "geometry"), (exposed, "gate_exposed_geometry")):
            previous = native_previous[key]
            source.require(all(result_geometry["arms"][arm] == previous["arms"][arm] for arm in (RAW, BOTH))
                           and result_geometry["pairwise"][pair_name] == previous["pairwise"][pair_name],
                           "Published geometry or native-stock paired interval differs")
        coverage = {arm: stock.calculate_coverage(source, geom, gt, ids, records, anchors, metrics, arm) for arm in ARMS}
        source.require(all(coverage[arm] == native_previous["coverage"][arm] for arm in (RAW, BOTH)),
                       "Published native-stock .50 associations differ")
        contrasts = {}
        for left, right in PAIRS:
            key = left + " minus " + right
            contrasts[left + "_minus_" + right] = dict(COCO_points=source.difference(metrics[left], metrics[right]),
                full_fixed_geometry=geometry["pairwise"][key], gate_exposed_geometry=exposed["pairwise"][key])
        after = contrasts[BOTH + "_minus_" + ADD]["COCO_points"]
        before = contrasts[DROP + "_minus_" + RAW]["COCO_points"]
        interaction = dict(definition="(stock - add_only) - (delete_only - raw)",
                           COCO_points={key: after[key]-before[key] if after[key] is not None and before[key] is not None else None for key in after},
                           geometry=geom_interaction)
        counts = sum((signed_counts[name] for name in names), Counter())
        views[view] = dict(images=len(ids), annotations=len(subset["annotations"]), candidates=100*len(ids),
                           metrics_fraction=metrics, geometry=geometry, gate_exposed_geometry=exposed, coverage=coverage,
                           signed_energies=energies, gate_exposed_changes=differing, counts=dict(counts), contrasts=contrasts,
                           descriptive_interaction=interaction, primary_gate_exposed_deletion=primary_deletion(exposed, differing))
        print(json.dumps(dict(phase="scored_view", view=view, images=len(ids))), flush=True)
    return dict(status=STATUS, phase="score", worker_sha256=source.digest(Path(__file__).read_bytes()), protocol_sha256=protocol,
                stock_score_sha256=old_result, stock_score_terminal_sha256=old_terminal, stock_worker_sha256=STOCK_WORKER,
                inference_result_sha256=result, inference_terminal_sha256=terminal, inference_worker_sha256=args.inference_worker_sha256,
                stock_native_parity=True, verified_signed_writes_before_labels=True, counts=totals,
                edits_sha256=infer["edits_sha256"], author_labels_sha256=scorer.GT_SHA, detector_forwards=0, segrefiner_forwards=0,
                optimizer_updates=0, bootstrap_draws=4096, bootstrap_seed=5217, primary_view="official",
                sensitivity_view="exclude_exact_fit_overlap", views=views, score_seconds=perf_counter()-start)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("stock-source", "stock-score", "stock-score-terminal", "factorial-source", "code", "owner", "capture", "classical-root",
                 "exposure-safe", "inference", "inference-terminal", "author-labels", "protocol", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("inference-worker-sha256", "runtime-sha256", "checkpoint-sha256", "protocol-sha256"):
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
