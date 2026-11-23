"""Score frozen learned and classical endpoint selections at the same dynamic edit count on the 74-image reserve."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from types import ModuleType

import numpy as np
import shapely


SEEDS = (5217, 5218, 5219)
LEARNED = tuple(f"{seed}/learned" for seed in SEEDS)
COMPARATORS = ("dynamic", "static")
BASELINES = ("raw", *COMPARATORS)
EDITED = (*COMPARATORS, *LEARNED)
ARMS = ("raw", *EDITED)
HISTORY_SHA = "a0ff8233617770fe95b828fd72369590855ce2314ee84006ebe93857cac1a07d"
INPUTS_SHA = "2c3b84f852bb4f8234d935347708a2e71eba4466d0eb111471bd254aa2f45e58"
REFERENCE = "INSTANCE_RSDD_PARENT_RESERVE_ACTUAL_SAFE_20261123.json"
REFERENCE_SHA = "737a5de1e28a2a9729c69d8c32a5a1dfb667affab989c81561c957a901bb3b2e"
STATUS = "INSTANCE_RSDD_CLASSICAL_FAIR_MATCH_SCORE_COMPLETE"
PROTOCOL = "d826e08a75572cb643a5b4880f92529dadb27a78fe14696451ca7a3ae6738693"
FIELDS = ("continuous_iou_points", "fp_per_truth_area_points", "fn_per_truth_area_points",
          "intersection_per_truth_area_points", "intersection_px2")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return sha256(data).hexdigest()


def source(path, expected):
    raw = path.read_bytes()
    require(digest(raw) == expected, "Frozen source changed: " + path.name)
    module = ModuleType(path.stem)
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def closed(directory, terminal_file, phase, status, args):
    body = (directory / "RESULT_SAFE.json").read_bytes()
    raw_terminal = terminal_file.read_bytes()
    result, terminal = json.loads(body), json.loads(raw_terminal)
    require(type(terminal["actual_exit"]) is int and terminal["actual_exit"] == 0
            and terminal["output_footer_verified"] is True and terminal["phase"] == phase
            and terminal["result_sha256"] == digest(body)
            and terminal["worker_sha256"] == result["worker_sha256"]
            and len(result["worker_sha256"]) == 64, "Successful observed footer required for " + phase)
    require(result["phase"] == phase and result["status"] == status and result["cohort"] == args.cohort
            and result["protocol_sha256"] == args.protocol_sha256 and result["target_labels_opened"] is False
            and result["model_forwards"] == result["detector_forwards"] == result["optimizer_updates"] == 0,
            "Unexpected label-free replay qualification for " + phase)
    return result, terminal["result_sha256"], digest(raw_terminal)


def exact_edit_lines(prepared, verified, expected):
    """Check stored bytes in lockstep, including row order, EOF and both full stream identities."""
    seal = sha256()
    with prepared.open("rb") as left, verified.open("rb") as right:
        for first, second in zip(left, right, strict=True):
            require(first == second, "Independent replay bytes differ")
            seal.update(first)
            yield json.loads(first)
    require(seal.hexdigest() == expected, "Closed edit stream digest differs")


def shared_budget(entries, native, native_area, core):
    require(set(entries) == set(EDITED), "All five policies must share each positive-budget candidate")
    k = entries["dynamic"]["swaps"]
    require(type(k) is int and 0 < k <= min(8, native_area // 10) and 32 <= native_area < 1024,
            "Unexpected dynamic swap count or area")
    for arm in EDITED:
        entry = entries[arm]
        require(set(entry) == {"rle", "swaps"} and type(entry["swaps"]) is int and entry["swaps"] == k,
                "Per-arm quota or edit fields differ")
        mask = core.checked_rle(entry["rle"], 512, 512)
        overlap = int(core.codec.area(core.codec.merge([native, mask], intersect=True)))
        require(int(core.codec.area(mask)) == native_area and native_area - overlap == k,
                "Candidate's actual added and removed pixel counts must equal its dynamic quota")
    return k


def checked_blocks(rows, lines, adapter, core):
    blocks = {}
    totals = Counter(mapped_candidates=0, eligible_candidates=0, positive_budget_candidates=0, swaps_per_arm=0)
    disagreements = {base: Counter({str(s): 0 for s in SEEDS}) for base in COMPARATORS}
    for (item, audit, pred, _, _), row in zip(rows, lines, strict=True):
        image = item["image_id"]
        require(set(row) == {"image_id", "changes"} and row["image_id"] == image and image not in blocks,
                "Unknown image, duplicate or out-of-order replay")
        native = adapter.native_masks(core, pred, audit, image)
        require(isinstance(row["changes"], dict), "Invalid candidate edit table")
        changes = {}
        for pos_str, entries in row["changes"].items():
            pos = int(pos_str)
            require(pos_str == str(pos) and 1 <= pos <= len(native), "Unknown candidate position")
            area = int(core.codec.area(native[pos - 1]))
            require(audit["candidates"][pos - 1]["mask_area"] == area, "Archived candidate area differs")
            k = shared_budget(entries, native[pos - 1], area, core)
            changes[pos] = entries
            totals.update(positive_budget_candidates=1, swaps_per_arm=k)
            for base in COMPARATORS:
                for seed, arm in zip(SEEDS, LEARNED, strict=True):
                    disagreements[base][str(seed)] += int(different_pixels(entries[arm]["rle"], entries[base]["rle"], area, core.codec))
        totals["mapped_candidates"] += len(native)
        totals["eligible_candidates"] += sum(32 <= c["mask_area"] < 1024 for c in audit["candidates"])
        blocks[image] = dict(stock=pred["coco"], ranks=[r["position"] for r in audit["candidates"]],
                             changes=changes, group=audit["acquisition_group"], sensor=audit["sensor"],
                             file=audit["source_image_name"], rgb=audit["rgb_sha256"])
    require(len(blocks) == 74 and totals["mapped_candidates"] == 7346 and totals["eligible_candidates"] == 4137,
            "Expected the complete reserve and its native census")
    census = {base: {seed: dict(different=n, equivalent=totals["mapped_candidates"] - n,
                                equivalent_positive_budget=totals["positive_budget_candidates"] - n)
                     for seed, n in values.items()} for base, values in disagreements.items()}
    return blocks, dict(totals), census


def different_pixels(a, b, area, codec):
    if a == b:
        return False
    return int(codec.area(codec.merge([a, b], intersect=True))) < area


def exposure_summary(counters, dates):
    return {k: dict(matches=sum(count.values()), groups=len(count), sensor_date_blocks=len({dates[g] for g in count}))
            for k, count in counters.items()}


def interval(values, blocks, reserve):
    ci = reserve.bootstrap(values, blocks)
    return {field: dict(delta=float(val), interval_95=bounds)
            for field, val, bounds in zip(FIELDS, values.mean(axis=0), ci, strict=True)}


def fixed_geometry(anchors, blocks, polygons, dates, core, reserve, history):
    group_values = defaultdict(list)
    counts = Counter(native_matches=len(anchors), eligible_matches=0, invalid_polygon_matches=0, area_excluded_matches=0)
    exposure = {base: {k: Counter() for k in (*map(str, SEEDS), "all_seeds")} for base in BASELINES}
    signs = {base: {str(s): Counter(improved=0, worsened=0, tied=0) for s in SEEDS} for base in BASELINES}
    both = {"all_seeds_both_comparators": Counter()}
    for gt_id, (image, pos, det) in anchors.items():
        polygon = polygons[gt_id]
        if not polygon.is_valid or polygon.area <= 0:
            counts["invalid_polygon_matches"] += 1
            continue
        native = det["segmentation"]
        area = int(core.codec.area(native))
        if not 32 <= area < 1024:
            counts["area_excluded_matches"] += 1
            continue
        counts["eligible_matches"] += 1
        group = blocks[image]["group"]
        edits = blocks[image]["changes"].get(pos, {})
        masks = {arm: history.rle_for(edits, arm, native) for arm in ARMS}
        values = {}
        for mask in masks.values():
            key = mask["counts"]
            if key in values:
                continue
            iou = core.geometric_iou(mask, polygon, shapely)
            intersection = iou * (area + polygon.area) / (1.0 + iou)
            values[key] = [100.0 * iou, 100.0 * (area - intersection) / polygon.area,
                           100.0 * (polygon.area - intersection) / polygon.area, 100.0 * intersection / polygon.area,
                           intersection]
        group_values[group].append([values[masks[arm]["counts"]] for arm in ARMS])
        all_changed = {}
        for base in BASELINES:
            flags = [different_pixels(masks[arm], masks[base], area, core.codec) for arm in LEARNED]
            for seed, arm, changed in zip(SEEDS, LEARNED, flags, strict=True):
                score = values[masks[arm]["counts"]]
                control = values[masks[base]["counts"]]
                require(abs((score[1] - control[1]) - (score[2] - control[2])) <= 1e-8
                        and abs(score[3] - control[3] + score[1] - control[1]) <= 1e-8,
                        "Equal signed edit quota violates fixed-area FP/FN/intersection identities")
                difference = score[0] - control[0]
                key = "improved" if difference > 1e-10 else "worsened" if difference < -1e-10 else "tied"
                signs[base][str(seed)][key] += 1
                if changed:
                    exposure[base][str(seed)][group] += 1
            all_changed[base] = all(flags)
            if all_changed[base]:
                exposure[base]["all_seeds"][group] += 1
        if all(all_changed[base] for base in COMPARATORS):
            both["all_seeds_both_comparators"][group] += 1
    parents = sorted(group_values)
    keys = [dates[p] for p in parents]
    out = dict(counts=dict(counts), represented_groups=len(parents), sensor_date_blocks=len(set(keys)),
               exposure={base: exposure_summary(c, dates) for base, c in exposure.items()},
               joint_exposure=exposure_summary(both, dates), fixed_match_direction=signs, direction_tolerance_points=1e-10,
               group_equal_arms={}, per_seed={}, pooled={}, pooled_parent_only_sensitivity={})
    if not parents:
        return out
    # All seeds and policies have the same fixed original anchors and parent weights.
    means = np.stack([np.mean(group_values[parent], axis=0) for parent in parents])
    out["group_equal_arms"] = {arm: dict(zip(FIELDS, means[:, j].mean(axis=0).tolist(), strict=True))
                               for j, arm in enumerate(ARMS)}
    for base in BASELINES:
        diff = np.stack([means[:, ARMS.index(arm)] - means[:, ARMS.index(base)] for arm in LEARNED])
        out["per_seed"][base] = {str(s): interval(d, keys, reserve) for s, d in zip(SEEDS, diff, strict=True)}
        out["pooled"][base] = interval(diff.mean(axis=0), keys, reserve)
        out["pooled_parent_only_sensitivity"][base] = interval(diff.mean(axis=0), parents, reserve)
        for j, field in enumerate(FIELDS):
            seed_effects = diff[:, :, j].mean(axis=1)
            out["pooled"][base][field].update(median_seed_delta=float(np.median(seed_effects)),
                                              seed_sd=float(np.std(seed_effects, ddof=1)))
    out["classical_against_native"] = {
        arm: interval(means[:, ARMS.index(arm)] - means[:, ARMS.index("raw")], keys, reserve)
        for arm in COMPARATORS}
    return out


def score_view(ids, gt, blocks, polygons, dates, core, evaluator, reserve, history):
    scoped = {i: block for i, block in blocks.items() if i in ids}
    selected = dict(gt, images=[r for r in gt["images"] if r["id"] in ids],
                    annotations=[r for r in gt["annotations"] if r["image_id"] in ids])
    metrics = {}
    for arm in ARMS:
        scores, matched = evaluator.score_one(selected, scoped, arm, anchors=arm == "raw")
        require(len(scores) == 13 and {"AP", "AP50", "AP75", "AP90", "AR100"} <= set(scores),
                "Incomplete original COCO metrics")
        metrics[arm] = {k: history.available(v) for k, v in scores.items()}
        if arm == "raw":
            anchors = matched
    effects = {base: {str(s): history.difference(metrics[arm], metrics[base]) for s, arm in zip(SEEDS, LEARNED)}
               for base in BASELINES}
    geo = fixed_geometry(anchors, scoped, polygons, dates, core, reserve, history)
    gates = {}
    for base in COMPARATORS:
        e = geo["exposure"][base]["all_seeds"]
        iou = geo["pooled"].get(base, {}).get(history.FIELDS[0])
        gates[base] = dict(all_seed_AP_positive=all(v["AP"] is not None and v["AP"] > 0 for v in effects[base].values()),
                           seed_mean_IoU_lower_positive=iou is not None and iou["interval_95"][0] > 0,
                           common_disagreement_exposure=e["matches"] >= 20 and e["sensor_date_blocks"] >= 8)
    return dict(images=len(ids), annotations=len(selected["annotations"]), source_groups=len({b["group"] for b in scoped.values()}),
                metrics_fraction=metrics, per_seed_differences_points=effects,
                seed_summaries_points={base: history.stats(list(e.values())) for base, e in effects.items()},
                classical_minus_raw_points={arm: history.difference(metrics[arm], metrics["raw"]) for arm in COMPARATORS},
                geometry=geo, gates=gates, joint_criterion=all(all(gate.values()) for gate in gates.values()))


def verify_native(view, reference, history):
    require(view["images"] == reference["images"] and view["annotations"] == reference["annotations"], "View census differs")
    expected = {k: history.available(v) for k, v in reference["metrics_fraction"]["raw"].items()}
    require(view["metrics_fraction"]["raw"] == expected, "Archived native COCO score differs")
    actual = view["geometry"]["group_equal_arms"]["raw"]
    previous = reference["geometry"]["group_equal_arms"]["raw"]
    require(all(abs(actual[name] - previous[name]) < 1e-9 for name in history.FIELDS), "Archived fixed-match geometry differs")
    return dict(exact_COCO_fields=len(expected), geometry_fields=len(history.FIELDS),
                max_geometry_difference_points=max(abs(actual[name] - previous[name]) for name in history.FIELDS))


def run(a):
    require(a.cohort == 74 and a.support_root is None, "Only the 74-image parent reserve is admitted")
    require(a.prepare != a.verify and a.prepare_terminal != a.verify_terminal and not a.output.exists(),
            "Separate observed executions and a fresh score directory required")
    require(PROTOCOL == a.protocol_sha256 == digest(a.protocol.read_bytes()),
            "Published protocol bytes differ")
    prepare, prepare_sha, prepare_term = closed(a.prepare, a.prepare_terminal, "prepare-edits",
                                              "INSTANCE_RSDD_CLASSICAL_FAIR_MATCH_PREPARED", a)
    verify, verify_sha, verify_term = closed(a.verify, a.verify_terminal, "verify-edits",
                                           "INSTANCE_RSDD_CLASSICAL_FAIR_MATCH_VERIFIED", a)
    require(verify["prepare_result_sha256"] == prepare_sha and verify["prepare_terminal_sha256"] == prepare_term
            and all(prepare[k] == verify[k] for k in ("worker_sha256", "edits_sha256", "input_lineage", "counts")),
            "Independent replay and prepare lineage differ")
    folder = a.code / "POINTREND_SIGNED_QUOTA_SWAP_20261123"
    h = source(folder / "score_quota_swap.py", HISTORY_SHA)
    inputs = source(folder / "inputs.py", INPUTS_SHA)
    rows, lineage = inputs.load_population(a)
    require(lineage == verify["input_lineage"] and lineage["cohort"] == 74, "Native replay sources differ")
    require(shapely.__version__ == "2.0.2" and shapely.geos_version_string == "3.11.2", "Original geometry runtime required")
    reserve = h.published_module(a.code / "INSTANCE_RSDD_ACQUISITION_RESERVE_20261123/score_reserve.py", h.RESERVE_SCORE)
    old = h.published_module(a.code / "INSTANCE_RSDD_FROZEN_HRSID_20261122/score_rsdd.py", reserve.HISTORIC)
    evaluator = old.module(a.code / "PSEG_RFDETR_FROZEN_SOURCE_HEAD_20261122/score.py", old.SCORE_SOURCE)
    adapter = old.module(a.code / "UEIKAP_RFDETR_CARRIER_20261122/score_rf.py", old.GEOMETRY_SOURCE)
    core = adapter.load_geometry(a.code / "UEIKAP_RESERVED_HULL_FROZEN_20261122/score.py")
    lines = exact_edit_lines(a.prepare / "EDITS_PRIVATE.jsonl", a.verify / "EDITS_PRIVATE.jsonl", verify["edits_sha256"])
    blocks, counts, census = checked_blocks(rows, lines, adapter, core)
    del rows
    tally = verify["counts"]
    require(tally["images"] == len(blocks) and tally["eligible_planes"] == counts["eligible_candidates"]
            and all(tally[k] == counts[k] for k in ("mapped_candidates", "positive_budget_candidates", "swaps_per_arm")),
            "Independently reconstructed candidate or signed budget totals differ from the verified replay")
    print(json.dumps(dict(phase="closed_edits_checked", images=len(blocks), **counts)), flush=True)
    # The polygon file is opened only after both observed exits, byte parity and every candidate's realized counts.
    gt, rekeyed, polygons, sets, dates, label_sha = h.views_and_labels(a, old, reserve, blocks)
    reference = json.loads(h.pin(a.code.parent / "results" / REFERENCE, REFERENCE_SHA))["score"]["views"]
    views, parity, cache = {}, {}, {}
    for name, ids in sets.items():
        key = frozenset(ids)
        if key not in cache:
            cache[key] = score_view(ids, gt, blocks, polygons, dates, core, evaluator, reserve, h)
        views[name] = cache[key]
        parity[name] = verify_native(views[name], reference[name], h)
        print(json.dumps(dict(phase="scored_view", view=name, images=len(ids))), flush=True)
    return dict(status=STATUS, phase="score", cohort=a.cohort, images=len(blocks), seeds=list(SEEDS), arms=list(ARMS),
                primary_view="full", secondary_view="date_separated", primary_joint_criterion=views["full"]["joint_criterion"],
                views=views, counts=counts, all_candidate_locations=census, replay_counts=tally,
                model_forwards=0, detector_forwards=0, optimizer_updates=0,
                target_labels_opened=True, author_labels_sha256=label_sha, author_ids_rekeyed=rekeyed,
                non_simple_polygons_retained_in_AP=sum(not p.is_valid for p in polygons.values()),
                historical_native_parity=parity, historical_score_sha256=REFERENCE_SHA, input_lineage=lineage,
                bootstrap_draws=4096, bootstrap_seed=5217, cluster_unit="sensor/date, equal parent weight",
                prepare_result_sha256=prepare_sha, prepare_terminal_sha256=prepare_term,
                verify_result_sha256=verify_sha, verify_terminal_sha256=verify_term,
                replay_worker_sha256=verify["worker_sha256"], edits_sha256=verify["edits_sha256"], protocol_sha256=a.protocol_sha256)


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--cohort", type=int, choices=(74,), required=True)
    for option in ("code", "stock-root", "point-root", "prepare", "prepare-terminal",
                   "verify", "verify-terminal", "author-labels", "protocol", "output"):
        p.add_argument("--" + option, required=True, type=Path)
    p.add_argument("--protocol-sha256", required=True)
    a = p.parse_args()
    a.support_root = None
    for key, value in vars(a).items():
        if isinstance(value, Path):
            setattr(a, key, value.resolve(strict=key != "output"))
    start = perf_counter()
    result = run(a)
    result.update(seconds=perf_counter() - start, worker_sha256=digest(Path(__file__).read_bytes()))
    raw = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    a.output.mkdir()
    with (a.output / "RESULT_SAFE.json").open("xb") as stream:
        require(stream.write(raw) == len(raw), "Incomplete aggregate result")
    print(json.dumps(dict(status=STATUS, result_sha256=digest(raw))), flush=True)


if __name__ == "__main__":
    main()
