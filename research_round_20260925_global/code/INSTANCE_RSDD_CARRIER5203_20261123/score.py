"""Score one frozen Instance-RSDD cohort on the second RF-DETR fit."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType

import numpy as np
import shapely


LOCK = "f7ac19ff63c2c84fb5e2aac772f5efa0bf16a8e8ac519b0a2469d74b6bfd05cf"
CHECKPOINT = "36205b47714057ee24a3916f2bc0d307779ad4aa60e5b39b949bbdf381f31054"
OLD_SCORE = "7039d7d0336f6a0cc3b64bfbe4b0c446b38b073cc90764f446a4dc6d09f90d85"
RESERVE_SCORE = "5ff671b2b83bb12e322b0c766897bf3cdacaedd233bf0d6edcdfcccf2c938df4"
MANIFEST = {74: "154d0d748b96df0d56ac249adcb8b6ae63275a09dfd7a0d7480839005e044338",
            740: "16a546e96d6e76d3afbfbe4f380b182505494818b35f1c8bcc960e6d4d1b2184"}
SEEDS = (5217, 5218, 5219)
HEAD_ARMS = ("fixed", "free_matched_count", "free_natural", "learned")
ARMS = ("raw", "dynamic", "static", *(f"{s}/{a}" for s in SEEDS for a in HEAD_ARMS))
FIELDS = ("continuous_iou_points", "intersection_per_truth_area_points",
          "fp_per_truth_area_points", "fn_per_truth_area_points")
COMPARE = {"independent_minus_fixed": ("free_matched_count", "fixed"),
           "natural_minus_raw": ("free_natural", "raw"),
           "learned_minus_dynamic": ("learned", "dynamic"),
           "learned_minus_static": ("learned", "static")}


def check(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return sha256(raw).hexdigest()


def pinned(path, expected):
    raw = path.read_bytes()
    check(digest(raw) == expected, "Incorrect source fingerprint: " + path.name)
    return raw


def load(path, expected):
    module = ModuleType(path.stem)
    module.__file__ = str(path)
    exec(compile(pinned(path, expected), str(path), "exec"), module.__dict__)
    return module


def closed(directory, terminal, phase, status, worker, cohort):
    raw = (directory / "RESULT_SAFE.json").read_bytes()
    result, result_sha = json.loads(raw), digest(raw)
    end = terminal.read_bytes()
    receipt = json.loads(end)
    check(type(receipt["actual_exit"]) is int and receipt["actual_exit"] == 0
          and receipt["output_footer_verified"] is True and receipt["phase"] == result["phase"] == phase
          and receipt["worker_sha256"] == result["worker_sha256"] == worker
          and receipt["result_sha256"] == result_sha and result["status"] == status
          and result["protocol_sha256"] == LOCK and result["checkpoint_sha256"] == CHECKPOINT
          and result["images"] == cohort and result["manifest_sha256"] == MANIFEST[cohort]
          and result["target_labels_opened"] is False, "Image-only phase did not close with exact inputs")
    return result, result_sha, digest(end)


def verified_options(entries, mask, core):
    area = int(core.codec.area(mask))
    check(32 <= area < 1024 and entries and set(entries) <= set(ARMS[1:]), "Unknown or ineligible edit")
    budget = {k: v["swaps"] for k, v in entries.items()}
    same_k = budget.get("dynamic", 0)
    check(budget.get("static", 0) == same_k and all(
        budget.get(f"{s}/learned", 0) == same_k
        and budget.get(f"{s}/fixed", 0) == budget.get(f"{s}/free_matched_count", 0) for s in SEEDS
    ), "Different realized budgets")
    for arm, edit in entries.items():
        k = budget[arm]
        after = core.checked_rle(edit["rle"], 512, 512)
        overlap = int(core.codec.area(core.codec.merge([after, mask], intersect=True)))
        check(set(edit) == {"swaps", "rle"} and type(k) is int and 1 <= k <= min(8, area//10)
              and int(core.codec.area(after)) == area and area-overlap == k,
              "Incorrect number of native additions or removals")


def inputs(a, adapter, core):
    cap, cs, ct = closed(a.capture, a.capture_terminal, "capture", "RSDD_CARRIER5203_CAPTURE_COMPLETE",
                         a.capture_worker_sha256, a.cohort)
    replay, rs, rt = closed(a.replay, a.replay_terminal, "replay", "RSDD_CARRIER5203_REPLAY_COMPLETE",
                            a.replay_worker_sha256, a.cohort)
    check(cap["independent_cpu_readback"] is True and cap["model_forward_calls"] == a.cohort
          and replay["capture_result_sha256"] == cs and replay["capture_terminal_sha256"] == ct
          and all(replay[k] == cap[k] for k in ("predictions_sha256", "audit_sha256", "roster_sha256", "readback_result_sha256"))
          and replay["seeds"] == list(SEEDS) and set(replay["arms"]) == set(ARMS[1:])
          and replay["exact_support_xor_area"] is True and replay["independent_signed_orders_exact"] is True
          and replay["optimizer_updates"] == replay["detector_forwards"] == cap["optimizer_updates"] == 0,
          "Replay is not from the independently verified capture")
    unroll = lambda directory, name, sha: map(json.loads, pinned(directory / name, sha).splitlines())
    sources = (unroll(a.capture, "PREDICTIONS_PRIVATE.jsonl", cap["predictions_sha256"]),
               unroll(a.capture, "CANDIDATE_AUDIT_PRIVATE.jsonl", cap["audit_sha256"]),
               unroll(a.replay, "EDITS_PRIVATE.jsonl", replay["edits_sha256"]))
    blocks, candidate_counts, counts = {}, Counter(), Counter()
    for pred, audit, edits in zip(*sources, strict=True):
        i = pred["image_id"]
        check(i == audit["image_id"] == edits["image_id"] and i not in blocks
              and audit["original_hw"] == [512, 512], "Non-unique or misordered candidate stream")
        masks = adapter.native_masks(core, pred, audit, i)
        changes = {}
        for pos, entries in edits["changes"].items():
            n = int(pos)
            check(str(n) == pos and 1 <= n <= len(masks), "Unknown compact candidate position")
            verified_options(entries, masks[n-1], core)
            changes[n] = entries
            for arm, entry in entries.items():
                candidate_counts[arm] += 1
                counts[arm] += entry["swaps"]
        blocks[i] = dict(stock=pred["coco"], ranks=[r["position"] for r in audit["candidates"]],
                         changes=changes, rgb=audit["rgb_sha256"], sensor=audit["sensor"],
                         group=audit["acquisition_group"], file=audit["source_image_name"])
    masks_total = sum(len(b["stock"]) for b in blocks.values())
    check(len(blocks) == a.cohort and masks_total == cap["counts"]["mapped"]
          and len({b["rgb"] for b in blocks.values()}) == a.cohort, "Complete exact capture population required")
    budget = {arm: dict(positive_candidates=candidate_counts[arm], swaps=counts[arm],
                       untouched_candidates=masks_total-candidate_counts[arm]) for arm in ARMS}
    check(replay["counts"]["mapped_candidates"] == masks_total and all(
        replay["counts_by_arm"][arm] == dict(changed_candidates=candidate_counts[arm], swaps=counts[arm],
                                             adds=counts[arm], drops=counts[arm]) for arm in ARMS[1:]
    ), "Replay signed-count receipt does not match the closed masks")
    chain = dict(capture_result_sha256=cs, capture_terminal_sha256=ct, replay_result_sha256=rs,
                 replay_terminal_sha256=rt, predictions_sha256=cap["predictions_sha256"],
                 audit_sha256=cap["audit_sha256"], edits_sha256=replay["edits_sha256"])
    return blocks, masks_total, budget, chain


def labels(a, blocks, old, reserve):
    full = set(blocks)
    gf = {i for i, b in blocks.items() if b["sensor"] == "GF-3"}
    views = dict(full=full, gf3=gf, terrasar_x=full-gf)
    parents = {b["group"] for b in blocks.values()}
    if a.cohort == 740:
        gt = {"info": {}, **json.loads(pinned(a.author_labels, old.AUTHOR_TEST))}
        check(len(gt["annotations"]) == 872 and len(gt["images"]) == 740 and len(gf) == 583
              and len(parents) == 112 and len({blocks[i]["group"] for i in gf}) == 76
              and sum(x["image_id"] in gf for x in gt["annotations"]) == 631,
              "Author test roster census differs")
        label_sha = old.AUTHOR_TEST
        dates = {p: p for p in parents}
    else:
        census = json.loads(pinned(a.code.parent / "results/INSTANCE_RSDD_PARENT_RESERVE_CENSUS_SAFE_20261123.json",
                                  reserve.CENSUS))
        check(Counter((b["group"], b["sensor"]) for b in blocks.values())
              == {(p["paper_scene"], p["sensor"]): p["images"] for p in census["per_parent"]},
              "Author reserve source group census differs")
        gt = reserve.select_labels(pinned(a.author_labels, reserve.TRAIN), blocks)
        label_sha = reserve.TRAIN
        dates = {p["paper_scene"]: (p["sensor"], p["calendar_date"]) for p in census["per_parent"]}
        selected = set(census["date_separated_sensitivity"]["parent_scene_ids"])
        views["date_separated"] = {i for i, b in blocks.items() if b["group"] in selected}
        check(len(views["date_separated"]) == 42 and len(gt["annotations"]) == 81, "Author reserve roster differs")
    check({x["id"] for x in gt["images"]} == full and all(x["height"] == x["width"] == 512
          and x["file_name"] == blocks[x["id"]]["file"] for x in gt["images"])
          and all(a["category_id"] == 1 and len(a["segmentation"]) == 1 for a in gt["annotations"]),
          "Original polygons do not match the frozen image roster")
    gt, rekeyed = old.coco_annotation_ids(gt)
    polygons = {r["id"]: shapely.Polygon(np.asarray(r["segmentation"][0], dtype=float).reshape(-1, 2))
                for r in gt["annotations"]}
    non_simple = sum(not p.is_valid for p in polygons.values())
    check(non_simple == (1 if a.cohort == 740 else 0), "Original contour validity census differs")
    return gt, polygons, dates, views, dict(annotation_sha256=label_sha, evaluator_ids_rekeyed=rekeyed,
                                           non_simple_polygons_retained_in_COCO=non_simple)


def arms_for(seed, comparison):
    first, second = COMPARE[comparison]
    return (f"{seed}/{first}", second if second in ("raw", "dynamic", "static") else f"{seed}/{second}")


def contour_values(mask, polygon, area, core):
    iou = core.geometric_iou(mask, polygon, shapely)
    inter = iou*(area+polygon.area)/(1+iou)
    return [100*iou, 100*inter/polygon.area, 100*(area-inter)/polygon.area, 100*(polygon.area-inter)/polygon.area]


def geometry(anchors, blocks, polygons, dates, core, reserve):
    groups, exposures = defaultdict(list), {s: Counter() for s in SEEDS}
    counts = Counter(native_matches=len(anchors), eligible_matches=0, outside_area=0, invalid_polygon=0)
    for gid, (i, pos, pred) in anchors.items():
        polygon = polygons[gid]
        if not polygon.is_valid or polygon.area <= 0:
            counts["invalid_polygon"] += 1
            continue
        native = pred["segmentation"]
        area = int(core.codec.area(native))
        if not 32 <= area < 1024:
            counts["outside_area"] += 1
            continue
        counts["eligible_matches"] += 1
        group, cache, values = blocks[i]["group"], {}, []
        entries = blocks[i]["changes"].get(pos, {})
        masks = {arm: entries.get(arm, {}).get("rle", native) for arm in ARMS}
        for mask in masks.values():
            key = mask["counts"]
            if key not in cache:
                cache[key] = contour_values(mask, polygon, area, core)
            values.append(cache[key])
        groups[group].append(values)
        for s, exposure in exposures.items():
            left, right = arms_for(s, "independent_minus_fixed")
            if masks[left] != masks[right]:
                exposure[group] += 1
    coverage = {s: dict(changed_matches=sum(v.values()), changed_groups=len(v),
                        changed_blocks=len({dates[g] for g in v})) for s, v in exposures.items()}
    keys = sorted(groups)
    result = dict(counts=dict(counts), represented_groups=len(keys), represented_blocks=len({dates[g] for g in keys}),
                  primary_exposure=coverage, group_equal={}, effects={})
    if not keys:
        return result
    matrix = np.stack([np.mean(groups[p], axis=0) for p in keys])
    arm = {name: matrix[:, j] for j, name in enumerate(ARMS)}
    result["group_equal"] = {name: dict(zip(FIELDS, values.mean(axis=0).tolist(), strict=True))
                             for name, values in arm.items()}
    for name in COMPARE:
        paired = {s: arm[arms_for(s, name)[0]] - arm[arms_for(s, name)[1]] for s in SEEDS}
        paired["head_mean"] = np.stack(list(paired.values())).mean(axis=0)
        result["effects"][name] = {s: {
            field: dict(delta=float(d), interval_95=ci)
            for field, d, ci in zip(FIELDS, values.mean(axis=0), reserve.bootstrap(values, [dates[p] for p in keys]), strict=True)
        } for s, values in paired.items()}
    return result


def evaluate(gt, ids, blocks, polygons, dates, evaluator, core, reserve):
    truth = dict(gt, images=[r for r in gt["images"] if r["id"] in ids],
                 annotations=[r for r in gt["annotations"] if r["image_id"] in ids])
    scoped = {i: b for i, b in blocks.items() if i in ids}
    metrics = {}
    for arm in ARMS:
        result, matches = evaluator.score_one(truth, scoped, arm, anchors=arm == "raw")
        metrics[arm] = {k: float(v) if np.isfinite(v) and v >= 0 else None for k, v in result.items()}
        if arm == "raw":
            anchors = matches
    measured = geometry(anchors, scoped, polygons, dates, core, reserve)
    comparisons = {}
    for name in COMPARE:
        rows = {}
        for s in SEEDS:
            first, second = (metrics[a] for a in arms_for(s, name))
            rows[s] = {k: 100*(first[k]-second[k]) if first[k] is not None and second[k] is not None else None for k in first}
        rows["head_mean"] = {k: float(np.mean([rows[s][k] for s in SEEDS]))
                             if all(rows[s][k] is not None for s in SEEDS) else None for k in rows[SEEDS[0]]}
        comparisons[name] = rows
    primary = comparisons["independent_minus_fixed"]
    finite = [primary[s]["AP"] for s in SEEDS]
    exposure = measured["primary_exposure"]
    interval = measured["effects"].get("independent_minus_fixed", {}).get("head_mean", {}).get(FIELDS[0], {})
    gate = dict(AP_positive_at_all_heads=all(v is not None and v>0 for v in finite),
                each_head_20_changed_in_8_groups=all(x["changed_matches"]>=20 and x["changed_groups"]>=8
                                                    for x in exposure.values()),
                matched_head_mean_iou_positive_lower=bool(interval and interval["interval_95"][0]>0))
    return dict(images=len(ids), annotations=len(truth["annotations"]), groups=len({b["group"] for b in scoped.values()}),
                metrics_fraction=metrics, comparisons_points=comparisons, geometry=measured,
                gate=gate, primary_mechanism_pass=all(gate.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--cohort", type=int, choices=(740, 74), required=True)
    for name in ("code", "capture", "capture-terminal", "replay", "replay-terminal", "protocol", "author-labels", "output"):
        parser.add_argument("--"+name, required=True, type=Path)
    for name in ("protocol-sha256", "capture-worker-sha256", "replay-worker-sha256"):
        parser.add_argument("--"+name, required=True)
    a = parser.parse_args()
    start = perf_counter()
    check(a.protocol_sha256 == LOCK == digest(a.protocol.read_bytes()) and not a.output.exists()
          and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and sys.flags.isolated and sys.dont_write_bytecode
          and np.__version__ == "1.26.4" and version("pycocotools") == "2.0.11"
          and shapely.__version__ == "2.0.2" and shapely.geos_version_string == "3.11.2", "Original CPU scoring runtime")
    old = load(a.code / "INSTANCE_RSDD_FROZEN_HRSID_20261122/score_rsdd.py", OLD_SCORE)
    reserve = load(a.code / "INSTANCE_RSDD_ACQUISITION_RESERVE_20261123/score_reserve.py", RESERVE_SCORE)
    evaluator = old.module(a.code / "PSEG_RFDETR_FROZEN_SOURCE_HEAD_20261122/score.py", old.SCORE_SOURCE)
    adapter = old.module(a.code / "UEIKAP_RFDETR_CARRIER_20261122/score_rf.py", old.GEOMETRY_SOURCE)
    core = adapter.load_geometry(a.code / "UEIKAP_RESERVED_HULL_FROZEN_20261122/score.py")
    blocks, mapped, budget, chain = inputs(a, adapter, core)
    gt, polygons, dates, views, label_info = labels(a, blocks, old, reserve)
    scores = {}
    for key, ids in views.items():
        scores[key] = evaluate(gt, ids, blocks, polygons, dates, evaluator, core, reserve)
        print(json.dumps(dict(phase="score", view=key, images=len(ids))), flush=True)
    primary = "gf3" if a.cohort == 740 else "full"
    result = dict(status="RSDD_CARRIER5203_SCORE_COMPLETE", phase="score", protocol_sha256=LOCK,
                  worker_sha256=digest(Path(__file__).read_bytes()), images=a.cohort, primary_view=primary,
                  checkpoint_sha256=CHECKPOINT, detector_seed=5203, head_seeds=SEEDS,
                  target_labels_opened=True, detector_forwards=0, optimizer_updates=0, signed_budgets_verified=True,
                  bootstrap_draws=4096, bootstrap_seed=5217, matched_equal_weight_unit="author parent",
                  block_mode="author acquisition" if a.cohort == 740 else "sensor/date",
                  mapped_candidates=mapped, budget=budget, views=scores, **chain, **label_info,
                  elapsed_seconds=perf_counter()-start)
    body = (json.dumps(result, allow_nan=False)+"\n").encode()
    a.output.mkdir()
    with (a.output / "RESULT_SAFE.json").open("xb") as f:
        f.write(body)
    print(json.dumps(dict(status=result["status"], result_sha256=digest(body))), flush=True)


if __name__ == "__main__":
    main()
