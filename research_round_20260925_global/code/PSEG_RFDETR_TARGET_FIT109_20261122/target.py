"""Replay and score a target-fitted head at the archived dynamic exchange counts."""
import argparse
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType

import numpy as np
from pycocotools import mask


PRELOCK = "11a74f36a4b662cffd1a78fcea7b77b9e5d5e18d4543cc191ac296642752fb02"
OLD_ROOT = "PSEG_RFDETR_FROZEN_SOURCE_HEAD_20261122"
REPLAY_SHA = "be1516489123153a018ebc721806190d83f05285617e7c108e9b3c1064f66655"
SCORE_SHA = "77a3fbfaa04ac08287eff4ec7bb827ce3914e61aa20dd4f1e6f9c1e3b8ccc9fa"
OLD_EDIT = "828ed385ab8dff38840a0e179a0fdf4f4bc0a192088d062aeae193fad750656f"
OLD_EDIT_EXIT = "3ad741a7b489839d6b6d0f50347342cf6bbf28160dddd8b268812ea3a2e0520c"
OLD_SCORE = "b3252decceb28c9da468b0e86fc42568343385d79b27304ad071678c4ac8035b"
DYNAMIC = "attraction_dynamic"
SOURCE = "learned_at_attraction_count"
TARGET = "target_at_attraction_count"
STATES = {"prepare-edits":"PSEG_TARGET_FIT109_DEV187_OVERLAY_PREPARED",
          "verify-edits":"PSEG_TARGET_FIT109_DEV187_OVERLAY_VERIFIED",
          "score":"PSEG_TARGET_FIT109_DEV187_SCORE_COMPLETE"}
HISTORY_FIELDS = {"protocol", "code", "old_edit_result", "old_edit_terminal", "old_edits", "stock", "capture"}
HEAD_FIELDS = {"images", "fit_result", "fit_result_sha", "fit_terminal", "fit_terminal_sha",
               "fit_code", "fit_code_sha", "weights"}
PREPARED_FIELDS = {"overlay", "overlay_sha", "prepared", "prepared_sha", "prepare_terminal", "prepare_terminal_sha"}
SCORE_FIELDS = {"verified", "verified_sha", "verify_terminal", "verify_terminal_sha", "overlay", "overlay_sha",
                "old_score", "prior", "gt", "shapely_site"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pin(path, digest):
    raw = path.read_bytes()
    require(sha256(raw).hexdigest() == digest, "Pinned input identity: " + path.name)
    return raw


def read_module(path, digest):
    module = ModuleType(path.stem)
    module.__file__ = str(path)
    exec(compile(pin(path, digest), str(path), "exec"), module.__dict__)
    return module


def rows(path, digest):
    return [json.loads(line) for line in pin(path, digest).splitlines()]


def readback_actions(q, k):
    require(q.ndim == 2 and q.shape[1] == 2 and 0 < k <= len(q) and q.dtype == np.float32
            and np.isfinite(q).all() and np.all((q >= 0) & (q <= 1)), "Native point probabilities")
    additions = sorted(range(len(q)), key=lambda i: (-q[i, 0], i))[:k]
    removals = sorted(range(len(q)), key=lambda i: (q[i, 1], i))[:k]
    return list(zip(additions, removals, strict=True))


def verify_xor(original, coords, entry, k):
    edit = mask.decode(entry["rle"]).astype(bool)
    require(entry["swaps"] == k and edit.shape == original.shape, "Exact dynamic count and canvas")
    additions = np.flatnonzero(edit & ~original)
    removals = np.flatnonzero(original & ~edit)
    require(len(additions) == len(removals) == k and np.isin(additions, coords[:, 0]).all()
            and np.isin(removals, coords[:, 1]).all(), "Endpoint XOR and foreground area")


def dynamic_edits(prior):
    return {rank: values[DYNAMIC] for rank, values in prior["changes"].items() if DYNAMIC in values}


def join_overlay(prior, overlay):
    expected, new = dynamic_edits(prior), overlay["changes"]
    require(prior["image_id"] == overlay["image_id"] and set(expected) == set(new)
            and all(new[r]["swaps"] == item["swaps"] for r, item in expected.items()),
            "Exact dynamic-count coverage")
    changed = {rank: values | ({TARGET: new[rank]} if rank in new else {})
               for rank, values in prior["changes"].items()}
    return prior | {"changes": changed}


def original_point_features(pair, sampler, native, box, score_, posterior, image, endpoints):
    data = pair.paired_points(native, box, int(native.sum()), score_, posterior, image, sampler)
    require(data is not None and np.array_equal(data[0], endpoints), "Exact archived endpoints")
    return data[0], data[1]


def observed(path, digest, result, phase=None, worker=None):
    event = json.loads(pin(path, digest))
    require(event["actual_exit"] == 0 and event["output_footer_verified"] is True
            and event["result_sha256"] == result and (phase is None or event["phase"] == phase)
            and (worker is None or event["worker_sha256"] == worker), "Observed worker exit and result")


def original_inputs(a, r):
    source = json.loads(pin(a.old_edit_result, OLD_EDIT))
    observed(a.old_edit_terminal, OLD_EDIT_EXIT, OLD_EDIT, "verify-edits", REPLAY_SHA)
    require(source["GT_opened"] is False and source["predictions_sha256"] == r.STOCK_SHA,
            "Archived verified native source results")
    stock = json.loads(pin(a.stock, r.STOCK_SHA))
    capture = rows(a.capture / "CAPTURE_PRIVATE.jsonl", source["capture_stream_sha256"])
    entries = rows(a.old_edits, source["edits_sha256"])
    require(len(stock) == 16161 and len(capture) == len(entries) == 187, "Complete archived candidate universe")
    return stock, capture, entries


def replay_target(a, r, stock, captures, previous, verify):
    import torch
    torch.set_num_threads(1)
    fit = json.loads(pin(a.fit_result, a.fit_result_sha))
    observed(a.fit_terminal, a.fit_terminal_sha, a.fit_result_sha, "fit", a.fit_code_sha)
    require(fit["status"] == "PSEG_TARGET_RF5201_FIT109_POINT_HEAD_COMPLETE_V1"
            and fit["protocol_sha256"] == PRELOCK and fit["worker_sha256"] == a.fit_code_sha
            and fit["optimizer_updates"] == 200 and fit["head_parameter_count"] == 1321
            and fit["seed"] == 5217 and fit["sample_seed"] == 20261112,
            "Observed fixed target training recipe")
    adapter = read_module(a.fit_code, a.fit_code_sha)
    predict = adapter.make_predictor(a.code, a.weights, fit["weights_sha256"])
    modules = r.methods(a.code)
    stored = None
    if verify:
        previous_run = json.loads(pin(a.prepared, a.prepared_sha))
        worker = sha256(Path(__file__).read_bytes()).hexdigest()
        observed(a.prepare_terminal, a.prepare_terminal_sha, a.prepared_sha, "prepare-edits", worker)
        require(previous_run["status"] == STATES["prepare-edits"] and previous_run["GT_opened"] is False
                and previous_run["overlay_sha256"] == a.overlay_sha
                and previous_run["fit_result_sha256"] == a.fit_result_sha, "Observed prelabel overlay")
        stored = rows(a.overlay, a.overlay_sha)
        require(len(stored) == len(captures), "Overlay preserves all native images")
    result = []
    for image_index, (row, prior) in enumerate(zip(captures, previous, strict=True)):
        require(row["image_id"] == prior["image_id"] and row["rgb_sha256"] == prior["rgb_sha256"],
                "Original image alignment")
        options = dynamic_edits(prior)
        altered = {}
        if options:
            image = modules.pair.image_features(r.load_rgb(a.images, row))
            pair, sampler = r.canvas(modules, tuple(row["original_hw"]))
            start, count = row["stock_prediction_start"], row["stock_prediction_count"]
            candidates = dict(zip(row["native_ranks"], zip(stock[start:start+count], row["native_boxes_xyxy"],
                                                          strict=True), strict=True))
            planes = r.load_array(a.capture, row["logits_ref"])
            logit_rows = dict(zip(row["logit_ranks"], planes, strict=True))
            endpoints = {x["rank"]: np.asarray(x["coordinates"], dtype=np.int32) for x in prior["points"]}
            overrides = {x["native_rank"]: x for x in row["posterior_overrides"]}
            for rank, dynamic in options.items():
                k, rank_id = dynamic["swaps"], int(rank)
                cand, box = candidates[rank_id]
                native = mask.decode(cand["segmentation"]).astype(bool)
                box = np.asarray(box, dtype=np.float32)
                override = overrides.get(rank_id)
                posterior = r.native_probability(logit_rows[rank_id], native,
                                                 None if override is None else r.load_array(a.capture, override))
                coords, features = original_point_features(pair, sampler, native, box, cand["score"],
                                                           posterior, image, endpoints[rank_id])
                q = predict(features)
                cap = min(8, int(native.sum()) // 10, len(coords))
                require(0 < k <= cap and q.shape == coords.shape and q.dtype == np.float32
                        and np.isfinite(q).all() and np.all((q >= 0) & (q <= 1)), "Locked count and predictions")
                actions = readback_actions(q, k) if verify else modules.free.choose_free(q, cap, fixed_count=k)
                altered[rank] = r.mask_edit(native, coords, actions)
                verify_xor(native, coords, altered[rank], k)
        item = {"image_id": row["image_id"], "changes": altered}
        join_overlay(prior, item)
        if verify:
            require(item == stored[image_index], "Independent recalculation of every predicted edit")
        result.append(item)
    count = sum(len(item["changes"]) for item in result)
    swaps = sum(v["swaps"] for item in result for v in item["changes"].values())
    require(count == 7142 and swaps == 51416, "Historical per-candidate and aggregate count parity")
    return result, dict(edited_candidates=count, swaps=swaps, fit_result_sha256=a.fit_result_sha,
                        target_weights_sha256=fit["weights_sha256"], GT_opened=False, TEST=False)


def score(a, s, stock, captures, previous):
    receipt = json.loads(pin(a.verified, a.verified_sha))
    worker = sha256(Path(__file__).read_bytes()).hexdigest()
    observed(a.verify_terminal, a.verify_terminal_sha, a.verified_sha, "verify-edits", worker)
    require(receipt["status"] == STATES["verify-edits"] and receipt["GT_opened"] is False
            and receipt["overlay_sha256"] == a.overlay_sha and receipt["edited_candidates"] == 7142
            and receipt["swaps"] == 51416, "Independently accepted target overlay")
    overlay = rows(a.overlay, a.overlay_sha)
    entries = [join_overlay(p, n) for p, n in zip(previous, overlay, strict=True)]
    historical = json.loads(pin(a.old_score, OLD_SCORE))
    rivals = ("raw", SOURCE, DYNAMIC)
    old_arms = s.ARMS
    s.ARMS = (*s.ARMS, TARGET)
    s.PAIRS = tuple((TARGET, opponent) for opponent in rivals)
    blocks, counts = s.cohort(stock, captures, entries)
    require(counts[TARGET+"_candidates"] == 7142 and counts[TARGET+"_swaps"] == 51416,
            "All verified target masks admitted to COCO scoring")
    prior = json.loads(pin(a.prior, s.PRIOR))
    gt = json.loads(pin(a.gt, prior["floating_gt_sha256"]))
    require(set(r["id"] for r in gt["images"]) == set(blocks) and len(gt["annotations"]) == 380
            and all([r["height"],r["width"]] == blocks[r["id"]]["hw"] for r in gt["images"]),
            "Original DEV float labels and native shapes")
    dedup = {}
    for image_id in sorted(blocks):
        dedup.setdefault(blocks[image_id]["rgb"], image_id)
    require(len(dedup) == 182, "Historical exact RGB groups")
    views = {}
    assignments = None
    for name, ids in (("full", set(blocks)), ("equal_rgb_deduplicated", set(dedup.values()))):
        subset = dict(gt, images=[row for row in gt["images"] if row["id"] in ids],
                      annotations=[row for row in gt["annotations"] if row["image_id"] in ids])
        included = {k: v for k, v in blocks.items() if k in ids}
        scores = {}
        for arm in s.ARMS:
            scores[arm], anchors = s.score_one(subset, included, arm, anchors=name=="full" and arm=="raw")
            if anchors:
                assignments = anchors
        require({arm:scores[arm] for arm in old_arms} == historical["views"][name]["arms_fraction"],
                "All 13 archived metrics exact for all six scored arms in both views")
        contrasts = {TARGET+" minus "+ref: {k:100*(v-scores[ref][k]) if min(v,scores[ref][k]) >= 0 else None
                                          for k,v in scores[TARGET].items()} for ref in rivals}
        views[name] = dict(images=len(ids), annotations=len(subset["annotations"]),
                           arms_fraction=scores, paired_contrasts_points=contrasts)
    geometry = s.report_geometry(assignments, gt, blocks, a.code, a.shapely_site)
    require((geometry["fixed_matches"], geometry["area_eligible"],geometry["groups"]) == (378,281,119),
            "Same fixed nonignored match and paired units")
    for arm in old_arms:
        require(geometry["group_equal_arms"][arm] == historical["geometry"]["group_equal_arms"][arm],
                "Exact archived continuous geometries")
    primary = TARGET + " minus raw"
    lower = geometry["paired_contrasts_points"][primary]["continuous_iou_points"]["interval_95"][0]
    full = views["full"]["paired_contrasts_points"]
    sensitivity = views["equal_rgb_deduplicated"]["paired_contrasts_points"]
    checks = dict(native_ap_at_least_1=full[primary]["AP"] >= 1,
                  source_and_dynamic_ap_positive=all(full[TARGET+" minus "+ref]["AP"] > 0 for ref in rivals[1:]),
                  all_dedup_contrasts_positive=all(x["AP"] > 0 for x in sensitivity.values()),
                  positive_lower_iou_against_native=lower > 0)
    return dict(views=views, geometry=geometry, readiness=checks, advance=all(checks.values()),
                archived_six_arms_exact=True, actual_edit_counts=counts, checked_overlay_sha256=a.overlay_sha,
                verified_overlay_receipt_sha256=a.verified_sha,
                fit_result_sha256=receipt["fit_result_sha256"], GT_opened=True, TEST=False)


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("phase", choices=STATES)
    p.add_argument("--inputs", required=True, type=Path)
    p.add_argument("--inputs-sha256", required=True)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    refs = json.loads(pin(args.inputs, args.inputs_sha256))
    stage_fields = {"prepare-edits": HEAD_FIELDS, "verify-edits": HEAD_FIELDS | PREPARED_FIELDS, "score": SCORE_FIELDS}
    require(set(refs) == HISTORY_FIELDS | stage_fields[args.phase], "Exact phase-specific closed input authority")
    a = argparse.Namespace(**{key: value if key.endswith("_sha") else Path(value) for key, value in refs.items()})
    require(not args.output.exists() and sys.flags.isolated and sys.dont_write_bytecode
            and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
            and version("pycocotools") == "2.0.11" and np.__version__ == "1.26.4", "Fresh isolated CPU execution")
    pin(a.protocol, PRELOCK)
    r = read_module(a.code / OLD_ROOT / "replay.py", REPLAY_SHA)
    stock, captures, previous = original_inputs(a, r)
    start = perf_counter()
    if args.phase == "score":
        scored = read_module(a.code / OLD_ROOT / "score.py", SCORE_SHA)
        summary = score(a, scored, stock, captures, previous)
    else:
        result, summary = replay_target(a, r, stock, captures, previous, args.phase == "verify-edits")
        data = b"".join(r.encoded(item) for item in result)
        summary["overlay_sha256"] = sha256(data).hexdigest()
        if args.phase == "verify-edits":
            require(a.overlay_sha == summary["overlay_sha256"], "Verified identical overlay bytes")
    args.output.mkdir()
    if args.phase == "prepare-edits":
        (args.output / "OVERLAY_PRIVATE.jsonl").write_bytes(data)
    summary.update(status=STATES[args.phase], protocol_sha256=PRELOCK, input_authority_sha256=args.inputs_sha256,
                   elapsed_seconds=perf_counter()-start,
                   evaluator_optimizer_updates=0, detector_forwards=0,
                   worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    output = (json.dumps(summary, indent=2, sort_keys=True, allow_nan=False)+"\n").encode()
    (args.output / "RESULT_SAFE.json").write_bytes(output)
    print(json.dumps(dict(status=summary["status"], result_sha256=sha256(output).hexdigest())), flush=True)


if __name__ == "__main__":
    main()
