"""Score an independently closed DEV705 same-count spatial-attraction comparison."""
import argparse
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from copy import deepcopy
from hashlib import sha256
from io import StringIO
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType

import numpy as np

PRELOCK_SHA = "fd11e933a9f5d89699c7b614cf1d3176a036ed61d2ed8f194cb4329923a4da80"
BASE_SCORE_SHA = "77159ea493a95973fe422dac1d303eaeba603e6d8af7ae179c72aecc2e4301b0"
BASE_ENDPOINT_SHA = "e32f2e6f45fb2e1a777765a57b021877e0803588285c6868aa2a915c5c6542e8"
BASE_SIDECAR_SHA = "277965bef4230ec7fadac30e103095af3ce68f99d1b69f74a8e348c391cef035"
ADAPTER = "DIFFSAR_NATIVE_CPU_COCO_ADAPTER_DRAFT_20261107_01/worker.py"
BASE = "DFINE_SEG_EXTERNAL_BASELINE_FIT32_E24_20261112/e24/dev705_free_endpoint"
NEW_ARMS = ("attraction_dynamic", "attraction_static_at_dynamic_count", "learned_at_attraction_count")
DYNAMIC, STATIC, LEARNED = NEW_ARMS
COMPARISONS = ((LEARNED, DYNAMIC), (LEARNED, STATIC), (STATIC, DYNAMIC))
FIELDS = ("continuous_iou_points", "fp_per_truth_points", "fn_per_truth_points")
VIEWS = ("primary", "sensitivity")


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(data):
    return sha256(data).hexdigest()


def pin(path, digest):
    data = path.read_bytes()
    need(sha(data) == digest, "Exact immutable input " + path.name)
    return data


def document(path, digest):
    return json.loads(pin(path, digest))


def load(path, digest, name):
    module = ModuleType(name)
    module.__file__ = str(path)
    exec(compile(pin(path, digest), str(path), "exec"), module.__dict__)
    return module


def new_budget(arms, area):
    present = set(arms).intersection(NEW_ARMS)
    if not present:
        return 0
    need(present == set(NEW_ARMS) and 32 <= area < 1024, "Exact three-arm native eligibility")
    counts = []
    for name in NEW_ARMS:
        edit = arms[name]
        need(set(edit) == {"rle", "swaps"} and type(edit["swaps"]) is int
             and 1 <= edit["swaps"] <= min(8, area // 10), "Native realized cap and edit shape")
        counts.append(edit["swaps"])
    need(len(set(counts)) == 1, "Equal actual swaps for dynamic, static and learned")
    return counts[0]


def read_phase(root, phase, expected, replay):
    term_path = root / ("observed_" + phase) / "TERMINAL.json"
    term_data = term_path.read_bytes()
    terminal = json.loads(term_data)
    need(terminal["actual_exit"] == 0 and terminal["output_footer_verified"] is True
         and terminal["phase"] == phase, "Real closed execution and checked footer " + phase)
    result = document(root / phase / "RESULT_SAFE.json", terminal["result_sha256"])
    need(result["status"] == expected and result["protocol_sha256"] == PRELOCK_SHA
         and result["worker_sha256"] == terminal["worker_sha256"]
         and result["sidecar_accepted_sha256"] == BASE_SIDECAR_SHA
         and result["predictions_sha256"] == replay.PREDICTIONS
         and result["candidate_audit_sha256"] == replay.AUDIT
         and result["images"] == 705 and result["candidates"] == 70500
         and result["detector_forwards"] == result["optimizer_updates"] == 0
         and result["DEV"] is True
         and result["GT_opened"] is result["AP"] is result["TEST"] is result["FIT"] is False,
         "Exact label-free phase scope " + phase)
    return result, terminal["result_sha256"], sha(term_data)


def read_label_free(a, old, replay, dev):
    prepared, prepared_sha, prepared_terminal_sha = read_phase(
        a.phase_root, "prepare-edits", "DFINE_DEV705_CLASSICAL_ATTRACTION_PRELABEL_COMPLETE", replay)
    verified, verified_sha, verified_terminal_sha = read_phase(
        a.phase_root, "verify-edits", "DFINE_DEV705_CLASSICAL_ATTRACTION_PRELABEL_INDEPENDENT_ACCEPTED", replay)
    need(prepared["worker_sha256"] == verified["worker_sha256"]
         and verified["prepare_result_sha256"] == prepared_sha
         and prepared["edits_sha256"] == verified["edits_sha256"]
         and verified["exact_original_policy_projection"] is True
         and verified["exact_new_arm_area_xor_support"] is True,
         "Independent five-arm identity and three-arm native edit acceptance")
    sidecar = replay.probability_receipt(a, dev)
    manifest = dev.read_manifest(a.dev_root)
    modes = (*old.MODES, *NEW_ARMS)
    edit_hash = sha256()
    counts = Counter()
    blocks = []
    with (a.phase_root / "prepare-edits" / replay.EDITS).open("rb") as edits:
        for item, coco, audit, _ in replay.source_blocks(a, dev, manifest, sidecar, False):
            line = edits.readline(4 * 1024**2)
            need(line.endswith(b"\n"), "Complete candidate block on each original image")
            edit_hash.update(line)
            row = json.loads(line)
            need(set(row) == {"image_id", "changes"} and row["image_id"] == item["image_id"],
                 "Original image identity and edit schema")
            changes = row["changes"]
            for key, arms in changes.items():
                rank = int(key)
                need(str(rank) == key and 1 <= rank <= len(coco) == len(audit) == 100
                     and arms and set(arms) <= set(modes[1:]), "Original native ranks and complete policy list")
                n = new_budget(arms, audit[rank - 1]["mask_area"])
                if n:
                    counts.update(edited=1, swaps=n)
            blocks.append((item, coco, audit, changes))
        need(not edits.read(1) and edit_hash.hexdigest() == verified["edits_sha256"],
             "Exact complete approved edit stream")
    need(len(blocks) == 705 and sum(len(block[1]) for block in blocks) == 70500, "All original 705 images")
    for name in NEW_ARMS:
        for field in ("edited", "swaps"):
            key = name + "_" + field
            need(prepared["counts"][key] == verified["counts"][key] == counts[field],
                 "Verified aggregate per-arm census " + key)
    provenance = dict(protocol_sha256=PRELOCK_SHA, prepared_sha256=prepared_sha,
                      prepared_terminal_sha256=prepared_terminal_sha, verified_sha256=verified_sha,
                      verified_terminal_sha256=verified_terminal_sha,
                      replay_worker_sha256=verified["worker_sha256"], replay_edits_sha256=verified["edits_sha256"],
                      sidecar_accepted_sha256=BASE_SIDECAR_SHA,
                      predictions_sha256=replay.PREDICTIONS, candidate_audit_sha256=replay.AUDIT)
    return manifest, blocks, counts, provenance


def field_delta(left, right):
    metrics = ("AP", "AP50", "AP75", "AP90", "APs", "AR100")
    return {key: 100 * (left[key] - right[key]) if min(left[key], right[key]) >= 0 else None
            for key in metrics}


def coco_views(gt, empty_ids, blocks, dev, old, adapter, prior_raw, prior_endpoint):
    result = {view: {"arms": {}, "paired_contrasts_points": {}} for view in VIEWS}
    for name in (*old.MODES, *NEW_ARMS):
        rows = [row for _, coco, _, changes in blocks for row in old.candidate_rows(coco, changes, name)]
        measured = dev.score_variants(deepcopy(gt), deepcopy(rows), empty_ids, adapter.evaluate_population)
        for view in VIEWS:
            result[view]["arms"][name] = measured[view]
    for view in VIEWS:
        arms = result[view]["arms"]
        need(arms["raw"] == prior_raw[view], "Exact original raw AP parity for " + view)
        for left, right in COMPARISONS:
            result[view]["paired_contrasts_points"][left + " minus " + right] = field_delta(arms[left], arms[right])
    for name in old.MODES:
        measured = result["primary"]["arms"][name]
        reference = prior_endpoint["coco_fraction"][name]
        need(all(measured[k] == v for k, v in reference.items()), "Exact original five-arm COCO parity " + name)
    return result


def threshold_transitions(left, right, threshold):
    new, old = left >= threshold, right >= threshold
    return dict(gained=int((new & ~old).sum()), lost=int((~new & old).sum()),
                retained=int((new & old).sum()), both_below=int((~new & ~old).sum()))


def geometry(gt, blocks, old, matcher, modules, original):
    truth, neutral, poly = modules.geometry.polygon_support(gt, modules.vector)
    annotation_groups = defaultdict(list)
    for ann in gt["annotations"]:
        annotation_groups[ann["image_id"]].append(ann)
    groups = defaultdict(list)
    all_pairs = []
    modes = (*old.MODES, *NEW_ARMS)
    counts = Counter(stock_matches=0, neutral_or_invalid=0, area_outside=0, eligible=0)
    edited = Counter()
    for im, (item, coco, audit, changes) in zip(gt["images"], blocks, strict=True):
        identity = im["id"]
        need(item["image_id"] == identity, "Exact original fixed geometry population")
        pool = [dict(rank=i, score=row["score"], rle=row["segmentation"], area=info["mask_area"])
                for i, (row, info) in enumerate(zip(coco, audit, strict=True), 1)]
        with redirect_stdout(StringIO()):
            matches, _, _ = matcher.freeze_coco(im, annotation_groups[identity], pool)
        counts["stock_matches"] += len(matches)
        for gt_id, rank in matches.items():
            if gt_id in neutral or gt_id not in truth:
                counts["neutral_or_invalid"] += 1
                continue
            source = pool[rank - 1]
            area = source["area"]
            if not 32 <= area < 1024:
                counts["area_outside"] += 1
                continue
            counts["eligible"] += 1
            edits = changes.get(str(rank), {})
            shape = truth[gt_id][0]
            raw = old.continuous(source["rle"], area, shape, modules)
            measured = []
            for mode in modes:
                if mode in edits:
                    edited[mode] += 1
                    measured.append(old.continuous(edits[mode]["rle"], area, shape, modules))
                else:
                    measured.append(raw)
            row = np.asarray(measured)
            groups[old.as_group(item)].append(row)
            all_pairs.append(row)
    need(dict(counts) == original["counts"] and len(groups) == original["source_groups"] == 27
         and len(all_pairs) == counts["eligible"] == 1447 and poly == original["polygon_support"],
         "Exact historical native matches, polygon disposition and source-prefix count")
    raw_pairs = np.asarray(all_pairs)
    source_means = np.asarray([np.mean(groups[g], axis=0) for g in sorted(groups)])
    draws = np.random.Generator(np.random.PCG64(5217)).integers(len(groups), size=(4096, len(groups)))
    by_arm = {}
    for i, name in enumerate(modes):
        group_mean = source_means[:, i, :].mean(axis=0)
        pair_values = raw_pairs[:, i, 0]
        by_arm[name] = dict(group_equal_iou_points=float(group_mean[0]),
                            fp_per_truth_points=float(group_mean[1]), fn_per_truth_points=float(group_mean[2]),
                            pair_equal_iou_points=float(pair_values.mean()), strict75=int((pair_values >= 75).sum()),
                            strict90=int((pair_values >= 90).sum()))
        if name in old.MODES:
            ref = original["metrics"][name]
            need(all(abs(value - ref[key]) <= 1e-10 for key, value in by_arm[name].items()),
                 "Exact original continuous geometry within roundoff " + name)
    contrasts = {}
    for left, right in COMPARISONS:
        ia, ib = modes.index(left), modes.index(right)
        delta = source_means[:, ia, :] - source_means[:, ib, :]
        interval = np.percentile(delta[draws].mean(axis=1), [2.5, 97.5], axis=0)
        values = {key: dict(delta=float(delta[:, i].mean()),
                            percentile95=interval[:, i].tolist()) for i, key in enumerate(FIELDS)}
        for threshold in (75, 90):
            values["strict" + str(threshold)] = threshold_transitions(
                raw_pairs[:, ia, 0], raw_pairs[:, ib, 0], threshold)
        contrasts[left + " minus " + right] = values
    return dict(counts=dict(counts), source_groups=len(groups), bootstrap_draws=4096, bootstrap_seed=5217,
                metrics=by_arm, edited_original_eligible_matches={m: edited[m] for m in modes},
                polygon_support=poly, pairwise=contrasts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("code", "dev-source", "dev-root", "sidecar", "sidecar-accepted", "phase-root",
                 "protocol", "prior-raw", "prior-stock-check", "prior-endpoint", "shapely-site", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    a = parser.parse_args()
    a.sidecar_accepted_sha = BASE_SIDECAR_SHA
    need(sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1",
         "Isolated CPU-only scoring")
    old = load(a.code / BASE / "score_endpoint.py", BASE_SCORE_SHA, "_original_dev_score")
    replay = old.load_worker(a.code / BASE / "dev_endpoint.py")
    dev = replay.load(a.dev_source, replay.DEV_SOURCE, "_original_dev_capture")
    dev.cpu_runtime()
    pin(a.protocol, PRELOCK_SHA)
    adapter = replay.load(a.code / ADAPTER, dev.SCORE_ADAPTER_SHA, "_original_dev_coco_adapter")
    polygon = replay.load(a.code / "MASK_RTDETR_DEV705_CONTINUOUS_AP_20261112/continuous_endpoint.py",
                          old.GEOM, "_original_dev_geometry")
    matcher = replay.load(a.code / "HRSID_FIT_HULL_CDF_20261112/stock_coco_match.py", old.MATCH, "_stock_match")
    sys.path.append(str(a.shapely_site.resolve(strict=True)))
    import shapely

    need(shapely.__version__ == "2.0.2" and shapely.geos_version_string == "3.11.2",
         "Frozen geometry runtime")
    modules = polygon.geometry_modules(a.code)
    prior_raw = document(a.prior_raw, old.HISTORICAL)
    prior_stock_check = document(a.prior_stock_check, old.HISTORICAL_INDEPENDENT)
    prior = document(a.prior_endpoint, BASE_ENDPOINT_SHA)
    need(prior_stock_check["exact_26_metrics"] is True
         and prior_stock_check["score_sha256"] == old.HISTORICAL
         and prior_raw["gt_sha256"] == dev.FILES["DEV_ORIGINAL_GT_PRIVATE.json"],
         "Previously independently verified historical score")
    manifest, blocks, counts, source = read_label_free(a, old, replay, dev)
    a.output.mkdir()
    prelabel = dict(source, status="DFINE_DEV705_CLASSICAL_SCORE_PRELABEL_VERIFIED", DEV=True,
                    FIT=False, TEST=False, GT_opened=False, AP=False, images=len(blocks), **dict(counts))
    prelabel_sha = dev.write_new(a.output / "PRELABEL_SAFE.json", prelabel)
    started = perf_counter()
    gt = dev.dev_json(a.dev_root, "DEV_ORIGINAL_GT_PRIVATE.json")
    empty = dev.verify_labels(gt, dev.dev_json(a.dev_root, "DEV_EMPTY_RASTERS_PRIVATE.json"), manifest)
    coco = coco_views(gt, empty, blocks, dev, old, adapter, prior_raw, prior)
    fixed = geometry(gt, blocks, old, matcher, modules, prior["geometry_points"])
    primary = LEARNED + " minus " + DYNAMIC
    gates = dict(primary_ap_positive=coco["primary"]["paired_contrasts_points"][primary]["AP"] > 0,
                 same_ap_direction_sensitivity=coco["sensitivity"]["paired_contrasts_points"][primary]["AP"] >= 0,
                 positive_lower_iou_bound=fixed["pairwise"][primary]["continuous_iou_points"]["percentile95"][0] > 0)
    result = dict(source, status="DFINE_DEV705_CLASSICAL_MATCHED_FROZEN_SCORE_COMPLETE",
                  prelabel_sha256=prelabel_sha, previous_score_sha256=BASE_ENDPOINT_SHA,
                  gt_sha256=prior_raw["gt_sha256"], source_sha256=sha(Path(__file__).read_bytes()),
                  DEV=True, FIT=False, TEST=False, GT_opened=True, AP=True, official_TEST_AP=False,
                  detector_forwards=0, optimizer_updates=0, images=len(blocks), candidates=70500,
                  geometry_points=fixed, coco_fraction=coco, candidates_edited_per_new_arm=counts["edited"],
                  candidate_swaps_per_new_arm=counts["swaps"], raw_exact_parity_both_views=True,
                  five_arm_exact_primary_parity=True, gate=gates, advance=all(gates.values()),
                  score_seconds=perf_counter() - started)
    result_sha = dev.write_new(a.output / "RESULT_SAFE.json", result)
    print(json.dumps(dict(status=result["status"], result_sha256=result_sha), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

