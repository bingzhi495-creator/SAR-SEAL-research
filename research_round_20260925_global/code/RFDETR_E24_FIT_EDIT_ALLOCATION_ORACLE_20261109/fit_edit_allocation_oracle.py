"""Measure equal-capacity repair bounds on the original SSDD FIT holdout."""
from collections import Counter, defaultdict
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import time
from types import ModuleType

import numpy as np


D = Path(__file__).resolve().parents[1]
RELEASE = D / "github_release_20261011/research_round_20260925_global/code"
MATERIAL = D / "RFDETR_E24_FIT_SELECTOR_MATERIALIZED_ACTUAL_PRIVATE_20261108_01"
MATERIAL_SHA = "742053fc509b9168b02727a8ed58120a471569b597916da764ea8a06e0fee9c4"
SPEC_SHA = "a8f1ebefc3e67ed2a857e7ae88beb1031aa55c37428de4e8588eaa93d387c617"
API = RELEASE / "RFDETR_E24_FIT_EQUAL_HEAD_SCOUT_ACTUAL_20261108/fit_selector_materialize.py"
API_SHA = "c3f19a7dab49729c37737bf12f56683ba9ff0bef1432607fdb42c4691234530e"
P71 = D / "RFDETR_E24_FIT_SELECTOR_SCOUT_ACTUAL_PRIVATE_20261108_01"
P71_ACCEPTANCE_SHA = "0b8d95e2d7a8546917beb85f37be4fd7eb0ecacd8f3d3eb2a6bed603d66b5364"
P71_HEAD_SHA = "1959137a3e232136d81b7f575ae47263b922d5ac8a2d4820588d2fa9017bd83b"
P71_SOLVER = RELEASE / "RFDETR_E24_FIT_EQUAL_HEAD_SOLVER_CANDIDATE_20261108/solver.py"
P71_SOLVER_SHA = "e375d7a06ef7bc652e51a47cc2c16ee06fc280a6172fbd4eb5ed16ba3ce5738c"
CAPTURE_API = RELEASE / "RFDETR_E24_FIT_NATIVE_CAPTURE_CANDIDATE_20261108/capture_post.py"
CAPTURE_API_SHA = "957a9e6845588c475da9bdce38ba6a1fc70f6c51ddbc65579ed4c2c376a8bdb9"
P106_API = RELEASE / "RFDETR_STRICT_FS_RANK_WITNESS_20261109/PIXEL_FLIP_ORACLE_CANDIDATE_20261109_01/coarse_fp_pixel_oracle.py"
P106_API_SHA = "11526696262c12f5396f4009cedbafdae0dc0cdd869dd894692bbaa48d1e059d"


def need(condition, message):
    if not condition:
        raise ValueError(message)


def pinned(path, digest):
    raw = path.read_bytes()
    need(sha256(raw).hexdigest() == digest, "Exact closed input " + path.name)
    return raw


def load_module(name, path, digest):
    module = ModuleType(name)
    module.__file__ = str(path)
    exec(compile(pinned(path, digest), str(path), "exec"), module.__dict__)
    return module


def allocation_summary(images):
    """Count attainable repairs under fixed caps and two equally funded pools."""
    all_rows = [row for image in images for row in image]

    def funded_pool(rows):
        capacity = sum(cap for cap, _ in rows)
        spent = count = 0
        for cost in sorted(cost for _, cost in rows if cost is not None):
            if spent + cost > capacity:
                break
            count += 1
            spent += cost
        return count, spent

    def section(count, spent):
        return dict(eligible=count, spent=spent, unused_capacity=capacity - spent)

    need(all(cap >= 0 and (cost is None or cost > 0) for cap, cost in all_rows),
         "Nonnegative per-candidate capacity and positive repair costs")
    capacity = sum(cap for cap, _ in all_rows)
    fixed = [cost for cap, cost in all_rows if cost is not None and cost <= cap]
    image_totals = [funded_pool(image) for image in images]
    fixed_result = section(len(fixed), sum(fixed))
    image_result = section(sum(n for n, _ in image_totals), sum(k for _, k in image_totals))
    global_result = section(*funded_pool(all_rows))
    need(fixed_result["eligible"] <= image_result["eligible"] <= global_result["eligible"],
         "Capacity pooling must contain the original feasible assignments")
    return dict(representatives=len(all_rows), feasible_without_total_cap=sum(cost is not None for _, cost in all_rows),
                capacity=capacity, fixed=fixed_result, per_image_pool=image_result, global_pool=global_result,
                per_image_pool_gain=image_result["eligible"] - fixed_result["eligible"],
                global_pool_gain=global_result["eligible"] - fixed_result["eligible"])




def prefix_repair(intersection, union, ordered_margin_changes, fixed_cap):
    """Return first successful prefix and actual outcome at the full fixed cap."""
    need(0 <= fixed_cap <= len(ordered_margin_changes), "Fixed capacity must be a valid prefix")
    margin = 10 * intersection - 9 * union + np.cumsum(ordered_margin_changes, dtype=np.int64)
    qualifying = np.flatnonzero(margin >= 0)
    minimum = int(qualifying[0] + 1) if len(qualifying) else None
    return minimum, bool(fixed_cap and margin[fixed_cap - 1] >= 0)




def select_representatives(iou, small):
    """Select one first native-ranked unique coarse candidate per missing small GT."""
    coverage = Counter({k: 0 for k in ("already_strict_gt", "unique_coarse_representatives",
                                      "ambiguous_only_gt", "no_candidate_ge050_gt")})
    pairs = []
    ge50 = iou >= .5
    unique = ge50.sum(axis=1) == 1
    for g in np.flatnonzero(small):
        if np.any(iou[:, g] >= .9):
            coverage["already_strict_gt"] += 1
            continue
        options = np.flatnonzero(ge50[:, g] & unique)
        if len(options):
            pairs.append((int(options[0]), int(g)))
            coverage["unique_coarse_representatives"] += 1
        else:
            name = "ambiguous_only_gt" if np.any(ge50[:, g]) else "no_candidate_ge050_gt"
            coverage[name] += 1
    return pairs, dict(coverage)


def examine_image(item, capture, material, annotations, api, codec, capture_post, oracle, capture_root, score_mean):
    document, sparse = api.captured_item(capture_root, item, capture)
    rows = document["records"]
    need(all(a["native_rank"] < b["native_rank"] for a, b in zip(rows, rows[1:])),
         "Original ascending native rank order")
    with np.load(BytesIO(sparse), allow_pickle=False) as arrays:
        need(set(arrays.files) == {"offsets", "positions", "logits"}, "Exact sparse array roles")
        offsets, positions, logits = (arrays[k] for k in ("offsets", "positions", "logits"))
    need(offsets.dtype == np.int64 and positions.dtype == np.uint32 and logits.dtype == np.float32
         and offsets.ndim == positions.ndim == logits.ndim == 1 and len(offsets) == len(rows)+1
         and offsets[0] == 0 and offsets[-1] == len(positions) == len(logits)
         and np.all(offsets[1:] >= offsets[:-1]), "Aligned original support spans")
    h, w = item["original_hw"]
    regular = [a for a in annotations if not a.get("iscrowd", 0)]
    candidates = [dict(size=r["segmentation"]["size"], counts=r["segmentation"]["counts"].encode("ascii"))
                  for r in rows]
    truths = [codec.merge(codec.frPyObjects(a["segmentation"], h, w)) for a in regular]
    matrix = (codec.iou(candidates, truths, [0]*len(truths)) if candidates and truths
              else np.zeros((len(candidates), len(truths))))
    need(matrix.shape == (len(candidates), len(truths)) and np.isfinite(matrix).all()
         and np.all((matrix >= 0) & (matrix <= 1)), "Finite original prediction and GT mask IoU")
    small = [a["area"] < 1024 for a in regular]
    pairs, coverage = select_representatives(matrix, small)
    if pairs:
        matched_rows = api.doc(material["match_ref"])["rows"]
        feature_bytes = api.checked(material["arrays_ref"], 16*1024**2)
        with np.load(BytesIO(feature_bytes), allow_pickle=False) as f:
            features = {k: f[k] for k in ("offsets", "positions", "features", "edit_sign", "valid")}
        need(len(rows) == len(matched_rows) and features["features"].shape == (len(positions), 3)
             and np.array_equal(features["offsets"], offsets) and np.array_equal(features["positions"], positions),
             "Frozen P71 aligned candidate geometry and support")
    repairs = []
    for index, gt_index in pairs:
        pred = capture_post._decode_mask(np, codec, rows[index]["segmentation"], [h, w]).astype(bool)
        truth = codec.decode(truths[gt_index]).astype(bool)
        lo, hi = map(int, offsets[index:index+2])
        coord, values = positions[lo:hi], logits[lo:hi]
        need(np.isfinite(values).all() and np.array_equal(coord, capture_post._band_positions(np, pred))
             and np.array_equal(values > 0, pred.ravel()[coord]), "Exact radius-two support and logit signs")
        geometry = oracle.measure_pair(pred, truth, lambda *_: coord)
        i, u = geometry["intersection"], geometry["union"]
        need(0 < i <= u and 2*i >= u and 10*i < 9*u and abs(i/u - matrix[index, gt_index]) < 1e-14,
             "Exact unique coarse pair geometry")
        cap = min(16, len(coord))
        was_predicted = pred.ravel()[coord]
        in_truth = truth.ravel()[coord]
        changes = np.where(was_predicted, np.where(in_truth, -10, 9), np.where(in_truth, 10, -9))
        order = np.lexsort((coord, np.abs(values)))
        first, forced = prefix_repair(i, u, changes[order], cap)
        best = geometry["band_min_edits"]
        need(first is None or (best is not None and best <= first <= len(coord)),
             "Order-free oracle must bound the same-band prefix")
        m = matched_rows[index]
        need(all(m[k] == rows[index][k] for k in ("native_rank", "query_id", "score"))
             and (m["start"], m["stop"]) == (lo, hi), "Exact selected P71 row identity")
        f = features["features"][lo:hi]
        direction = features["edit_sign"][lo:hi]
        need(np.array_equal(f[:, 0], np.abs(values)) and np.array_equal(direction, np.where(was_predicted, -1, 1)),
             "Frozen P71 feature magnitude and edit direction")
        mean = score_mean(dict(abs_logit=f[:, 0], mean_contrast=f[:, 1], edit_sign=direction,
                               valid=features["valid"][lo:hi]), rows[index]["score"])
        mean_first, mean_forced = prefix_repair(i, u, changes[np.lexsort((coord, -mean))], cap)
        need(mean_first is None or (best is not None and best <= mean_first <= len(coord)),
             "Order-free oracle must bound the same-band mean proposal")
        repairs.append((cap, best, first, forced, mean_first, mean_forced))
    return dict(repairs=repairs, coverage=coverage, candidates=len(rows), annotations=len(annotations),
                regular_small=sum(small))


def main():
    import argparse
    import re

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    need(re.fullmatch(r"[a-f0-9]{40}", args.protocol) and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
         and "torch" not in sys.modules and not args.output_dir.exists(), "Fresh prelocked CPU-only execution")
    spec = json.loads(pinned(MATERIAL/"SPEC.json", SPEC_SHA))
    accepted = json.loads(pinned(MATERIAL/"ROOT_INDEPENDENT_MATERIALIZATION_ACCEPTED_PRIVATE.json", MATERIAL_SHA))
    need(spec["schema"] == "Root-FIT741-selector-materialization-v1"
         and spec["DEV"] is spec["TEST"] is accepted["DEV"] is accepted["TEST"] is False
         and accepted["status"] == "ACTUAL_FULL_FIT741_FIXED_MATCH_FEATURES_ACCEPTED"
         and spec["capture_acceptance"] == accepted["capture_acceptance_ref"],
         "Original independently accepted FIT-only input roles")
    api = load_module("_fit_materialization_closed", API, API_SHA)
    capture_post = load_module("_fit_capture_post_closed", CAPTURE_API, CAPTURE_API_SHA)
    oracle = load_module("_fit_p106_geometry_closed", P106_API, P106_API_SHA)
    _, _, _, codec = api.runtime(spec)
    p71 = json.loads(pinned(P71/"ROOT_INDEPENDENT_FIT_SCOUT_ACCEPTED_PRIVATE.json", P71_ACCEPTANCE_SHA))
    need(p71["status"] == "ACTUAL_FIT741_EQUAL_HEAD_SCOUT_ACCEPTED" and p71["DEV"] is p71["TEST"] is False
         and p71["material_acceptance_ref"]["sha256"] == MATERIAL_SHA,
         "Original independently verified P71 source and holdout")
    heads = api.doc(api.doc(p71["original_result_ref"])["model_ref"])
    need(api.file_ref(P71/"HEAD_MODELS_PRIVATE.json")["sha256"] == P71_HEAD_SHA
         and heads["models"]["mean"]["branch"] == "mean", "Frozen original P71 mean head")
    solver = load_module("_fit_p71_solver_closed", P71_SOLVER, P71_SOLVER_SHA)
    def score_mean(features, detector_score):
        return solver.predict_head(heads["models"]["mean"], features, detector_score)

    fit = api.doc(spec["image_manifest"])
    items = fit["items"]
    captures = api.doc(spec["capture_index"])
    materials = api.doc(accepted["material_index_ref"])
    need(len(items) == len(captures) == len(materials) == 741
         and fit["logical_counts"] == {"head_fit":516,"calibration":114,"head_holdout":111},
         "Original FIT741 partitions")
    annotations = defaultdict(list)
    for a in api.original_gt(spec, items)["annotations"]:
        annotations[a["image_id"]].append(a)
    holdout = []
    for item, capture, material in zip(items, captures, materials, strict=True):
        need(all(item[k] == capture[k] == material[k] for k in ("partition", "image_id", "dataset_index")),
             "Exactly the original immutable partition and image order")
        if item["partition"] == "head_holdout":
            holdout.append(examine_image(item, capture, material, annotations[item["image_id"]], api, codec,
                                         capture_post, oracle, Path(spec["capture_root"]), score_mean))
    need(len(holdout) == 111 and sum(r["candidates"] for r in holdout) == 9083
         and sum(r["annotations"] for r in holdout) == 292, "Exact original holdout population")
    coverage = Counter()
    for r in holdout:
        coverage.update(r["coverage"])
    need(sum(coverage.values()) == sum(r["regular_small"] for r in holdout), "Exhaustive small-GT availability categories")
    band, prefix, mean = (allocation_summary([[(x[0], x[index]) for x in r["repairs"]] for r in holdout])
                          for index in (1, 2, 4))
    forced, mean_forced = (sum(x[index] for r in holdout for x in r["repairs"]) for index in (3, 5))
    for ordered, exact in ((prefix, forced), (mean, mean_forced)):
        need(band["capacity"] == ordered["capacity"]
             and all(ordered[k]["eligible"] <= band[k]["eligible"] for k in ("fixed", "per_image_pool", "global_pool"))
             and exact <= ordered["fixed"]["eligible"], "Equal caps and valid nested opportunity bounds")
    report = dict(status="ACTUAL_E24_FIT_ONLY_EDIT_CAPACITY_ORACLE", prelock_commit=args.protocol,
        holdout_images=111, frozen_candidates=9083, holdout_GT=292, small_GT_coverage=dict(coverage),
        images_with_representatives=sum(bool(r["repairs"]) for r in holdout),
        budget_rule="sum of min(16, original band span) over one selected representative per missing small GT",
        representative_rule="first native-ranked candidate having exactly one GT at IoU>=.50; "
                            "small GT with any original IoU>=.90 candidate excluded",
        support="fixed radius-two clipped both-sign window", small="source integer area < 1024",
        free_correct_pixel_order_bound=band, fixed_native_uncertainty_prefix_bound=prefix,
        native_forced_full_fixed_prefix_reaches_090=forced, fixed_p71_mean_prefix_bound=mean,
        p71_mean_forced_full_fixed_prefix_reaches_090=mean_forced, p71_mean_head_sha256=P71_HEAD_SHA,
        p71_mean_prefix_order="descending original P71 mean head probability then ascending flat pixel index",
        fixed_prefix_order="ascending original absolute logit then ascending original flat pixel index",
        capacity_is_maximum_not_forced_spend=True, ground_truth_used_for_selection_and_bounds=True,
        predictions_created=False, AP=False, original_DEV=False, TEST=False, training=False,
        elapsed_seconds=time.monotonic() - start)
    args.output_dir.mkdir()
    api.put(args.output_dir/"RESULT_SAFE.json", report)
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

