"""Compare equal per-image edit budgets using one frozen boundary proposal order."""
import argparse
from collections import Counter
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


D = Path(__file__).resolve().parent.parent
OLD = D / "RFDETR_E24_FIT_SELECTOR_SCOUT_ACTUAL_PRIVATE_20261108_01"
MATERIAL = D / "RFDETR_E24_FIT_SELECTOR_MATERIALIZED_ACTUAL_PRIVATE_20261108_01"
RELEASE = D / "github_release_20261011/research_round_20260925_global/code"
BASE_SCOUT = RELEASE / "RFDETR_E24_FIT_BILATERAL_QUALITY_20261109/bilateral_quality_scout.py"
FIT_HELPER_DIR = RELEASE / "RFDETR_E24_FIT_EQUAL_HEAD_SCOUT_ACTUAL_20261108"
MATERIAL_API = FIT_HELPER_DIR / "fit_selector_materialize.py"
BASE_SCOUT_SHA = "de7bafabb5887d52f9e541f5875f53260f395b9e8a23bf69429d06539832e2d0"
ARMS = ("fixed", "confidence", "jaccard")
MAX_PREFIX = 32
FIXED_PREFIX = 16
TOP_RETAINED = 10


def need(condition, message):
    if not condition:
        raise ValueError(message)


def pinned(path, expected):
    raw = path.read_bytes()
    need(sha256(raw).hexdigest() == expected, "Exact closed input " + path.name)
    return raw


def pinned_json(path, expected):
    return json.loads(pinned(path, expected))


def load_module(path, digest, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(pinned(path, digest), str(path), "exec"), module.__dict__)
    return module


def exact_allocate(curves, budget):
    """Maximize separable prefix utility at an exact integer cardinality."""
    need(0 <= budget <= sum(len(c) - 1 for c in curves), "Exact budget must be feasible")
    state = np.full(budget + 1, -np.inf)
    state[0] = 0.0
    decisions = []
    for curve in curves:
        need(len(curve) > 0 and np.isfinite(curve).all(), "Finite prefix utility")
        best = np.full_like(state, -np.inf)
        choice = np.zeros(len(state), dtype=np.int32)
        for k, utility in enumerate(curve[:budget + 1]):
            trial = state[:budget + 1 - k] + utility
            dst = best[k:]
            improve = trial > dst
            dst[improve] = trial[improve]
            choice[k:][improve] = k
        state = best
        decisions.append(choice)
    need(np.isfinite(state[budget]), "Exact budget solution exists")
    selected = []
    remaining = budget
    for choice in reversed(decisions):
        k = int(choice[remaining])
        selected.append(k)
        remaining -= k
    need(remaining == 0, "Exact budget backtracking")
    return tuple(reversed(selected))


def prefix_expected_iou(area, p, direction, order):
    """Change in the ratio of expected intersection to expected union."""
    need(area > 0 and len(p) == len(direction) and np.all((0 <= p) & (p <= 1))
         and np.all((direction == 1) | (direction == -1)), "Finite binary edit geometry")
    addition = direction == 1
    i0 = area - float(p[~addition].sum())
    u0 = area + float(p[addition].sum())
    di = np.where(addition, p, p - 1)
    du = np.where(addition, 1 - p, -p)
    i = i0 + np.cumsum(di[order], dtype=np.float64)
    u = u0 + np.cumsum(du[order], dtype=np.float64)
    need((i >= -1e-9).all() and (u >= -1e-9).all(), "Consistent nonnegative expected mask geometry")
    q = np.divide(np.maximum(i, 0), u, out=np.zeros_like(i), where=u > 0)
    return np.concatenate(([0.0], np.clip(q, 0, 1) - i0 / u0))


def candidate_curves(rows, records, arrays, probability, codec):
    scores = np.array([r["score"] for r in records])
    selected = np.argsort(-scores, kind="stable")[:TOP_RETAINED]
    indices, orders, confidences, overlaps, capacities = [], [], [], [], []
    for i in selected:
        lo, hi = map(int, arrays["offsets"][i:i+2])
        if lo == hi:
            continue
        rle = records[i]["segmentation"]
        area = int(codec.area(dict(size=rle["size"], counts=rle["counts"].encode("ascii"))))
        need(area > 0 and hi - lo == rows[i]["stop"] - rows[i]["start"], "Exact nonempty supported native mask")
        p = probability[lo:hi]
        pos = arrays["positions"][lo:hi]
        order = np.lexsort((pos, -p))[:MAX_PREFIX]
        confidence = np.concatenate(([0.0], np.cumsum(2 * p[order] - 1, dtype=np.float64)))
        overlap = prefix_expected_iou(area, p, arrays["edit_sign"][lo:hi], order)
        indices.append(int(i))
        orders.append(order)
        confidences.append(confidence)
        overlaps.append(overlap)
        capacities.append(min(FIXED_PREFIX, len(order)))
    return indices, orders, confidences, overlaps, tuple(capacities)


def flip_mask(codec, segmentation, positions):
    if not len(positions):
        return segmentation
    raw = dict(size=segmentation["size"], counts=segmentation["counts"].encode("ascii"))
    canvas = np.ascontiguousarray(codec.decode(raw), dtype=np.uint8)
    flat = canvas.ravel()
    need(len(positions) <= MAX_PREFIX and len(np.unique(positions)) == len(positions)
         and positions.min() >= 0 and positions.max() < len(flat), "Exact bounded pixel support")
    flat[positions] ^= 1
    q = codec.encode(np.asfortranarray(canvas))
    need(q["size"] == segmentation["size"], "Preserve original output geometry")
    return dict(size=q["size"], counts=q["counts"].decode("ascii"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", required=True)
    args = parser.parse_args()
    start = time.monotonic()
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and "torch" not in sys.modules
         and len(args.protocol) == 40 and all(c in "0123456789abcdef" for c in args.protocol),
         "Exact CPU-only prospective FIT scope")
    args.output_dir.mkdir()
    base = load_module(BASE_SCOUT, BASE_SCOUT_SHA, "_closed_bilateral_helpers")
    api = load_module(MATERIAL_API, base.API_SHA, "_closed_fit_materializer")
    sys.path.insert(0, str(FIT_HELPER_DIR))
    scout = importlib.import_module("fit_selector_scout")
    need(sha256(Path(scout.__file__).read_bytes()).hexdigest() == "961ee68d5f067c9f7ffa08d2650a2b45554c76b60aa9554e661a13b58b86b6d4", "Exact previous FIT helper")
    spec = pinned_json(MATERIAL / "SPEC.json", base.SPEC_SHA)
    accepted = pinned_json(MATERIAL / "ROOT_INDEPENDENT_MATERIALIZATION_ACCEPTED_PRIVATE.json", base.MATERIAL_SHA)
    old_spec = pinned_json(OLD / "SPEC.json", "a45dd6bd99a097d69adb7f85c6007d2c53674b014009722f6cf50ba3a0c78534")
    old_accept = pinned_json(OLD / "ROOT_INDEPENDENT_FIT_SCOUT_ACCEPTED_PRIVATE.json",
                             "0b8d95e2d7a8546917beb85f37be4fd7eb0ecacd8f3d3eb2a6bed603d66b5364")
    old_heads = pinned_json(OLD / "HEAD_MODELS_PRIVATE.json",
                            "1959137a3e232136d81b7f575ae47263b922d5ac8a2d4820588d2fa9017bd83b")
    old_result = api.doc(old_accept["original_result_ref"])
    need(old_result["model_ref"]["sha256"] ==
         "1959137a3e232136d81b7f575ae47263b922d5ac8a2d4820588d2fa9017bd83b"
         and old_accept["status"] == "ACTUAL_FIT741_EQUAL_HEAD_SCOUT_ACCEPTED"
         and old_accept["material_acceptance_ref"]["sha256"] == base.MATERIAL_SHA
         and spec["DEV"] is spec["TEST"] is old_accept["DEV"] is old_accept["TEST"] is False,
         "Same independently closed original mean proposer")
    solver = api.load(old_spec["sources"]["solver"], "_closed_mean_solver")
    _, coco, evaluator, codec = api.runtime(spec)
    fit = api.doc(spec["image_manifest"])["items"]
    material = api.doc(accepted["material_index_ref"])
    capture = api.doc(spec["capture_index"])
    gt = api.original_gt(spec, fit)
    small = {r["id"] for r in gt["annotations"] if not r["iscrowd"] and r["area"] < 1024}
    means = old_heads["models"]["mean"]
    predictions = {k: [] for k in ("native", *ARMS)}
    selected_images = set()
    counts = {k: Counter() for k in ARMS}
    arm_seconds = {k: 0.0 for k in ARMS}
    chosen_count, edit_budget, multi_candidate_images = 0, 0, 0
    for item, meta, index in zip(fit, material, capture, strict=True):
        need(all(item[k] == meta[k] == index[k] for k in ("image_id", "dataset_index", "partition")),
             "Exact original capture and holdout identities")
        if item["partition"] != "head_holdout":
            continue
        image_id = item["image_id"]
        selected_images.add(image_id)
        rows, a = scout.load_image(np, meta)
        document, _ = api.captured_item(Path(spec["capture_root"]), item, index)
        records = document["records"]
        raw = scout.identities(image_id, document, rows)
        predictions["native"].extend(raw)
        scores = solver.predict_head(means, scout.features(a), scout.detector_scores(np, rows, a))
        idx, order, p, q, fixed = candidate_curves(rows, records, a, scores, codec)
        budget = sum(fixed)
        chosen_count += len(idx)
        multi_candidate_images += len(idx) >= 2
        edit_budget += budget
        need(len(idx) <= TOP_RETAINED, "Exact common retained prefix eligibility")
        for arm, utility in (("fixed", None), ("confidence", p), ("jaccard", q)):
            t = time.monotonic()
            k = fixed if utility is None else exact_allocate(utility, budget)
            need(sum(k) == budget and len(k) == len(idx), "Identical actual edit count for each image and arm")
            counts[arm]["images_with_different_allocation_than_fixed"] += k != fixed
            counts[arm]["flips_shifted_from_fixed"] += sum(abs(x-y) for x,y in zip(k,fixed,strict=True)) // 2
            counts[arm]["candidate_prefixes_above_16"] += sum(n > FIXED_PREFIX for n in k)
            edited = list(raw)
            for i, j, n in zip(idx, order, k, strict=True):
                lo = int(a["offsets"][i])
                pos = a["positions"][lo+j[:n]]
                edited[i] = dict(raw[i], segmentation=flip_mask(codec, raw[i]["segmentation"], pos))
                counts[arm]["actual_flips"] += n
                counts[arm]["edited_candidates"] += n > 0
                if rows[i]["status"] == "matched":
                    y0 = int(a["label_offsets"][i])
                    correct = int(a["y"][y0+j[:n]].sum())
                    counts[arm]["original_matched_correct_flips"] += correct
                    counts[arm]["original_matched_harmful_flips"] += n - correct
                    if rows[i]["GT_annotation_id"] in small:
                        counts[arm]["original_matched_small_net_correct_flips"] += 2 * correct - n
                else:
                    counts[arm]["original_unmatched_or_ignored_flips"] += n
            predictions[arm].extend(edited)
            arm_seconds[arm] += time.monotonic() - t
    scope = {**gt, "images":[r for r in gt["images"] if r["id"] in selected_images],
             "annotations":[r for r in gt["annotations"] if r["image_id"] in selected_images]}
    need(len(selected_images) == 111 and len(scope["annotations"]) == 292
         and all(len(p) == 9083 for p in predictions.values())
         and all(v["actual_flips"] == edit_budget for v in counts.values()),
         "Full fixed FIT holdout and same actual pixel budgets")
    metrics = {k: base.all_metrics(coco, evaluator, scope, p) for k, p in predictions.items()}
    n, j = metrics["native"], metrics["jaccard"]
    need(abs(n["AP"] - .7247336651207286) < 1e-14 and abs(n["AP75"] - .9016320389088902) < 1e-14,
         "Exact original frozen native holdout anchor")
    ap_delta = 100 * (j["AP"] - max(metrics[k]["AP"] for k in ("native","fixed","confidence")))
    passed = (ap_delta >= .1 and j["AP75"] >= n["AP75"] and
        counts["jaccard"]["original_matched_small_net_correct_flips"] >
        max(counts[k]["original_matched_small_net_correct_flips"] for k in ("fixed", "confidence")))
    result = dict(status="ACTUAL_EQUAL_PIXEL_BUDGET_FIT_SCOUT_COMPLETE",
        decision="PROVISIONAL_FIT_UTILITY_PASS" if passed else "STOP_FIXED_ALLOCATOR_RECIPE",
        protocol=args.protocol,holdout_images=111,holdout_GT=292,retained_candidates=9083,
        common_eligible_candidates=chosen_count,multi_candidate_images=multi_candidate_images,actual_edit_budget_each=edit_budget,
        top_retained_per_image=TOP_RETAINED,shared_prefix_cap=MAX_PREFIX,
        fixed_per_candidate=FIXED_PREFIX,metrics=metrics,counts={k:dict(v) for k,v in counts.items()},
        estimated_jaccard_delta_AP_over_strongest_control_pp=ap_delta,per_arm_selection_and_edit_seconds=arm_seconds,
        total_seconds=time.monotonic()-start,segmentation_updates=0,frozen_existing_mean_selector=True,
        original_DEV=False,TEST=False)
    api.put(args.output_dir/"RESULT_SAFE.json",result)
    print(json.dumps(result,sort_keys=True),flush=True)


if __name__ == "__main__":
    main()

