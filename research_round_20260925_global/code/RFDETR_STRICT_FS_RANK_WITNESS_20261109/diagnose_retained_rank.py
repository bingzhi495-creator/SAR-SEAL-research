"""Audit retained high-IoU masks and original score ranks on closed DEV187."""
import numpy as np
from scipy.optimize import linear_sum_assignment


CUTOFFS = (1, 10, 100)
IOU = 0.90


def witness(iou):
    """Summarize an IoU matrix whose rows already follow the original scores."""
    qualified = iou >= IOU
    first = np.zeros(iou.shape[1], dtype=np.int64)
    best = np.zeros(iou.shape[1], dtype=np.float64)
    if len(iou):
        first = np.where(qualified.any(axis=0), qualified.argmax(axis=0) + 1, 0)
        best = iou.max(axis=0)
    counts = {}
    for cutoff in CUTOFFS:
        top = qualified[:cutoff]
        row, col = linear_sum_assignment(top, maximize=True)
        counts[str(cutoff)] = int(top[row, col].sum())
    return {"first_rank": first, "best_iou": best, "injective": counts}


def compare_first_ranks(f, s):
    """Separate presence changes from score-rank changes on paired GT."""
    both = (f > 0) & (s > 0)
    gained = (f == 0) & (s > 0)
    lost = (f > 0) & (s == 0)
    shifts = {}
    for k in CUTOFFS:
        s_only_inside = int(np.count_nonzero(gained & (s <= k)))
        f_only_inside = int(np.count_nonzero(lost & (f <= k)))
        both_entered = int(np.count_nonzero(both & (f > k) & (s <= k)))
        both_left = int(np.count_nonzero(both & (f <= k) & (s > k)))
        shifts[str(k)] = dict(
            S_only_inside=s_only_inside, F_only_inside=f_only_inside,
            both_entered=both_entered, both_left=both_left,
            net_S_minus_F=s_only_inside - f_only_inside + both_entered - both_left,
        )
    return dict(
        availability=dict(both=int(both.sum()), S_only=int(gained.sum()),
                          F_only=int(lost.sum()), neither=int(((f == 0) & (s == 0)).sum())),
        both_rank_direction=dict(S_earlier=int(np.count_nonzero(both & (s < f))),
                                 S_later=int(np.count_nonzero(both & (s > f))),
                                 same=int(np.count_nonzero(both & (s == f)))),
        cutoff_decomposition=shifts,
    )


def inside(rank, cutoff):
    return int(np.count_nonzero((rank > 0) & (rank <= cutoff)))


def median_or_none(values):
    return float(np.median(values)) if len(values) else None


def main():
    from collections import defaultdict
    from hashlib import sha256
    from importlib.metadata import version
    import json
    import math
    import os
    from pathlib import Path
    import sys

    from pycocotools import mask

    root = Path(__file__).resolve().parent.parent
    helper = root / "RFDETR_E24_STRICT_OPTIMIZER_EVAL_CANDIDATE_20261109_01"
    helper_sha = "bfb6f0148e106a15c8d503796b594c51489b8f75140f641b2f8bbcf8a367f135"
    if sha256((helper / "verify_optimizer_eval_cpu.py").read_bytes()).hexdigest() != helper_sha:
        raise ValueError("Original fixed DEV verification helper changed")
    sys.path.insert(0, str(helper))
    from verify_optimizer_eval_cpu import CONTROL, GT, OUT, OWNER, PAIR, need, raw_ref, read, read_worker_result

    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and version("pycocotools") == "2.0.11",
         "Pinned CPU-only DEV mask runtime required")
    eval_sha = "a0ea5a22c60162c3076269d1d01ab18be53ab7de838420da7a6658cd90b452e7"
    rank_sha = "e1605021c825b23845d1dba31147dea1fb01a4640d2583d968e1a8ad04448a6f"
    iou_sha = "30fd9508657a1b86db1c01e2c566192fb59e639f749e82e8bee66b116f74e46e"
    prior_files = {
        "ROOT_SAFE_FIXED_EVAL_RESULT.json": eval_sha,
        "ROOT_SAFE_STRICT_COVERAGE_SCORE_CUTOFF.json": rank_sha,
        "ROOT_SAFE_FIXED_EVAL_IOU_DECOMPOSITION.json": iou_sha,
    }
    for name, digest in prior_files.items():
        need(sha256((PAIR / name).read_bytes()).hexdigest() == digest, "P101 safe evidence changed")
    prior, prior_rank, prior_iou = (read(PAIR / name) for name in prior_files)
    controller = read(OWNER / "EVAL_CONTROLLER_TERMINAL_PRIVATE.json")
    wait = read(CONTROL / "TERMINAL_PRIVATE.json")
    need(wait["exit"] == controller["exit"] == 0 and wait["original_OS_wait"] is controller["original_OS_wait"] is True
         and wait["guard"] is None and controller["automatic_retry"] is False
         and Path(wait["stdout_ref"]["path"]) == CONTROL / "WORKER.stdout"
         and Path(wait["stderr_ref"]["path"]) == CONTROL / "WORKER.stderr"
         and prior["CPU_recompute"] is True and prior["TEST"] is prior_rank["TEST"] is prior_iou["TEST"] is False,
         "Original closed internal DEV with independent scoring required")
    raw_ref(wait["stderr_ref"], 1024**2)
    observed = read_worker_result(raw_ref(wait["stdout_ref"], 1024**2), OUT / "RESULT.json")
    need(observed["protocol_commit"] == prior["protocol_commit"] == prior_rank["protocol_commit"]
         == prior_iou["protocol_commit"] and observed["TEST"] is False and observed["new_training"] is False
         and observed["integer_GT_ref"]["sha256"] == GT, "Exact P101 internal DEV sources")
    gt = json.loads(raw_ref(observed["integer_GT_ref"], 1024**2))
    images = {row["id"]: (row["height"], row["width"]) for row in gt["images"]}
    need(len(images) == 187 and len(gt["images"]) == 187 and len(gt["annotations"]) == 380
         and len({row["id"] for row in gt["annotations"]}) == 380, "Fixed DEV187/380 population")
    truths = defaultdict(list)
    for row in gt["annotations"]:
        image_id = row["image_id"]
        h, w = images[image_id]
        need(row["category_id"] == 1 and row["area"] > 0 and row.get("iscrowd", 0) == 0, "Expected ordinary ship GT")
        seg = row["segmentation"]
        if isinstance(seg, list):
            seg = mask.merge(mask.frPyObjects(seg, h, w))
        elif isinstance(seg["counts"], list):
            seg = mask.frPyObjects(seg, h, w)
        truths[image_id].append((row["area"] < 1024, seg))
    small = np.array([small for image_id in images for small, _ in truths[image_id]])
    need(int(small.sum()) == 274, "Expected recorded-area small GT count")
    strata = {"all": np.ones_like(small), "small": small, "other": ~small}
    ranks, best, arms = {}, {}, {}
    expected_geometry = prior["retained_geometry"]
    for arm in ("E24", "F", "S"):
        pred_ref = observed["arms"][arm]["predictions_ref"]
        need(Path(pred_ref["path"]) == OUT / f"{arm}_PREDICTIONS.json"
             and pred_ref["sha256"] == expected_geometry["prediction_sha256"][arm],
             "Expected P101 retained predictions only")
        predictions = json.loads(raw_ref(pred_ref, 256*1024**2))
        need(len(predictions) == prior["arms"][arm]["retained_predictions"], "Exact retained prediction count")
        by_image = defaultdict(list)
        for row in predictions:
            need(row["image_id"] in images and row["category_id"] == 1 and math.isfinite(row["score"]),
                 "Valid original DEV prediction")
            by_image[row["image_id"]].append(row)
        arm_ranks, arm_best = [], []
        injective = {str(k): 0 for k in CUTOFFS}
        for image_id in images:
            rows = sorted(by_image[image_id], key=lambda row: -row["score"])
            gt_rows = [rle for _, rle in truths[image_id]]
            need(len(rows) <= 100, "Original retained top100 scope")
            overlap = (
                mask.iou([row["segmentation"] for row in rows], gt_rows, [0] * len(gt_rows))
                if rows and gt_rows else np.zeros((len(rows), len(gt_rows)))
            )
            case = witness(overlap)
            arm_ranks.extend(case["first_rank"])
            arm_best.extend(case["best_iou"])
            for key, count in case["injective"].items():
                injective[key] += count
        need(injective == prior_rank["maximum_injective_score_ranked_coverage"][arm],
             "Reproduce P101 one-to-one coverage at all fixed cutoffs")
        ranks[arm], best[arm] = np.asarray(arm_ranks), np.asarray(arm_best)
        groups = {}
        for name, select in strata.items():
            rank = ranks[arm][select]
            covered = rank[rank > 0]
            groups[name] = dict(
                GT=int(select.sum()), independent_first_rank_counts={str(k): inside(rank, k) for k in CUTOFFS},
                median_first_rank_given_available=median_or_none(covered),
                median_best_iou=median_or_none(best[arm][select]),
            )
        free = expected_geometry["coverage"][arm]["0.90"]
        for name, key in (("all", "per_gt_max"), ("small", "small_per_gt_max"), ("other", "other_per_gt_max")):
            need(groups[name]["independent_first_rank_counts"]["100"] == free[key],
                 "Reproduce exact P101 score-free opportunities")
        arms[arm] = dict(groups=groups, maximum_injective_all=injective, AP90=prior_iou["AP_by_IoU"][arm]["0.90"])
    pairs = {}
    for name, select in strata.items():
        f, s = ranks["F"][select], ranks["S"][select]
        both = (f > 0) & (s > 0)
        row = compare_first_ranks(f, s)
        expected = expected_geometry["S_vs_F_per_GT"]["0.90"][name]
        need(row["availability"]["S_only"] == expected["gained"] and row["availability"]["F_only"] == expected["lost"],
             "Reproduce exact P101 paired opportunity changes")
        for cutoff in CUTOFFS:
            need(row["cutoff_decomposition"][str(cutoff)]["net_S_minus_F"] == inside(s, cutoff) - inside(f, cutoff),
                 "Each paired cutoff must balance")
        row.update(median_first_rank_for_both={arm: median_or_none(rank[both]) for arm, rank in (("F", f), ("S", s))},
                   median_S_minus_F_rank_for_both=median_or_none(s[both] - f[both]),
                   median_S_minus_F_best_iou_for_both=median_or_none((best["S"] - best["F"])[select][both]))
        pairs[name] = row
    report = dict(
        status="EXPLORATORY_P101_CLOSED_DEV187_RETAINED_MASK_RANK_WITNESS",
        source_commit="4d2954ebce27f73cde209fedfdf188a3c40bf07b", protocol_commit=prior["protocol_commit"],
        source_evidence_sha256=prior_files, integer_GT_sha256=GT,
        predictions_sha256=expected_geometry["prediction_sha256"], images=len(images), GT_instances=len(small),
        DEV=True, TEST=False, new_training=False, CPU_only=True, official_metrics_unchanged=True,
        IoU=IOU, score_cutoffs=list(CUTOFFS), small_definition="recorded integer annotation area < 1024",
        ranking="descending original score within image; ties use retained file order",
        independent_witness="earliest rank with retained mask IoU >= 0.90 for each GT; one mask may witness multiple GT",
        paired_limit="same GT across arms; each arm may have different qualifying masks, so shared availability does not fix geometry",
        arms=arms, S_vs_F=pairs,
    )
    output = Path(__file__).resolve().with_name("RANK_WITNESS_SAFE.json")
    with output.open("x", encoding="utf-8", newline="\n") as file:
        json.dump(report, file, indent=2, sort_keys=True, allow_nan=False)
        file.write("\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
