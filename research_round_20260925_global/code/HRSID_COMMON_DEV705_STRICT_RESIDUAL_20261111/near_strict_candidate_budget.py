"""Compare exact edit budgets for best-IoU versus all closed DEV705 candidates."""
import argparse
from collections import Counter
from contextlib import redirect_stdout
from hashlib import sha256
from io import StringIO
import json
import os
from pathlib import Path
from types import ModuleType

import numpy as np
from pycocotools import mask
from pycocotools.coco import COCO
from scipy.ndimage import maximum_filter, minimum_filter
from scipy.optimize import linear_sum_assignment


BUCKETS = (0, 4, 8, 16)
ARMS = ("native", "center")
METHODS = ("best_anywhere", "all_anywhere", "best_one_pixel", "all_one_pixel")
P311_SHA = "6fbefd4e62e7715518c1e460db9b0c32f5c407f11adef8de6cd5737b65346b54"
PRIOR_SHA = "280ae9abe55e67d1e85c0e3a22406e5489cf295f1a37c81068644d21ea8c7e7d"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def minimum_edits(deficit, fn_capacity, fp_capacity):
    """Optimal number when all truth-correct changes cost one bit each."""
    additions = min(fn_capacity, (deficit + 9)//10)
    deletions = (max(0, deficit - 10*additions) + 8)//9
    return additions+deletions if deletions <= fp_capacity else None


def assign_budget(matrix, budget):
    """Maximum one-to-one coverage, with minimum edit count within it."""
    rows, cols = matrix.shape
    if not rows or not cols:
        return 0, 0
    dummy = (budget+1)*(rows+1)
    costs = np.full((rows, cols+rows), dummy, dtype=np.int64)
    costs[:, :cols] = np.where(matrix <= budget, matrix, 2*dummy)
    selected, matches = linear_sum_assignment(costs)
    real = matches < cols
    totals = matrix[selected[real],matches[real]]
    require((totals <= budget).all(), "An inadmissible edge was assigned")
    return int(real.sum()), int(totals.sum())


def radius_one_capacity(pred, truth):
    require(pred.shape == truth.shape and pred.ndim == 2 and pred.dtype == truth.dtype == bool,
            "Exact Boolean canvas geometry required")
    out_band = maximum_filter(pred, size=3, mode="constant", cval=0) & ~pred
    in_band = ~minimum_filter(pred, size=3, mode="constant", cval=0) & pred
    return int((truth & out_band).sum()), int((~truth & in_band).sum())


def anchors(overlaps):
    """Freeze existing .90 true positives in native candidate and GT tie order."""
    used_targets = np.zeros(overlaps.shape[1], dtype=bool)
    used_candidates = []
    for candidate, row in enumerate(overlaps):
        eligible = np.flatnonzero((row >= .90) & ~used_targets)
        if len(eligible):
            selected = eligible[np.argmax(row[eligible])]
            used_targets[selected] = True
            used_candidates.append(candidate)
    return used_candidates


def cached_bitmap(cache, key, row):
    if key not in cache:
        cache[key] = mask.decode(row).astype(bool, copy=False)
    return cache[key]


def selected_window(pred_rle, gt_rle):
    """Crop to complete support plus a one-pixel halo within the image."""
    bboxes = np.array([mask.toBbox(rle) for rle in (pred_rle,gt_rle)], dtype=np.int64)
    x0, y0 = np.maximum(0, bboxes[:,:2].min(axis=0)-1)
    x1, y1 = np.minimum(pred_rle["size"][::-1], (bboxes[:,:2] + bboxes[:,2:]).max(axis=0)+1)
    require(x0 < x1 and y0 < y1, "Nonempty Boolean support crop required")
    return np.s_[y0:y1,x0:x1]


def image_costs(prefix, targets, overlaps, eligible, reserved, best):
    costs = {name: np.full((len(eligible),len(prefix)),max(BUCKETS)+1,dtype=np.int64) for name in METHODS}
    if not len(eligible) or not prefix:
        return costs, 0
    pred_areas = mask.area(prefix).astype(np.int64)
    gt_areas = mask.area(targets).astype(np.int64)
    images = overlaps[:,eligible]
    sums = pred_areas[:,None] + gt_areas[eligible][None,:]
    intersection = np.rint(images*sums/(1+images)).astype(np.int64)
    union = sums-intersection
    require(np.array_equal(intersection / union, images), "IoU does not recover the exact integer intersection")
    deficit = np.maximum(0,9*union-10*intersection)
    additions = np.minimum(gt_areas[eligible][None,:]-intersection,(deficit+9)//10)
    deletions = (np.maximum(0,deficit-10*additions)+8)//9
    need = additions + deletions
    require((deletions <= pred_areas[:,None]-intersection).all(), "Invalid unconstrained edit capacities")
    need[reserved,:] = max(BUCKETS)+1
    pred_cache, gt_cache = {}, {}
    admitted = 0
    for candidate, target_row in zip(*np.nonzero(need <= max(BUCKETS)),strict=True):
        target_index = eligible[target_row]
        pred_rle, gt_rle = prefix[candidate],targets[target_index]
        exact_intersection = int(mask.area(mask.merge([pred_rle,gt_rle],intersect=True)))
        require(exact_intersection == intersection[candidate,target_row],
                "Admitted edge differs from independent integer RLE overlap")
        admitted += 1
        cost = int(need[candidate,target_row])
        costs["all_anywhere"][target_row,candidate] = cost
        best_edge = candidate == best[target_index]
        if best_edge:
            costs["best_anywhere"][target_row,candidate] = cost
        window = selected_window(pred_rle,gt_rle)
        pred = cached_bitmap(pred_cache,candidate,pred_rle)[window]
        gt = cached_bitmap(gt_cache,target_index,gt_rle)[window]
        require(int(pred.sum()) == pred_areas[candidate] and int(gt.sum()) == gt_areas[target_index],
                "Padded support crop discarded foreground")
        band = minimum_edits(int(deficit[candidate,target_row]),*radius_one_capacity(pred,gt))
        if band is not None and band <= max(BUCKETS):
            require(band >= cost, "Restricted band beat the unrestricted minimum")
            costs["all_one_pixel"][target_row,candidate] = band
            if best_edge:
                costs["best_one_pixel"][target_row,candidate] = band
    return costs, admitted


def census(gt, native, center, p311, coverage):
    coco = COCO()
    coco.dataset = gt
    with redirect_stdout(StringIO()):
        coco.createIndex()
    require(len(coco.imgs) == 705 and len(coco.anns) == 1782 and [x["id"] for x in gt["categories"]] == [1],
            "Exact original ground truth population required")
    ordered = gt["images"]
    ranked = {arm:coverage.group_rows(rows,ordered) for arm,rows in zip(ARMS,(native,center),strict=True)}
    totals = {arm:{k:{name:Counter() for name in METHODS} for k in BUCKETS} for arm in ARMS}
    anchor_counts = Counter()
    near_strata, exclusions = Counter(), Counter(empty=0,crowd=0,ignore=0)
    prefix_count, checks = 0, Counter()
    for i,image in enumerate(ordered):
        targets, sizes = p311.image_targets(coco,image,exclusions)
        nonempty = {arm:p311.eligible_prefix(ranked[arm][i]) for arm in ARMS}
        quota = min(map(len,nonempty.values()))
        prefix_count += quota
        if not quota or not targets:
            continue
        prefix = {arm:nonempty[arm][:quota] for arm in ARMS}
        overlaps = {arm:mask.iou(rows,targets,[0]*len(targets)) for arm,rows in prefix.items()}
        best = {arm:rows.argmax(axis=0) for arm,rows in overlaps.items()}
        near = np.ones(len(targets),dtype=bool)
        reserved = {arm:anchors(rows) for arm,rows in overlaps.items()}
        for arm in ARMS:
            vals = overlaps[arm][best[arm],np.arange(len(targets))]
            near &= (vals >= .75) & (vals < .90)
            anchor_counts[arm] += len(reserved[arm])
        eligible = np.flatnonzero(near)
        near_strata.update(sizes[eligible].tolist())
        if not len(eligible):
            continue
        for arm in ARMS:
            costs, count = image_costs(prefix[arm],targets,overlaps[arm],eligible,reserved[arm],best[arm])
            checks[arm] += count
            for k in BUCKETS:
                per_image = {name:assign_budget(cost,k) for name,cost in costs.items()}
                require(per_image["all_anywhere"][0] >= max(per_image["best_anywhere"][0],per_image["all_one_pixel"][0])
                        and per_image["all_one_pixel"][0] >= per_image["best_one_pixel"][0]
                        and per_image["best_anywhere"][0] >= per_image["best_one_pixel"][0],
                        "Matching violates nested edge support")
                for name,(matches,edits) in per_image.items():
                    totals[arm][k][name].update(matches=matches,edits=edits,images_with_recovery=int(matches>0))
    require(prefix_count == 41482 and sum(near_strata.values()) == 827
            and anchor_counts == {"native":134,"center":271}
            and exclusions == {"empty":4,"crowd":0,"ignore":0}, "Prior fixed target, quota and strict anchors differ")
    return dict(images=705, equal_prefix_rows_per_arm=prefix_count, near_targets=sum(near_strata.values()),
                near_size_counts=dict(near_strata), exclusions=dict(exclusions), anchor_counts=dict(anchor_counts),
                exact_intersection_checks_at_max_budget=dict(checks), result_by_arm_and_budget=totals)


def load_source(path, expected, name):
    raw = path.read_bytes()
    require(sha256(raw).hexdigest() == expected, "Exact closed source or report required")
    module = ModuleType(name)
    module.__file__ = str(path)
    exec(compile(raw,str(path),"exec"),module.__dict__)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("code-root","pair","native-arm-receipt","center-arm-receipt","accepted-result",
                "prior-result","p311-source","output"):
        parser.add_argument("--"+arg,type=Path,required=True)
    for arg in ("pair-sha256","training-protocol","accepted-result-sha256"):
        parser.add_argument("--"+arg,required=True)
    a = parser.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and a.output.parent.resolve(strict=True)==a.output.parent,
            "CPU-only closed DEV diagnostic required")
    p311 = load_source(a.p311_source,P311_SHA,"_exact_p311")
    prior = a.prior_result.read_bytes()
    require(sha256(prior).hexdigest() == PRIOR_SHA
            and json.loads(prior)["common_near_strict"]["all"]["same_gt_pairs"] == 827,
            "Exact prior closed near-strict population required")
    require((a.pair_sha256,a.training_protocol,a.accepted_result_sha256)
            == (p311.PAIR_SHA,p311.TRAINING_PROTOCOL,p311.PRIMARY_SHA),"Exact closed experiment identity required")
    closed = p311.load_closed(a.code_root)
    coverage = closed.load_source(a.code_root/p311.COVERAGE,p311.COVERAGE_SHA,"_exact_candidate_order")
    driver = closed.load_gate(a.code_root).load_driver()
    primary = driver.read_json(a.accepted_result,a.accepted_result_sha256)
    require(primary["status"] == p311.PRIMARY_STATUS and primary["training_seed"] == 5202
            and primary["official_TEST_opened"] is False, "Exact prior independent acceptance required")
    gt,native,center,proof = closed.verified_inputs(a.code_root,a.pair,a.pair_sha256,
        native_arm_receipt=a.native_arm_receipt,center_arm_receipt=a.center_arm_receipt,
        training_protocol=a.training_protocol)
    require(proof == json.loads(prior)["provenance"], "Prior frozen prediction provenance differs")
    result = census(gt,native,center,p311,coverage)
    report = dict(status="HRSID_COMMON_DEV705_NEAR_STRICT_CANDIDATE_BUDGET_ACTUAL",
                  original_near_result_sha256=PRIOR_SHA, original_pair_sha256=a.pair_sha256,
                  truth_assisted=True, anchor_policy="freeze raw native-rank IoU .90 matches",
                  per_instance_budgets=BUCKETS, edited_outputs_saved=False,
                  COCO_AP=False, official_TEST_opened=False, model_forward_calls=0, **result)
    raw = (json.dumps(report,sort_keys=True,allow_nan=False)+"\n").encode()
    with a.output.open("xb") as file:
        file.write(raw)
    print(json.dumps(dict(status=report["status"],result_sha256=sha256(raw).hexdigest())),flush=True)


if __name__ == "__main__":
    main()
