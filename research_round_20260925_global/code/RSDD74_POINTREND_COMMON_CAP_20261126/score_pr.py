"""Independently verify and evaluate the frozen PointRend-by-writer crossover."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import sys
from time import perf_counter
from types import ModuleType

import numpy as np
from scipy.optimize import linear_sum_assignment


HISTORICAL_SCORE = "bb751d3d42875b4cc9e980534697430283428b9ecfbb1c3c9bc09d26b63028e5"
HISTORICAL_RENDER = "34d4e588dc757dd60c975ebe9498d479a80265bf568422d57b3fa0fa1f6975f5"
BOUNDED_RENDER = "4a015eef90344a4d06bd77fedb5fd5a1ea9d5742811f7de01a96dcb6a24ac994"
BOUNDED_RENDER_TERMINAL = "92fea4ab053d319397b823bafe3cab7991c793c55a17802df0bd6bb64b11bcee"
BOUNDED_SCORE = "648f9509bf889d11da57a5680e14dee95438f0318a32a93eae0085cab6a1c97c"
BOUNDED_SCORE_TERMINAL = "844d94c6532599bd97aa8c513683be11ad40bf2ac576210895fad18e7dd34fef"
SPARSE = "cb9b8e839de85588fdf116ce5554ed6cadbc20bdbe76e38180561528bfd870d5"
POINTREND_PRIOR_RENDER = "dbc6e2dbce37cdd8fea34ccefbce16af0ed1ce58db2495de682e278921626fc8"
SOLVER = "99817f192798f52f03efe5e3ac422a453019bf458d33096a03844cb60360436c"
BOUNDARY = "b477c918a1168eb6b734313b18c4df8e338a591305c164a444eb61a560e873c5"
RENDER_STATUS = "RSDD74_POINTREND_COMMON_CAP_RENDER_COMPLETE"
VERIFY_STATUS = "RSDD74_POINTREND_COMMON_CAP_VERIFY_COMPLETE"
SCORE_STATUS = "RSDD74_POINTREND_COMMON_CAP_SCORE_COMPLETE"
LEARNED = ("fixed", "bounded", "free_matched_count")
POINTREND = ("pr_fixed", "pr_bounded", "pr_independent")
MASK_ARMS = (*LEARNED, *POINTREND)
ARMS = ("raw", *MASK_ARMS)
FIELDS = ("continuous_iou_points", "fp_per_truth_area_points", "fn_per_truth_area_points",
          "boundary_iou1_points", "boundary_iou2_points")
PAIR_CONTRASTS = {
    "pr_I_minus_L": ("pr_independent", "pr_bounded"),
    "pr_L_minus_F": ("pr_bounded", "pr_fixed"),
    "pr_I_minus_F": ("pr_independent", "pr_fixed"),
    "I_minus_L": ("free_matched_count", "bounded"),
    "L_minus_F": ("bounded", "fixed"),
    "I_minus_F": ("free_matched_count", "fixed"),
}
INTERACTION = "writer_head_interaction"
VIEWS = {
    "full": (74, 81, 81, 47, 12, 11),
    "date_separated": (42, 47, 47, 32, 8, 8),
}


def need(condition, detail):
    if not condition:
        raise ValueError(detail)


def digest(raw):
    return sha256(raw).hexdigest()


def pin(path, expected):
    raw = Path(path).read_bytes()
    need(digest(raw) == expected, "Exact sealed input: " + Path(path).name)
    return raw


def module(path, expected):
    loaded = ModuleType(Path(path).stem)
    loaded.__file__ = str(path)
    sys.modules[loaded.__name__] = loaded
    exec(compile(pin(path, expected), str(path), "exec"), loaded.__dict__)
    return loaded


def calculate_contrast(values, names, name):
    """Contrasts on the final axis, with learned minus PR interaction."""
    if name == INTERACTION:
        return (calculate_contrast(values, names, "I_minus_L")
                - calculate_contrast(values, names, "pr_I_minus_L"))
    high, low = PAIR_CONTRASTS[name]
    return values[..., names.index(high)] - values[..., names.index(low)]


def squared_distance(additions, removals, width):
    ay, ax = np.divmod(np.asarray(additions, dtype=np.int64), width)
    dy, dx = np.divmod(np.asarray(removals, dtype=np.int64), width)
    return (ay[:, None] - dy[None, :]) ** 2 + (ax[:, None] - dx[None, :]) ** 2


def cap_feasible(additions, removals, width, cap):
    if len(additions) != len(removals):
        return False
    distances = squared_distance(additions, removals, width)
    try:
        rows, cols = linear_sum_assignment(np.where(distances <= cap, distances, np.inf))
    except ValueError:
        return False
    return bool(len(rows) == len(additions) and np.all(distances[rows, cols] <= cap))


def slot_rows(before, after):
    need(before.shape == after.shape and before.ndim == 2 and before.shape[1] == 2,
         "Identical two-sided support indexing")
    additions = np.flatnonzero(~before[:, 0] & after[:, 0])
    removals = np.flatnonzero(before[:, 1] & ~after[:, 1])
    need(len(additions) == len(removals), "Exact signed additions and removals")
    return additions, removals


def memberships(sparse, rle, coordinates):
    height, width = rle["size"]
    fortran = (coordinates % width)*height + coordinates//width
    runs = np.cumsum(sparse.decode_runs(rle["counts"]), dtype=np.int64)
    return (np.searchsorted(runs, fortran, side="right") % 2).astype(bool)


def check_slot(stock, record, original_edits, old_bounded, sparse):
    required = {"cap_squared", "coordinates", "sf_pair_indices", *MASK_ARMS}
    need(set(record) == required, "Exactly six policies and shared support/cap metadata")
    need(record["fixed"] == original_edits["fixed"]
         and record["free_matched_count"] == original_edits["free_matched_count"]
         and record["bounded"] == old_bounded, "Original learned F/L/I native RLE and budgets")
    k = original_edits["fixed"]["swaps"]
    width = stock["size"][1]
    coords = np.asarray(record["coordinates"])
    fixed_rows = np.asarray(record["sf_pair_indices"])
    cap = record["cap_squared"]
    need(type(k) is int and 0 < k <= 8 and type(cap) is int and cap > 0
         and coords.ndim == 2 and coords.shape[1] == 2 and k <= len(coords) <= 32
         and coords.dtype.kind in "iu" and np.unique(coords).size == coords.size
         and np.all((coords >= 0) & (coords < stock["size"][0]*width))
         and fixed_rows.dtype.kind in "iu" and fixed_rows.shape == (k,)
         and np.unique(fixed_rows).size == k
         and np.all((fixed_rows >= 0) & (fixed_rows < len(coords))),
         "Exact dimensions, original native coordinates, and inherited k/cap")
    before = memberships(sparse, stock, coords)
    need(not before[:, 0].any() and before[:, 1].all(),
         "Original native background/foreground support signs")
    pair_distances = squared_distance(coords[:, 0], coords[:, 1], width).diagonal()
    need(int(pair_distances[fixed_rows].max()) == cap, "Cap is the largest realized archived F pair")
    for arm in MASK_ARMS:
        policy = record[arm]
        need(set(policy) == {"swaps", "rle"} and policy["swaps"] == k
             and policy["rle"]["size"] == stock["size"],
             "Original count and native canvas preserved on every policy")
        after = memberships(sparse, policy["rle"], coords)
        ai, di = slot_rows(before, after)
        need(len(ai) == len(di) == k and
             sparse.xor_rle(stock, np.r_[coords[ai, 0], coords[di, 1]]) == policy["rle"],
             "Every binary mask exactly reconstructs from signed support point flips")
        if arm in (LEARNED[0], POINTREND[0]):
            need(np.array_equal(ai, di) and np.all(pair_distances[ai] <= cap),
                 "The distance-first comparator selects only original cap-qualified pair rows")
        if arm == LEARNED[0]:
            need(set(ai) == set(fixed_rows), "Archived F row indices exactly reproduce the F edit")
        if arm in (LEARNED[1], POINTREND[1]):
            need(cap_feasible(coords[ai, 0], coords[di, 1], width, cap),
                 "Independent perfect assignment validates every realized bounded endpoint set")
    return k


def stream_rows(path, expected):
    return [json.loads(line) for line in pin(path, expected).splitlines()]


def new_render(a, past):
    root = a.render_root
    result = past.closed(root / "render/RESULT_SAFE.json", root / "observed_render/TERMINAL.json",
                         a.render_sha, a.render_terminal_sha, a.render_worker_sha,
                         "prepare-edits", RENDER_STATUS)
    flags = ("exact_original_native_masks_and_scores", "exact_archived_pointrend_replay_bytes",
             "exact_original_learned_fixed_bounded_independent", "exact_signed_counts_and_common_cap")
    counts = result["counts"]
    need(result["protocol_sha256"] == a.protocol_sha and result["prelock_commit"] == a.prelock_commit
         and result["original_pointrend_result_sha256"] == POINTREND_PRIOR_RENDER
         and result["learned_bounded_result_sha256"] == BOUNDED_RENDER
         and result["learned_bounded_terminal_sha256"] == BOUNDED_RENDER_TERMINAL
         and result["solver_sha256"] == SOLVER
         and all(result[key] is True for key in flags)
         and result["label_files_opened"] == result["detector_forwards"] == result["optimizer_updates"] == 0
         and tuple(counts[x] for x in ("images", "mapped_candidates", "positive_candidates", "swaps"))
         == (74, 7346, 4092, 30841), "Exact observed label-free render lineage and original budget")
    return stream_rows(root / "render/EDITS_PRIVATE.jsonl", result["edits_sha256"]), result


def sources(a):
    past = module(a.code / "LEARNED_BOUNDED_PAIRING_20261126/rsdd/score.py", HISTORICAL_SCORE)
    original = past.module(a.code / "INSTANCE_RSDD_FROZEN_HRSID_20261122/score_rsdd.py", past.ORIGINAL_SCORE_SHA)
    evaluator = original.module(a.code / "PSEG_RFDETR_FROZEN_SOURCE_HEAD_20261122/score.py", original.SCORE_SOURCE)
    adaptor = original.module(a.code / "UEIKAP_RFDETR_CARRIER_20261122/score_rf.py", original.GEOMETRY_SOURCE)
    core = adaptor.load_geometry(a.code / "UEIKAP_RESERVED_HULL_FROZEN_20261122/score.py")
    sparse = module(a.code / "RFDETR_SPARSE_RLE_EQUIVALENCE_20261120/sparse_rle.py", SPARSE)
    blocks, prior, reserve, dates, nested = past.historical_reserve(a, original, core, adaptor)
    baseline = past.closed(
        a.bounded / "reserve74_5217_render/RESULT_SAFE.json",
        a.bounded / "observed_reserve74_5217_render/TERMINAL.json", BOUNDED_RENDER,
        BOUNDED_RENDER_TERMINAL, HISTORICAL_RENDER, "prepare-edits", past.RENDER_STATUS)
    historical = past.closed(
        a.bounded / "reserve74_score/RESULT_SAFE.json",
        a.bounded / "observed_reserve74_score/TERMINAL.json", BOUNDED_SCORE,
        BOUNDED_SCORE_TERMINAL, HISTORICAL_SCORE, "score", past.STATUS)
    need(baseline["prior"]["prior_verified_result_sha256"] == prior["receipt_sha256"]["replay"]
         and baseline["counts"]["images"] == 74 and baseline["counts"]["positive_candidates"] == 4092
         and baseline["counts"]["swaps"] == 30841, "Shared original reserve capture and bounded replay lineage")
    old_rows = stream_rows(a.bounded / "reserve74_5217_render/BOUNDED_PRIVATE.jsonl",
                           baseline["actual_bounded_stream_sha256"])
    fresh_rows, fresh = new_render(a, past)
    need(fresh["learned_bounded_edits_sha256"] == baseline["actual_bounded_stream_sha256"]
         and fresh["prior_verified_result_sha256"] == baseline["prior"]["prior_verified_result_sha256"]
         and fresh["predictions_sha256"] == baseline["prior"]["prior_candidate_sha256"]
         and fresh["audit_sha256"] == baseline["prior"]["prior_candidate_audit_sha256"],
         "New and original bounded render attach the same native candidate and edit streams")
    need(len(old_rows) == len(fresh_rows) == 74, "Complete zipped native roster")
    total = Counter()
    for iid, old, now in zip(blocks, old_rows, fresh_rows, strict=True):
        block = blocks[iid]
        need(set(now) == {"image_id", "changes"} and old["image_id"] == now["image_id"] == iid,
             "Every positive slot bound to original image order")
        historical_changes = block["changes"]
        required = {j for j, edits in historical_changes.items() if "fixed" in edits}
        need(set(old["changes"]) == set(now["changes"]) == set(map(str, required)),
             "Every original positive slot and all native pass-throughs preserved")
        merged = {}
        for key, row in now["changes"].items():
            j = int(key)
            stock = block["stock"][j-1]["segmentation"]
            swaps = check_slot(stock, row, historical_changes[j], old["changes"][key], sparse)
            merged[j] = {arm: row[arm] for arm in MASK_ARMS}
            total.update(positive_candidates=1, swaps=swaps)
        block["changes"] = merged
        total.update(images=1, mapped_candidates=len(block["stock"]))
    need(dict(total) == dict(images=74, mapped_candidates=7346, positive_candidates=4092, swaps=30841),
         "Whole roster, phase and count checks")
    return past, original, evaluator, core, reserve, dates, nested, blocks, historical, fresh, total


def compare_effect(delta, interval, count, groups):
    covered = count >= 20 and groups >= 8
    if covered and delta >= .30 and interval[0] > 0:
        decision = "positive"
    elif covered and delta <= -.30 and interval[1] < 0:
        decision = "negative"
    else:
        decision = "inconclusive"
    return dict(decision=decision, change_coverage=covered)


def masks_by_arm(orig, variants):
    return [orig, *(variants.get(arm, {}).get("rle", orig) for arm in MASK_ARMS)]


def positive_reference_rings(core, boundary, ann):
    refs = core.codec.frPyObjects(ann["segmentation"], 512, 512)
    mask = core.codec.decode(core.codec.merge(refs)).astype(bool)
    rings = [boundary.boundary_ring(mask, width) for width in (1,2)]
    need(all(ring.any() for ring in rings), "Positive reference rings on original eligible matches")
    return rings


def interaction_changes(masks, core):
    idx = [ARMS.index(x) for x in (*PAIR_CONTRASTS["pr_I_minus_L"], *PAIR_CONTRASTS["I_minus_L"])]
    pix = [core.codec.decode(masks[x]).astype(np.int8, copy=False) for x in idx]
    return bool(np.any((pix[0]-pix[1]) != (pix[2]-pix[3])))


def collect_geometry(gt, blocks, anchors, core, shapely, boundary):
    labels = {r["id"]: r for r in gt["annotations"]}
    groups = defaultdict(list)
    exposed = {key: (set(),set()) for key in (*PAIR_CONTRASTS, INTERACTION)}
    eligible = 0
    for gid, (iid, rank, native) in anchors.items():
        orig = native["segmentation"]
        area = int(core.codec.area(orig))
        ann = labels[gid]
        polygon = shapely.Polygon(np.asarray(ann["segmentation"][0]).reshape(-1,2))
        if not polygon.is_valid or polygon.area <= 0 or not 32 <= area < 1024:
            continue
        block = blocks[iid]
        group = block["group"]
        masks = masks_by_arm(orig, block["changes"].get(rank, {}))
        refs = positive_reference_rings(core, boundary, ann)
        eligible += 1
        cache, values = {}, []
        for rle in masks:
            compact = rle["counts"]
            if compact not in cache:
                iou = core.geometric_iou(rle, polygon, shapely)
                intersection = iou*(area+polygon.area)/(1+iou)
                raster = core.codec.decode(rle)
                bio = [boundary.ring_iou(boundary.boundary_ring(raster, width), ref)
                       for width,ref in enumerate(refs,1)]
                need(np.isfinite(bio).all(), "Defined native Boundary IoU on the common eligible hulls")
                cache[compact] = [100*iou, 100*(area-intersection)/polygon.area,
                                  100*(polygon.area-intersection)/polygon.area, *[100*x for x in bio]]
            values.append(cache[compact])
        groups[group].append(values)
        for key, (high, low) in PAIR_CONTRASTS.items():
            if masks[ARMS.index(high)] != masks[ARMS.index(low)]:
                exposed[key][0].add(gid)
                exposed[key][1].add(group)
        if interaction_changes(masks, core):
            exposed[INTERACTION][0].add(gid)
            exposed[INTERACTION][1].add(group)
    return groups, exposed, eligible


def measure(gt, blocks, dates, archived, evaluator, core, shapely, boundary, bootstrap, compare):
    ids = set(blocks)
    subset = dict(gt, images=[r for r in gt["images"] if r["id"] in ids],
                  annotations=[r for r in gt["annotations"] if r["image_id"] in ids])
    native, anchors = evaluator.score_one(subset, blocks, "raw", anchors=True)
    compare(native, archived["native_metrics"])
    scores = {"raw": native, **{arm: evaluator.score_one(subset, blocks, arm, anchors=False)[0]
                               for arm in MASK_ARMS}}
    for arm in LEARNED:
        compare(scores[arm], archived["head_metrics"]["5217"][arm])
    groups, exposure, count = collect_geometry(subset, blocks, anchors, core, shapely, boundary)
    parents = sorted(groups)
    day = [dates[p] for p in parents]
    group_means = np.asarray([np.asarray(groups[p]).mean(axis=0) for p in parents])
    # Contrast helpers use the arm axis last. Every effect uses the same parent/date draws.
    values = group_means.transpose(0,2,1)
    ap_points = np.asarray([scores[arm]["AP"] * 100 for arm in ARMS])
    contrasts = {}
    for key in (*PAIR_CONTRASTS, INTERACTION):
        dy = calculate_contrast(values, ARMS, key)
        date_ci, parent_ci = bootstrap(dy, day), bootstrap(dy, parents)
        hulls, groups_with_change = exposure[key]
        changed_blocks = len({dates[parent] for parent in groups_with_change})
        ap = float(calculate_contrast(ap_points, ARMS, key))
        if key == INTERACTION:
            direction = ("positive" if date_ci[0][0] > 0 else
                         "negative" if date_ci[0][1] < 0 else "uncertain")
            decision = dict(continuous_interaction_direction=direction)
        else:
            decision = compare_effect(ap, date_ci[0],len(hulls),changed_blocks)
        contrasts[key] = dict(
            ap_points=ap, changed_hulls=len(hulls), changed_parent_groups=len(groups_with_change),
            changed_analysis_blocks=changed_blocks,
            geometry={metric: dict(delta=float(mean), interval_95=ci, parent_interval_95=pi)
                      for metric, mean, ci, pi in zip(FIELDS, dy.mean(axis=0), date_ci, parent_ci, strict=True)},
            **decision,
        )
    for key in ("I_minus_L", "L_minus_F", "I_minus_F"):
        before, after = archived["pooled_contrasts"][key], contrasts[key]
        need(abs(before["ap_points"]-after["ap_points"]) < 1e-10 and
             all(before[x] == after[x] for x in ("changed_hulls", "changed_parent_groups", "changed_analysis_blocks")),
             "Exactly reproduce original learned pooled signed contrast")
        for field in FIELDS[:3]:
            need(all(np.allclose(before["geometry"][field][k], after["geometry"][field][k], atol=1e-10, rtol=0)
                     for k in ("delta", "interval_95", "parent_interval_95")),
                 "Exactly reproduce original learned continuous/FP/FN scores and blocked bootstrap")
    return dict(images=len(ids), author_hulls=len(subset["annotations"]), original_native_matches=len(anchors),
                eligible_original_matches=count, represented_parent_groups=len(parents),
                represented_analysis_blocks=len(set(day)),
                mask_metrics_fraction=scores, parent_equal_arm_points={
                    arm:dict(zip(FIELDS, group_means[:,ix].mean(axis=0).tolist(), strict=True))
                    for ix,arm in enumerate(ARMS)
                }, contrasts=contrasts,
                writer_supported_for_both_heads=all(contrasts[key]["decision"]=="positive"
                                                     for key in ("I_minus_L","pr_I_minus_L")),
                historical_learned_metrics_reproduced=True)


def finish(a, status, started, **details):
    payload = dict(status=status, phase=a.phase, worker_sha256=digest(Path(__file__).read_bytes()),
                   protocol_sha256=a.protocol_sha, prelock_commit=a.prelock_commit,
                   render_worker_sha256=a.render_worker_sha, render_result_sha256=a.render_sha,
                   render_terminal_sha256=a.render_terminal_sha, prior_bounded_score_sha256=BOUNDED_SCORE,
                   elapsed_seconds=perf_counter()-started, detector_forwards=0, optimizer_updates=0, **details)
    raw = (json.dumps(payload, sort_keys=True, allow_nan=False, indent=2)+"\n").encode()
    a.output.mkdir()
    (a.output/"RESULT_SAFE.json").write_bytes(raw)
    print(json.dumps(dict(status=status, result_sha256=digest(raw))), flush=True)


def run(a):
    started = perf_counter()
    need(sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
         and np.__version__ == "1.26.4" and version("pycocotools") == "2.0.11"
         and a.output.parent.is_dir() and not a.output.exists()
         and re.fullmatch("[0-9a-f]{40}", a.prelock_commit) is not None,
         "Fresh isolated original scoring environment and published protocol")
    pin(a.protocol, a.protocol_sha)
    pin(a.render_root/"render_pr.py", a.render_worker_sha)
    if a.phase == "score":
        past = module(a.code/"LEARNED_BOUNDED_PAIRING_20261126/rsdd/score.py",HISTORICAL_SCORE)
        verified = past.closed(a.verify_root/"verify/RESULT_SAFE.json",
                               a.verify_root/"observed_verify/TERMINAL.json",a.verify_sha,
                               a.verify_terminal_sha,digest(Path(__file__).read_bytes()),
                               "verify-edits", VERIFY_STATUS)
        need(verified["protocol_sha256"] == a.protocol_sha and verified["prelock_commit"] == a.prelock_commit
             and verified["render_result_sha256"] == a.render_sha and verified["render_terminal_sha256"] == a.render_terminal_sha
             and verified["target_label_files_opened"] == 0 and verified["exact_signed_and_cap_readback"],
             "Independent pre-label replay proof belongs to this exact run")
    past, original, evaluator, core, reserve, dates, nested, blocks, archived, fresh, totals = sources(a)
    if a.phase == "verify":
        finish(a, VERIFY_STATUS, started, counts=dict(totals), target_label_files_opened=0,
               exact_signed_and_cap_readback=True, edits_sha256=fresh["edits_sha256"])
        return
    need(verified["counts"] == dict(totals) and verified["edits_sha256"] == fresh["edits_sha256"],
         "Independent closed readback matches this scoring stream")
    if a.shapely_site is not None:
        sys.path.append(str(a.shapely_site))
    import shapely
    need(shapely.__version__ == "2.0.2" and shapely.geos_version_string == "3.11.2",
         "The original continuous polygon runtime")
    boundary = module(a.code/"INSTANCE_RSDD_NATIVE_BOUNDARY_IOU_20261126/boundary_metric.py",BOUNDARY)
    gt, _ = original.coco_annotation_ids(reserve.select_labels(pin(a.author_labels,reserve.TRAIN),blocks))
    views = {}
    for name, selected in (("full", blocks), ("date_separated", {i:blocks[i] for i in nested})):
        report = measure(gt,selected,dates,archived["views"][name],evaluator,core,shapely,boundary,
                         reserve.bootstrap,past.close_enough)
        names = ("images","author_hulls","original_native_matches","eligible_original_matches",
                 "represented_parent_groups","represented_analysis_blocks")
        need(tuple(report[x] for x in names) == VIEWS[name], "Exact unchanged author/match/group roster")
        views[name] = report
    finish(a,SCORE_STATUS,started,views=views,target_label_files_opened=1,author_reference_sha256=reserve.TRAIN,
           verify_result_sha256=a.verify_sha,verify_terminal_sha256=a.verify_terminal_sha,
           counts=dict(totals),exposed_historical_reserve_only=True,
           interaction_formula="(I-L)-(PR_I-PR_L)",paired_resamples=4096,ap_inference="pooled_point_only",
           exact_signed_and_cap_readback=True)


def main():
    parser=ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument("phase",choices=("verify","score"))
    for arg in ("code","archive","bounded","render-root","protocol","author-labels","output"):
        parser.add_argument("--"+arg,type=Path,required=True)
    for arg in ("protocol-sha","prelock-commit","render-sha","render-terminal-sha","render-worker-sha"):
        parser.add_argument("--"+arg,required=True)
    for arg in ("verify-root","shapely-site"):
        parser.add_argument("--"+arg,type=Path)
    for arg in ("verify-sha","verify-terminal-sha"):
        parser.add_argument("--"+arg)
    a=parser.parse_args()
    if a.phase == "score":
        need(all((a.verify_root,a.verify_sha,a.verify_terminal_sha)), "Qualified prior independent readback")
    run(a)


if __name__ == "__main__":
    main()

