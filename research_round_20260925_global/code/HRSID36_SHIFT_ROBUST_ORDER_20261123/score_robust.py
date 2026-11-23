"""Compare five-view robust and mean endpoint order on the closed HRSID FIT core36."""
import argparse
from collections import defaultdict
from contextlib import redirect_stdout
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
from statistics import mean
import sys
from time import perf_counter
from types import ModuleType

import numpy as np
from pycocotools import mask as codec


LOCK = "8cd0ea42c1033871e2810a58ec1b0600b70eefecc19b15c23e1b9e9a1e7c5316"
ORIGINAL = "d4c5c81d3bd73197fdc17cc0db26e38f1b3b663ec507518fab1cccbb9d27e044"
REQUEST = "ccac34ef91dc1aa49873d957c3355da5f691c87b522b974e62f3ad72a226f5d0"
PRIOR = "768e833dba8f37c9db86e8b8d9cafbda189a00a8c758784b0ebdfe28f6677075"
CAPTURE = "f3855f18529586348b4b423d180adaf8b0ebd46cd37aec5dbe7fd17c2a3fad8f"
READBACK = "f50ed715c14f00c436c73569dbf11ae66cc89df5a2cded8807452c10ab0376c5"
STATUS = "HRSID36_SHIFT_ROBUST_ORDER_SCORE_COMPLETE"
ARMS = ("zero", "native", "mean", "robust")
FIELDS = ("iou", "fp", "fn", "intersection_px2", "intersection_per_truth_area_points")


def need(ok, message):
    if not ok:
        raise ValueError(message)


def robust_order(logits, inside):
    values = np.asarray(logits, dtype=np.float64)
    inside = np.asarray(inside, dtype=bool)
    need(values.shape == (len(inside), 5) and np.isfinite(values).all(), "Five finite original logit views required")
    ranks = np.partition(values, (1, 3), axis=1)
    # The historical editor maximizes its foreground score for additions and minimizes it for removals.
    return np.where(inside, ranks[:, 3], ranks[:, 1])


def paired_interval(groups, delta):
    names, inverse = np.unique(groups, return_inverse=True)
    delta = np.asarray(delta, dtype=np.float64)
    need(delta.ndim == 2 and len(names) and len(delta) == len(groups) and np.isfinite(delta).all(), "Paired source groups")
    sums = np.zeros((len(names), delta.shape[1]))
    np.add.at(sums, inverse, delta)
    sizes = np.bincount(inverse)
    rng = np.random.Generator(np.random.PCG64(5217))
    draws = rng.integers(len(names), size=(4096, len(names)))
    sampled = sums[draws].sum(axis=1) / sizes[draws].sum(axis=1)[:, None]
    return delta.mean(axis=0), np.quantile(sampled, [.025, .975], axis=0).T


def admit(root, code):
    folder = code / "HRSID_FIT_FIXED_QUERY_SHIFT_PHASE_SCOUT_20261112"
    path = folder / "score_shift_phase.py"
    raw = path.read_bytes()
    need(sha256(raw).hexdigest() == ORIGINAL, "Exact historical FIT-only scoring source required")
    old = ModuleType("original_shift_phase")
    old.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), old.__dict__)
    old.exact(folder / "PROTOCOL.md", old.SOURCE_DIGESTS["PROTOCOL.md"])
    math = old.load(folder / "phase_probe_math.py", old.SOURCE_DIGESTS["phase_probe_math.py"], "_phase_math")
    readout = old.load(folder / "phase_readout.py", old.SOURCE_DIGESTS["phase_readout.py"], "_phase_readout")
    anchor = old.load(root / "P379_HRSID_FIT_LARGE_COMPLEMENTARY_QUERY_DRAFT_PRIVATE_20261111_01/score.py",
                      old.MATCH_SHA, "_fit_phase_match")
    geom = old.load(code / "MASK_RTDETR_DEV705_CONTINUOUS_AP_20261112/continuous_endpoint.py",
                    old.GEOMETRY_SHA, "_fit_phase_geometry").geometry_modules(code)

    def doc(file, sha, cap=2**20):
        return json.loads(old.exact(file, sha, cap))

    cap = root / "HRSID_FIT114_CORE36_CENTER5202_SHIFT_BAND_ACTUAL_PRIVATE_20261112_02/center"
    verify = root / "HRSID_FIT114_CORE36_CENTER5202_SHIFT_BAND_DRAFT_PRIVATE_20261112_02"
    result = doc(cap / "capture/RESULT_SAFE.json", CAPTURE)
    check = doc(verify / "INDEPENDENT_READBACK_SAFE.json", READBACK)
    wait = doc(verify / "INDEPENDENT_TOOL_WAIT_PRIVATE.json", "08d630451ad923d4b4eae33e494aa65ebd056ed2aa0104d2b329614dcb4e57d0")
    old.exact(verify / "ORIGINAL_TOOL_WAIT_PRIVATE.json", check["original_wait_sha256"])
    old.exact(cap / "capture/POINT_LOGITS_INDEX_PRIVATE.jsonl", result["point_index_sha256"], 8*2**20)
    need(wait["independent_result_sha256"] == READBACK and wait["exact_tool_wait"]["exit_code"] == 0 and
         result["views_dx_dy"] == [[0,0],[-1,0],[1,0],[0,-1],[0,1]] and result["model_forward_calls"] == 180 and
         result["FIT"] is True and result["GT_opened"] is result["TEST"] is result["DEV"] is False and
         result["point_index_sha256"] == check["point_index_sha256"] and check["zero_replay_bitwise"] == 267782,
         "Original registered-view capture and independent readback required")
    prior_root = root / "HRSID_FIT_FIXED_QUERY_SHIFT_PHASE_SCOUT_ACTUAL_20261112_02"
    prior = doc(prior_root / "RESULT_SAFE.json", PRIOR)
    closed = doc(prior_root / "ORIGINAL_WAIT_PRIVATE.json", "1681f90e06931c1161146a049ed300e0a10efafd3bb57bb0e8de277b79e48bb3")
    draft = root / "HRSID_FIT_FIXED_QUERY_SHIFT_PHASE_SCOUT_DRAFT_20261112_02"
    terminal = doc(draft / "ORIGINAL_TOOL_TERMINAL_PRIVATE.json", "636c93c31c1e06e0dbec61ef7a0e0ba52f85e0ca608c3232dc91ab3658360251")
    need(closed["original_os_exit"] == terminal["exit_code"] == 0 and closed["process_group_closed"] is True and
         PRIOR in terminal["output"] and prior["request_sha256"] == REQUEST, "Previous scoring closure required")
    request = doc(draft / "REQUEST_PRIVATE.json", REQUEST, 8*2**20)
    views = request["images"]
    need(request["FIT"] is True and request["TEST"] is request["DEV"] is False and request["original_capture_accepted"] is True
         and request["forward_calls"] == 180 and result["support_sha256"] == request["original_support_sha"] == old.SUPPORT_SHA,
         "Registered FIT roster required")
    blocks = [json.loads(line) for line in old.exact(Path(request["prediction_path"]), old.PREDICTIONS_SHA, 32*2**20).splitlines()]
    need(len(blocks) == len(views) == len({r["group"] for r in views}) == len({r["image_id"] for r in views}) == 36 and
         [r["ordinal"] for r in views] == list(range(1,37)), "Same image and source roster required")
    routed = old.reconstruct_support(views, blocks, anchor)
    need(sum(r[2] for r in routed) == 267782 and sum(r[3] for r in routed) == 1, "Native signed support parity")
    return old, math, readout, anchor, geom, prior, views, routed


def score_hull(example, logits, math, readout):
    current = example.shape
    q = example.target_occupancy
    scores = {"native":example.features["native_order"], "mean":example.features["mean_order"],
              "robust":robust_order(logits, example.inside)}
    metrics, selections = {}, {}
    for arm in ARMS:
        k = 0 if arm == "zero" else example.budget
        add, remove = math.edits(scores.get(arm, scores["native"]), example.inside, k)
        choice = (frozenset(add.tolist()), frozenset(remove.tolist()))
        need(len(choice[0]) == len(choice[1]) == k and not choice[0].intersection(choice[1]), "Realized sign budgets differ")
        inter = example.intersection + (q[add].sum() - q[remove].sum())
        row = readout.contour_metrics(example, inter)
        row.update(intersection_px2=inter, intersection_per_truth_area_points=100*inter/example.gt_area, edit_pairs=k,
                   component_changes=0, neighbor_overlaps=0)
        if k:
            updated = current.copy()
            a, d = example.points_xy[add], example.points_xy[remove]
            updated[a[:,1], a[:,0]], updated[d[:,1], d[:,0]] = True, False
            need(np.count_nonzero(updated ^ current) == 2*k and updated.sum() == current.sum(), "Actual Hamming/count mismatch")
            row["component_changes"] = int(readout.label(updated,readout.CONNECTIVITY)[1] != readout.label(current,readout.CONNECTIVITY)[1])
            if example.neighbor_masks:
                rle = codec.encode(np.asfortranarray(updated, dtype=np.uint8))
                ious = codec.iou([rle], example.neighbor_masks, [0]*len(example.neighbor_masks))
                row["neighbor_overlaps"] = int(np.any(ious >= .20))
        metrics[arm], selections[arm] = row, choice
    return metrics, selections["mean"] != selections["robust"]


def evaluate(root, old, math, readout, anchor, geom, views, routed):
    fit = root / "HRSID_FIT_NATIVE_COCO_LAYOUT_ACTUAL_PRIVATE_20261109_01/train/_annotations.coco.json"
    labels = json.loads(old.exact(fit, old.LABEL_SHA, 32*2**20))
    need(len(labels["images"]) == 2937 and len(labels["annotations"]) == 9260, "Exact HRSID source FIT annotations required")
    ids = {v["image_id"] for v in views}
    images = {im["id"]:im for im in labels["images"] if im["id"] in ids}
    by_image = defaultdict(list)
    for annotation in labels["annotations"]:
        if annotation["image_id"] in images:
            by_image[annotation["image_id"]].append(annotation)
    subset = dict(images=list(images.values()), categories=labels["categories"], annotations=[a for rows in by_image.values() for a in rows])
    targets, neutral, _ = geom.geometry.polygon_support(subset, geom.vector)
    records = []
    matches = 0
    for view, (pool, points, _, _) in zip(views, routed, strict=True):
        iid = view["image_id"]
        with redirect_stdout(StringIO()):
            original, _, _ = anchor.freeze_coco(images[iid], by_image[iid], pool)
        matches += len(original)
        by_rank = {rank:key for key,rank in original.items()}
        for index, candidate in enumerate(pool):
            rank = candidate["rank"]
            key = by_rank.get(rank)
            if rank not in points or key is None or key in neutral:
                continue
            truth = targets[key][0]
            rle = candidate["rle"]
            stock = codec.decode(rle).astype(bool)
            rectangles = geom.geometry.boxes_from_counts(geom.shift.runs(rle["counts"]), *rle["size"])
            intersection = geom.geometry.overlap_area(rectangles, truth)
            xy, inside, logits = points[rank]
            need(np.array_equal(logits[:,0] > 0, inside), "Unshifted logit side differs on an original fixed match")
            ex = readout.example(view["ordinal"],stock,xy,inside,logits,truth,intersection,[p["rle"] for p in pool[:index]],math)
            row, different = score_hull(ex, logits, math, readout)
            records.append((view["group"], row, different))
    need(matches == 190 and len(records) == 166, "Historical native association or eligible fixed-hull roster differs")
    return records


def summarize(records, prior):
    groups = [r[0] for r in records]
    metrics = {}
    for arm in ARMS:
        cells = [r[1][arm] for r in records]
        metrics[arm] = {f:mean(float(v[f]) for v in cells) for f in FIELDS}
        metrics[arm].update({f:sum(v[f] for v in cells) for f in ("edit_pairs","component_changes","neighbor_overlaps")})
        metrics[arm]["strict"] = sum(int(v["iou"] >= 90) for v in cells)
        if arm != "robust":
            for key, expected in prior["metrics"][arm].items():
                need(metrics[arm][key] == expected, "Archived metric parity differs: " + arm + "/" + key)
    need(all(metrics[arm]["edit_pairs"] == 1209 for arm in ("mean","native","robust")), "All active arms require 1,209 swaps")
    delta = np.array([[row["robust"][f]-row["mean"][f] for f in FIELDS] for _,row,_ in records])
    point, band = paired_interval(groups, delta)
    contrast = {name:dict(mean=float(d), interval_95=b.tolist()) for name,d,b in zip(FIELDS,point,band,strict=True)}
    changed = [(g,r) for g,r,differs in records if differs]
    n_changed = len(changed)
    g_changed = len({g for g,_ in changed})
    vs_stock = metrics["robust"]["iou"]-metrics["zero"]["iou"]
    supported = n_changed >= 20 and g_changed >= 8 and point[0] >= .20 and band[0,0] > 0 and vs_stock > 0
    crossings = {base:dict(gains=sum(int(row[base]["iou"] < 90 <= row["robust"]["iou"]) for _,row,_ in records),
                           losses=sum(int(row["robust"]["iou"] < 90 <= row[base]["iou"]) for _,row,_ in records))
                 for base in ("mean","native","zero")}
    return dict(metrics=metrics, robust_minus_mean=contrast, robust_minus_unedited_iou_points=vs_stock,
                differing_fixed_hulls=n_changed, differing_source_groups=g_changed, represented_source_groups=len(set(groups)),
                strict90_crossings_robust_vs=crossings, historical_controls_exact=True, passed=bool(supported),
                decision="ADVANCE_TO_NEW_SOURCE_VALIDATION" if supported else "STOP_FIXED_ROBUST_RULE")


def main():
    import shapely
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("archive-root","code","protocol","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--protocol-sha256",required=True)
    a = p.parse_args()
    need(sys.flags.isolated and sys.dont_write_bytecode and shapely.__version__ == "2.0.2" and
         shapely.geos_version_string == "3.11.2" and np.__version__ == "1.26.4", "Isolated original CPU geometry runtime required")
    need(a.protocol.name == "HRSID36_SHIFT_ROBUST_ORDER_PRELOCK_20261123.md" and
         sha256(a.protocol.read_bytes()).hexdigest() == a.protocol_sha256 == LOCK and not a.output.exists(),
         "Exact source-scout protocol and fresh output required")
    start = perf_counter()
    root,code = a.archive_root.resolve(strict=True),a.code.resolve(strict=True)
    old,math,readout,anchor,geom,prior,views,routed = admit(root,code)
    result = summarize(evaluate(root,old,math,readout,anchor,geom,views,routed),prior)
    result.update(status=STATUS,phase="score",protocol_sha256=LOCK,worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                  historical_score_sha256=PRIOR, capture_result_sha256=CAPTURE, independent_readback_sha256=READBACK,
                  request_sha256=REQUEST, FIT=True, DEV=False, TEST=False, AP=False, source_fit_labels_opened=True,
                  target_labels_opened=False, images=36, original_matches=190, fixed_hulls=166, original_sparse_response_rows=267782,
                  bootstrap_seed=5217, bootstrap_draws=4096, bootstrap_unit="represented source prefix, equal hull weights",
                  model_forward_calls=0, optimizer_updates=0, worker_seconds=perf_counter()-start)
    data=(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    a.output.mkdir()
    with (a.output/"RESULT_SAFE.json").open("xb") as output:
        need(output.write(data)==len(data),"Complete score receipt required")
    print(json.dumps(dict(status=STATUS,result_sha256=sha256(data).hexdigest())),flush=True)


if __name__=="__main__":
    main()
