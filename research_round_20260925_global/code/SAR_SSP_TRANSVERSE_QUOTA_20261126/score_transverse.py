"""Read back short-axis quota masks, then score the exact historical SAR-SSP population."""
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
from zipfile import ZipFile

import numpy as np
from pycocotools import mask as codec


TRANSVERSE_CORE = "888fb7963aa73ea43c970edd92a87edebd3684f8df5334887af69218c8ca5eb0"
PROTOCOL_SHA = "94172ed035225cdd352b5622424f4b1f41698f4d66c55698d0d250f7aace933e"
PRELOCK_COMMIT = "a9906f6914e889333db8f0c09669c733ac6f7779"
RENDER_WORKER = "9010ebe81ef4c1a5f57bba999fbeb6418efd23c02c8a8af4e6bc14121c44869b"
BASE_WORKER = "83ff61406235b142b86c2bcdbebbc92b3b17a7d3eb48032cd6c6619ec7e6f8a2"
BASE_RENDER = "9cf1753d4818df64717217831d29421f329eae21014b4656e52be83ecefd1701"
BASE_RENDER_TERMINAL = "1ea6b73fc7f977ad7f3cce632c97083fb2d3fa6b8c153ce4480dd212ec67c582"
BASE_SCORE = "a857f06b37ca65e38205e24c5fd024e837f533dbf1ba6c448fed5d2e4727e837"
BASE_SCORE_TERMINAL = "faa2c69b75cb23bfd5077a338a6682a9fd6233e6c66720fc713c1f64108f57c3"
RENDER = "SAR_SSP_TRANSVERSE_QUOTA_RENDER_COMPLETE"
VERIFIED = "SAR_SSP_TRANSVERSE_QUOTA_READBACK_COMPLETE"
SCORED = "SAR_SSP_TRANSVERSE_QUOTA_SCORE_COMPLETE"
N, B, F = "native", "fixed_iou", "variable_iou"
T, A, L = "transverse_iou", "area_iou", "longitudinal_iou"
HISTORICAL = (B,F,"near_iou","bce_fixed_robust")
ARMS = (B,F,T,A,L)
PAIRS = ((T,F),(T,A),(T,L),(T,B),(A,F),(L,F),(F,B))


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(raw):
    return sha256(raw).hexdigest()


def pin(path, expected):
    raw = path.read_bytes()
    require(digest(raw) == expected, "Exact bound artifact " + path.name)
    return raw


def closed(result, terminal, result_sha, terminal_sha, worker, phase, status):
    r = json.loads(pin(result, result_sha))
    t = json.loads(pin(terminal, terminal_sha))
    require(t["phase"] == phase and t["actual_exit"] == 0 and t["output_footer_verified"] is True
            and t["worker_sha256"] == r["worker_sha256"] == worker and t["result_sha256"] == result_sha
            and r["status"] == status, "Fresh actual archived natural close: " + status)
    return r


def load(path, expected):
    mod = ModuleType(path.stem)
    mod.__file__ = str(path)
    exec(compile(pin(path, expected), str(path), "exec"), mod.__dict__)
    return mod


def reconstruct(stock, coords, choice, k):
    require(type(k) is int and 1 <= k <= 8 and coords.ndim == 2 and coords.shape[1] == 2
            and len(coords) <= 32 and coords.dtype.kind in "iu"
            and np.all((coords >= 0) & (coords < stock.size)) and len(np.unique(coords)) == coords.size,
            "Exact Legal original signed endpoint coordinates")
    flat = stock.ravel(order="C")
    require(not flat[coords[:,0]].any() and flat[coords[:,1]].all(), "Exact immutable mask signs")
    add, drop = choice["action"]
    require(len(add) + len(drop) == 2*k and [len(add),len(drop)] == choice["counts"]
            and all(all(type(i) is int and 0 <= i < len(coords) for i in side) and len(set(side)) == len(side)
                    for side in (add,drop)), "Exact total and unique switched endpoints")
    rendered = flat.copy()
    rendered[coords[np.asarray(add, dtype=np.intp),0]] = True
    rendered[coords[np.asarray(drop, dtype=np.intp),1]] = False
    written = codec.encode(np.asfortranarray(rendered.reshape(stock.shape), dtype=np.uint8))
    written["counts"] = written["counts"].decode("ascii")
    require(written == choice["rle"] and int(codec.area(written)) == int(stock.sum()) + len(add)-len(drop),
            "Independently reconstructed exact original-shape RLE and signed area")
    return written


def verify_layer(stock, coords, k, layer, options):
    from scipy.ndimage import label

    points = np.column_stack(stock.nonzero())
    centered = points - points.mean(axis=0)
    vals, axes = np.linalg.eigh(centered.T @ centered / len(points))
    minor, major = np.maximum(vals, 0.)
    aspect = float(np.sqrt(major / minor)) if minor > 0 else None
    span = np.s_[points[:,0].min():points[:,0].max()+1, points[:,1].min():points[:,1].max()+1]
    linked = label(stock[span], structure=np.ones((3,3), dtype=bool))[1] == 1
    disposition = "aspect" if major <= 0 or major < 4*minor else "eligible" if linked else "disconnected"
    eligible = disposition == "eligible"
    require(layer["eligible"] is eligible and layer["disposition"] == disposition
            and (layer["axis_ratio"] == aspect if aspect is None else
                 abs(layer["axis_ratio"]-aspect)<1e-10), "Independently verified original native PCA / 8-connection")
    n = len(coords)
    quotas = sorted(range(max(0,2*k-n),min(n,2*k)+1), key=lambda a:(abs(a-k),a))
    admitted = layer["admitted_quotas"]
    amounts = layer["admitted_counts"]
    require(layer["feasible_quotas"] == len(quotas) and set(admitted) == set(amounts) == {T,A,L}
            and len(set(amounts.values())) == 1, "Same quota menu size for all three controls")
    size = amounts[T]
    require(1 <= size <= len(quotas), "Nonempty subset of exact feasible signed totals")
    for arm in (T,A,L):
        menu = admitted[arm]
        require(menu == [a for a in quotas if a in menu] and len(menu) == size and k in menu
                and options[arm]["counts"][0] in menu, "Admissible original quota order, B retained and valid action")
    if not eligible:
        require(all(options[x] == options[F] for x in (T,A,L)) and size == len(quotas)
                and layer["width_by_arm"] == layer["longitudinal_variance_by_arm"] == {},
                "Exact original free fallback outside native geometry gate")
        return eligible, False
    require(admitted[A] == quotas[:size], "Same-sized central area menu")
    for arm in ARMS:
        decoded = codec.decode(options[arm]["rle"]).astype(bool)
        changed_points = np.column_stack(decoded.nonzero())
        projected = (changed_points - changed_points.mean(axis=0)) @ axes
        actual = np.var(projected, axis=0)
        expected = (layer["width_by_arm"][arm],layer["longitudinal_variance_by_arm"][arm])
        require(np.allclose(actual, expected, rtol=1e-11, atol=1e-10), "Independent direct PCA projected variance")
    observed, limit = (layer["width_by_arm"][arm] for arm in (T,B))
    require(observed <= limit + 1e-12*max(1.,abs(observed),abs(limit)),
            "Chosen T respects its original B short-axis cap")
    return eligible, size < len(quotas)


def validate_masks(base, setup, historical, rendered):
    old, m, args, roster, capture, checked, _, scoring, _ = setup
    old_swaps = scoring.read_stream(base.OLD / "prepare/EDITS_PRIVATE.jsonl", checked["edits_sha256"])
    records, geometry_state = {}, {}
    counts = Counter()
    arm_counts = {arm:Counter(adds=0,drops=0,expanded=0,shrank=0,balanced=0) for arm in ARMS}
    require(len(roster) == len(historical) == len(rendered) == len(old_swaps) == 232,
            "Original 232 chip rows present in every stream")
    for j, (item, stock, _, _) in enumerate(m.legacy.blocks(args, m.author, roster, capture, False)):
        iid = item["image_id"]
        hist, now, warm = historical[j], rendered[j], old_swaps[j]
        require(iid == hist["image_id"] == now["image_id"] == warm["image_id"] and iid not in records
                and item["rgb_sha256"] == now["rgb_sha256"] == hist["rgb_sha256"] == warm["rgb_sha256"]
                and len(stock) == 100, "Exact original native frame, JPEG decode and candidate roster")
        warm_masks = warm["changes"]
        rows, old_rows = now["edits"], hist["edits"]
        positive = {r for r, masks in warm_masks.items() if "attraction_static_at_dynamic_count" in masks}
        require(rows.keys() == old_rows.keys() == positive, "Identical original positive/k-zero partition")
        ordered = []
        for rank, candidate in enumerate(stock, 1):
            hit = str(rank)
            old_row = old_rows.get(hit)
            rle = candidate["segmentation"]
            changes = {}
            if old_row is not None:
                row = rows[hit]
                k = row["k"]
                coords = np.asarray(row["coordinates"], dtype=np.int32)
                require(k == old_row["k"] == warm_masks[hit]["attraction_static_at_dynamic_count"]["swaps"]
                        and row["coordinates"] == old_row["coordinates"]
                        and set(row["arms"]) == {*HISTORICAL,T,A,L}
                        and all(row["arms"][arm] == old_row["arms"][arm] for arm in HISTORICAL),
                        "Exact archived B, F and companions with untouched endpoints and quota")
                native = codec.decode(rle).astype(bool, order="C")
                layer = row["transverse_geometry"]
                require(type(layer["eligible"]) is bool, "Immutable label-free geometry eligibility")
                for arm in ARMS:
                    info = row["arms"][arm]
                    written = reconstruct(native, coords, info, k)
                    adds, drops = info["counts"]
                    require(scoring.signed_counts(rle, written, int(codec.area(rle))) == (adds,drops),
                            "Independently recovered actual signed counts")
                    if arm == B:
                        require(adds == drops == k, "Equal budget reference holds both signs")
                    changes[arm] = written
                    arm_counts[arm].update(adds=adds,drops=drops,expanded=int(adds>drops),
                                           shrank=int(adds<drops),balanced=int(adds==drops))
                eligible, constrained = verify_layer(native,coords,k,layer,row["arms"])
                counts.update(positive_candidates=1, k=k, geometry_eligible=int(eligible),
                              constrained_menu_candidates=int(constrained))
                layer = dict(eligible=eligible, constrained=constrained)
                for left, right in PAIRS:
                    counts[left+"__vs__"+right+"_changed_candidates"] += int(changes[left] != changes[right])
                geometry_state[(iid,rank)] = layer
            ordered.append(dict(native=rle,score=candidate["score"],masks=changes))
        records[iid] = ordered
    require(len(records)==232 and sum(map(len,records.values()))==23200
            and counts["positive_candidates"]==10393 and counts["k"]==73559
            and all(x["adds"]+x["drops"] == 147118 for x in arm_counts.values()),
            "No candidate/score/count losses before target reference")
    return records, geometry_state, dict(counts), dict(arm_counts)


def baseline_parity(current, previous):
    for arm in (N,B,F):
        a, b = current["coco_mask_metric_points"][arm], previous["coco_mask_metric_points"][arm]
        require(a == b, "Exact previously closed native/B/F COCO fields")
    for view in ("fixed_native_match_all_truth","fixed_native_match_nonedge_sensitivity"):
        a, b = current[view], previous[view]
        for key in ("hulls","groups","matched_hulls","unmatched_hulls","matched_groups"):
            require(a[key] == b[key], "Exact historical author and native-match denominator")
        for arm in (N,B,F):
            require(a["arms"][arm].keys() == b["arms"][arm].keys()
                    and all(abs(x-b["arms"][arm][key]) <= 1e-12 for key,x in a["arms"][arm].items()),
                    "Exact historical fixed-anchor IoU / foreground-error metrics")


def exposure(annotations, groups, matches, records, geometry, nonedge=None):
    def summarize(picks):
        return dict(hulls=len(picks),groups=len({group for _,group in picks}))
    scopes = defaultdict(list)
    changed = defaultdict(list)
    for annotation in annotations:
        aid = annotation["id"]
        if nonedge is not None and aid in nonedge:
            continue
        if aid not in matches:
            continue
        img, rank = matches[aid]
        group = groups[img]
        scopes["all_native_matches"].append((aid,group))
        layer = geometry.get((img,rank))
        if layer is None:
            continue
        scopes["positive_original_k"].append((aid,group))
        if layer["eligible"]:
            scopes["transverse_geometry_eligible"].append((aid,group))
        if layer["constrained"]:
            scopes["strictly_constrained_menu"].append((aid,group))
        masks = records[img][rank-1]["masks"]
        for left,right in PAIRS:
            if masks[left] != masks[right]:
                key = left+"__vs__"+right
                changed[key].append((aid,group))
                if layer["eligible"]:
                    changed[key+"_geometry_eligible"].append((aid,group))
    base = {name:summarize(scopes[name]) for name in ("all_native_matches","positive_original_k","transverse_geometry_eligible","strictly_constrained_menu")}
    for left,right in PAIRS:
        key = left+"__vs__"+right
        base[key] = dict(changed_positive=summarize(changed[key]),
                         changed_geometry_eligible=summarize(changed[key+"_geometry_eligible"]))
    return base


def foreground_deltas(annotations, groups, matches, records, fixed, omit=None):
    names = (N,*ARMS)
    values = defaultdict(list)
    for ann in annotations:
        if omit is not None and ann["id"] in omit:
            continue
        area = [0.] * len(names)
        if ann["id"] in matches:
            image, rank = matches[ann["id"]]
            row = records[image][rank-1]
            area = [100*int(codec.area(row["masks"].get(arm,row["native"]))) / ann["area"] for arm in names]
        values[groups[ann["image_id"]]].append(area)
    parent = np.asarray([np.mean(values[key], axis=0) for key in sorted(values)])
    draw = np.random.Generator(np.random.PCG64(5217)).integers(len(parent),size=(4096,len(parent)))
    output = {}
    for left,right in PAIRS:
        key = left+"__vs__"+right
        measured = fixed["comparisons"][key]
        fp = measured["false_positive_area_over_truth_points"]
        fn = measured["false_negative_area_over_truth_points"]
        d = parent[:,names.index(left)]-parent[:,names.index(right)]
        mean = float(d.mean())
        require(abs(mean-(fp["delta"]-fn["delta"])) <= 1e-12, "Exact signed area identity on fixed author hull")
        output[key] = dict(net_predicted_area_points=dict(delta=mean,
                           percentile95=np.percentile(d[draw].mean(axis=1),[2.5,97.5]).tolist()),
                           intersection_points=dict(delta=-fn["delta"], percentile95=[-fn["percentile95"][1],
                                                                                      -fn["percentile95"][0]]))
    return output



def decisions(result, scope):
    geom = result["fixed_native_match_all_truth"]["comparisons"]
    ap = result["coco_mask_deltas_points"]
    out = {}
    for right in (F,B,A,L):
        key = T+"__vs__"+right
        hits = scope[key]["changed_geometry_eligible"]
        coverage = hits["hulls"] >= 20 and hits["groups"] >= 8
        change = geom[key]["iou_points"]
        fp = geom[key]["false_positive_area_over_truth_points"]
        ap_delta = ap[key]["AP"]
        standard = right in (F,B)
        passes = bool(coverage and change["percentile95"][0] > 0 and
                      ((ap_delta >= .30 and change["delta"] >= .15) if standard else
                       (fp["percentile95"][1] < 0 and ap_delta >= -.10)))
        out[key] = dict(coverage=coverage,criteria_pass=passes)
    return dict(contrast=out,mitigation=out[T+"__vs__"+F]["criteria_pass"],
                recovery=out[T+"__vs__"+B]["criteria_pass"],
                specificity=all(out[T+"__vs__"+x]["criteria_pass"] for x in (A,L)))


def run(c):
    start = perf_counter()
    require(sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
            and np.__version__ == "1.26.4" and version("pycocotools") == "2.0.11"
            and c.prelock_commit == PRELOCK_COMMIT and c.protocol_sha == PROTOCOL_SHA
            and c.render_worker_sha == RENDER_WORKER and not c.output.exists(), "Fresh qualified CPU phase")
    pin(c.protocol,c.protocol_sha)
    base = load(c.base_run/"transfer.py",BASE_WORKER)
    old_render = closed(c.base_run/"render/RESULT_SAFE.json",c.base_run/"observed_render/TERMINAL.json",
                        BASE_RENDER,BASE_RENDER_TERMINAL,BASE_WORKER,"capture",base.RENDER_STATUS)
    old_score = closed(c.base_run/"score/RESULT_SAFE.json",c.base_run/"observed_score/TERMINAL.json",
                       BASE_SCORE,BASE_SCORE_TERMINAL,BASE_WORKER,"verify",base.SCORE_STATUS)
    require(old_score["render_result_sha256"] == BASE_RENDER and old_score["original_control_parity"] is True
            and old_render["gate_labels_opened"] is False and old_score["gate_labels_opened"] is True,
            "Exact historical frozen render and author score")
    render = closed(c.writer/"render/RESULT_SAFE.json",c.writer/"observed_render/TERMINAL.json",
                    c.render_result_sha,c.render_terminal_sha,c.render_worker_sha,"capture",RENDER)
    require(render["protocol_sha256"]==c.protocol_sha and render["prelock_commit"]==c.prelock_commit
            and render["original_render_sha256"]==BASE_RENDER and render["original_score_sha256"]==BASE_SCORE
            and render["original_render_terminal_sha256"]==BASE_RENDER_TERMINAL
            and render["original_score_terminal_sha256"]==BASE_SCORE_TERMINAL
            and render["original_worker_sha256"]==BASE_WORKER
            and render["transverse_core_sha256"]==TRANSVERSE_CORE
            and render["all_archived_actions_rle_and_quota_exact"] is True
            and render["gate_labels_opened"] is False
            and render["detector_forwards"] == render["optimizer_updates"] == 0,
            "Original exact label-free transverse and quota control closure")
    work = base.setup()
    old,m,_,roster,_,_,_,scoring,_ = work
    historical = scoring.read_stream(c.base_run/"render/RENDER_PRIVATE.jsonl",old_render["raw_sha256"])
    rendered = scoring.read_stream(c.writer/"render/RENDER_PRIVATE.jsonl",render["raw_sha256"])
    records, state, counts, budgets = validate_masks(base,work,historical,rendered)
    require(counts["geometry_eligible"] == render["transverse_geometry"]["eligible"]
            and all(all(value == render["signed_arm_counts"][arm].get(key,0) for key,value in budget.items())
                    for arm,budget in budgets.items()), "Independent geometry and all signed-arm aggregate parity")
    worker = digest(Path(__file__).read_bytes())
    report = dict(protocol_sha256=c.protocol_sha,prelock_commit=c.prelock_commit,worker_sha256=worker,
                  source_transfer_render_sha256=BASE_RENDER,source_transfer_score_sha256=BASE_SCORE,
                  render_result_sha256=c.render_result_sha,render_terminal_sha256=c.render_terminal_sha,
                  raw_sha256=render["raw_sha256"],original_masks_scores_and_b_f_exact=True,
                  independent_position_and_rle_readback=True,original_images=232,original_candidates=23200,
                  gate_labels_opened=False,detector_forwards=0,optimizer_updates=0,counts=counts,signed_counts=budgets)
    if c.phase=="score":
        prior = closed(c.verify/"RESULT_SAFE.json",c.verify_terminal,c.verify_result_sha,c.verify_terminal_sha,
                       worker,"verify-edits",VERIFIED)
        require(all(report[key] == prior[key] for key in report), "Independent label-free proof and exact readback parity")
        print(json.dumps({"stage":"INDEPENDENT_TRANSVERSE_WRITEBACK_VERIFIED_PRELABEL"}),flush=True)
        require(old.file_hash(base.ARCHIVE)==scoring.ARCHIVE_SHA,"Exact original public author archive")
        names = {row["image_id"]:(Path(row["image_ref"]["path"]).name,row["image_ref"]["sha256"]) for row in roster}
        with ZipFile(base.ARCHIVE) as package:
            truth, groups, edge = scoring.exact_ground_truth(package,names)
        scoring.ARMS = {arm:arm for arm in ARMS}
        scoring.PRIMARY, scoring.SECONDARY = PAIRS[:2]
        scoring.COMPARISONS = PAIRS
        original_matches = scoring.native_matches
        matches = {}
        def capture_native_matches(*args):
            result = original_matches(*args)
            matches.update(result)
            return result
        scoring.native_matches = capture_native_matches
        evaluate = scoring.load_evaluator(base.CODE/scoring.EVALUATOR_PATH)
        result = scoring.score(truth,groups,edge,records,evaluate)
        baseline_parity(result,old_score["scoring"])
        geometry = result["fixed_native_match_all_truth"]
        require((geometry["hulls"],geometry["matched_hulls"],geometry["unmatched_hulls"],geometry["groups"]) ==
                (139,88,51,23) and len(set(groups.values())) == 31 and len(edge)==48, "Exact all-hull parent denominator")
        full = exposure(truth["annotations"],groups,matches,records,state)
        nested = exposure(truth["annotations"],groups,matches,records,state,nonedge=edge)
        report.update(status=SCORED,gate_labels_opened=True,verify_result_sha256=c.verify_result_sha,
                      verify_terminal_sha256=c.verify_terminal_sha,scoring=result,
                      exposure_full=full,exposure_nonedge=nested,old_native_b_f_metric_parity=True,
                      preregistered_decisions=decisions(result, full),
                      signed_foreground_deltas_all_truth=foreground_deltas(truth["annotations"],groups,matches,records,geometry),
                      signed_foreground_deltas_nonedge=foreground_deltas(truth["annotations"],groups,matches,records,
                          result["fixed_native_match_nonedge_sensitivity"],omit=edge),
                      bootstrap_draws=4096,bootstrap_seed=5217,prospective=False)
    else:
        report.update(status=VERIFIED)
    report["elapsed_seconds"] = perf_counter()-start
    raw = (json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    c.output.mkdir()
    (c.output/"RESULT_SAFE.json").write_bytes(raw)
    print(json.dumps(dict(status=report["status"],result_sha256=digest(raw))),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("phase",choices=("verify-edits","score"))
    for key in ("base-run","writer","protocol","output"):
        p.add_argument("--"+key,type=Path,required=True)
    for key in ("protocol-sha","prelock-commit","render-result-sha","render-terminal-sha","render-worker-sha"):
        p.add_argument("--"+key,required=True)
    for key in ("verify","verify-terminal"):
        p.add_argument("--"+key,type=Path)
    for key in ("verify-result-sha","verify-terminal-sha"):
        p.add_argument("--"+key)
    run(p.parse_args())


if __name__ == "__main__":
    main()

