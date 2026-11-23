"""Score frozen, matched-budget boundary corrections against their archived D-FINE controls."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType

import numpy as np
from pycocotools import mask as codec


ARMS = ("raw", "SS", "LL", "DD", "C", "CM", "Q")
CONTROLS = ("C", "CM", "SS", "LL", "DD", "raw")
INDEX = {arm: i for i, arm in enumerate(ARMS)}
MEASURES = ("continuous_iou_points", "fp_per_truth_pct", "fn_per_truth_pct",
            "intersection_per_truth_pct", "added_intersection_per_truth_pct",
            "removed_intersection_signed_per_truth_pct")
PRIOR = "ATTRACTION_SIGNED_SIDES_20261123/dfine_sides.py"
PRIOR_SHA = "cca7746728efc1ac5388364f9c4c47784b7665c418e99c8c62e02f68fcb15603"
PROTOCOL_SHA = "7e5c54f9e0ea3489d158daaa114a643d1100175ce4f818696ead95f816d7b933"


def need(condition, message):
    if not condition:
        raise ValueError(message)


def bytes_checked(path, expected):
    raw = path.read_bytes()
    need(sha256(raw).hexdigest() == expected, "Digest mismatch: " + path.name)
    return raw


def load_original(code):
    path = code / PRIOR
    module = ModuleType("inherited_dfine_evaluation")
    module.__file__ = str(path)
    exec(compile(bytes_checked(path, PRIOR_SHA), str(path), "exec"), module.__dict__)
    # Keep the original detector matching, group definitions, author geometry, COCO
    # evaluators, sensitive views and equality checks. Only the policy panel changes.
    module.ARMS = ARMS
    module.add_measurements = add_measurements
    module.effects = effects
    return module


def closed(result_file, terminal_file, phase):
    terminal_raw = terminal_file.read_bytes()
    terminal = json.loads(terminal_raw)
    need(terminal["phase"] == phase and type(terminal["actual_exit"]) is int and
         terminal["actual_exit"] == 0 and terminal["output_footer_verified"] is True,
         "Expected closed observed " + phase)
    result_sha = terminal["result_sha256"]
    result = json.loads(bytes_checked(result_file, result_sha))
    return result, terminal, result_sha, sha256(terminal_raw).hexdigest()


def bound_replay(a, prior):
    fit, fit_event, fit_sha, fit_term = closed(a.fit_result, a.fit_terminal, "fit")
    need(fit["status"] == "ATTRACTION_LOCAL_CORRECTION_SOURCE_RIDGE_COMPLETE" and
         fit["protocol_sha256"] == PROTOCOL_SHA and fit["worker_sha256"] == fit_event["worker_sha256"],
         "Frozen source ridge fit")
    feature, ft, feature_sha, feature_term = closed(a.feature_result, a.feature_terminal, "prepare-pairs")
    need(feature["status"] == "ATTRACTION_LOCAL_CORRECTION_TARGET_PAIRS_COMPLETE" and
         feature["protocol_sha256"] == PROTOCOL_SHA and feature["cohort"] == a.cohort and
         feature["images"] == prior.PINS[a.cohort]["images"] and feature["target_labels_opened"] is False and
         feature["detector_forwards"] == 0 and feature["worker_sha256"] == ft["worker_sha256"],
         "Closed label-free target feature preparation")
    replay, terminal, replay_sha, replay_term = closed(
        a.phase_root / "replay/RESULT_SAFE.json", a.phase_root / "observed_replay/TERMINAL.json", "replay")
    need(replay["status"] == "ATTRACTION_LOCAL_CORRECTION_REPLAY_COMPLETE" and
         replay["protocol_sha256"] == PROTOCOL_SHA and replay["cohort"] == a.cohort and
         replay["images"] == prior.PINS[a.cohort]["images"] and replay["target_labels_opened"] is False and
         replay["detector_forwards"] == replay["head_forwards"] == replay["optimizer_updates"] == 0 and
         replay["original_pure_arms_exact"] is True and replay["worker_sha256"] == terminal["worker_sha256"] and
         replay["feature_result_sha256"] == feature_sha and replay["feature_terminal_sha256"] == feature_term and
         replay["fit_result_sha256"] == fit_sha and replay["fit_terminal_sha256"] == fit_term,
         "Observed label-free replay and exact source fit required")
    return replay, dict(replay_result_sha256=replay_sha, replay_terminal_sha256=replay_term,
                        fit_result_sha256=fit_sha, fit_terminal_sha256=fit_term,
                        feature_result_sha256=feature_sha, feature_terminal_sha256=feature_term,
                        feature_pairs_sha256=feature["pairs_sha256"], gate_model_sha256=feature["gate_model_sha256"],
                        replay_worker_sha256=replay["worker_sha256"], edits_sha256=replay["edits_sha256"])


def same_native_quota(native, current, area, swaps):
    need(current["size"] == native["size"] and isinstance(current["counts"], str) and
         int(codec.area(current)) == area and
         int(codec.area(codec.merge([native, current], intersect=True))) == area-swaps,
         "Edited RLE must add and remove the original realized per-mask count")


def signed_changes(a, b, native, area, k):
    """Count displaced add endpoints and displaced drop endpoints between two k-swap masks."""
    if a == b:
        return (0, 0)
    common = codec.merge([a, b], intersect=True)
    kept_common = int(codec.area(codec.merge([common, native], intersect=True)))
    added_common = int(codec.area(common)) - kept_common
    return (k-added_common, area-k-kept_common)


def count_decisions(entries, native, area, k, totals):
    masks = {arm:entry["rle"] for arm,entry in entries.items()}
    shared = codec.merge([masks["SS"], masks["LL"]], intersect=True)
    either = codec.merge([masks["SS"], masks["LL"]], intersect=False)
    shared_area = int(codec.area(shared))
    for arm in ("C", "CM", "Q"):
        need(int(codec.area(codec.merge([shared, masks[arm]], intersect=True))) == shared_area and
             int(codec.area(codec.merge([either, masks[arm]], intersect=True))) == area,
             "Retain common decisions and use only archived disputed positions")
    accepted = {arm:signed_changes(masks["SS"], masks[arm], native, area, k) for arm in ("LL", "C", "CM", "Q")}
    need(accepted["CM"] == accepted["Q"], "Same accepted count on each candidate and sign")
    for side_i,side in enumerate(("additions","removals")):
        possible = accepted["LL"][side_i]
        totals[side]["disputed_pairs"] += possible
        for arm in ("C", "CM", "Q"):
            n = accepted[arm][side_i]
            need(0 <= n <= possible, "Accepted choices within signed disputed positions")
            totals[side]["accepted_"+arm] += n
    for arm in ("C", "CM"):
        displaced = signed_changes(masks["Q"], masks[arm], native, area, k)
        for i,side in enumerate(("additions","removals")):
            both2 = accepted["Q"][i] + accepted[arm][i] - displaced[i]
            need(both2 % 2 == 0 and 0 <= both2//2 <= min(accepted[arm][i],accepted["Q"][i]),
                 "Overlap of the two accepted disagreement-pair selections")
            totals[side]["accepted_by_Q_and_"+arm] += both2//2


def load_records(a, prior, replay, base):
    p = prior.PINS[a.cohort]
    old = prior.historical(a, "verify-edits")
    need(old["GT_opened"] is False, "Archived attraction masks must be label-free")
    historical = a.root / p["history"] / "prepare-edits/EDITS_PRIVATE.jsonl"
    newly = a.phase_root / "replay/EDITS_PRIVATE.jsonl"
    state, records, totals, digest = [], {}, Counter(), sha256()
    decisions = {k:Counter() for k in ("additions","removals")}
    with historical.open("rb") as history:
        for item, pred, audit, updated in prior.rows(a, newly, replay["edits_sha256"]):
            raw = history.readline()
            need(raw.endswith(b"\n"), "Complete archived image record")
            digest.update(raw)
            archived = json.loads(raw)
            iid = item["image_id"]
            need(archived["image_id"] == iid and iid not in records, "Unique aligned native image")
            positive = {}
            for key, action in archived["changes"].items():
                counts = [action.get(prior.NAMES[arm], {}).get("swaps", 0) for arm in ("SS", "LL", "DD")]
                need(len(set(counts)) == 1 and type(counts[0]) is int and counts[0] >= 0, "Archived count consensus")
                if counts[0]:
                    positive[key] = counts[0]
            changes = updated["changes"]
            need(set(changes) == set(positive), "Exactly the archived positive-budget native candidates")
            for key, k in positive.items():
                rank = int(key)
                need(1 <= rank <= 100 and str(rank) == key, "Native rank ordinal")
                native = pred[rank-1]["segmentation"]
                area = audit[rank-1]["mask_area"]
                arms = changes[key]["arms"]
                need(set(arms) == set(ARMS[1:]) and int(codec.area(native)) == area, "Complete mask panel")
                for arm in ("SS", "LL", "DD"):
                    need(arms[arm]["rle"] == archived["changes"][key][prior.NAMES[arm]]["rle"],
                         "Archived pure RLE unchanged: " + arm)
                seen = set()
                for arm in ARMS[1:]:
                    entry = arms[arm]
                    need(type(entry["swaps"]) is int and entry["swaps"] == k and
                         entry["rle"]["size"] == native["size"], "All arms inherit the same quota and canvas")
                    packed = entry["rle"]["counts"]
                    if packed not in seen:
                        same_native_quota(native, entry["rle"], area, k)
                        seen.add(packed)
                count_decisions(arms, native, area, k, decisions)
                totals.update(positive_candidates=1, swaps=k)
            if a.cohort == "dev705":
                name, group = item["file_name"], base.as_group(item)
            else:
                need(archived["rgb_sha256"] == updated["rgb_sha256"], "Archived decoded image grouping")
                name, group = item["image_ref"]["path"], updated["rgb_sha256"]
            need(isinstance(group, str) and group, "Complete inherited group")
            state.append(item)
            records[iid] = dict(native=pred, audit=audit, edit=changes, name=name, group=group)
        need(not history.read(1) and digest.hexdigest() == old["edits_sha256"], "Complete original edit provenance")
    need(len(records) == p["images"] and totals["positive_candidates"] == p["positive"] and
         totals["swaps"] == p["swaps"], "Full archived roster and exact common budget")
    return state, records, totals, decisions, old["edits_sha256"]


def add_measurements(records, matches, truth, neutral, base, modules):
    measured = {}
    for gt_id, (iid, rank) in matches.items():
        rec = records[iid]
        area = rec["audit"][rank-1]["mask_area"]
        if gt_id in neutral or gt_id not in truth or not 32 <= area < 1024:
            continue
        shape = truth[gt_id][0]
        native = rec["native"][rank-1]["segmentation"]
        arms = rec["edit"].get(str(rank), {}).get("arms", {})
        cache, union_cache, values, masks = {}, {}, [], []
        for arm in ARMS:
            entry = arms.get(arm)
            mask = entry["rle"] if entry else native
            key = mask["counts"]
            if key not in cache:
                iou, fp, fn = base.continuous(mask, area, shape, modules)
                joint = 100.-fn
                added = removed = 0.
                if entry:
                    # U = native union edited. J(U)-J(native) is added truth coverage.
                    # J(edited)-J(U) is the signed truth coverage lost through removals.
                    union = codec.merge([native, mask], intersect=False)
                    union["counts"] = union["counts"].decode("ascii")
                    union_key = union["counts"]
                    if union_key not in union_cache:
                        union_fn = base.continuous(union, area+entry["swaps"], shape, modules)[2]
                        union_cache[union_key] = 100.-union_fn
                    j_union = union_cache[union_key]
                    native_joint = cache[native["counts"]][3]
                    added = j_union - native_joint
                    removed = joint - j_union
                    tol = 1e-7 * 100./shape.area
                    need(added >= -tol and removed <= tol and abs(joint-native_joint-added-removed) <= tol,
                         "Signed true intersection and union bounds")
                cache[key] = (iou, fp, fn, joint, added, removed)
            values.append(cache[key])
            masks.append(mask)
        q = masks[INDEX["Q"]]
        exposure = tuple(q["counts"] != masks[INDEX[name]]["counts"] and
                         int(codec.area(codec.merge([q, masks[INDEX[name]]], intersect=True))) != area
                         for name in CONTROLS)
        measured[gt_id] = (iid, rec["group"], np.array(values, dtype=np.float64), exposure)
    return measured


def effects(data):
    groups = defaultdict(list)
    changed_matches = Counter()
    changed_groups = defaultdict(set)
    for _, group, values, changed in data:
        groups[group].append(values)
        for other, flag in zip(CONTROLS, changed):
            if flag:
                changed_matches[other] += 1
                changed_groups[other].add(group)
    by_group = np.array([np.mean(groups[group], axis=0) for group in sorted(groups)])
    need(len(by_group) > 0 and np.isfinite(by_group).all(), "Finite group-equal continuous geometry")
    draws = np.random.Generator(np.random.PCG64(5217)).integers(len(by_group), size=(4096,len(by_group)))
    arm_metrics = {arm: dict(zip(MEASURES, by_group[:,i,:].mean(axis=0).tolist())) for i,arm in enumerate(ARMS)}
    contrasts = {}
    for other in CONTROLS:
        delta = by_group[:,INDEX["Q"],:] - by_group[:,INDEX[other],:]
        interval = np.percentile(delta[draws].mean(axis=1), [2.5, 97.5], axis=0)
        details = {name: dict(points=float(delta[:,i].mean()), interval=interval[:,i].tolist())
                   for i,name in enumerate(MEASURES)}
        iou = delta[:,0]
        contrasts["Q minus "+other] = dict(measures=details, changed_matches=changed_matches[other],
            changed_groups=len(changed_groups[other]), coverage_adequate=changed_matches[other]>=20 and len(changed_groups[other])>=8,
            groups_improved=int((iou>0).sum()), groups_worse=int((iou<0).sum()), groups_equal=int((iou==0).sum()))
    return dict(eligible_matches=len(data), groups=len(by_group), arm_means=arm_metrics,
                arm_mean_iou_points={arm:v["continuous_iou_points"] for arm,v in arm_metrics.items()},
                contrasts=contrasts, bootstrap_draws=4096, bootstrap_seed=5217)


def gate(ap, geometry):
    comparisons = {}
    for other in CONTROLS:
        c = geometry["contrasts"]["Q minus "+other]
        delta = 100.*(ap["Q"]["AP"]-ap[other]["AP"])
        comparisons[other] = dict(ap_points=delta, ap_positive=delta>0,
            iou_lower_positive=c["measures"]["continuous_iou_points"]["interval"][0]>0,
            coverage_adequate=c["coverage_adequate"], changed_matches=c["changed_matches"], changed_groups=c["changed_groups"])
    mechanism = all(comparisons[arm]["ap_positive"] and comparisons[arm]["iou_lower_positive"] and
                    comparisons[arm]["coverage_adequate"] for arm in ("C", "CM"))
    refs = ("SS", "LL", "DD")
    best_ap = max(ap[arm]["AP"] for arm in refs)
    ap_margin = 100.*(ap["Q"]["AP"] - best_ap)
    practical_contrasts = ap_margin>0 and all(comparisons[arm]["iou_lower_positive"] for arm in refs)
    practical = mechanism and practical_contrasts
    repetition = ("full_repetition" if practical and ap_margin>=1 else
                  "at_most_one_locked_repetition" if practical else "does_not_advance")
    return dict(comparisons=comparisons, mechanism_passed=mechanism,
                highest_reference_ap_arms=[arm for arm in refs if ap[arm]["AP"]==best_ap],
                ap_over_highest_reference_points=ap_margin, practical_contrasts_passed=practical_contrasts,
                practical_advancement_passed=practical, repetition=repetition)


def run(a):
    need(a.protocol_sha256 == PROTOCOL_SHA, "Locked protocol digest")
    bytes_checked(a.protocol, PROTOCOL_SHA)
    prior = load_original(a.code)
    replay, lineage = bound_replay(a, prior)
    base = prior.module(a, "base")
    state, records, totals, decisions, archive_sha = load_records(a, prior, replay, base)
    # Only beyond this point are the complete closed replay, native and historic pure
    # streams checked. The inherited evaluator now opens the author's scoring labels.
    sys.path.append(str(a.shapely_site.resolve(strict=True)))
    import shapely
    need(shapely.__version__ == "2.0.2" and shapely.geos_version_string == "3.11.2",
         "Frozen continuous geometry runtime")
    modules = prior.module(a, "polygon").geometry_modules(a.code)
    historical = prior.historical(a, "score")
    ap, geom = prior.collect(a, state, records, modules, base, historical)
    primary = "primary" if a.cohort=="dev705" else "exclude_exact_fit_overlap"
    gates = {view:gate(ap[view],v) for view,v in geom.items()}
    # For HRSID the inherited empty-raster sensitivity changes COCO only; exactly
    # the same continuous geometries and native fixed matches are used.
    if a.cohort == "dev705":
        gates["sensitivity"] = gate(ap["sensitivity"], geom["primary"])
    return dict(status="ATTRACTION_LOCAL_CORRECTION_SCORE_COMPLETE", cohort=a.cohort,
        protocol_sha256=PROTOCOL_SHA, images=len(records), native_candidates=100*len(records),
        arm_order=list(ARMS), original_pure_rle_exact=True, archived_pure_arms_coco_exact=True,
        archived_pure_arms_iou_within_1e_minus_9=True, historical_score_terminal_sha256=prior.PINS[a.cohort]["scored"],
        archived_edits_sha256=archive_sha, target_labels_opened=True, detector_forwards=0, head_forwards=0,
        optimizer_updates=0, counts=dict(totals), side_decisions=decisions, coco_fraction=ap, geometry=geom,
        primary_view=primary, gates=gates, primary_decision=gates[primary], **lineage)


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--cohort", choices=("dev705","pseg232"), required=True)
    for key in ("code","root","manifest","phase-root","protocol","fit-result","fit-terminal",
                "feature-result","feature-terminal","shapely-site","output"):
        p.add_argument("--"+key, type=Path, required=True)
    p.add_argument("--gt", type=Path)
    p.add_argument("--protocol-sha256", required=True)
    a = p.parse_args()
    need((a.cohort=="pseg232") == bool(a.gt), "Supply the frozen GT path for PSeg only")
    need(a.output == a.phase_root/"score" and not a.output.exists(), "Fresh observed score destination")
    a.output.mkdir()
    started = perf_counter()
    result = run(a)
    result.update(worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(), seconds=perf_counter()-started)
    raw = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
    (a.output/"RESULT_SAFE.json").write_bytes(raw)
    print(json.dumps(dict(status=result["status"],result_sha256=sha256(raw).hexdigest())), flush=True)


if __name__=="__main__":
    main()
