"""Score the frozen fractional-target comparison on the additional 74 Instance-RSDD images."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
from time import perf_counter
from types import ModuleType

import numpy as np
import shapely


STATUS = "RFDETR_FRACTIONAL_TIE_RESERVE74_SCORE_COMPLETE"
REPLAY_STATUS = "RFDETR_FRACTIONAL_TIE_RESERVE74_REPLAY_COMPLETE"
PROTOCOL = "270342431d38e48e48e011921c8791133af14746e7eba4d1622d186910cdeefa"
PORTABLE = "45512c959eb914ae7713264ecff48eb437a2a52729810b34308abc52079148df"
CAPTURE = "f43fb04035045b1e30b7ae06ced778e66fcc4cd3e34077f83ebff6096dacc71d"
STOCK_RESULT = "b31d4d1c718aa6cac9b73c61778aafb94404a897233928d8b78fa678d560f083"
STOCK_EDITS = "353362e0efabc0374eaaf0b9c27fdabbcad195402a1090230c4ef56ca2a10398"
HISTORICAL = "737a5de1e28a2a9729c69d8c32a5a1dfb667affab989c81561c957a901bb3b2e"
SEEDS = (5217,5218,5219)
TARGETS = ("F","T","H","C")
LABELS = tuple(f"{s}/{t}" for s in SEEDS for t in TARGETS)
HISTORY = ("raw","fixed","free_matched_count","free_natural")
ARMS = HISTORY + LABELS
FIELDS = ("continuous_iou_points","fp_per_truth_area_points","fn_per_truth_area_points")


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(data):
    return sha256(data).hexdigest()


def pin(path, wanted):
    raw = path.read_bytes()
    require(digest(raw) == wanted,"Fingerprint differs for "+path.name)
    return raw


def portable_eval(code):
    path = code/"INSTANCE_RSDD_B5_REPRO_20261123/portable_eval.py"
    source = ModuleType("reserve_b5")
    source.__file__ = str(path)
    exec(compile(pin(path,PORTABLE),str(path),"exec"),source.__dict__)
    return source


def checked_inputs(args, b5, reserve, adapter, core, protocol_sha):
    terminal_raw = args.replay_terminal.read_bytes()
    term = json.loads(terminal_raw)
    require(type(term["actual_exit"]) is int and term["actual_exit"] == 0 and term["output_footer_verified"] is True,
            "Independent replay must be closed before target scoring")
    result = json.loads(pin(args.replay/b5.RESULT,term["result_sha256"]))
    require(result["status"] == REPLAY_STATUS and result["target_labels_opened"] is False and result["GT_opened"] is False
            and all(result[k] is True for k in ("historical_5217_F_exact","independent_support_exact","independent_ranking_exact","signed_xor_exact"))
            and result["model_forwards"] == result["optimizer_updates"] == 0 and result["protocol_sha256"] == protocol_sha,
            "Required frozen replay checks or prelock differ")
    require(tuple(result["seeds"]) == SEEDS and tuple(result["arms"]) == TARGETS and set(result["weights_sha256"]) == set(LABELS)
            and all(re.fullmatch("[a-f0-9]{64}",h) for h in (*result["weights_sha256"].values(),result["source_result_sha256"],result["source_terminal_sha256"])),
            "Frozen source fits differ")
    capture = json.loads(pin(args.capture/b5.RESULT,CAPTURE))
    stock = json.loads(pin(args.stock/b5.RESULT,STOCK_RESULT))
    require(result["capture_result_sha256"] == CAPTURE and result["budget_edits_sha256"] == stock["edits_sha256"] == STOCK_EDITS
            and all(result[k] == capture[k] == stock[k] for k in ("predictions_sha256","audit_sha256","checkpoint_sha256",
                                                                 "roster_sha256","manifest_sha256","images")),
            "Budget or carrier differs from published B5")
    native = b5.module(args.code/b5.NATIVE/"capture_instance_rsdd.py",b5.CAPTURE_SOURCE)
    items, manifest_sha = b5.load_intake(args,native)
    require(manifest_sha == capture["manifest_sha256"],"Unexpected reserve image manifest")
    _, pairs = b5.read_capture(args,items,manifest_sha)
    old = [json.loads(line) for line in pin(args.stock/b5.EDITS,STOCK_EDITS).splitlines()]
    blocks = b5.scoring_blocks(items,pairs,old,reserve,core,adapter)
    raw = pin(args.replay/b5.EDITS,result["edits_sha256"])
    totals = Counter()
    for (image,block), line in zip(blocks.items(),raw.splitlines(),strict=True):
        row = json.loads(line)
        require(set(row) == {"changes","image_id"} and row["image_id"] == image,"Replay order differs")
        changes = row["changes"]
        expected = {str(j) for j,arms in block["changes"].items() if "fixed" in arms}
        require(set(changes) == expected,"Edited candidate sets differ from the fixed-budget reference")
        for key, updates in changes.items():
            j = int(key)
            earlier = block["changes"][j]
            k = earlier["fixed"]["swaps"]
            require(set(updates) == set(LABELS) and updates["5217/F"] == earlier["free_matched_count"],
                    "All twelve target/seed arms and the archived fractional output required")
            original = block["stock"][j-1]["segmentation"]
            area = int(core.codec.area(original))
            for arm, updated in updates.items():
                require(set(updated) == {"rle","swaps"} and type(updated["swaps"]) is int and updated["swaps"] == k,
                        "Different per-candidate quota")
                rle = core.checked_rle(updated["rle"],512,512)
                actual_drop = area-int(core.codec.area(core.codec.merge([original,rle],intersect=True)))
                require(int(core.codec.area(rle)) == area and actual_drop == k,"Realized additions/deletions differ")
            earlier.update(updates)
            totals["budget_swaps"] += k
            totals["budget_candidates"] += 1
    totals["images"] = len(blocks)
    totals["mapped_candidates"] = sum(len(b["stock"]) for b in blocks.values())
    totals["eligible_planes"] = capture["counts"]["eligible"]
    require(totals == dict(images=74,mapped_candidates=7346,eligible_planes=4137,budget_candidates=4092,budget_swaps=30841)
            and all(result[k] == v for k,v in totals.items()) and set(result["counts_by_arm"]) == set(LABELS),
            "Prelocked cohort or budget totals differ")
    for values in result["counts_by_arm"].values():
        require(values["changed_candidates"] == totals["budget_candidates"] and values["adds"] == values["drops"] == totals["budget_swaps"],
                "Replay per-arm counts differ")
    return blocks,result,term["result_sha256"],digest(terminal_raw),dict(totals)


def interval_record(deltas, keys, reserve):
    ci = reserve.bootstrap(deltas,keys)
    return {f:dict(delta=float(x),sensor_date_interval_95=i)
            for f,x,i in zip(FIELDS,deltas.mean(axis=0),ci,strict=True)}


def geometry(anchors, polygons, blocks, dates, core, reserve):
    per_parent = defaultdict(list)
    exposed = set()
    counts = dict(original_matches=len(anchors), eligible_matches=0, positive_fixed_edits=0,
                  invalid_polygon_matches=0,area_excluded_matches=0)
    for gid,(image,rank,pred) in anchors.items():
        poly = polygons[gid]
        if not poly.is_valid or poly.area <= 0:
            counts["invalid_polygon_matches"] += 1
            continue
        native_mask = pred["segmentation"]
        area = int(core.codec.area(native_mask))
        if not 32<=area<1024:
            counts["area_excluded_matches"] += 1
            continue
        counts["eligible_matches"] += 1
        block = blocks[image]
        updates = block["changes"].get(rank,{})
        if "fixed" in updates:
            counts["positive_fixed_edits"] += 1
            exposed.add(block["group"])
        values,cache = [],{}
        for name in ARMS:
            rle = updates.get(name,{}).get("rle",native_mask)
            key = rle["counts"]
            if key not in cache:
                iou = core.geometric_iou(rle,poly,shapely)
                intersection = iou*(poly.area+area)/(1+iou)
                cache[key] = (100*iou,100*(area-intersection)/poly.area,100*(poly.area-intersection)/poly.area)
            values.append(cache[key])
        per_parent[block["group"]].append(values)
    parents = sorted(per_parent)
    keys = [dates[p] for p in parents]
    out = dict(counts=counts,represented_parents=len(parents),represented_sensor_date_blocks=len(set(keys)),
               exposed_parents=len(exposed),exposed_sensor_date_blocks=len({dates[p] for p in exposed}),
               group_equal_arms={},seed_mean_arms={},per_seed_contrasts={},pooled_contrasts={})
    if not parents:
        return out
    means = np.asarray([np.mean(per_parent[p],axis=0) for p in parents])
    out["group_equal_arms"] = {arm:dict(zip(FIELDS,means[:,j].mean(axis=0).tolist(),strict=True)) for j,arm in enumerate(ARMS)}
    by_target = {t:means[:,[ARMS.index(f"{s}/{t}") for s in SEEDS],:] for t in TARGETS}
    for t, rows in by_target.items():
        out["seed_mean_arms"][t] = dict(zip(FIELDS,rows.mean(axis=1).mean(axis=0).tolist(),strict=True))
    for other in TARGETS[1:]:
        delta = by_target["F"] - by_target[other]  # parent x seed x geometry metric
        name = "F minus "+other
        out["per_seed_contrasts"][name] = {str(s):interval_record(delta[:,j],keys,reserve) for j,s in enumerate(SEEDS)}
        f_indices = [ARMS.index(f"{s}/F") for s in SEEDS]
        control_indices = [ARMS.index(f"{s}/{other}") for s in SEEDS]
        averaged = []
        for parent in parents:
            match_values = np.asarray(per_parent[parent])
            match_delta = match_values[:,f_indices,:] - match_values[:,control_indices,:]
            averaged.append(match_delta.mean(axis=1).mean(axis=0))
        out["pooled_contrasts"][name] = interval_record(np.asarray(averaged),keys,reserve)
        for j, field in enumerate(FIELDS):
            out["pooled_contrasts"][name][field]["between_seed_sd_points"] = float(delta.mean(axis=0)[:,j].std(ddof=1))
    return out


def metric_delta(first, second):
    return {k:100*(v-second[k]) if v is not None and second[k] is not None else None for k,v in first.items()}


def mean_metrics(rows):
    return {k:float(np.mean([r[k] for r in rows])) if all(r[k] is not None for r in rows) else None for k in rows[0]}


def score_view(ids, gt, blocks, polygons, dates, evaluator, core, reserve):
    scoped = {i:b for i,b in blocks.items() if i in ids}
    selected = dict(gt,images=[r for r in gt["images"] if r["id"] in ids],
                    annotations=[r for r in gt["annotations"] if r["image_id"] in ids])
    metrics = {}
    for arm in ARMS:
        raw, matches = evaluator.score_one(selected,scoped,arm,anchors=arm=="raw")
        metrics[arm] = {k:float(v) if np.isfinite(v) and v>=0 else None for k,v in raw.items()}
        require(len(raw) == 13 and {"AP","AP75","AP90","AR100"}<=set(raw),"Expected 13 original COCO mask metrics")
        if arm=="raw":
            anchors = matches
    means = {t:mean_metrics([metrics[f"{s}/{t}"] for s in SEEDS]) for t in TARGETS}
    per_seed,pooled,spread = {},{},{}
    for other in TARGETS[1:]:
        name = "F minus "+other
        per_seed[name] = {str(s):metric_delta(metrics[f"{s}/F"],metrics[f"{s}/{other}"]) for s in SEEDS}
        pooled[name] = mean_metrics(list(per_seed[name].values()))
        spread[name] = {k:float(np.std([r[k] for r in per_seed[name].values()],ddof=1)) if v is not None else None
                        for k,v in pooled[name].items()}
    geo = geometry(anchors,polygons,scoped,dates,core,reserve)
    iou = geo["pooled_contrasts"].get("F minus T",{}).get(FIELDS[0])
    gates = dict(exposure=geo["counts"]["positive_fixed_edits"]>=20 and geo["exposed_parents"]>=8,
                 each_seed_ap_positive=all(p["AP"] is not None and p["AP"]>0 for p in per_seed["F minus T"].values()),
                 pooled_iou_lower_positive=iou is not None and iou["sensor_date_interval_95"][0]>0)
    return dict(images=len(scoped),annotations=len(selected["annotations"]),parents=len({b["group"] for b in scoped.values()}),
                metrics_fraction=metrics,seed_mean_metrics_fraction=means,per_seed_contrasts_points=per_seed,
                mean_contrasts_points=pooled,between_seed_sd_points=spread,geometry=geo,criteria=gates,within_recipe_criterion=all(gates.values()))


def history_parity(view, historical):
    for arm in HISTORY:
        scores = view["metrics_fraction"][arm]
        require(scores == historical["metrics_fraction"][arm],"Recomputed original COCO results differ")
        now = view["geometry"]["group_equal_arms"][arm]
        old = historical["geometry"]["group_equal_arms"][arm]
        require(all(abs(now[f]-old[f])<=1e-10 for f in FIELDS),"Original parent-balanced geometry differs")
    require(view["metrics_fraction"]["5217/F"] == view["metrics_fraction"]["free_matched_count"]
            and view["geometry"]["group_equal_arms"]["5217/F"] == view["geometry"]["group_equal_arms"]["free_matched_count"],
            "Archived independent mask and seed 5217 F must give identical scores")


def run(args):
    pin(args.protocol,PROTOCOL)
    require(shapely.__version__ == "2.0.2" and shapely.geos_version_string == "3.11.2","Original geometry runtime required")
    b5 = portable_eval(args.code)
    reserve = b5.module(args.code/b5.RESERVE/"score_reserve.py",b5.SCORE_SOURCE)
    old = b5.module(args.code/b5.NATIVE/"score_rsdd.py",reserve.HISTORIC)
    evaluator = old.module(args.code/"PSEG_RFDETR_FROZEN_SOURCE_HEAD_20261122/score.py",old.SCORE_SOURCE)
    adapter = old.module(args.code/b5.UEIKAP/"score_rf.py",old.GEOMETRY_SOURCE)
    core = adapter.load_geometry(args.code/"UEIKAP_RESERVED_HULL_FROZEN_20261122/score.py")
    blocks, replay, replay_sha, terminal_sha, totals = checked_inputs(args,b5,reserve,adapter,core,PROTOCOL)
    results = args.code.parent/"results"
    census = json.loads(pin(results/"INSTANCE_RSDD_PARENT_RESERVE_CENSUS_SAFE_20261123.json",reserve.CENSUS))
    historic = json.loads(pin(results/"INSTANCE_RSDD_PARENT_RESERVE_ACTUAL_SAFE_20261123.json",HISTORICAL))
    require(Counter((b["group"],b["sensor"]) for b in blocks.values())==
            {(p["paper_scene"],p["sensor"]):p["images"] for p in census["per_parent"]},"Parent roster differs")
    # The author annotations are first opened after the independent, observed replay and signed edit checks.
    gt,rekeyed = old.coco_annotation_ids(reserve.select_labels(pin(args.author_train,reserve.TRAIN),blocks))
    require(len(gt["images"])==74 and len(gt["annotations"])==81,"Selected author cohort differs")
    polygons = {r["id"]:shapely.Polygon(np.asarray(r["segmentation"][0]).reshape(-1,2)) for r in gt["annotations"]}
    dates = {p["paper_scene"]:(p["sensor"],p["calendar_date"]) for p in census["per_parent"]}
    dated = set(census["date_separated_sensitivity"]["parent_scene_ids"])
    full = set(blocks)
    gf = {i for i,b in blocks.items() if b["sensor"]=="GF-3"}
    subsets = dict(full=full,date_separated={i for i,b in blocks.items() if b["group"] in dated},
                   gf3=gf,terrasar_x=full-gf,full_deduplicated=old.representatives(blocks,full))
    require([len(ids) for ids in subsets.values()] == [74,42,42,32,74],"Five fixed views differ")
    cache, views = {},{}
    for name, ids in subsets.items():
        key = frozenset(ids)
        if key not in cache:
            cache[key] = score_view(ids,gt,blocks,polygons,dates,evaluator,core,reserve)
        views[name] = cache[key]
        history_parity(views[name],historic["score"]["views"][name])
    return dict(status=STATUS,protocol_sha256=PROTOCOL,replay_result_sha256=replay_sha,replay_terminal_sha256=terminal_sha,
                source_result_sha256=replay["source_result_sha256"],source_terminal_sha256=replay["source_terminal_sha256"],
                weights_sha256=replay["weights_sha256"],edits_sha256=replay["edits_sha256"],budget_edits_sha256=STOCK_EDITS,
                capture_result_sha256=CAPTURE,stock_result_sha256=STOCK_RESULT,checkpoint_sha256=b5.CHECKPOINT,
                manifest_sha256=replay["manifest_sha256"],roster_sha256=b5.ROSTER,predictions_sha256=replay["predictions_sha256"],
                audit_sha256=replay["audit_sha256"],census_sha256=reserve.CENSUS,author_train_sha256=reserve.TRAIN,
                historical_score_sha256=HISTORICAL,historical_scores_verified=True,seed5217_fractional_matches_archived=True,
                per_candidate_realized_counts_verified=True,realized_budget_per_arm=totals,per_arm_edits=replay["counts_by_arm"],
                evaluator_annotation_ids_rekeyed=rekeyed,non_simple_polygons_retained_in_ap=sum(not p.is_valid for p in polygons.values()),
                full_primary_criterion=views["full"]["within_recipe_criterion"],
                date_separated_sensitivity_criterion=views["date_separated"]["within_recipe_criterion"],
                cluster_draws=4096,cluster_seed=5217,seeds=list(SEEDS),views=views,detector_forwards=0,optimizer_updates=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for key in ("code","roster","intake","capture","stock","replay","replay-terminal","protocol","author-train","output"):
        parser.add_argument("--"+key,required=True,type=Path)
    args = parser.parse_args()
    require(not args.output.exists(),"New score directory required")
    started = perf_counter()
    result = run(args)
    result.update(elapsed_seconds=perf_counter()-started,worker_sha256=digest(Path(__file__).read_bytes()))
    raw = (json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    args.output.mkdir()
    (args.output/"RESULT_SAFE.json").write_bytes(raw)
    print(json.dumps({"status":STATUS,"result_sha256":digest(raw)}),flush=True)


if __name__=="__main__":
    main()
