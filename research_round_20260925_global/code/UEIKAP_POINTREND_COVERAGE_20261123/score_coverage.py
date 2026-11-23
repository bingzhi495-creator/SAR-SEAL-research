"""Measure independent mask coverage and continuous fixed-box geometry on UEIKAP."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from types import ModuleType

import numpy as np
import shapely


PROTOCOL = "e0cad345351d1cfc23b6777d84abfeffd9fc7e2c82d69242813356af33715225"
HEAD = "7401de08367ef31b909ea9c92daa27be02d4efb04460acbf96eab33648fdd7cc"
OLD_SCORER = "b50f3035c1575fc417ced6b67a8ffeeb04a9cbe933f5c531b9b310c7c4792711"
OLD_POPULATION = "3781bb4601d17519953a783794fdf860fbdab808d169514e7be9ced9c84b5eac"
OLD_LINEAGE = "05d8b47f3d4b9351c1ffd12ac32b3f4f6e91b05daf87a1c6d148536e67f97492"
OLD_BOX = "b423494646b255247b18ed66a0ace695e6088ec831e8473fce1300704646064f"
ARMS = ("N", "C", "P")
RENDERED = {"coarse_paste": "C", "point_paste": "P"}
CONTRASTS = (("P", "N"), ("P", "C"), ("C", "N"))
THRESHOLDS = (.50, .75)
METRICS = ("continuous_iou_points", "intersection_per_hull_pct", "false_positive_per_hull_pct",
           "false_negative_per_hull_pct")


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(raw):
    return sha256(raw).hexdigest()


def load(path, fingerprint):
    raw = path.read_bytes()
    require(digest(raw) == fingerprint, "Historical source changed: " + path.name)
    obj = ModuleType(path.stem)
    obj.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), obj.__dict__)
    return obj


def closed(root, phase, status):
    event_raw = (root / ("observed_" + phase) / "TERMINAL.json").read_bytes()
    event = json.loads(event_raw)
    require(event["phase"] == phase and type(event["actual_exit"]) is int and event["actual_exit"] == 0
            and event["output_footer_verified"] is True, "Observed successful " + phase + " is required")
    raw = (root / phase / "RESULT_SAFE.json").read_bytes()
    result = json.loads(raw)
    require(digest(raw) == event["result_sha256"] and result["worker_sha256"] == event["worker_sha256"]
            and result["protocol_sha256"] == PROTOCOL and result["status"] == status,
            "Bound worker or result differs for " + phase)
    return result, digest(raw), digest(event_raw)


def render_inputs(args, lineage):
    basis, basis_sha, basis_term = closed(args.render, "basis", "UEIKAP_POINTREND_COVERAGE_BASIS_COMPLETE")
    require((basis["images"],basis["groups"],basis["model_forwards"],basis["optimizer_updates"]) == (342,13,342,0)
            and basis["exact_original_candidates_and_masks"] is basis["exact_original_eligible_logits"] is
            basis["exact_all_query_basis_contraction"] is True and basis["GT_opened"] is basis["target_labels_opened"] is False
            and basis["checkpoint_sha256"] == lineage["checkpoint_sha256"]
            and basis["candidate_audit_sha256"] == lineage["audit_sha256"]
            and basis["capture_receipt_sha256"] == lineage["capture_result_sha256"], "Native feature parity differs")
    start, start_sha, start_term = closed(args.render, "prepare-edits", "UEIKAP342_POINTREND_COVERAGE_EDITS_PREPARED")
    verified, ver_sha, ver_term = closed(args.render, "verify-edits", "UEIKAP342_POINTREND_COVERAGE_EDITS_VERIFIED")
    for doc in (start, verified):
        require(doc["images"] == 342 and doc["groups"] == 13 and doc["fit_weight_sha256"] == HEAD
                and doc["target_labels_opened"] is doc["GT_opened"] is False and doc["optimizer_updates"] == 0
                and doc["old_lineage"] == lineage and doc["basis_result_sha256"] == basis_sha
                and doc["basis_terminal_sha256"] == basis_term, "Unexpected source for PointRend masks")
    require(verified["exact_independent_render"] is True and verified["edits_sha256"] == start["edits_sha256"]
            and start["counts"] == verified["counts"] and verified["prepare_result_sha256"] == start_sha
            and verified["prepare_terminal_sha256"] == start_term, "Independent render parity differs")
    raw = (args.render/"prepare-edits/EDITS_PRIVATE.jsonl").read_bytes()
    require(digest(raw) == verified["edits_sha256"] and raw.endswith(b"\n"), "Prepared mask ledger changed")
    rows = [json.loads(line) for line in raw.splitlines()]
    require(len(rows) == 342, "Incomplete render population")
    refs = dict(basis_result_sha256=basis_sha, basis_terminal_sha256=basis_term, prepare_result_sha256=start_sha,
                prepare_terminal_sha256=start_term, verify_result_sha256=ver_sha, verify_terminal_sha256=ver_term,
                edits_sha256=verified["edits_sha256"], fit_weight_sha256=HEAD)
    return rows, verified["counts"], refs


def image_masks(core, stock, audit, row, tally):
    require(row["image_id"] == audit["image_id"] and set(row) == {"image_id", "changes"}
            and len(stock) == len(audit["candidates"]), "Misaligned original queries or edit row")
    arms = dict(N=stock, C=stock.copy(), P=stock.copy())
    for key, updates in row["changes"].items():
        pos = int(key)
        require(str(pos) == key and 1 <= pos <= len(stock) and updates and set(updates) <= set(RENDERED),
                "Unexpected compact ship ordinal or render arm")
        native = stock[pos-1]
        area = int(core.codec.area(native))
        x0,y0,x1,y1 = audit["candidates"][pos-1]["box_xyxy"]
        require(32 <= area < 1024 and x1 > x0 and y1 > y0, "Changed unsupported candidate")
        for mode, update in updates.items():
            require(set(update) == {"rle","adds","drops"} and type(update["adds"]) is type(update["drops"]) is int,
                    "Malformed signed mask update")
            mask = core.checked_rle(update["rle"], core.HEIGHT, core.WIDTH)
            target_area = int(core.codec.area(mask))
            common = int(core.codec.area(core.codec.merge([native,mask],intersect=True)))
            adds, drops = target_area-common, area-common
            require((adds,drops) == (update["adds"],update["drops"]) and adds+drops > 0,
                    "Original signed whole-mask area or changed status differs")
            arms[RENDERED[mode]][pos-1] = mask
            tally.update({mode+"_candidates": 1, mode+"_adds": adds, mode+"_drops": drops,
                          mode+"_empty": int(target_area == 0)})
    return arms


def geometry(core, polygon, masks):
    values, cache = [], {}
    for mask in masks:
        key = mask["counts"]
        if key not in cache:
            iou = core.geometric_iou(mask, polygon, shapely)
            area, truth = int(core.codec.area(mask)), polygon.area
            intersection = iou*(area+truth)/(1+iou)
            require(-1e-8 <= intersection <= min(area,truth)+1e-8, "Invalid continuous intersection")
            cache[key] = [100*iou, 100*intersection/truth, 100*max(0.,area-intersection)/truth,
                          100*max(0.,truth-intersection)/truth]
        values.append(cache[key])
    return np.asarray(values)


def collect(core, box, population, labels, audits, render, intake):
    groups, tally = defaultdict(list), Counter()
    for image_id, ((stock,_), label, audit, row) in enumerate(zip(population, labels, audits, render, strict=True),1):
        require(image_id == label["image_id"] == audit["image_id"] == row["image_id"], "Image roster differs")
        arms = image_masks(core, stock, audit, row, tally)
        labeled, classes = core.parse_yolo(core.label_path(intake,label["label_ref"]),core.HEIGHT,core.WIDTH,shapely)
        require(labeled and classes == label["count_by_class"], "Admitted hull labels differ")
        polygons, rasters = zip(*labeled)
        boxes = [v["box_xyxy"] for v in audit["candidates"]]
        require(all(len(b) == 4 and all(np.isfinite(v) and 0 <= v <= 512 for v in b) for b in boxes),
                "Original clipped XYXY boxes differ")
        association = box.match_box(boxes,[p.bounds for p in polygons])
        ranked = np.zeros((len(polygons),len(ARMS),len(THRESHOLDS)),dtype=int)
        by_box_raster = np.zeros((len(polygons),len(ARMS)))
        for j, arm in enumerate(ARMS):
            overlaps = core.codec.iou(arms[arm],list(rasters),[False]*len(rasters)) if stock else None
            for h, rank in association.items():
                by_box_raster[h,j] = overlaps[rank-1,h]
            for k, threshold in enumerate(THRESHOLDS):
                for h, rank in core.match_native(arms[arm], list(rasters), threshold).items():
                    ranked[h,j,k] = rank
        tally.update(images=1, mapped_candidates=len(stock), labeled_hulls=len(labeled), excluded_wakes=classes.get("1",0))
        for h, poly in enumerate(polygons):
            rank = association.get(h,0)
            fixed = (geometry(core,poly,[arms[arm][rank-1] for arm in ARMS]) if rank
                     else np.zeros((len(ARMS),len(METRICS))))
            native_rank = ranked[h,0,0]
            native_iou = (fixed[0,0] if native_rank == rank else
                          100*core.geometric_iou(stock[native_rank-1],poly,shapely) if native_rank else 0.)
            groups[label["acquisition_group"]].append(dict(rank=ranked[h],box=rank,weak=bool(rank and by_box_raster[h,0]<.5),
                raster=by_box_raster[h],fixed=fixed,native_iou=native_iou))
    return groups, tally


def sample_indices(n):
    return np.random.Generator(np.random.PCG64(5217)).integers(n,size=(4096,n))


def paired(values, groups, draws):
    means = np.asarray(values,dtype=float)
    point = float(means.mean())
    return dict(mean_points=point, paired_95_interval_points=np.percentile(means[draws].mean(axis=1),[2.5,97.5]).tolist(),
                groups_improved=int((means>0).sum()), groups_equal=int((means==0).sum()), groups_worse=int((means<0).sum()),
                by_family_points=dict(zip(groups,means.tolist(),strict=True)))


def coverage(grouped, threshold_pos):
    groups = sorted(grouped)
    ranks = [np.asarray([r["rank"][:,threshold_pos] for r in grouped[g]]) for g in groups]
    means = np.stack([100*(arr>0).mean(axis=0) for arr in ranks])
    draws = sample_indices(len(groups))
    by_group = {g:dict(hulls=len(arr),matched=dict(zip(ARMS,np.count_nonzero(arr,axis=0).tolist(),strict=True)),
                       arm_coverage_points=dict(zip(ARMS,means[i].tolist(),strict=True)))
                for i,(g,arr) in enumerate(zip(groups,ranks,strict=True))}
    reports = {}
    for a,b in CONTRASTS:
        i,j = ARMS.index(a),ARMS.index(b)
        events = {}
        for g,arr in zip(groups,ranks,strict=True):
            after,before = arr[:,i]>0, arr[:,j]>0
            both = after & before
            events[g] = dict(gained=int((after & ~before).sum()),lost=int((~after & before).sum()),
                             retained=int(both.sum()),both_below=int((~after & ~before).sum()),
                             retained_different_query=int((both & (arr[:,i]!=arr[:,j])).sum()))
        totals = Counter()
        for entry in events.values():
            totals.update(entry)
        reports[a+"_minus_"+b] = dict(**paired(means[:,i]-means[:,j],groups,draws),**totals,
            changed_hulls=totals["gained"]+totals["lost"], changed_families=sum(v["gained"]+v["lost"]>0 for v in events.values()),
            transitions_by_family=events)
    return dict(arm_coverage_points=dict(zip(ARMS,means.mean(axis=0).tolist(),strict=True)),
                matched_hulls={a:sum(g["matched"][a] for g in by_group.values()) for a in ARMS},
                hulls=sum(len(v) for v in ranks),families=len(groups),by_family=by_group,contrasts=reports)


def fixed_summary(grouped, subset, all_denominator):
    keys = [tag+"_050_"+outcome for tag in ("raster","continuous") for outcome in ("retained","lost","gained","both_below")]
    groups,arrays,transitions = [],[],{a:Counter(dict.fromkeys(keys,0)) for a in ARMS}
    actual = Counter()
    for g in sorted(grouped):
        active = [r for r in grouped[g] if r[subset]]
        if not active and not all_denominator:
            continue
        groups.append(g)
        actual[g] = len(active)
        denominator = len(grouped[g]) if all_denominator else len(active)
        values = np.zeros((len(ARMS),len(METRICS)))
        for row in active:
            values += row["fixed"]
            for i,a in enumerate(ARMS):
                tally = transitions[a]
                for tag,judge in (("raster",row["raster"]>=.5),("continuous",row["fixed"][:,0]>=50.)):
                    before,after = bool(judge[0]),bool(judge[i])
                    key = ("retained" if after else "lost") if before else ("gained" if after else "both_below")
                    tally[tag+"_050_"+key] += 1
        arrays.append(values/denominator)
    if not groups:
        return None
    means = np.stack(arrays)
    draws = sample_indices(len(groups))
    result = {}
    for a,b in CONTRASTS:
        diff = means[:,ARMS.index(a)]-means[:,ARMS.index(b)]
        result[a+"_minus_"+b] = {metric:paired(diff[:,i],groups,draws) for i,metric in enumerate(METRICS)}
    return dict(active_hulls=sum(actual.values()),active_families=sum(n>0 for n in actual.values()),active_by_family=dict(actual),
                denominator_hulls=sum(len(grouped[g]) if all_denominator else actual[g] for g in groups),families=len(groups),
                arm_means={a:dict(zip(METRICS,means[:,i].mean(axis=0).tolist(),strict=True)) for i,a in enumerate(ARMS)},
                contrasts=result, transitions_from_N_on_active_hulls=transitions)


def summarize(groups, historical, tally, rendered_counts):
    coverage_by_iou = {f"{t:.2f}":coverage(groups,i) for i,t in enumerate(THRESHOLDS)}
    primary = coverage_by_iou["0.50"]
    native_hulls = [r for rows in groups.values() for r in rows if r["rank"][0,0]]
    native_groups = [rows for rows in groups.values() if any(r["rank"][0,0] for r in rows)]
    native_raw = np.mean([np.mean([r["native_iou"] for r in rows]) for rows in groups.values()])
    native_matched = np.mean([np.mean([r["native_iou"] for r in rows if r["rank"][0,0]]) for rows in native_groups])
    require((len(groups),tally["images"],tally["labeled_hulls"],tally["excluded_wakes"],tally["mapped_candidates"],
             len(native_hulls),len(native_groups),primary["matched_hulls"]["N"]) == (13,342,528,396,33975,121,10,121),
            "Historical native hull accounting differs")
    for value,pop in ((native_raw,"primary_all_labeled_hulls"),(native_matched,"originally_matched_hulls_sensitivity")):
        require(abs(value-historical[pop]["arm_mean_iou_points"]["raw"]) <= 1e-9, "Historical continuous geometry differs")
    for mode in RENDERED:
        for suffix in ("candidates","adds","drops","empty"):
            k = mode+"_"+suffix
            require(tally[k] == rendered_counts.get(k,0), "Rendered native signed pixel census differs: "+k)
    all_box = fixed_summary(groups,"box",True)
    checks = {}
    for control in ("N","C"):
        contrast = "P_minus_"+control
        x = primary["contrasts"][contrast]
        y = all_box["contrasts"][contrast]["continuous_iou_points"]
        checks[contrast] = dict(coverage_lower_positive=x["paired_95_interval_points"][0] > 0,
            coverage_change_20_hulls_8_families=x["changed_hulls"]>=20 and x["changed_families"]>=8,
            fixed_box_all_hulls_iou_lower_positive=y["paired_95_interval_points"][0]>0)
    return dict(counts=dict(tally),released_hull_coverage=coverage_by_iou,
                fixed_box_all_hulls=all_box,fixed_box_matched_only=fixed_summary(groups,"box",False),
                weak_box_on_all_hulls=fixed_summary(groups,"weak",True),weak_box_matched_only=fixed_summary(groups,"weak",False),
                decision=dict(contrasts=checks,advance=all(all(check.values()) for check in checks.values())),
                historic_native_match_and_continuous_means_reproduced=True)


def run(args):
    require(digest(args.protocol.read_bytes()) == args.protocol_sha256 == PROTOCOL,"Frozen prelock differs")
    require(np.__version__ == "1.26.4" and shapely.__version__ == "2.0.2" and shapely.geos_version_string=="3.11.2",
            "Historical raster and continuous runtime required")
    old = load(args.code/"UEIKAP_ADDITIONAL_FAMILIES_20261123/score_additional.py",OLD_SCORER)
    hist = load(args.code/"ATTRACTION_SIGNED_SIDES_20261123/score_ueikap.py",OLD_POPULATION)
    original = load(args.code/"ATTRACTION_SIGNED_SIDES_20261123/replay_ueikap.py",OLD_LINEAGE)
    box = load(args.code/"UEIKAP_RFDETR_BOX_ASSOCIATION_20261122/box_association.py",OLD_BOX)
    historical = json.loads(hist.pinned(args.root/"score/RESULT_SAFE.json",hist.OLD_SCORE))
    original_evt = old.observed(args.root/"observed_score/TERMINAL.json",hist.OLD_SCORE_TERMINAL,hist.OLD_SCORE,"score")
    require(original_evt["worker_sha256"] == historical["inputs"]["worker_sha256"] == OLD_SCORER,"Historical worker differs")
    lineage,_,_ = original.old_inputs(args.root)
    render,render_counts,render_refs = render_inputs(args,lineage)
    rf,core = old.methods(args.code)
    population,labels,_,_ = hist.old_population(args,old,core,rf,historical)
    audits = core.stream_pinned(args.root/"capture/CANDIDATE_AUDIT_PRIVATE.jsonl",lineage["audit_sha256"])
    groups,tally = collect(core,box,population,labels,audits,render,args.root/"intake")
    report = summarize(groups,historical,tally,render_counts)
    return dict(**report,status="UEIKAP342_POINTREND_COVERAGE_SCORE_COMPLETE",phase="score",images=342,
                groups=13,labeled_hulls=528,arms=list(ARMS),thresholds=list(THRESHOLDS),
                bootstrap_draws=4096,bootstrap_seed=5217,bootstrap_unit="acquisition family",
                protocol_sha256=PROTOCOL,worker_sha256=digest(Path(__file__).read_bytes()),
                old_score_sha256=hist.OLD_SCORE,old_score_terminal_sha256=hist.OLD_SCORE_TERMINAL,
                render_lineage=render_refs,historical_lineage=lineage,render_counts=render_counts,
                target_labels_opened=True,previously_evaluated_cohort=True,whole_scene_AP=False,
                geographic_holdout=False,detector_forwards=0,head_forwards=0,optimizer_updates=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ("code","root","render","protocol","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--protocol-sha256",required=True)
    args = parser.parse_args()
    require(not args.output.exists(),"Supply a fresh score directory")
    started = perf_counter()
    result = run(args)
    result["seconds"] = perf_counter()-started
    raw = (json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n").encode("utf-8")
    args.output.mkdir()
    with (args.output/"RESULT_SAFE.json").open("xb") as handle:
        handle.write(raw)
    print(json.dumps(dict(status=result["status"],result_sha256=digest(raw))),flush=True)


if __name__ == "__main__":
    main()
