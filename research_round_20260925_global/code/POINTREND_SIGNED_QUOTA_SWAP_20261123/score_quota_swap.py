"""Score the two-by-two signed-quota PointRend position comparison on either frozen cohort."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType

import numpy as np
import shapely


SEEDS = (5217,5218,5219)
CONTEXT = ("local","full")
BASE = {"local":"p_local","full":"point_paste"}
CONTROLS = ("raw","learned_common","point_equal","point_paste","p_local")
LEARNED = tuple(f"{seed}/{c}" for seed in SEEDS for c in CONTEXT)
ARMS = CONTROLS + LEARNED
NEW = ("p_local",) + LEARNED
FIELDS = ("continuous_iou_points","fp_per_truth_area_points","fn_per_truth_area_points")
OLD_SCORES = {
    74:("INSTANCE_RSDD_RESERVE_POINTREND_ACTUAL_SAFE_20261123.json","c54b9a53a5bfd3f6a54b6d31d93a0789eb176b07bd2cb52c58c6f2ceb4e77b27"),
    740:("INSTANCE_RSDD_POINTREND_TRANSFER_SCORE_SAFE_20261122.json","2b3791a794511bb437b9cd47830fb3b01f19bb30cb68107a224acc33e5a75c31"),
}
H10 = ("INSTANCE_RSDD_POINTREND_SUPPORT_FACTOR_SCORE_SAFE_20261123.json","bd6f0974727513732cc6f70b724f72ec59f18fdc21a052e62ac3f20454e05b5e")
RESERVE_SCORE = "5ff671b2b83bb12e322b0c766897bf3cdacaedd233bf0d6edcdfcccf2c938df4"
STATUS = "POINTREND_SIGNED_QUOTA_SWAP_SCORE_COMPLETE"
PROTOCOL = "f51a2404008b90be6a13f28513fc6ba62d776f7ab64873a4d549fe1772d530d8"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return sha256(data).hexdigest()


def pin(path, sha):
    raw = path.read_bytes()
    require(digest(raw)==sha, "Input digest differs: "+path.name)
    return raw


def shared_inputs():
    # The observer launches the worker in Python isolated mode.
    return published_module(Path(__file__).resolve().with_name("inputs.py"),
                            "2c3b84f852bb4f8234d935347708a2e71eba4466d0eb111471bd254aa2f45e58")


def published_module(path, sha):
    mod = ModuleType(path.stem)
    mod.__file__ = str(path)
    exec(compile(pin(path,sha),str(path),"exec"),mod.__dict__)
    return mod


def rle_for(updates, arm, native):
    return updates.get(arm,{}).get("rle",native)


def matched_edits(updates, original, core):
    """Validate each archived native-relative signed edit count, including full-mask pastes."""
    native_area = int(core.codec.area(original))
    signed = {}
    for arm in (*NEW,"point_paste"):
        change = updates.get(arm)
        if change:
            after = core.checked_rle(change["rle"],512,512)
            common = int(core.codec.area(core.codec.merge([original,after],intersect=True)))
            adds,drops = int(core.codec.area(after))-common,native_area-common
            require(type(change["adds"]) is int and type(change["drops"]) is int and
                    (adds,drops)==(change["adds"],change["drops"]) and adds+drops>0, "Native signed pixel counts differ")
            signed[arm] = (adds,drops)
        else:
            signed[arm] = (0,0)
    for c in CONTEXT:
        require(all(signed[f"{s}/{c}"]==signed[BASE[c]] for s in SEEDS),"Per-sign quota differs within context")
    return signed


def prepare_blocks(rows, edit_bytes, core, adapter, cohort):
    blocks = {}
    totals = {arm:Counter(adds=0,drops=0,changed_candidates=0) for arm in (*NEW,"point_paste")}
    for (item,audit,pred,point,h10),line in zip(rows,edit_bytes.splitlines(),strict=True):
        i = item["image_id"]
        edit = json.loads(line)
        require(set(edit)=={"image_id","changes"} and i==edit["image_id"]==point["image_id"] and i not in blocks,
                "Frozen replay identity or ordering differs")
        native = adapter.native_masks(core,pred,audit,i)
        require(isinstance(edit["changes"],dict) and all(k.isdigit() and str(int(k))==k and 1<=int(k)<=len(native)
                                                         for k in edit["changes"]), "Unknown replay position")
        changes = {}
        for j, mask in enumerate(native,1):
            k = str(j)
            prior = point["changes"].get(k,{})
            supplied = edit["changes"].get(k,{})
            require(set(supplied)<=set(NEW) and all(set(x)=={"rle","adds","drops"} for x in supplied.values()),
                    "Unknown new arm or signed edit fields")
            u = {arm:prior[arm] for arm in ("point_paste","learned_common","point_equal") if arm in prior}
            u.update(supplied)
            if cohort==740:
                require(h10 is not None and h10["image_id"]==i,"Historical H10 source required")
                ref = rle_for(h10["changes"].get(k,{}),"p_on_support",mask)
                require(rle_for(u,"p_local",mask)==ref,"P inside S and native outside must reproduce archived H10")
            if u:
                signs = matched_edits(u,mask,core)
                for arm,(adds,drops) in signs.items():
                    totals[arm].update(adds=adds,drops=drops,changed_candidates=int(adds+drops>0))
                changes[j] = u
        blocks[i] = dict(stock=pred["coco"],ranks=[r["position"] for r in audit["candidates"]],changes=changes,
                         group=audit["acquisition_group"],sensor=audit["sensor"],file=audit["source_image_name"],rgb=audit["rgb_sha256"])
    require(len(blocks)==cohort,"Incomplete target roster")
    return blocks,totals


def views_and_labels(args, old, reserve, blocks):
    cohort = int(args.cohort)
    dates = {}
    if cohort==74:
        census = json.loads(pin(args.code.parent/"results/INSTANCE_RSDD_PARENT_RESERVE_CENSUS_SAFE_20261123.json",reserve.CENSUS))
        require(Counter((b["group"],b["sensor"]) for b in blocks.values())==
                {(p["paper_scene"],p["sensor"]):p["images"] for p in census["per_parent"]},"Historical 74-parent census differs")
        sha = reserve.TRAIN
        gt = reserve.select_labels(pin(args.author_labels,sha),blocks)
        dates = {p["paper_scene"]:(p["sensor"],p["calendar_date"]) for p in census["per_parent"]}
        selected_dates = set(census["date_separated_sensitivity"]["parent_scene_ids"])
    else:
        sha = old.AUTHOR_TEST
        gt = {"info":{},**json.loads(pin(args.author_labels,sha))}
        require(len(gt["images"])==740 and len(gt["annotations"])==872 and
                {r["id"] for r in gt["images"]}==set(blocks) and
                all(r["file_name"]==blocks[r["id"]]["file"] and (r["width"],r["height"])==(512,512) for r in gt["images"]),
                "Official 740 roster differs")
        dates = {b["group"]:b["group"] for b in blocks.values()}
    gt,rekeyed = old.coco_annotation_ids(gt)
    require(all(r["category_id"]==1 and len(r["segmentation"])==1 for r in gt["annotations"]), "Original hull polygons required")
    polygons = {r["id"]:shapely.Polygon(np.asarray(r["segmentation"][0]).reshape(-1,2)) for r in gt["annotations"]}
    full = set(blocks)
    gf = {i for i,b in blocks.items() if b["sensor"]=="GF-3"}
    sets = dict(full=full,gf3=gf,terrasar_x=full-gf,full_deduplicated=old.representatives(blocks,full))
    if cohort==74:
        sets["date_separated"] = {i for i,b in blocks.items() if b["group"] in selected_dates}
        require(len(sets["date_separated"])==42 and len(gf)==42 and len(gt["annotations"])==81,"Expected reserve populations")
    else:
        sets["gf3_deduplicated"] = old.representatives(blocks,gf)
        require(len(gf)==583,"Expected historical Gaofen-3 subset")
    return gt,rekeyed,polygons,sets,dates,sha


def checked_intersections(stats, tol=1e-7):
    """Unions are measured in native square pixels. Recover polygon intersections from continuous IoU."""
    maximum = 0.0
    for s in SEEDS:
        delta, signs = [], []
        for c in CONTEXT:
            a, b = stats[f"{s}/{c}"], stats[BASE[c]]
            require(a[0]==b[0],"Areas must match within a background")
            delta.append(a[1]-b[1])
            signs.append(np.sign(a[2]-b[2]))
        error = abs(delta[0]-delta[1])
        require(error<=tol,"Signed intersection gain depends on the outside mask")
        if min(abs(x) for x in delta)>tol:
            require(signs[0]==signs[1]==np.sign(delta[0]),"IoU direction depends on the outside mask")
        maximum=max(maximum,error)
    return maximum


def measure_matches(anchors,blocks,polygons,core):
    groups = defaultdict(list)
    counts = Counter(original_matches=len(anchors),invalid_polygons=0,area_excluded=0,eligible_matches=0)
    exposure = {name:Counter() for name in (*map(str,SEEDS),"any_seed","all_seeds")}
    max_error = 0.0
    for gid,(i,j,det) in anchors.items():
        poly = polygons[gid]
        original = det["segmentation"]
        if not poly.is_valid or poly.area<=0:
            counts["invalid_polygons"]+=1
            continue
        if not 32<=int(core.codec.area(original))<1024:
            counts["area_excluded"]+=1
            continue
        counts["eligible_matches"]+=1
        group = blocks[i]["group"]
        updates = blocks[i]["changes"].get(j,{})
        cache,stats,values={},{},[]
        for arm in ARMS:
            mask = rle_for(updates,arm,original)
            k=mask["counts"]
            if k not in cache:
                area=int(core.codec.area(mask))
                iou=core.geometric_iou(mask,poly,shapely)
                overlap=iou*(area+poly.area)/(1+iou)
                cache[k]=(area,overlap,iou)
            area,overlap,iou=stats[arm]=cache[k]
            values.append([100*iou,100*(area-overlap)/poly.area,100*(poly.area-overlap)/poly.area])
        max_error=max(max_error,checked_intersections(stats))
        changed=[]
        for seed in SEEDS:
            flags=[]
            for c in CONTEXT:
                a,b=(rle_for(updates,x,original) for x in (f"{seed}/{c}",BASE[c]))
                flags.append(a!=b and int(core.codec.area(core.codec.merge([a,b],intersect=True)))<stats[BASE[c]][0])
            require(flags[0]==flags[1],"Actual position disagreement differs across backgrounds")
            changed.append(flags[0])
            if flags[0]:
                exposure[str(seed)][group]+=1
        if any(changed):
            exposure["any_seed"][group]+=1
        if all(changed):
            exposure["all_seeds"][group]+=1
        groups[group].append(values)
    return groups,dict(counts),{k:dict(matches=sum(x.values()),groups=len(x)) for k,x in exposure.items()},max_error


def interval(values,keys,reserve):
    bounds=reserve.bootstrap(values,keys)
    return {f:dict(delta=float(x),interval_95=ci) for f,x,ci in zip(FIELDS,values.mean(axis=0),bounds,strict=True)}


def effect_arrays(array):
    """Input has observations x arms x fields. Return seed x observations x fields."""
    local=np.stack([array[:,ARMS.index(f"{s}/local")]-array[:,ARMS.index("p_local")] for s in SEEDS])
    full=np.stack([array[:,ARMS.index(f"{s}/full")]-array[:,ARMS.index("point_paste")] for s in SEEDS])
    return dict(full=full,local=local,interaction=full-local)


def geometry(anchors,blocks,polygons,dates,core,reserve):
    groups,counts,exposure,error=measure_matches(anchors,blocks,polygons,core)
    parents=sorted(groups)
    keys=[dates[p] for p in parents]
    result=dict(counts=counts,actual_changed_exposure=exposure,represented_groups=len(parents),resampled_blocks=len(set(keys)),
                max_cross_context_intersection_error_px2=error,intersection_tolerance_px2=1e-7,
                group_equal_arms={},per_seed={},pooled={})
    if not parents:
        return result
    means=np.stack([np.mean(groups[g],axis=0) for g in parents])
    result["group_equal_arms"]={arm:dict(zip(FIELDS,means[:,j].mean(axis=0).tolist(),strict=True)) for j,arm in enumerate(ARMS)}
    match_effects=[effect_arrays(np.asarray(groups[g])) for g in parents]
    for key,delta in effect_arrays(means).items():
        result["per_seed"][key]={str(s):interval(delta[i],keys,reserve) for i,s in enumerate(SEEDS)}
        # Collapse seeds on each original fixed match before aggregating parents and drawing whole clusters.
        pooled=np.stack([g[key].mean(axis=0).mean(axis=0) for g in match_effects])
        result["pooled"][key]=interval(pooled,keys,reserve)
        seed_means=delta.mean(axis=1)
        for j,field in enumerate(FIELDS):
            result["pooled"][key][field].update(median_seed_delta=float(np.median(seed_means[:,j])),
                                                seed_sd=float(np.std(seed_means[:,j],ddof=1)))
    return result


def available(v):
    return float(v) if v is not None and np.isfinite(v) and v>=0 else None


def difference(a,b):
    return {k:100*(v-b[k]) if v is not None and b[k] is not None else None for k,v in a.items()}


def stats(rows):
    out={}
    for k in rows[0]:
        v=[r[k] for r in rows]
        if any(x is None for x in v):
            out[k]=dict(mean=None,median=None,sd=None)
        else:
            out[k]=dict(mean=float(np.mean(v)),median=float(np.median(v)),sd=float(np.std(v,ddof=1)))
    return out


def score_view(ids,gt,blocks,polygons,dates,evaluator,core,reserve):
    scoped={i:b for i,b in blocks.items() if i in ids}
    selected=dict(gt,images=[r for r in gt["images"] if r["id"] in ids],
                  annotations=[r for r in gt["annotations"] if r["image_id"] in ids])
    metrics={}
    for arm in ARMS:
        scores,matched=evaluator.score_one(selected,scoped,arm,anchors=arm=="raw")
        require(len(scores)==13 and {"AP","AP90","AR100"}<=set(scores),"Expected original COCO metrics")
        metrics[arm]={k:available(v) for k,v in scores.items()}
        if arm=="raw":
            anchors=matched
    effects={c:{str(s):difference(metrics[f"{s}/{c}"],metrics[BASE[c]]) for s in SEEDS} for c in CONTEXT}
    effects["interaction"]={}
    for s in map(str,SEEDS):
        a,b=effects["full"][s],effects["local"][s]
        effects["interaction"][s]={k:v-b[k] if v is not None and b[k] is not None else None for k,v in a.items()}
    summary={c:stats(list(d.values())) for c,d in effects.items()}
    geo=geometry(anchors,scoped,polygons,dates,core,reserve)
    exp=geo["actual_changed_exposure"]["any_seed"]
    iou=geo["pooled"].get("full",{}).get(FIELDS[0])
    gate=dict(exposure=exp["matches"]>=20 and exp["groups"]>=8,
              each_seed_ap_positive=all(r["AP"] is not None and r["AP"]>0 for r in effects["full"].values()),
              pooled_iou_lower_positive=iou is not None and iou["interval_95"][0]>0)
    return dict(images=len(ids),annotations=len(selected["annotations"]),source_groups=len({r["group"] for r in scoped.values()}),
                metrics_fraction=metrics,per_seed_contrasts_points=effects,contrast_seed_summaries_points=summary,
                learned_seed_metrics_fraction={c:stats([metrics[f"{s}/{c}"] for s in SEEDS]) for c in CONTEXT},
                geometry=geo,criteria=gate,within_recipe_criterion=all(gate.values()))


def control_parity(view,reference,local=None):
    for arm in ("raw","point_equal","learned_common","point_paste"):
        require(view["metrics_fraction"][arm]=={k:available(v) for k,v in reference["metrics_fraction"][arm].items()},
                "Original COCO control differs")
        current=view["geometry"]["group_equal_arms"][arm]
        expected=reference["geometry"]["group_equal_arms"][arm]
        require(all(abs(current[f]-expected[f])<1e-9 for f in FIELDS),"Original acquisition-balanced geometry differs")
    if local:
        require(view["metrics_fraction"]["p_local"]=={k:available(v) for k,v in local["metrics_fraction"]["p_on_support"].items()},
                "Historical P-local COCO differs")
        expected=local["geometry"]["means_points"]["p_on_support"]
        require(all(abs(view["geometry"]["group_equal_arms"]["p_local"][k]-expected[h])<1e-9 for k,h in
                    zip(FIELDS,("continuous_iou","fp_over_truth","fn_over_truth"),strict=True)),
                "Historical P-local geometry differs")


def run(args,inputs):
    cohort=int(args.cohort)
    pin(args.protocol,PROTOCOL)
    traw=args.replay_terminal.read_bytes()
    term=json.loads(traw)
    require(type(term["actual_exit"]) is int and term["actual_exit"]==0 and term["output_footer_verified"] is True,
            "Replay must be observed terminated with a verified footer before target labels")
    receipt=json.loads(pin(args.replay/"RESULT_SAFE.json",term["result_sha256"]))
    require(receipt["status"]=="POINTREND_SIGNED_QUOTA_SWAP_REPLAY_COMPLETE" and receipt["cohort"]==receipt["images"]==cohort
            and receipt["protocol_sha256"]==PROTOCOL and receipt["target_labels_opened"] is False and receipt["GT_opened"] is False
            and all(receipt[k] is True for k in ("exact_signed_quotas","exact_cross_context_xor","exact_native_support",
                                                 "archived_5217_rank_parity","point_full_preserved"))
            and receipt["historical_H10_exact"] is (True if cohort==740 else None),"Frozen mask proof or cohort differs")
    rows,lineage=inputs.load_population(args)
    require(receipt["input_lineage"]==lineage and lineage["cohort"]==cohort and set(receipt["weights_sha256"])==set(map(str,SEEDS)),
            "Replay not bound to the named archived inputs or seeds")
    require(shapely.__version__=="2.0.2" and shapely.geos_version_string=="3.11.2","Original geometry runtime required")
    reserve=published_module(args.code/"INSTANCE_RSDD_ACQUISITION_RESERVE_20261123/score_reserve.py",RESERVE_SCORE)
    old=published_module(args.code/"INSTANCE_RSDD_FROZEN_HRSID_20261122/score_rsdd.py",reserve.HISTORIC)
    evaluator=old.module(args.code/"PSEG_RFDETR_FROZEN_SOURCE_HEAD_20261122/score.py",old.SCORE_SOURCE)
    adapter=old.module(args.code/"UEIKAP_RFDETR_CARRIER_20261122/score_rf.py",old.GEOMETRY_SOURCE)
    core=adapter.load_geometry(args.code/"UEIKAP_RESERVED_HULL_FROZEN_20261122/score.py")
    blocks,totals=prepare_blocks(rows,pin(args.replay/"EDITS_PRIVATE.jsonl",receipt["edits_sha256"]),core,adapter,cohort)
    del rows
    for s in map(str,SEEDS):
        expected=receipt["counts_by_seed"][s]
        for c in CONTEXT:
            require(all(expected[c+"_"+f]==totals[s+"/"+c][f] for f in ("adds","drops","changed_candidates")),
                    "Rendered versus scored signed counts differ")
    require(totals["point_paste"]["adds"]==receipt["counts"]["point_total_adds"] and
            totals["point_paste"]["drops"]==receipt["counts"]["point_total_drops"],"Whole-mask edit totals differ")
    # Author labels are read only after checking the successful observed replay and all scored edit streams.
    gt,rekeyed,polygons,sets,dates,author_sha=views_and_labels(args,old,reserve,blocks)
    filename,ref_digest=OLD_SCORES[cohort]
    result_path=args.code.parent/"results"
    ref=json.loads(pin(result_path/filename,ref_digest))
    h10=json.loads(pin(result_path/H10[0],H10[1]))["views"] if cohort==740 else {}
    views,cache={},{}
    for name,ids in sets.items():
        key=frozenset(ids)
        if key not in cache:
            cache[key]=score_view(ids,gt,blocks,polygons,dates,evaluator,core,reserve)
        views[name]=cache[key]
        control_parity(views[name],ref["views"][name],h10.get(name))
    primary="full" if cohort==74 else "gf3"
    return dict(status=STATUS,cohort=cohort,primary_view=primary,primary_criterion=views[primary]["within_recipe_criterion"],
                protocol_sha256=PROTOCOL,replay_result_sha256=term["result_sha256"],replay_terminal_sha256=digest(traw),
                edits_sha256=receipt["edits_sha256"],old_score_sha256=ref_digest,old_support_score_sha256=H10[1] if cohort==740 else None,
                weights_sha256=receipt["weights_sha256"],input_lineage=lineage,author_labels_sha256=author_sha,
                old_scores_verified=True,old_h10_verified=cohort==740,signed_counts=totals,replay_counts=receipt["counts"],
                counts_by_seed=receipt["counts_by_seed"],seeds=list(SEEDS),bootstrap_draws=4096,bootstrap_seed=5217,
                cluster_unit="sensor and date" if cohort==74 else "acquisition group",views=views,
                author_ids_rekeyed=rekeyed,non_simple_polygons_retained_in_ap=sum(not p.is_valid for p in polygons.values()),
                model_forwards=0,optimizer_updates=0)


def main():
    inputs=shared_inputs()
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    inputs.add_input_arguments(parser)
    for name in ("replay","replay-terminal","protocol","author-labels","output"):
        parser.add_argument("--"+name,required=True,type=Path)
    args=parser.parse_args()
    require(not args.output.exists(),"Use a fresh score directory")
    started=perf_counter()
    result=run(args,inputs)
    result.update(elapsed_seconds=perf_counter()-started,worker_sha256=digest(Path(__file__).read_bytes()))
    raw=(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    args.output.mkdir()
    (args.output/"RESULT_SAFE.json").write_bytes(raw)
    print(json.dumps(dict(status=STATUS,result_sha256=digest(raw))),flush=True)


if __name__=="__main__":
    main()
