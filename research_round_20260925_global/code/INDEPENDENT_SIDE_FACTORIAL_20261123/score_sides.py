"""Score which side of a fixed-count boundary swap benefits from independent selection."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from types import ModuleType

import numpy as np
import shapely


SIDES = ("FF", "IF", "FI", "II")
PAIRS = (("IF","FF"), ("FI","FF"), ("II","FF"), ("II","IF"), ("II","FI"), ("IF","FI"))
PRIMARY = ("II_minus_IF", "II_minus_FI")
FIELDS = ("continuous_iou_points", "fp_per_truth_area_points", "fn_per_truth_area_points", "intersection_px2")
EVALUATOR = "a0ff8233617770fe95b828fd72369590855ce2314ee84006ebe93857cac1a07d"
INPUTS = "ae600c98bcb353a1a2b4dce4a4b5b2aa9edca46b15a7ddb68fe50ec5b549ff0f"
REPLAYER = "aa590649fb2fa0a70d7fd239b26e9f865efac2d8c0dfe86c4525933f6720f214"
REFERENCE = {
    (74,5217): ("INSTANCE_RSDD_PARENT_RESERVE_ACTUAL_SAFE_20261123.json",
                "737a5de1e28a2a9729c69d8c32a5a1dfb667affab989c81561c957a901bb3b2e"),
    (740,5217): ("INSTANCE_RSDD_FROZEN_HRSID_5217_SAFE_20261122.json",
                 "59936bb4ceb70db1af4801f8d31d1ba84f223adb688759c795d8a71e58f7da4b"),
    (740,5218): ("INSTANCE_RSDD_FROZEN_HRSID_5218_SAFE_20261122.json",
                 "a8cf7537d8a42c6c17cc08b11d4699354b3eedc3fb26cb2ee24150eef7b1d90a"),
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return sha256(raw).hexdigest()


def load_source(path, expected=None):
    raw = path.read_bytes()
    require(expected is None or digest(raw)==expected, "Unexpected source identity: "+path.name)
    mod = ModuleType(path.stem)
    mod.__file__ = str(path)
    exec(compile(raw,str(path),"exec"),mod.__dict__)
    return mod


def arms(seeds):
    return ("raw", *(f"{s}/{arm}" for s in seeds for arm in SIDES))


def check_mask_sets(codec, masks, native, k):
    """Verify equal signed budgets and that each hybrid uses its named foreground/background side."""
    area = int(codec.area(native))
    inside = {}
    for arm, mask in masks.items():
        inside[arm] = codec.merge([mask,native],intersect=True)
        require(int(codec.area(mask))==area and area-int(codec.area(inside[arm]))==k,
                "Realized signed budget or total area differs")
    for left, right in (("FF","IF"), ("FI","II")):
        require(inside[left]==inside[right],"Identical removal choice required")
    for left, right in (("FF","FI"), ("IF","II")):
        shared = codec.merge([masks[left],masks[right]],intersect=True)
        stock_overlap = codec.merge([shared,native],intersect=True)
        require(int(codec.area(shared))-int(codec.area(stock_overlap))==k, "Identical addition choice required")


def checked_blocks(rows, edit_bytes, seeds, adapter, core):
    blocks = {}
    numbers = {s: Counter() for s in seeds}
    for (item,audit,pred,old),line in zip(rows,edit_bytes.splitlines(),strict=True):
        image = item["image_id"]
        row = json.loads(line)
        require(set(row)=={"image_id","changes"} and row["image_id"]==image and image not in blocks,
                "New replay row changed image order or format")
        native = adapter.native_masks(core,pred,audit,image)
        require(set(old)==set(map(str,seeds)) and all(v["image_id"]==image for v in old.values()),
                "Historical seed or candidate image identity differs")
        old_by_seed = {s:old[str(s)]["changes"] for s in seeds}
        expected = {str(j) for j in range(1,len(native)+1)
                    if any("fixed" in hist.get(str(j),{}) for hist in old_by_seed.values())}
        require(set(row["changes"])==expected,"New candidate set differs from the original fixed budget")
        changes = {}
        for key, entries in row["changes"].items():
            pos = int(key)
            mask = native[pos-1]
            checked = {}
            for seed, hist in old_by_seed.items():
                previous = hist.get(key,{})
                ff, ii = previous.get("fixed"), previous.get("free_matched_count")
                require((ff is None)==(ii is None), "Historical fixed budget differs")
                if ff is None:
                    continue
                k = ff["swaps"]
                require(type(k) is int and 0<k<=8 and k==ii["swaps"], "Historical shared positive budget")
                masks = {}
                for arm in SIDES:
                    label = f"{seed}/{arm}"
                    entry = entries[label]
                    require(set(entry)=={"rle","adds","drops"} and type(entry["adds"]) is type(entry["drops"]) is int
                            and entry["adds"]==entry["drops"]==k, "Side arm signed counts differ")
                    masks[arm] = core.checked_rle(entry["rle"],512,512)
                    checked[label] = entry
                require(masks["FF"]==ff["rle"] and masks["II"]==ii["rle"], "Historical FF or II RLE differs")
                check_mask_sets(core.codec,masks,mask,k)
                numbers[seed].update(positive_budget_candidates=1,swaps_per_arm=k)
            require(set(checked)==set(entries),"Undefined arm or unexpected seed on this candidate")
            changes[pos]=checked
        blocks[image] = dict(stock=pred["coco"],ranks=[c["position"] for c in audit["candidates"]],changes=changes,
            group=audit["acquisition_group"],sensor=audit["sensor"],file=audit["source_image_name"],rgb=audit["rgb_sha256"])
    return blocks,numbers


def linear_effects(array):
    """Input [...,4,fields], arm order FF/IF/FI/II. Differences are always left minus right."""
    effect = {a+"_minus_"+b: array[...,SIDES.index(a),:] - array[...,SIDES.index(b),:] for a,b in PAIRS}
    effect["interaction"] = effect["II_minus_IF"] - effect["FI_minus_FF"]
    return effect


def geometry(anchors, blocks, polygons, dates, seeds, core, reserve):
    group_values = defaultdict(list)
    labels = arms(seeds)
    exposures = {c:{k:Counter() for k in (*map(str,seeds),"any_seed","all_seeds")} for c in PRIMARY}
    counts = Counter(native_matches=len(anchors),eligible_matches=0,invalid_geometry=0,area_excluded=0)
    max_error = 0.0
    for gid,(i,pos,det) in anchors.items():
        original, poly = det["segmentation"], polygons[gid]
        if not poly.is_valid or poly.area<=0:
            counts["invalid_geometry"]+=1
            continue
        area = int(core.codec.area(original))
        if not 32<=area<1024:
            counts["area_excluded"]+=1
            continue
        counts["eligible_matches"]+=1
        updates, group = blocks[i]["changes"].get(pos,{}), blocks[i]["group"]
        cache, tokens, scores = {}, {}, []
        for arm in labels:
            rle = updates.get(arm,{}).get("rle",original)
            key = tokens[arm] = rle["counts"]
            if key not in cache:
                iou = core.geometric_iou(rle,poly,shapely)
                inter = iou*(area+poly.area)/(1+iou)
                cache[key] = (100*iou,100*(area-inter)/poly.area,100*(poly.area-inter)/poly.area,inter)
            scores.append(cache[key])
        score_array=np.asarray(scores,dtype=np.float64)
        effects=linear_effects(score_array[1:].reshape(len(seeds),len(SIDES),len(FIELDS)))
        error = float(np.max(np.abs(effects["interaction"][:,-1])))
        require(error<=1e-7,"Independent sides have a nonzero polygon-intersection interaction")
        max_error=max(max_error,error)
        group_values[group].append(score_array)
        for key in PRIMARY:
            a,b = key.split("_minus_")
            flags = [tokens[f"{s}/{a}"] != tokens[f"{s}/{b}"] for s in seeds]
            for name,on in zip((*map(str,seeds),"any_seed","all_seeds"),(*flags,any(flags),all(flags)),strict=True):
                if on:
                    exposures[key][name][group]+=1
    summary = {c:{k:dict(matches=sum(g.values()),groups=len(g),blocks=len({dates[p] for p in g}))
                  for k,g in by_seed.items()} for c,by_seed in exposures.items()}
    parents = sorted(group_values)
    keys = [dates[p] for p in parents]
    out = dict(counts=dict(counts),represented_groups=len(parents),resampled_blocks=len(set(keys)),
        actual_disagreement_exposure=summary,max_intersection_interaction_px2=max_error,intersection_tolerance_px2=1e-7,
        group_equal_arms={},per_seed={},pooled={})
    if not parents:
        return out
    means = np.stack([np.mean(group_values[p],axis=0) for p in parents])
    out["group_equal_arms"] = {key:dict(zip(FIELDS,means[:,j].mean(axis=0).tolist(),strict=True))
                               for j,key in enumerate(labels)}
    effects = linear_effects(means[:,1:].reshape(len(parents),len(seeds),len(SIDES),len(FIELDS)))

    def interval(values):
        limits = reserve.bootstrap(values,keys)
        return {f:dict(delta=float(mean),interval_95=ci) for f,mean,ci in zip(FIELDS,values.mean(axis=0),limits,strict=True)}

    for key,d in effects.items():
        out["per_seed"][key] = {str(s):interval(d[:,j]) for j,s in enumerate(seeds)}
        # Per-match seed averaging commutes with the mean within each parent: the same anchors enter every seed.
        out["pooled"][key] = interval(d.mean(axis=1))
    return out


def metric_contrasts(metrics, seeds, history):
    effects = {a+"_minus_"+b:{} for a,b in PAIRS}
    effects["interaction"]={}
    for seed in seeds:
        for a,b in PAIRS:
            effects[a+"_minus_"+b][str(seed)] = history.difference(metrics[f"{seed}/{a}"],metrics[f"{seed}/{b}"])
        lhs,rhs=(effects[k][str(seed)] for k in ("II_minus_IF","FI_minus_FF"))
        effects["interaction"][str(seed)]={k:None if lhs[k] is None or rhs[k] is None else lhs[k]-rhs[k] for k in lhs}
    summary={}
    for comparison, by_seed in effects.items():
        summary[comparison]={}
        for k in next(iter(by_seed.values())):
            values=[v[k] for v in by_seed.values()]
            defined=all(x is not None for x in values)
            summary[comparison][k]=dict(mean=float(np.mean(values)) if defined else None,
                median=float(np.median(values)) if defined else None,
                sample_sd=float(np.std(values,ddof=1)) if defined and len(values)>1 else None)
    return effects,summary


def score_view(ids, gt, blocks, polygons, dates, seeds, evaluator, core, reserve, history):
    scoped={i:b for i,b in blocks.items() if i in ids}
    dataset=dict(gt,images=[r for r in gt["images"] if r["id"] in ids],
                 annotations=[r for r in gt["annotations"] if r["image_id"] in ids])
    metrics={}
    for arm in arms(seeds):
        measured,matched=evaluator.score_one(dataset,scoped,arm,anchors=arm=="raw")
        require(len(measured)==13 and {"AP","AP90","AR100"}<=set(measured), "Original COCO mask metrics differ")
        metrics[arm]={k:history.available(v) for k,v in measured.items()}
        if arm=="raw":
            anchors=matched
    geo=geometry(anchors,scoped,polygons,dates,seeds,core,reserve)
    effects,summary=metric_contrasts(metrics,seeds,history)
    gates={}
    for name in PRIMARY:
        e=geo["actual_disagreement_exposure"][name]
        iou=geo["pooled"].get(name,{}).get(FIELDS[0])
        gates[name]=dict(exposure_20_matches_8_groups=any(e[str(s)]["matches"]>=20 and e[str(s)]["groups"]>=8 for s in seeds),
                         all_seed_AP_positive=all(v["AP"] is not None and v["AP"]>0 for v in effects[name].values()),
                         pooled_IoU_lower_positive=iou is not None and iou["interval_95"][0]>0)
    return dict(images=len(ids),annotations=len(dataset["annotations"]),source_groups=len({r["group"] for r in scoped.values()}),
        metrics_fraction=metrics,per_seed_contrasts_points=effects,contrast_summaries_points=summary,geometry=geo,
        primary_gates=gates,joint_criterion=all(all(g.values()) for g in gates.values()))


def check_references(view,name,references,cohort,history):
    counts=Counter()
    for seed,reference in references.items():
        old=reference["views"][name]
        require(view["images"]==old["images"] and view["annotations"]==old["annotations"], "Cohort drift")
        old_geometry=old["geometry"] if cohort==74 else old["geometry"]["stats"]
        for arm,prior in (("raw","raw"),(f"{seed}/FF","fixed"),(f"{seed}/II","free_matched_count")):
            values={k:history.available(v) for k,v in old["metrics_fraction"][prior].items()}
            require(view["metrics_fraction"][arm]==values, "Historical complete COCO vector differs for "+arm)
            current=view["geometry"]["group_equal_arms"][arm]
            require(all(abs(current[k]-old_geometry["group_equal_arms"][prior][k])<1e-9 for k in FIELDS[:3]),
                    "Historical original-mask fixed-match geometry differs for "+arm)
            counts.update(COCO_fields=len(values),geometry_means=3)
    return counts


def run(a,inputs):
    h = load_source(a.code/"POINTREND_SIGNED_QUOTA_SWAP_20261123/score_quota_swap.py",EVALUATOR)
    cohort = int(a.cohort)
    seeds = (5217,) if cohort==74 else (5217,5218)
    h.pin(a.protocol,a.protocol_sha256)
    traw=a.replay_terminal.read_bytes()
    terminal=json.loads(traw)
    require(terminal["phase"] == "replay" and type(terminal["actual_exit"]) is int
            and terminal["actual_exit"] == 0 and terminal["output_footer_verified"] is True,
            "Closed replay and independent matching footer required before scoring")
    receipt=json.loads(h.pin(a.replay/"RESULT_SAFE.json",terminal["result_sha256"]))
    require(receipt["status"]=="INDEPENDENT_SIDE_FACTORIAL_REPLAY_COMPLETE" and receipt["cohort"]==receipt["images"]==cohort
            and receipt["protocol_sha256"]==a.protocol_sha256 and receipt["worker_sha256"]==terminal["worker_sha256"]==REPLAYER
            and receipt["input_reader_sha256"] == INPUTS
            and tuple(receipt["seeds"])==seeds and tuple(receipt["arms"])==SIDES and receipt["GT_opened"] is False
            and receipt["target_labels_opened"] is False and receipt["model_forwards"]==receipt["optimizer_updates"]==0
            and all(receipt[k] is True for k in ("original_arms_exact","per_candidate_signed_counts_exact","cross_context_signed_xor_exact")),
            "Unqualified replay receipt")
    rows,lineage=inputs.load_population(a)
    require(receipt["input_lineage"]==lineage and shapely.__version__=="2.0.2" and shapely.geos_version_string=="3.11.2",
            "Frozen native stream or geometry runtime differs")
    reserve=h.published_module(a.code/"INSTANCE_RSDD_ACQUISITION_RESERVE_20261123/score_reserve.py",h.RESERVE_SCORE)
    old=h.published_module(a.code/"INSTANCE_RSDD_FROZEN_HRSID_20261122/score_rsdd.py",reserve.HISTORIC)
    evaluator=old.module(a.code/"PSEG_RFDETR_FROZEN_SOURCE_HEAD_20261122/score.py",old.SCORE_SOURCE)
    adapter=old.module(a.code/"UEIKAP_RFDETR_CARRIER_20261122/score_rf.py",old.GEOMETRY_SOURCE)
    core=adapter.load_geometry(a.code/"UEIKAP_RESERVED_HULL_FROZEN_20261122/score.py")
    blocks,budgets=checked_blocks(rows,h.pin(a.replay/"EDITS_PRIVATE.jsonl",receipt["edits_sha256"]),seeds,adapter,core)
    del rows
    require(len(blocks)==cohort, "Incomplete native cohort")
    for seed, totals in budgets.items():
        require(all(receipt["counts_by_seed"][str(seed)][name]==v for name,v in totals.items()),
                "Reconstructed positive counts and swaps differ from the replay")
    gt,rekeyed,polygons,sets,dates,annotation_sha=h.views_and_labels(a,old,reserve,blocks)
    refs,ref_shas={},{}
    for s in seeds:
        filename,stamp=REFERENCE[cohort,s]
        ref=json.loads(h.pin(a.code.parent/"results"/filename,stamp))
        refs[s]=ref["score"] if cohort==74 else ref
        ref_shas[s]=stamp
    views,cache,parity={},{},{}
    for name,ids in sets.items():
        key=frozenset(ids)
        if key not in cache:
            cache[key]=score_view(ids,gt,blocks,polygons,dates,seeds,evaluator,core,reserve,h)
        views[name]=cache[key]
        parity[name]=check_references(views[name],name,refs,cohort,h)
        print(json.dumps(dict(phase="scored_view",view=name,images=len(ids))),flush=True)
    primary="full" if cohort==74 else "gf3"
    return dict(status="INDEPENDENT_SIDE_FACTORIAL_SCORE_COMPLETE",cohort=cohort,seeds=list(seeds),arms=list(SIDES),
        arm_letters="first is addition, second is deletion; F fixed pairing, I independent",
        primary_view=primary,primary_joint_criterion=views[primary]["joint_criterion"],views=views,
        protocol_sha256=a.protocol_sha256,input_lineage=lineage,source_score_sha256=ref_shas,
        historical_parity=parity,original_FF_II_exact=True,budgets_by_seed=budgets,
        edits_sha256=receipt["edits_sha256"],replay_result_sha256=terminal["result_sha256"],
        replay_terminal_sha256=digest(traw),author_labels_sha256=annotation_sha,author_ids_rekeyed=rekeyed,
        non_simple_polygons_retained_in_ap=sum(not p.is_valid for p in polygons.values()),
        bootstrap_draws=4096,bootstrap_seed=5217,cluster_unit="sensor/date" if cohort==74 else "acquisition group",
        optimizer_updates=0,model_forwards=0)


def main():
    inputs=load_source(Path(__file__).with_name("side_inputs.py"), INPUTS)
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    inputs.add_input_arguments(parser)
    for name in ("replay","replay-terminal","author-labels","protocol","output"):
        parser.add_argument("--"+name,required=True,type=Path)
    parser.add_argument("--protocol-sha256",required=True)
    args=parser.parse_args()
    for k,v in vars(args).items():
        if isinstance(v,Path):
            setattr(args,k,v.resolve(strict=k!="output"))
    require(not args.output.exists(),"A fresh score output directory is required")
    start=perf_counter(); result=run(args,inputs)
    result.update(seconds=perf_counter()-start,worker_sha256=digest(Path(__file__).read_bytes()))
    raw=(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n").encode()
    args.output.mkdir()
    with (args.output/"RESULT_SAFE.json").open("xb") as out:
        out.write(raw)
    print(json.dumps(dict(status=result["status"],result_sha256=digest(raw))),flush=True)


if __name__=="__main__":
    main()
