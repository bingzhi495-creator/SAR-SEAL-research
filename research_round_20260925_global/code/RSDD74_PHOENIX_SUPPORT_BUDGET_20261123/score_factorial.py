"""Score the four frozen Phoenix support and total-edit-budget policies on Instance-RSDD 74."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from types import ModuleType, SimpleNamespace

import numpy as np
from pycocotools import mask as codec
import shapely


ARMS = ("nc","nt","wc","wt")
PAIRS = (("wc","wt"),("nc","wc"),("nc","nt"),("nt","wt"))
BASES = ("raw","phoenix","historical_lvis")
INTERACTION = "(wc-wt)-(nc-nt)"
AREA_WORKER = "a7f2a920dd43955fb810d325cc79b518aaa63194f88e25d70777ab7d5180caf9"
AREA_SCORE = "5eade7925b66b56772c576cd37c92f08ba49ec8eac69458affb8031c78ceba82"
AREA_TERM = "1581eb12702bd1bb0e79b5fee7fbe6fa33fe1fa644e9471a0289ecf84ed8f07e"
WHOLE_SCORE = "7c27a55ff2f21206e8aa3e5eca05ec5af218afe09074a23f0d786c67efee3b9f"
WHOLE_TERM = "9300c0b18069c44a49757ecc0e53b49d85ec9074978ad28ae03bb1f1fdd98b7d"
STATUS = "RSDD74_PHOENIX_SUPPORT_BUDGET_SCORE_COMPLETE"
LOCK = "f9ad7f1bd128fea19b88ef6e97efa2679ad942565691da2bd408dab8b711ee9f"
INFER_STATUS = "RSDD74_PHOENIX_SUPPORT_BUDGET_INFERENCE_COMPLETE"
INFER_WORKER = "8d46b1d16bcba274576f03448e7cf74b9d332a9bf8848b090aa659409fa91c80"
LOCK_NAME = "RSDD74_PHOENIX_SUPPORT_BUDGET_PRELOCK_20261123.md"


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def digest(data):
    return sha256(data).hexdigest()


def check_candidate(cand, probs, old_free, old_lvis, entries, check_area):
    require(set(entries) == set(ARMS) and all(set(v)=={"rle"} for v in entries.values())
            and entries["nc"]["rle"] == old_free and entries["wt"]["rle"] == old_lvis,
            "Frozen narrow capped or full-grid threshold mask differs")
    native = codec.decode(cand["segmentation"]).astype(bool)
    coords = np.asarray(cand["coords"])
    prob = np.asarray(probs, dtype=np.float64)
    k = cand["k"]
    require(native.shape == (512,512) and 32 <= int(native.sum()) < 1024 and type(k) is int and
            1 <= k <= min(8,len(coords),int(native.sum())//10) and coords.shape == prob.shape == (len(coords),2)
            and coords.dtype.kind in "iu" and np.all((coords>=0)&(coords<native.size)) and np.unique(coords).size==coords.size
            and not native.flat[coords[:,0]].any() and native.flat[coords[:,1]].all() and np.isfinite(prob).all()
            and np.all((0<=prob)&(prob<=1)) and not np.any(prob==.5),
            "Positive frozen signed support and archived finite probabilities without endpoint ties at one half required")
    masks,counts={},{}
    for arm in ARMS:
        value=entries[arm]["rle"]
        area=check_area(value)
        mask=codec.decode(value).astype(bool)
        canonical=codec.encode(np.asfortranarray(mask,dtype=np.uint8))
        require(canonical["counts"].decode("ascii")==value["counts"],"Canonical whole-image RLE required")
        added=int(np.count_nonzero(mask & ~native))
        dropped=int(np.count_nonzero(native & ~mask))
        require(area == int(native.sum())+added-dropped and (arm not in ("wc","nc") or added+dropped == 2*k),
                "Written area or realized total edit count differs")
        masks[arm]=mask
        counts[arm]=dict(adds=added,drops=dropped,flips=added+dropped,empty_outputs=int(area==0),
                         changed_candidates=int(added+dropped>0),signed_area=added-dropped,
                         candidates_below_2k=int(added+dropped<2*k),candidates_equal_2k=int(added+dropped==2*k),
                         candidates_above_2k=int(added+dropped>2*k))
    expected=native.copy(order="C")
    expected.flat[coords]=prob>.5
    require(np.array_equal(expected,masks["nt"]) and np.array_equal(masks["wt"].flat[coords],prob>.5),
            "Threshold writer differs from archived probabilities at original endpoint coordinates")
    utilities=np.stack((prob[:,0]-.5,.5-prob[:,1]),axis=1)
    for arm in ARMS:
        changed=masks[arm]^native
        signed=utilities[changed.flat[coords]]
        counts[arm].update(narrow_flips=len(signed),outside_narrow_flips=counts[arm]["flips"]-len(signed),
                           known_narrow_positive=int(np.count_nonzero(signed>0)),
                           known_narrow_nonpositive=int(np.count_nonzero(signed<=0)),
                           also_changed_by_wt=int(np.count_nonzero(changed & (masks["wt"]^native))))
    # All source posteriors on the signed narrow domain are available. The subset of
    # these sites selected by the whole-grid cap must be a stable prefix on each side.
    wc_edit=masks["wc"] ^ native
    flags=wc_edit.flat[coords]
    for col,order in enumerate((np.argsort(-prob[:,0],kind="stable"),np.argsort(prob[:,1],kind="stable"))):
        subset=flags[order,col]
        require(not np.any(subset[1:] & ~subset[:-1]), "Whole-grid cap skips a better sampled site or tie ordinal")
    if np.any(flags) and not np.all(flags):
        require(utilities[flags].min() >= utilities[~flags].max()-1e-13,
                "Whole-grid cap skips a higher signed utility on the sampled support")
    # If capped and threshold edit sets cross, their asymmetric differences must all
    # be zero-margin cells. The complete writer provides the independent global bound.
    threshold_edits=masks["wt"] ^ native
    outside=int(np.count_nonzero(wc_edit & ~threshold_edits))
    missed=int(np.count_nonzero(threshold_edits & ~wc_edit))
    counts["crossing_tie_cells_required"]=(outside+missed) if outside and missed else 0
    return counts


def sources(a):
    file=a.code/"RSDD74_PHOENIX_AREA_ABLATION_20261123/score_ablation.py"
    raw=file.read_bytes()
    require(digest(raw)==AREA_WORKER,"Frozen area-ablation scorer required")
    area=ModuleType("archived_area")
    area.__file__=str(file)
    exec(compile(raw,str(file),"exec"),area.__dict__)
    whole,sides,deps=area.sources(a)
    whole.ARMS=(*BASES,*ARMS)
    whole.MATCHED=(*BASES,"nc")
    whole.PAIRS=PAIRS
    return area,whole,sides,deps


def archive_args(a):
    m=a.matched_root
    return SimpleNamespace(code=a.code,stock_root=a.stock_root,point_root=a.point_root,cohort=74,support_root=None,
        phx_edits=m/"edits",phx_edits_terminal=m/"observed_edits/TERMINAL.json",phx_score=m/"score",
        phx_score_terminal=m/"observed_score/TERMINAL.json",request=m/"request",request_terminal=m/"observed_request/TERMINAL.json",
        inference=m/"inference",inference_terminal=m/"observed_inference/TERMINAL.json",
        endpoints=m/"inference/ENDPOINT_PROBABILITIES_PRIVATE.jsonl",writer=a.area_root/"writer",
        writer_terminal=a.area_root/"observed_edits/TERMINAL.json")


def load_inputs(a, area, whole, sides, deps):
    phx,api=deps[:2]
    aa=archive_args(a)
    blocks,prior,chain,_,_=area.observed_inputs(aa,sides,deps)
    a_score,_,_=api.observed(a.area_root/"score",a.area_root/"observed_score/TERMINAL.json","score",area.STATUS,
                             AREA_SCORE,AREA_TERM,AREA_WORKER)
    require(all(a_score[k]==v for k,v in chain.items()) and a_score["historical_balanced_masks_and_scores_exact"] is True,
            "Original native, matched and narrow-free anchors differ")
    w_score,_,_=api.observed(a.whole_root/"score",a.whole_root/"observed_score/TERMINAL.json","score",whole.STATUS,
                             WHOLE_SCORE,WHOLE_TERM,area.WHOLE_WORKER)
    whole_infer,_,_=api.observed(a.whole_root/"inference",a.whole_root/"observed_inference/TERMINAL.json","inference",
                                 "PHOENIX_RSDD74_WHOLE_INFERENCE_COMPLETE",w_score["inference_result_sha256"],
                                 w_score["inference_terminal_sha256"],whole.INFER_WORKER)
    require(w_score["historical_references_exact"] is True and w_score["input_lineage"]==chain["input_lineage"]
            and whole_infer["masks_sha256"]==w_score["masks_sha256"] and whole_infer["target_labels_opened"] is False,
            "Original whole-mask reference must share native detections")
    infer,isha,iterm=api.observed(a.inference,a.inference_terminal,"inference",INFER_STATUS,worker=INFER_WORKER)
    expected=dict(request_sha256=chain["request_sha256"],endpoint_sha256=chain["endpoint_sha256"],
                  area_edits_sha256=chain["edits_sha256"],whole_masks_sha256=w_score["masks_sha256"])
    require(all(infer[k]==v for k,v in expected.items()) and infer["phase"]=="inference" and infer["protocol_sha256"]==LOCK
            and infer["target_labels_opened"] is False and infer["native_detector_forwards"]==infer["optimizer_updates"]==0
            and infer["baseline_worker_sha256"]==phx.INFER_SHA and infer["author_commit"]==phx.AUTHOR
            and infer["weight_sha256"]==phx.WEIGHT_SHA and infer["endpoint_probabilities_exact"] is True
            and infer["input_lineage"]==chain["input_lineage"]
            and infer["nc_matches_archived_free"] is infer["wt_matches_archived_lvis"] is True
            and tuple(infer[k] for k in ("workers","threads","seed","batch_size","refine_iters","smooth_kernel"))==(4,2,5202,2,5,1)
            and infer["runtime"]==whole_infer["runtime"],"Closed exact native inference and archived masks required")
    result_bytes=api.pin(a.masks or a.inference/"FACTORIAL_EDITS_PRIVATE.jsonl",infer["masks_sha256"])
    archive=api.pin(a.whole_root/"inference/WHOLE_MASKS_PRIVATE.jsonl",w_score["masks_sha256"])
    request=json.loads(api.pin(aa.request/"REQUEST_PRIVATE.json",expected["request_sha256"]))
    require(infer["source_replay"]==request["source_replay"],"Original candidate replay lineage differs")
    endpoint_bytes=api.pin(aa.endpoints,expected["endpoint_sha256"])
    totals=Counter()
    per_arm={arm:Counter() for arm in ARMS}
    stream=lambda b:map(json.loads,b.splitlines())
    for j,(image_id,original,req,post,received) in enumerate(zip(blocks,stream(archive),request["images"],
                                                            stream(endpoint_bytes),stream(result_bytes),strict=True),1):
        b=blocks[image_id]
        require(req["image_id"]==post["image_id"]==received["image_id"]==image_id and req["decoded_rgb_sha256"]==b["rgb"]
                and type(received["ordinal"]) is int and received["ordinal"]==original["ordinal"]==j
                and set(received)=={"image_id","ordinal","changes"} and set(original)=={"ordinal","lvis","coco"},
                "Ordinal, image, native identity or ledger schema differs")
        positive=sorted(b["changes"])
        require(set(received["changes"])=={str(x) for x in positive} and [c["position"] for c in req["candidates"]]==positive,
                "All and only original admitted candidates must appear")
        full_positions=[p for p,det in enumerate(b["stock"],1) if int(codec.area(det["segmentation"]))>0]
        for pos,entry in zip(full_positions,original["lvis"],strict=True):
            require(type(entry["position"]) is int and pos==entry["position"],"Historical whole-mask native order differs")
            b["changes"].setdefault(pos,{})["historical_lvis"]={"rle":entry["segmentation"]}
        for pos,cand,probs in zip(positive,req["candidates"],post["candidates"],strict=True):
            entries=received["changes"][str(pos)]
            archived=b["changes"][pos]
            require(pos==probs["position"]==cand["position"] and cand["segmentation"]==b["stock"][pos-1]["segmentation"],
                    "Wrong candidate position or original segmentation")
            summary=check_candidate(cand,probs["probabilities"],archived["free"]["rle"],archived["historical_lvis"]["rle"],
                                    entries,whole.native_area)
            for arm in ARMS:
                per_arm[arm].update(summary[arm])
            totals["minimum_crossing_ties"]+=summary["crossing_tie_cells_required"]
            totals.update(decoded_candidates=1,endpoint_pairs=len(cand["coords"]),native_count_sum=cand["k"])
            archived.update(entries)
        totals.update(images=1,image_encodes=int(bool(positive)),decoder_batches=(len(positive)+1)//2,
                      native_candidates=len(b["stock"]),historical_lvis_nonempty=len(full_positions))
    require((totals["images"],totals["native_candidates"],totals["historical_lvis_nonempty"],totals["decoded_candidates"],
             totals["native_count_sum"],totals["endpoint_pairs"])==(74,7346,6079,4107,29900,129621),
            "Original common population or endpoint census differs")
    real=infer["counts"]
    require(all(totals[k]==real[k] for k in ("images","image_encodes","decoder_batches","decoded_candidates",
                "endpoint_pairs","native_count_sum")) and real["exact_endpoint_probabilities"]==real["endpoint_probabilities"]==2*129621
            and real["exact_nc_masks"]==real["exact_wt_masks"]==4107 and real["threshold_disagreements_away_from_half"]==0
            and real["whole_count_rank_certificates"]==4107 and real["endpoint_probability_differences"]==0
            and real["candidate_refinement_iterations"]==20535
            and real["threshold_half_narrow_endpoint_cells"]==0 and totals["minimum_crossing_ties"]<=real["threshold_half_cells"],
            "Remote posterior or threshold/cap geometry verification disagrees with closed endpoints")
    for arm,c in per_arm.items():
        source=infer["counts_by_arm"][arm]
        require(all(source[k]==c[k] for k in ("adds","drops","flips","changed_candidates","empty_outputs",
                "outside_narrow_flips","candidates_below_2k","candidates_equal_2k","candidates_above_2k")),
                "Per-arm foreground, changed or empty mask count differs")
        require(source["signed_area_delta"]==c["signed_area"] and source["inside_narrow_flips"]==c["narrow_flips"],
                "Signed area or realized support membership differs")
    lineage=dict(native_input_lineage=chain["input_lineage"],matched_score_sha256=chain["historical_score_sha256"],
                 area_score_sha256=AREA_SCORE,area_score_terminal_sha256=AREA_TERM,area_writer_sha256=chain["writer_result_sha256"],
                 whole_score_sha256=WHOLE_SCORE,whole_score_terminal_sha256=WHOLE_TERM,whole_inference_sha256=w_score["inference_result_sha256"],
                 inference_result_sha256=isha,inference_terminal_sha256=iterm,inference_worker_sha256=INFER_WORKER,
                 masks_sha256=infer["masks_sha256"],**expected)
    return blocks,prior,a_score,w_score,lineage,dict(all_candidates=totals,by_arm=per_arm),infer


def interval_for_nonpositive_counts(counts, real):
    """Derive strict gain bounds from local signed samples and the author's exact global threshold ties."""
    c=counts["by_arm"]
    zero=real["counts"]["threshold_zero_utility_flipped_cells"]
    wt=c["wt"]["flips"]
    require(0<=zero<=min(real["counts"]["threshold_half_cells"],wt),"Invalid zero-margin threshold census")
    out={}
    for arm in ARMS:
        n=c[arm]["flips"]
        overlap=c[arm]["also_changed_by_wt"]
        # Every strict positive belongs to WT. At most 'zero' of that set can be zero-margin.
        lo=max(c[arm]["known_narrow_positive"],overlap-min(zero,overlap))
        hi=min(n-c[arm]["known_narrow_nonpositive"],overlap-max(0,zero-(wt-overlap)))
        require(0<=lo<=hi<=n,"Contradictory local and whole threshold margin counts")
        out[arm]=dict(positive_min=lo,positive_max=hi,nonpositive_min=n-hi,nonpositive_max=n-lo)
    return out


def verify_margin_counts(counts,receipt,bounds):
    exact={}
    for arm in ARMS:
        c=counts["by_arm"][arm]
        m=receipt["counts_by_arm"][arm]
        positive,zero,negative=(m[k+"_selected_flips"] for k in ("positive","zero","negative"))
        bound=bounds[arm]
        require(all(type(n) is int and n>=0 for n in (positive,zero,negative)) and positive+zero+negative==c["flips"]
                and zero+negative==m["nonpositive_selected_flips"] and bound["positive_min"]<=positive<=bound["positive_max"]
                and bound["nonpositive_min"]<=zero+negative<=bound["nonpositive_max"]
                and zero<=receipt["counts"]["threshold_half_cells"] and negative>=c["known_narrow_nonpositive"],
                "Full-grid margin partition disagrees with native masks or known endpoint utilities")
        require((arm not in ("nc","nt") or (zero==0 and positive==c["known_narrow_positive"]))
                and (arm not in ("nt","wt") or negative==0)
                and (arm!="wt" or zero==receipt["counts"]["threshold_zero_utility_flipped_cells"]),
                "Archived endpoint or author threshold margin classification differs")
        exact[arm]=dict(positive=positive,zero=zero,negative=negative,nonpositive=zero+negative)
    return exact


def interaction_summary(parent_values,keys,cfa,reserve):
    nc,nt,wc,wt=(parent_values[:,ARMS.index(arm)] for arm in ARMS)
    return cfa.interval((wc-wt)-(nc-nt),keys,reserve)


def reference_view(prior,area,whole):
    def joined(table):
        return dict(raw=table(prior)["raw"],phoenix=table(prior)["phoenix"],
                    nc=table(area)["free"],historical_lvis=table(whole)["lvis"])
    metrics=joined(lambda v:v["metrics_fraction"])
    geom=joined(lambda v:v["geometry"]["group_equal_arms"])
    return dict(prior,metrics_fraction=metrics,geometry=dict(prior["geometry"],group_equal_arms=geom))


def fixed_factorial(anchors,blocks,polygons,dates,whole,core,cfa,reserve):
    groups=defaultdict(list)
    changed={pair:Counter() for pair in PAIRS}
    for truth,(image,pos,det) in anchors.items():
        poly=polygons[truth]
        native=det["segmentation"]
        if not poly.is_valid or poly.area<=0 or not 32<=int(codec.area(native))<1024:
            continue
        block=blocks[image]
        changes=block["changes"].get(pos,{})
        masks={arm:changes.get(arm,{}).get("rle",native) for arm in ARMS}
        cache={}
        for m in masks.values():
            if m["counts"] not in cache:
                cache[m["counts"]]=whole.geometry_vector(core.geometric_iou(m,poly,shapely),int(codec.area(m)),poly.area)
        groups[block["group"]].append([cache[masks[arm]["counts"]] for arm in ARMS])
        for pair in PAIRS:
            if masks[pair[0]]["counts"]!=masks[pair[1]]["counts"]:
                changed[pair][block["group"]]+=1
    exposures={a+"_minus_"+b:dict(matches=sum(c.values()),parents=len(c),sensor_date_blocks=len({dates[k] for k in c}))
               for (a,b),c in changed.items()}
    parents=sorted(groups)
    require(bool(parents),"Previously qualified nonempty fixed-hull cohort required")
    means=np.stack([np.mean(groups[k],axis=0) for k in parents])
    stats=interaction_summary(means,[dates[k] for k in parents],cfa,reserve)
    return stats,exposures,means.mean(axis=0),sum(len(g) for g in groups.values())


def score_view(ids,gt,blocks,polygons,dates,ref,whole,deps):
    _,_,cfa,_,reserve,_,evaluator,_,core=deps
    out=whole.score_view(ids,gt,blocks,polygons,dates,ref,deps)
    scope={i:b for i,b in blocks.items() if i in ids}
    part=dict(gt,images=[im for im in gt["images"] if im["id"] in ids],
              annotations=[ann for ann in gt["annotations"] if ann["image_id"] in ids])
    _,fixed=evaluator.score_one(part,scope,"raw",anchors=True)
    geometry,exposure,means,fixed_count=fixed_factorial(fixed,scope,polygons,dates,whole,core,cfa,reserve)
    require(fixed_count==out["geometry"]["counts"]["eligible_matches"],"Factorial population differs from paired comparisons")
    for j,arm in enumerate(ARMS):
        require(np.allclose(means[j],[out["geometry"]["group_equal_arms"][arm][f] for f in cfa.FIELDS],
                            atol=1e-9,rtol=0),"Interaction anchor means differ from original evaluator")
    for name,e in exposure.items():
        contrast=out["contrasts"][name]
        bound=out["geometry"]["paired"][name]["continuous_iou_points"]["interval_95"]
        ap=contrast["COCO_points"]["AP"]
        minimum=1.0 if name=="wc_minus_wt" else .30
        supported=e["matches"]>=20 and e["sensor_date_blocks"]>=8
        contrast["criteria"]=dict(differing_fixed_hulls=e["matches"],differing_parents=e["parents"],
            differing_sensor_date_blocks=e["sensor_date_blocks"],absolute_ap_threshold=minimum,exposure_supported=supported,
            supported_advantage=bool(supported and ap is not None and ap>=minimum and bound[0]>0),
            supported_loss=bool(supported and ap is not None and ap<=-minimum and bound[1]<0))
    left=out["contrasts"]["wc_minus_wt"]["COCO_points"]
    right=out["contrasts"]["nc_minus_nt"]["COCO_points"]
    effect={field:None if left[field] is None or right[field] is None else left[field]-right[field] for field in left}
    out["interaction"]=dict(definition=INTERACTION,COCO_points=effect,paired_geometry=geometry)
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for opt in ("code","stock-root","point-root","matched-root","area-root","whole-root","inference","inference-terminal",
                "protocol","output","author-labels"):
        parser.add_argument("--"+opt,required=True,type=Path)
    parser.add_argument("--masks",type=Path)
    parser.add_argument("--protocol-sha256",required=True)
    a=parser.parse_args()
    for name,path in vars(a).items():
        if isinstance(path,Path):
            setattr(a,name,path.resolve(strict=name!="output"))
    require(a.protocol.name==LOCK_NAME and a.protocol_sha256==LOCK==digest(a.protocol.read_bytes()) and
            not a.output.exists(),"Published factorial protocol and fresh score directory required")
    require(shapely.__version__=="2.0.2" and shapely.geos_version_string=="3.11.2","Historical geometry runtime")
    a.cohort,a.support_root=74,None
    start=perf_counter()
    area,whole,sides,deps=sources(a)
    phx,api,_,history,reserve,old,_,_,_=deps
    blocks,prior,ascore,wscore,chain,counts,inference=load_inputs(a,area,whole,sides,deps)
    margins=interval_for_nonpositive_counts(counts,inference)
    exact_margins=verify_margin_counts(counts,inference,margins)
    print(json.dumps(dict(phase="all_mask_arms_verified",candidates=counts["all_candidates"]["native_candidates"])),flush=True)
    # Original target polygons are opened only after verifying all observed input identities and complete mask ledgers.
    gt,rekeyed,polygons,sets,dates,labels_sha=history.views_and_labels(a,old,reserve,blocks)
    views,cache={},{}
    for name,ids in sets.items():
        key=frozenset(ids)
        ref=reference_view(prior["views"][name],ascore["views"][name],wscore["views"][name])
        if key not in cache:
            cache[key]=score_view(ids,gt,blocks,polygons,dates,ref,whole,deps)
        views[name]=cache[key]
        phx.parity(views[name],ref,whole.MATCHED)
        print(json.dumps(dict(phase="scored_view",view=name,images=len(ids))),flush=True)
    require(len(views)==5 and views["full"]["geometry"]["counts"]["eligible_matches"]==47 and
            views["date_separated"]["geometry"]["counts"]["eligible_matches"]==32,"Expected five frozen populations")
    full=views["full"]
    def summary(name):
        return dict(**full["contrasts"][name],paired_continuous_iou_points=full["geometry"]["paired"][name]["continuous_iou_points"])
    result=dict(status=STATUS,phase="score",worker_sha256=digest(Path(__file__).read_bytes()),protocol_sha256=LOCK,
                **chain,counts=counts,endpoint_probabilities_checked=inference["counts"]["exact_endpoint_probabilities"],
                certified_whole_grid_optima=inference["counts"]["whole_count_rank_certificates"],
                threshold_half_cells=inference["counts"]["threshold_half_cells"],margin_count_bounds=margins,
                selected_signed_margins=exact_margins,
                inference_seconds=inference["seconds"],inference_summed_worker_seconds=inference["summed_worker_seconds"],
                score_seconds=perf_counter()-start,score_detector_forwards=0,score_phoenix_forwards=0,optimizer_updates=0,
                target_labels_opened=True,author_labels_sha256=labels_sha,author_ids_rekeyed=rekeyed,
                native_unchanged_outside_common_population=True,narrow_threshold_projection_exact=True,
                historical_raw_balanced_free_and_full_lvis_exact=True,cluster_unit="sensor/date; equal parent weight",
                bootstrap_draws=4096,bootstrap_seed=5217,arms=list(whole.ARMS),primary_view="full",
                primary_contrast="wc_minus_wt",primary=summary("wc_minus_wt"),
                secondary={a+"_minus_"+b:summary(a+"_minus_"+b) for a,b in PAIRS[1:]},
                interaction=full["interaction"],views=views)
    a.output.mkdir()
    receipt=api.write(a.output/"RESULT_SAFE.json",api.encode(result))
    print(json.dumps(dict(status=STATUS,result_sha256=receipt)),flush=True)


if __name__=="__main__":
    main()
