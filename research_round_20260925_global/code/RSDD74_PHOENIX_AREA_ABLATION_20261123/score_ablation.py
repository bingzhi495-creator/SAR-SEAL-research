"""Verify and score frozen Phoenix endpoint masks at three signed area budgets."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from math import fsum
from pathlib import Path
from time import perf_counter
from types import ModuleType

import numpy as np
from pycocotools import mask as codec
import shapely


POLICIES = ("balance","relaxed","free")
PAIRS = (("balance","free"),("relaxed","balance"))
WHOLE_WORKER = "79c04a5c4761891c6d7f8bdfefdfd6089b8c10b07bda594048dd708d2de1c281"
WRITER_WORKER = "e42306bb001b9cf0dfb4732908cb9ab92afde0ef20d67d77a2f1814fc2c6b465"
LOCK = "ae73858a9dec877575a800616b8da17b73717358042b81aa86016678b42db57e"
LOCK_NAME = "RSDD74_PHOENIX_AREA_ABLATION_PRELOCK_20261123.md"
STATUS = "RSDD74_PHOENIX_AREA_ABLATION_SCORE_COMPLETE"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(data):
    return sha256(data).hexdigest()


def selected_ordinals(probabilities, k, policy):
    """Enumerate feasible integer allocations and rank each side by the original posterior."""
    require(policy in POLICIES and probabilities.ndim == 2 and probabilities.shape[1] == 2
            and np.isfinite(probabilities).all() and np.all((0 <= probabilities) & (probabilities <= 1))
            and type(k) is int and 0 < k <= min(8,len(probabilities)), "Invalid frozen endpoint probabilities or quota")
    n = len(probabilities)
    asc_add = sorted(range(n), key=lambda i:-float(probabilities[i,0]))
    asc_drop = sorted(range(n), key=lambda i:float(probabilities[i,1]))
    best = None
    for na in range(max(0,2*k-n), min(n,2*k)+1):
        nd = 2*k-na
        imbalance = abs(na-nd)
        if (policy == "balance" and imbalance) or (policy == "relaxed" and imbalance > 2):
            continue
        add, drop = asc_add[:na], asc_drop[:nd]
        utility = fsum([float(probabilities[i,0])-.5 for i in add]+[.5-float(probabilities[i,1]) for i in drop])
        key = (utility, -imbalance, -na)
        if best is None or key > best[0]:
            best = (key, add, drop)
    return best[1], best[2]


def verify_candidate(candidate, probabilities, entries, archived):
    q = np.asarray(probabilities, dtype=np.float64)
    coords = np.asarray(candidate["coords"])
    k, stock = candidate["k"], candidate["segmentation"]
    native = codec.decode(stock).astype(bool)
    require(native.shape == (512,512) and 32 <= int(native.sum()) < 1024 and
            coords.shape == q.shape and coords.ndim == 2 and coords.shape[1] == 2 and coords.dtype.kind in "iu"
            and np.all((coords >= 0) & (coords < native.size)) and np.unique(coords).size == coords.size
            and not np.any(native.flat[coords[:,0]]) and np.all(native.flat[coords[:,1]])
            and archived["swaps"] == k and set(entries) == set(POLICIES)
            and entries["balance"]["rle"] == archived["rle"], "Frozen signed support, area, quota or balance differs")
    splits = {}
    for arm in POLICIES:
        add, drop = selected_ordinals(q, k, arm)
        out = native.copy(order="C")
        out.flat[coords[add,0]], out.flat[coords[drop,1]] = True, False
        encoded = codec.encode(np.asfortranarray(out,dtype=np.uint8))
        expected = {"size":encoded["size"],"counts":encoded["counts"].decode("ascii")}
        require(entries[arm] == {"rle":expected} and len(add)+len(drop) == 2*k,
                "Written mask differs from independently maximized signed endpoint split")
        splits[arm] = len(add),len(drop)
    return splits


def sources(a):
    path = a.code/"RSDD74_PHOENIX_WHOLEMASK_20261123/score_whole.py"
    data = path.read_bytes()
    require(digest(data) == WHOLE_WORKER, "Frozen variable-area geometry evaluator differs")
    whole = ModuleType("whole_mask_geometry")
    whole.__file__ = str(path)
    exec(compile(data,str(path),"exec"),whole.__dict__)
    sides,deps = whole.frozen_sources(a)
    whole.ARMS = ("raw",*POLICIES)
    whole.MATCHED = ("raw","balance")
    whole.PAIRS = PAIRS
    return whole,sides,deps


def observed_inputs(a, sides, deps):
    phx, api, cfa, _, _, _, _, adapter, core = deps
    prior, ref_sha, rt = api.observed(a.phx_score, a.phx_score_terminal,"score",phx.SCORE,
                                     sides.PHX_SCORE,sides.PHX_TERM,sides.PHX_WORKER)
    old, esha, eterm = api.observed(a.phx_edits,a.phx_edits_terminal,"edits",phx.EDIT,
                                    prior["edits_result_sha256"],prior["edits_terminal_sha256"],sides.PHX_WORKER)
    require(old["target_labels_opened"] is False and old["independent_ranking_exact"] is old["archived_arms_exact"] is True
            and old["edits_sha256"] == prior["edits_sha256"] and prior["historical_arms_exact"] is True,
            "Original matched outcome and frozen mask ledger required")
    req, rsha, rterm = api.observed(a.request,a.request_terminal,"request",api.STATUS,
                                    old["request_result_sha256"],old["request_terminal_sha256"],phx.BUILDER)
    infer, isha, iterm = api.observed(a.inference,a.inference_terminal,"inference",phx.INFER,
                                      old["inference_result_sha256"],old["inference_terminal_sha256"],phx.INFER_SHA)
    writer,wsha,wterm = api.observed(a.writer,a.writer_terminal,"edits","RSDD74_PHOENIX_AREA_ABLATION_EDITS_COMPLETE",
                                     worker=WRITER_WORKER)
    chain = dict(request_sha256=req["request_sha256"],request_result_sha256=rsha,request_terminal_sha256=rterm,
                 endpoint_sha256=infer["endpoint_sha256"],inference_result_sha256=isha,inference_terminal_sha256=iterm,
                 archived_edits_sha256=old["edits_sha256"],archived_edits_result_sha256=esha,archived_edits_terminal_sha256=eterm)
    require(all(writer[k]==v for k,v in chain.items()) and writer["protocol_sha256"] == LOCK and writer["phase"] == "edits"
            and writer["target_labels_opened"] is False and writer["balanced_rles_exact"] is True
            and writer["phoenix_forwards"] == writer["detector_forwards"] == writer["optimizer_updates"] == 0
            and req["protocol_sha256"] == infer["protocol_sha256"] == old["protocol_sha256"] == api.LOCK
            and infer["target_labels_opened"] is req["target_labels_opened"] is False
            and old["request_sha256"] == infer["request_sha256"] == req["request_sha256"]
            and old["endpoint_sha256"] == infer["endpoint_sha256"],"Writer and historical endpoint lineage differ")
    loader = cfa.source(a.code/"POINTREND_SIGNED_QUOTA_SWAP_20261123/inputs.py",cfa.INPUTS_SHA)
    rows,lin = loader.load_population(a)
    require(writer["input_lineage"] == lin == prior["input_lineage"] == old["input_lineage"] ==
            req["input_lineage"] == infer["input_lineage"], "Original reserve capture or candidate roster differs")
    request = json.loads(api.pin(a.request/"REQUEST_PRIVATE.json",chain["request_sha256"]))
    require(request["input_lineage"] == lin and request["source_replay"] == req["source_replay"] == old["source_replay"]
            and request["status"] == api.REQUEST and request["protocol_sha256"] == api.LOCK,
            "Original label-free request signature differs")
    endpoints = api.pin(a.endpoints,chain["endpoint_sha256"])
    writer_stream = api.pin(a.writer/"EDITS_PRIVATE.jsonl",writer["edits_sha256"])
    archived = api.pin(a.phx_edits/"EDITS_PRIVATE.jsonl",old["edits_sha256"])
    records = [json.loads(line) for line in archived.splitlines()]
    cfa.EDITED = (*cfa.EDITED,"phoenix")
    blocks,budget,_ = cfa.checked_blocks(rows,records,adapter,core)
    require(budget == dict(mapped_candidates=7346,eligible_candidates=4137,positive_budget_candidates=4107,
                           swaps_per_arm=29900),"Frozen original candidate census differs")
    counts = {arm:Counter(adds=0,drops=0,flips=0,expanded=0,contracted=0,area_preserved=0) for arm in POLICIES}
    disagreements = Counter({f"{a}_minus_{b}":0 for a,b in PAIRS})
    images = changed = pairs = 0
    for row,img,scores,original,new in zip(rows,request["images"],map(json.loads,endpoints.splitlines()),records,
                                          map(json.loads,writer_stream.splitlines()),strict=True):
        item,audit,pred = row[:3]
        image = item["image_id"]
        require(img["image_id"] == scores["image_id"] == original["image_id"] == new["image_id"] == image
                and img["decoded_rgb_sha256"] == item["rgb_sha256"] and set(new) == {"image_id","changes"}
                and set(scores) == {"image_id","candidates"} and set(new["changes"]) == set(original["changes"]),
                "Writer, archived scores or native image order differs")
        positions = sorted(map(int,original["changes"]))
        for pos,candidate,scored in zip(positions,img["candidates"],scores["candidates"],strict=True):
            entries = new["changes"][str(pos)]
            info,det = audit["candidates"][pos-1],pred["coco"][pos-1]
            require(set(scored) == {"position","probabilities"} and scored["position"] == candidate["position"] == pos
                    and candidate["segmentation"] == det["segmentation"] and candidate["native_rank"] == info["native_rank"]
                    and candidate["query_id"] == info["query_id"] and candidate["score"] == info["score"] == det["score"],
                    "Original ordered endpoint/candidate identity differs")
            splits = verify_candidate(candidate,scored["probabilities"],entries,original["changes"][str(pos)]["phoenix"])
            for arm,(add,drop) in splits.items():
                counts[arm].update(adds=add,drops=drop,flips=add+drop,expanded=int(add>drop),
                                   contracted=int(add<drop),area_preserved=int(add==drop))
            for left,right in PAIRS:
                disagreements[left+"_minus_"+right] += int(entries[left]["rle"] != entries[right]["rle"])
            blocks[image]["changes"][pos].update(entries)
            changed += 1
            pairs += len(candidate["coords"])
        images += 1
    require((images,changed,pairs) == (74,4107,129621)
            and all(c["flips"] == 59800 and c["expanded"]+c["contracted"]+c["area_preserved"] == 4107
                    for c in counts.values()) and counts["balance"]["adds"] == counts["balance"]["drops"] == 29900,
            "Incomplete arm, writable support or exact signed-budget census")
    require(writer["counts"] == dict(images=74,positive_budget_candidates=4107,endpoint_pairs=129621,
                swaps_per_balanced_arm=29900,mapped_candidates=7346,untouched_candidates=3239,total_flips_per_arm=59800),
            "Writer's original candidate exposure disagrees with independent verification")
    for arm,c in counts.items():
        expected = dict(adds=c["adds"],drops=c["drops"],flips=c["flips"],changed_candidates=changed,
                        more_add_candidates=c["expanded"],more_drop_candidates=c["contracted"],
                        unchanged_quota=c["area_preserved"],signed_net_area=c["adds"]-c["drops"])
        require(all(writer["counts_by_arm"][arm][k] == v for k,v in expected.items()), "Actual signed policy totals differ")
    chain.update(input_lineage=lin,historical_score_sha256=ref_sha,historical_score_terminal_sha256=rt,
                 writer_result_sha256=wsha,writer_terminal_sha256=wterm,writer_worker_sha256=WRITER_WORKER,
                 edits_sha256=writer["edits_sha256"])
    totals = dict(images=images,native_candidates=budget["mapped_candidates"],positive_budget_candidates=changed,
                  zero_budget_candidates=budget["mapped_candidates"]-changed,endpoint_pairs=pairs,per_arm=counts,
                  differing_native_candidates=disagreements)
    return blocks,prior,chain,totals,writer


def exposures(anchors, blocks, polygons, dates):
    counts = {pair:Counter() for pair in PAIRS}
    for gt_id,(image,position,det) in anchors.items():
        poly = polygons[gt_id]
        native = det["segmentation"]
        if not poly.is_valid or poly.area <= 0 or not 32 <= int(codec.area(native)) < 1024:
            continue
        block = blocks[image]
        edits = block["changes"].get(position,{})
        for pair in PAIRS:
            left,right = (edits.get(arm,{}).get("rle",native) for arm in pair)
            # Independently reconstructed canonical RLEs give unique representations of these masks.
            if left["counts"] != right["counts"]:
                counts[pair][block["group"]] += 1
    return {a+"_minus_"+b:dict(matches=sum(c.values()),parents=len(c),sensor_date_blocks=len({dates[p] for p in c}))
            for (a,b),c in counts.items()}


def historic_alias(reference):
    """The byte-identical balanced writer is called phoenix in the archived score."""
    geom = reference["geometry"]
    return dict(reference,
                metrics_fraction=dict(reference["metrics_fraction"],balance=reference["metrics_fraction"]["phoenix"]),
                geometry=dict(geom,group_equal_arms=dict(geom["group_equal_arms"],balance=geom["group_equal_arms"]["phoenix"])))


def score_view(ids, gt, blocks, polygons, dates, prior, whole, deps):
    reference = historic_alias(prior)
    out = whole.score_view(ids,gt,blocks,polygons,dates,reference,deps)
    selected = {key:value for key,value in blocks.items() if key in ids}
    part = dict(gt,images=[im for im in gt["images"] if im["id"] in ids],
                annotations=[a for a in gt["annotations"] if a["image_id"] in ids])
    evaluator = deps[6]
    _, anchors = evaluator.score_one(part,selected,"raw",anchors=True)
    require(len(anchors) == out["geometry"]["counts"]["native_matches"], "Fixed native anchors differ")
    observed = exposures(anchors,selected,polygons,dates)
    for name,e in observed.items():
        contrast = out["contrasts"][name]
        paired = out["geometry"]["paired"].get(name)
        bounds = paired["continuous_iou_points"]["interval_95"] if paired else None
        ap = contrast["COCO_points"]["AP"]
        supported = e["matches"] >= 20 and e["sensor_date_blocks"] >= 8
        contrast["criteria"] = dict(differing_fixed_hulls=e["matches"],differing_parents=e["parents"],
            differing_sensor_date_blocks=e["sensor_date_blocks"],exposure_supported=supported,
            supported_advantage=bool(supported and ap is not None and ap>=.30 and bounds and bounds[0]>0),
            supported_loss=bool(supported and ap is not None and ap<=-.30 and bounds and bounds[1]<0))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ("code","stock-root","point-root","phx-edits","phx-edits-terminal","phx-score","phx-score-terminal",
                 "request","request-terminal","inference","inference-terminal","endpoints","writer","writer-terminal",
                 "author-labels","protocol","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--protocol-sha256",required=True)
    a = parser.parse_args()
    for key,v in vars(a).items():
        if isinstance(v,Path):
            setattr(a,key,v.resolve(strict=key!="output"))
    require(a.protocol.name == LOCK_NAME and a.protocol_sha256 == digest(a.protocol.read_bytes()) == LOCK and not a.output.exists(),
            "Published area-ablation protocol and fresh output required")
    require(shapely.__version__ == "2.0.2" and shapely.geos_version_string == "3.11.2","Historical geometry runtime")
    a.cohort,a.support_root = 74,None
    start = perf_counter()
    whole,sides,deps = sources(a)
    phx,api,_,h,reserve,old,_,_,_ = deps
    blocks,prior,chain,counts,writer = observed_inputs(a,sides,deps)
    print(json.dumps(dict(phase="all_signed_masks_independently_verified",images=len(blocks))),flush=True)
    # No target polygons are read until every observed worker, original endpoint and written mask has passed.
    gt,rekeyed,polygons,sets,dates,label_sha = h.views_and_labels(a,old,reserve,blocks)
    views,cache = {},{}
    for name,ids in sets.items():
        key = frozenset(ids)
        if key not in cache:
            cache[key] = score_view(ids,gt,blocks,polygons,dates,prior["views"][name],whole,deps)
        views[name] = cache[key]
        phx.parity(views[name],historic_alias(prior["views"][name]),("raw","balance"))
        print(json.dumps(dict(phase="scored_view",view=name,images=len(ids))),flush=True)
    require(len(views)==5 and views["full"]["geometry"]["counts"]["eligible_matches"]==47
            and views["date_separated"]["geometry"]["counts"]["eligible_matches"]==32,"Frozen view denominators differ")
    def summary(key):
        v=views["full"]
        return dict(**v["contrasts"][key],paired_continuous_iou_points=v["geometry"]["paired"][key]["continuous_iou_points"])
    result = dict(status=STATUS,phase="score",protocol_sha256=LOCK,worker_sha256=digest(Path(__file__).read_bytes()),
                  **chain,counts=counts,arms=list(whole.ARMS),independent_endpoint_splits_exact=True,
                  historical_balanced_masks_and_scores_exact=True,target_labels_opened=True,author_labels_sha256=label_sha,
                  author_ids_rekeyed=rekeyed,detector_forwards=0,phoenix_forwards=0,optimizer_updates=0,
                  bootstrap_draws=4096,bootstrap_seed=5217,cluster_unit="sensor/date; equal parent weight",
                  primary_view="full",primary_contrast="balance_minus_free",secondary_contrast="relaxed_minus_balance",
                  primary=summary("balance_minus_free"),secondary=summary("relaxed_minus_balance"),
                  views=views,score_seconds=perf_counter()-start,writer_seconds=writer["seconds"])
    a.output.mkdir()
    stamp = api.write(a.output/"RESULT_SAFE.json",api.encode(result))
    print(json.dumps(dict(status=STATUS,result_sha256=stamp)),flush=True)


if __name__=="__main__":
    main()
