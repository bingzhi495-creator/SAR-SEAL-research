"""Freeze image-only FIT109 boundary ranks, then evaluate their fixed-count effects."""
import argparse
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from hashlib import sha256
from io import StringIO
import importlib.metadata
import json
import os
from pathlib import Path
from time import perf_counter
from types import ModuleType
from zipfile import ZipFile

import numpy as np
from pycocotools import mask as codec


LOCK = "c8395c6ce19f46003535a01d7c4f3b10084b9ec80085c8c3b87cd9b4b05c9a37"
BUNDLE = "af12a4e86ea36e9d7fe6ac47b17c684a4812baba6aa9be019f889f833e0f577c"
AFFINITY = "1757685e62a834787f3e5b9eda1ea7d3dde9606895b22b32bddb3f1ec0bc4f2b"
CLASSICAL = "acb7e00e7dfd2c60e5afc4397e073a6fabcb6a4ed977f1c8eaf89115724b52ed"
TOPOLOGY = "8eb162a3bd544929843d7f2aeef174f1f1ba03a2c593b6bcda49289cd94ad3c6"
REFERENCE = "f78d431c749c4a58c3a14dd3ad2e248c5580093a6cbe59e47c4e2c35d6f7e7e6"
READBACK = "2c3c48fca43236558774762dc7475412e634897a96f3921fbbb14487d5585bf4"
EDIT_STATUS = "FIT109_IMAGE_SHELL_EDITS_COMPLETE"
SCORE_STATUS = "FIT109_IMAGE_SHELL_SCORE_COMPLETE"
FULL, RADIAL, GRAY = "aligned", "radial", "appearance_only"
IMAGE_ARMS = (FULL, RADIAL, GRAY)
LEDGER = "ACTIONS_PRIVATE.jsonl"


def need(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return sha256(raw).hexdigest()


def load(path, expected):
    raw = path.read_bytes()
    need(digest(raw) == expected, "Frozen module changed: " + path.name)
    mod = ModuleType(path.stem)
    mod.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), mod.__dict__)
    return mod


def sources(a):
    cdir = a.code / "FIT109_CLASSICAL_ATTRACTION_20261121"
    compare = load(cdir / "compare.py", CLASSICAL)
    pilot = load(a.code / "FIT109_BOUNDARY_SIMPLE_POINT_PILOT_20261121/pilot.py", compare.PILOT_SHA)
    affinity = load(Path(__file__).with_name("affinity.py"), AFFINITY)
    for name, version in pilot.VERSIONS.items():
        need(importlib.metadata.version(name) == version, "Original runtime required: "+name)
    a.helper, a.helper_sha256 = cdir / "spatial_attraction.py", TOPOLOGY
    a.private_root = pilot.PRIVATE
    modules = pilot.sources(a)
    return compare, pilot, affinity, modules


def checked_actions(coords, selected, k):
    values = np.asarray(selected)
    need(values.shape == (k,2) and np.issubdtype(values.dtype,np.integer), "Expected pairs of integer endpoint ordinals")
    arr = values.astype(np.intp,copy=False)
    need(arr.shape == (k,2) and np.all((arr >= 0) & (arr < len(coords)))
         and len(np.unique(arr[:,0])) == len(np.unique(arr[:,1])) == k, "Invalid signed endpoint ordinal or count")
    return arr.tolist()


def make_ledger(a, compare, affinity, m, blocks, planned):
    records, costs, counts = [], Counter(), Counter()
    ordered = (compare.DYNAMIC,compare.STATIC,*IMAGE_ARMS)
    with ZipFile(a.bundle) as archive:
        roster = m.pair.image_roster(archive,blocks,m.base)
        for block, plan in zip(blocks,planned,strict=True):
            start = perf_counter()
            rgb = m.pair.read_png(archive,roster[block["image_id"]],m.base)
            costs["verified_image_read"] += perf_counter()-start
            start = perf_counter()
            ranks = affinity.rank_image(rgb)
            costs["image_rank_shared"] += perf_counter()-start
            selections = {}
            for rank,(candidate,row) in enumerate(zip(block["candidates"][:100],block["coco"][:100],strict=True),1):
                area = candidate["mask_area"]
                need(candidate["native_rank"]==rank and row["score"]==candidate["score"] and area==int(codec.area(row["segmentation"])),
                     "Frozen candidate, score or area differs")
                counts["slots"] += 1
                data = plan["data"].get(rank)
                if data is None:
                    continue
                coords = data["coordinates"]
                need(32 <= area < 1024, "Endpoint support outside frozen gate")
                counts["supported"] += 1
                stock = codec.decode(row["segmentation"]).astype(bool,copy=False)
                t = perf_counter()
                dynamic, static = m.topology.select(stock,coords,min(8,area//10,len(coords)))
                costs["legacy_dynamic_and_static"] += perf_counter()-t
                k = len(dynamic)
                if not k:
                    continue
                t = perf_counter()
                scores = affinity.endpoint_scores(stock,coords,ranks,m.topology)
                costs["shared_aligned_radial_static_scores"] += perf_counter()-t
                t = perf_counter()
                gray = ranks.ravel()[coords]
                costs["appearance_endpoint_gather"] += perf_counter()-t
                picks = {compare.DYNAMIC:dynamic, compare.STATIC:static}
                for arm, values in ((FULL,scores["full"]),(RADIAL,scores["shell"]),(GRAY,gray),(compare.STATIC,scores["static"])):
                    t = perf_counter()
                    action = affinity.select_scores(values,k,m.topology)
                    costs["select_"+arm] += perf_counter()-t
                    if arm == compare.STATIC:
                        need(action == static, "Normalized static ranks differ from the historical native order")
                    picks[arm] = action
                selections[str(rank)] = {arm:checked_actions(coords,picks[arm],k) for arm in ordered}
                counts.update(positive_budget_candidates=1,swaps_per_arm=k)
            records.append(dict(image_id=block["image_id"],ordinal=block["ordinal"],changes=selections))
            if len(records) in (1,25,50,75,100,109):
                print(json.dumps(dict(phase="edits",images=len(records))),flush=True)
    need(len(records)==109 and counts["slots"]==10900 and counts["positive_budget_candidates"]==6159
         and counts["swaps_per_arm"]==43624, "Complete source candidate or realized count differs")
    counts["no_op_candidates"] = counts["slots"]-counts["positive_budget_candidates"]
    return records, dict(counts), dict(costs)


def closed_edits(a, m, worker):
    raw = (a.edits / "RESULT_SAFE.json").read_bytes()
    result, term_raw = json.loads(raw), a.edits_terminal.read_bytes()
    term = json.loads(term_raw)
    need(type(term["actual_exit"]) is int and term["actual_exit"]==0 and term["output_footer_verified"] is True
         and term["phase"]==result["phase"]=="edits" and term["worker_sha256"]==result["worker_sha256"]==worker
         and term["result_sha256"]==digest(raw) and result["status"]==EDIT_STATUS and result["protocol_sha256"]==LOCK
         and result["target_labels_opened"] is False and result["point_optimizer_updates"]==0
         and result["affinity_sha256"]==AFFINITY, "Terminal of the frozen image-only decisions required")
    rows = [json.loads(line) for line in m.base.exact(a.edits/LEDGER,result["actions_sha256"],64*1024**2).splitlines()]
    return rows,result,digest(raw),digest(term_raw)


def indexed_actions(records, blocks, planned, compare):
    arms = (compare.DYNAMIC,compare.STATIC,*IMAGE_ARMS)
    actions = {arm:[] for arm in arms}
    positives, swaps = 0, 0
    for rec, b, p in zip(records,blocks,planned,strict=True):
        need((rec["image_id"],rec["ordinal"])==(b["image_id"],b["ordinal"]), "Historical order differs")
        changes = rec["changes"]
        need(set(changes) <= {str(r) for r in p["data"]}, "Edits outside original point support")
        for rank in range(1,101):
            item = changes.get(str(rank))
            if item is None:
                for arm in arms:
                    actions[arm].append([])
                continue
            need(set(item)==set(arms), "Missing edit policy")
            coords = p["data"][rank]["coordinates"]
            k = len(item[compare.DYNAMIC])
            need(0<k<=min(8,b["candidates"][rank-1]["mask_area"]//10,len(coords)), "Historical budget violation")
            positives += 1
            swaps += k
            for arm in arms:
                ids = checked_actions(coords,item[arm],k)
                actions[arm].append([tuple(pair) for pair in ids])
    need((positives,swaps)==(6159,43624) and all(len(v)==10900 for v in actions.values()), "Frozen denominator differs")
    return actions


def restore_comparator(slots, m, compare, preset, historical, previous):
    prior, hook, seen, expected, extra, histogram = compare.observe(slots,m)
    m.learner.probabilities = hook
    t = perf_counter()
    try:
        original, training, usage = m.free.fit(slots,m.learner,m.math,previous)
    finally:
        m.learner.probabilities = prior
    elapsed = perf_counter()-t
    need(len(seen)==len(expected) and training==historical["training"] and usage==historical["usage"],
         "Historical held-out occupancy fit or candidate scores differ")
    need(extra[compare.DYNAMIC]==preset[compare.DYNAMIC] and extra[compare.STATIC]==preset[compare.STATIC],
         "Prelabel legacy choices disagree with the original independent replay")
    original.update(extra)
    original.update(preset)
    return original,training,elapsed


def raster_targets(blocks, gt, annotations, slots, a, m):
    matcher = m.base.load(a.code/"HRSID_FIT_HULL_CDF_20261112/stock_coco_match.py",m.base.MATCH,"_shell_raw_match")
    targets = {}
    for img_i,(b,im) in enumerate(zip(blocks,gt["images"],strict=True)):
        first = img_i*100
        pool=[dict(rank=i+1,score=s["row"]["score"],area=s["area"],rle=s["row"]["segmentation"])
              for i,s in enumerate(slots[first:first+100])]
        with redirect_stdout(StringIO()):
            matches,rles,_=matcher.freeze_coco(im,annotations[im["id"]],pool)
        for ann,rank in matches.items():
            i=first+rank-1
            if slots[i]["raw"] is None:
                continue
            truth = rles[ann]
            mask = slots[i]["row"]["segmentation"]
            inter = int(codec.area(codec.merge([truth,mask],intersect=True)))
            decoded = codec.decode(truth).astype(np.uint8,copy=False)
            data = slots[i]["data"]
            targets[i] = (int(codec.area(truth)),inter,decoded.ravel()[data["coordinates"]] if data is not None else None)
    need(len(targets)==412, "Raster and continuous geometry must share all 412 original anchors")
    return targets


def signed_delta(q, pairs):
    if not pairs:
        return 0.
    idx = np.asarray(pairs,dtype=np.intp)
    return float(np.sum(q[idx[:,0],0].astype(np.float64)-q[idx[:,1],1]))


def selected_set(action):
    return tuple(frozenset(x[j] for x in action) for j in (0,1))


FIELDS = ("intersection_per_polygon_area_points","continuous_iou_points","fp_per_polygon_area_points",
          "fn_per_polygon_area_points","raster_intersection_px","raster_intersection_per_truth_points",
          "raster_iou_points","continuous_intersection_px2")


def geometry(slots, actions, arms, raster):
    groups,exposed=defaultdict(list), {v:Counter() for v in arms if v!=FULL}
    for i,(rt,ri,rq) in raster.items():
        s=slots[i]
        gt_area,baseline=s["raw"]
        values=[]
        chosen=selected_set(actions[FULL][i])
        for arm in arms:
            act=actions[arm][i]
            inter=baseline+signed_delta(s["truth"],act)
            r_inter=ri+signed_delta(rq,act)
            need(-1e-8<=inter<=min(gt_area,s["area"])+1e-8 and 0<=r_inter<=min(s["area"],rt),"Balanced geometry invalid")
            inter=float(np.clip(inter,0,min(gt_area,s["area"])))
            values.append((100*inter/gt_area,100*inter/(gt_area+s["area"]-inter),100*(s["area"]-inter)/gt_area,
                           100*(gt_area-inter)/gt_area,r_inter,100*r_inter/rt,100*r_inter/(rt+s["area"]-r_inter),inter))
            if arm!=FULL and chosen!=selected_set(act):
                exposed[arm][s["group"]]+=1
        groups[s["group"]].append(values)
    keys=sorted(groups)
    need(len(keys)==93, "Historical source-group coverage differs")
    g=np.stack([np.mean(groups[k],axis=0) for k in keys])
    draws=np.random.Generator(np.random.PCG64(5217)).integers(len(keys),size=(4096,len(keys)))
    means={arm:dict(zip(FIELDS,g[:,i].mean(axis=0).tolist(),strict=True)) for i,arm in enumerate(arms)}
    contrasts={}
    for b,e in exposed.items():
        delta=g[:,arms.index(FULL)]-g[:,arms.index(b)]
        ci=np.percentile(delta[draws].mean(axis=1),[2.5,97.5],axis=0).T
        contrasts[b]=dict(changed_matches=sum(e.values()),changed_groups=len(e),
                          fields={field:dict(delta=float(x),interval_95=interval.tolist())
                                  for field,x,interval in zip(FIELDS,delta.mean(axis=0),ci,strict=True)})
    minus_raw={}
    for i,arm in enumerate(arms):
        delta=g[:,i]-g[:,arms.index("raw")]
        ci=np.percentile(delta[draws].mean(axis=1),[2.5,97.5],axis=0).T
        minus_raw[arm]={field:dict(delta=float(x),interval_95=interval.tolist())
                        for field,x,interval in zip(FIELDS,delta.mean(axis=0),ci,strict=True)}
    return dict(pairs=len(raster),source_groups=len(keys),group_equal=means,minus_raw=minus_raw,
                aligned_minus=contrasts,bootstrap_draws=4096,bootstrap_seed=5217,pointwise_intervals=True)


def verify_archive(path):
    need(path.stat().st_size==151320868,"Expected original FIT114 ZIP byte count")
    h=sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(4*1024**2),b""):
            h.update(block)
    need(h.hexdigest()==BUNDLE,"Expected original FIT114 ZIP digest")


def historical_reference(a):
    directory=a.code.parent/"results"
    ref_path=a.closed_result or directory/"FIT109_CLASSICAL_ATTRACTION_ACTUAL_SAFE_20261121.json"
    readback=directory/"FIT109_CLASSICAL_ATTRACTION_READBACK_SAFE_20261121.json"
    raw,read=ref_path.read_bytes(),readback.read_bytes()
    reference,receipt=json.loads(raw),json.loads(read)
    need(digest(raw)==REFERENCE and digest(read)==READBACK and receipt["score_sha256"]==REFERENCE
         and receipt["status"]=="FIT109_ATTRACTION_CLOSED_SUMMARIES_INDEPENDENTLY_VERIFIED"
         and reference["status"]=="FIT109_ATTRACTION_MATCHED_COUNT_ACTUAL" and reference["historical_parity"],
         "Previously accepted independent source comparison required")
    terminal=None
    if a.closed_terminal:
        term_raw=a.closed_terminal.read_bytes()
        term=json.loads(term_raw)
        terminal=digest(term_raw)
        if terminal!=READBACK:
            need(type(term["actual_exit"]) is int and term["actual_exit"]==0 and term["output_footer_verified"] is True
                 and term["result_sha256"]==REFERENCE, "Original source comparison terminal required")
    return reference,terminal


def materialize(a,m,pilot,compare,slots,actions,gt,arms):
    # Expose standard small-object AP from the very same COCOeval accumulation.
    m.free.adapt_function(a.official,pilot.REFERENCES["official"][1],m.official,"official","geometry_for",
                          (("return dict(AP=", "return dict(APs=(float(ev.stats[3])*100 if ev.stats[3]>=0 else None),AP="),))
    scores,budgets,elapsed={},{},{}
    with compare.verify_materialization(m.base) as counters:
        for arm in arms:
            start=perf_counter()
            m.gray.MODES=(arm,)
            one,cost=m.gray.materialize(slots,actions,m.base,m.official,gt)
            scores.update(one)
            budgets.update(cost)
            elapsed[arm]=perf_counter()-start
            print(json.dumps(dict(phase="score",coco_completed=arm)),flush=True)
    return scores,budgets,elapsed,dict(counters)


def decision(coco,geo,compare):
    by_arm={}
    for arm,contrast in geo["aligned_minus"].items():
        ap=coco[FULL]["AP"]-coco[arm]["AP"]
        lower=contrast["fields"]["intersection_per_polygon_area_points"]["interval_95"][0]
        by_arm[arm]=dict(AP_difference_points=ap,intersection_lower95_points=lower,
                         changed_matches=contrast["changed_matches"],changed_groups=contrast["changed_groups"],
                         exposure=contrast["changed_matches"]>=20 and contrast["changed_groups"]>=8,
                         positive_lower95=lower>0,AP_retained=ap>=0)
    primary=all(all(by_arm[arm][k] for k in ("exposure","positive_lower95","AP_retained"))
                for arm in (compare.STATIC,RADIAL))
    best=max(by_arm,key=lambda arm:coco[arm]["AP"])
    ap_gain=by_arm[best]["AP_difference_points"]
    practical=primary and ap_gain>0 and by_arm[compare.LEARNED]["positive_lower95"]
    route="no_advancement"
    if practical:
        route="full_replication" if ap_gain>=1 else "one_fixed_recipe_confirmation"
    return dict(aligned_minus=by_arm,mechanism_joint_pass=primary,practical_pass=practical,
                strongest_baseline_by_AP=best,AP_gain_over_strongest_points=ap_gain,route=route)


def verify_parity(coco,budgets,geo,reference,compare):
    for arm in ("raw",compare.DYNAMIC,compare.STATIC,compare.LEARNED):
        old=reference["cohorts"][arm]
        need({k:coco[arm][k] for k in old}==old and budgets[arm]==reference["budgets"][arm],
             "Historical COCO or budget parity failed for "+arm)
        old_geometry=reference["geometry"]["metrics"][arm]
        for name,prior in (("continuous_iou_points","group_equal_iou_points"),
                           ("fp_per_polygon_area_points","fp_per_truth_points"),
                           ("fn_per_polygon_area_points","fn_per_truth_points")):
            need(abs(geo["group_equal"][arm][name]-old_geometry[prior])<1e-8,
                 "Historical continuous geometry parity failed for "+arm)
    need(all(b==budgets[compare.STATIC] for arm,b in budgets.items() if arm!="raw"),
         "Realized edit counts, roles or fallbacks differ")


def run(a,compare,pilot,affinity,m):
    worker=digest(Path(__file__).read_bytes())
    start=perf_counter()
    staged=None
    if a.mode=="score":
        need(a.edits and a.edits_terminal,"Scoring requires a closed image-only phase")
        staged=closed_edits(a,m,worker)
    _,blocks=m.base.read_core(a)
    receipt,planned=m.pair.sealed_points(a,m.base,m.math,blocks)
    need(receipt["points_manifest_sha256"]==pilot.INPUTS["points_manifest"][1],"Exact historical frozen endpoints")
    a.output.mkdir()
    result=dict(phase=a.mode,protocol_sha256=LOCK,worker_sha256=worker,affinity_sha256=AFFINITY,
                legacy_worker_sha256=CLASSICAL,pilot_sha256=compare.PILOT_SHA,legacy_topology_sha256=TOPOLOGY,
                image_bundle_sha256=BUNDLE,point_manifest_sha256=receipt["points_manifest_sha256"],
                FIT=True,DEV=False,TEST=False,detector_forwards=0,detector_optimizer_updates=0,
                new_policy_optimizer_updates=0,images=109,versions=pilot.VERSIONS,
                original_source_sha256={k:v[1] for k,v in pilot.REFERENCES.items()},
                original_input_sha256={k:v[1] for k,v in pilot.INPUTS.items()})
    if a.mode=="edits":
        verify_archive(a.bundle)
        rows,counts,costs=make_ledger(a,compare,affinity,m,blocks,planned)
        indexed_actions(rows,blocks,planned,compare)
        actions_hash=m.base.seal(a.output/LEDGER,b"".join(m.base.encoded(row) for row in rows))
        result.update(status=EDIT_STATUS,GT_opened=False,target_labels_opened=False,point_optimizer_updates=0,
                      actions_sha256=actions_hash,counts=counts,costs_seconds=costs)
    else:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        rows,edits,edits_sha,terminal_sha=staged
        preset=indexed_actions(rows,blocks,planned,compare)
        reference,original_terminal=historical_reference(a)
        historical=json.loads(pilot.pinned(a.historical,pilot.INPUTS["historical"][1]))
        previous=json.loads(pilot.pinned(a.previous,pilot.INPUTS["previous"][1]))
        gt,annotations=m.pair.read_labels(a,m.base,blocks)
        slots,scope=m.pair.fixed_slots(a,m.base,blocks,planned,gt,annotations)
        need(scope==historical["match_scope"]==reference["fixed_original_matches"],"Original source matching differs")
        actions,training,fit_seconds=restore_comparator(slots,m,compare,preset,historical,previous)
        arms=("raw",compare.DYNAMIC,compare.STATIC,compare.LEARNED,*IMAGE_ARMS)
        coco,budgets,per_arm,xor=materialize(a,m,pilot,compare,slots,actions,gt,arms)
        t=perf_counter()
        raster=raster_targets(blocks,gt,annotations,slots,a,m)
        geo=geometry(slots,actions,arms,raster)
        geometry_seconds=perf_counter()-t
        verify_parity(coco,budgets,geo,reference,compare)
        need(xor==dict(verified_masks=6159*(len(arms)-1),verified_changed_pixels=2*43624*(len(arms)-1)),
             "Signed mask XOR audit did not cover every materialized mask")
        result.update(status=SCORE_STATUS,GT_opened=True,target_labels_opened=False,source_FIT_labels_opened=True,
                      edits_result_sha256=edits_sha,edits_terminal_sha256=terminal_sha,actions_sha256=edits["actions_sha256"],
                      historical_comparison_sha256=REFERENCE,historical_readback_sha256=READBACK,
                      historical_terminal_sha256=original_terminal,historical_parity=True,original_match_scope=scope,
                      point_optimizer_updates=1000,original_occupancy_replay_updates=1000,original_occupancy_training=training,
                      candidate_counts=edits["counts"],verified_xor=xor,cohorts=coco,budgets=budgets,geometry=geo,
                      gates=decision(coco,geo,compare),costs_seconds=dict(image_only=edits["costs_seconds"],
                      image_only_phase=edits["elapsed_seconds"],historical_fit_and_comparator_replay=fit_seconds,
                      per_arm_mask_materialization_and_COCO=per_arm,fixed_geometry_all_arms=geometry_seconds))
    result["elapsed_seconds"]=perf_counter()-start
    value=m.base.seal(a.output/"RESULT_SAFE.json",m.base.encoded(result))
    print(json.dumps(dict(status=result["status"],result_sha256=value)),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("check","edits","score"))
    for key in ("code","protocol","bundle","output","edits","edits-terminal","closed-result","closed-terminal"):
        parser.add_argument("--"+key,type=Path,required=key in ("code","protocol"))
    parser.add_argument("--protocol-sha256",required=True)
    a=parser.parse_args()
    for key,value in vars(a).copy().items():
        if isinstance(value,Path):
            setattr(a,key,value.resolve())
    need(a.protocol_sha256==LOCK and digest(a.protocol.read_bytes())==LOCK,"Current signed prelock changed")
    need(os.environ.get("CUDA_VISIBLE_DEVICES")=="-1","Original CPU-only scope")
    compare,pilot,affinity,m=sources(a)
    if a.mode=="check":
        print(json.dumps(dict(status="FIT109_IMAGE_SHELL_IMPORTS_VERIFIED",private_payloads_opened=0)),flush=True)
        return
    need(a.output and not a.output.exists() and (a.mode=="score" or a.bundle),"Fresh output and original inputs required")
    run(a,compare,pilot,affinity,m)


if __name__=="__main__":
    main()
