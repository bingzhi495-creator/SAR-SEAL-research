"""Readback and equal-signed-budget D-FINE re-pairing on the fixed RSDD reserve."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace


REPLAY = "f1b6e83e98851bf867bd72f3ecb082ca2997632bdd1aedaaf64110cf06b04fd5"
HISTORICAL_SCORE = "bb751d3d42875b4cc9e980534697430283428b9ecfbb1c3c9bc09d26b63028e5"
BOUNDARY = "b477c918a1168eb6b734313b18c4df8e338a591305c164a444eb61a560e873c5"
SPARSE = "cb9b8e839de85588fdf116ce5554ed6cadbc20bdbe76e38180561528bfd870d5"
MATCH = "99817f192798f52f03efe5e3ac422a453019bf458d33096a03844cb60360436c"
INTAKE = "da9d92ea16e74f988d46d9855d26b530111b4c5e650fef513895c5d2e4441065"
INTAKE_TERMINAL = "b97ba047bf9422346f210b2a513be606cdc648cccdc0e8545affdd510f9ffaeb"
MANIFEST = "154d0d748b96df0d56ac249adcb8b6ae63275a09dfd7a0d7480839005e044338"
HEAD = "eca85da01c53545485928ae3c502844839083d4ebff896e3df1f91a072d9f725"
REFIT_ACCEPT = "dca00310ba7398790a874e2f32c7fa26324fed1b8e071372831a85e7a83ecd3a"
CAPTURE_STATUS = "DFINE_E24_RSDD74_NATIVE_CAPTURE_COMPLETE"
ACCEPTED_STATUS = "DFINE_E24_RSDD74_NATIVE_CAPTURE_ACCEPTED"
RENDER_STATUS = "DFINE_RSDD74_COMMON_CAP_RENDER_COMPLETE"
VERIFY_STATUS = "DFINE_RSDD74_COMMON_CAP_VERIFY_COMPLETE"
SCORE_STATUS = "DFINE_RSDD74_COMMON_CAP_SCORE_COMPLETE"
ARMS = ("raw", "fixed", "bounded", "free_matched_count")
CUTS = (("I_minus_L", 3, 2), ("L_minus_F", 2, 1), ("I_minus_F", 3, 1),
        ("F_minus_N",1,0), ("L_minus_N",2,0), ("I_minus_N",3,0))
METRICS = ("continuous_iou_points", "fp_per_truth_area_points", "fn_per_truth_area_points",
           "boundary_iou1_points", "boundary_iou2_points")


def need(ok, detail):
    if not ok:
        raise ValueError(detail)


def digest(raw):
    return sha256(raw).hexdigest()


def pin(path, expected):
    raw = Path(path).read_bytes()
    need(digest(raw) == expected, "Exact immutable asset: " + Path(path).name)
    return raw


def module(path, expected, name):
    obj = ModuleType(name)
    obj.__file__ = str(path)
    sys.modules[name] = obj
    exec(compile(pin(path, expected), str(path), "exec"), obj.__dict__)
    return obj


def closed(result_path, terminal_path, result_sha, terminal_sha, phase, status):
    record = json.loads(pin(result_path, result_sha))
    terminal = json.loads(pin(terminal_path, terminal_sha))
    need(terminal["phase"] == phase and terminal["actual_exit"] == 0
         and terminal["output_footer_verified"] is True and terminal["result_sha256"] == result_sha
         and terminal["worker_sha256"] == record.get("worker_sha256", record.get("export_source_sha256"))
         and record["status"] == status,
         "Exact observed child closure and bound result for " + phase)
    return record


def encoded(row):
    return (json.dumps(row, sort_keys=True, allow_nan=False, separators=(",", ":"))+"\n").encode("ascii")


def load_capture(a):
    cap = closed(a.capture_root/"capture/RESULT_SAFE.json", a.capture_root/"observed_capture/TERMINAL.json",
                 a.capture_sha, a.capture_terminal_sha, "capture", CAPTURE_STATUS)
    accepted = closed(a.capture_root/"verify-capture/RESULT_SAFE.json",
                      a.capture_root/"observed_verify-capture/TERMINAL.json",
                      a.capture_readback_sha, a.capture_readback_terminal_sha, "verify-capture",
                      ACCEPTED_STATUS)
    need(cap["protocol_sha256"] == accepted["protocol_sha256"] == a.capture_protocol_sha
         and cap["prelock_commit"] == accepted["prelock_commit"] == a.capture_commit
         and cap["manifest_sha256"] == accepted["manifest_sha256"] == MANIFEST
         and cap["intake_result_sha256"] == accepted["intake_result_sha256"] == INTAKE
         and cap["intake_terminal_sha256"] == accepted["intake_terminal_sha256"] == INTAKE_TERMINAL
         and cap["ema_sha256"] == accepted["ema_sha256"]
         == "82ca80e2cb9a0b9673a136fad3d7de3097b7327f4f76e3a2d3e7184ab07e7fd8"
         and cap["e24_acceptance_sha256"] == "7401f031c996298caff1761c6ef691fae0c301e65c87c15f51d186dcf81f25d8"
         and cap["original_author_postprocess_parity"] is accepted["original_author_postprocess_parity"] is True
         and accepted["capture_result_sha256"] == a.capture_sha
         and accepted["capture_terminal_sha256"] == a.capture_terminal_sha
         and accepted["independent_probability_and_candidate_census"] is True
         and cap["images"] == cap["probability_files"] == accepted["images"] == 74
         and all(cap[k] == accepted[k] for k in
                 ("e24_acceptance_sha256", "predictions_sha256", "candidate_audit_sha256",
                  "probability_index_sha256", "probability_rows", "rows_at_100"))
         and all(row[key] == 0 for row in (cap, accepted)
                 for key in ("label_files_opened", "optimizer_updates")), "Qualified native reserve capture")
    path = a.intake_root/"intake"
    intake = json.loads(pin(path/"RESULT_SAFE.json", INTAKE))
    event = json.loads(pin(a.intake_root/"observed_intake/TERMINAL.json", INTAKE_TERMINAL))
    doc = json.loads(pin(path/"IMAGE_MANIFEST_PRIVATE.json", MANIFEST))
    rows = doc["items"]
    need(intake["status"] == "INSTANCE_RSDD_RESERVE74_IMAGE_INTAKE_COMPLETE_V1"
         and intake["image_manifest_sha256"] == MANIFEST and intake["target_polygon_rows_opened"] == 0
         and event["actual_exit"] == 0 and event["output_footer_verified"] is True
         and event["result_sha256"] == INTAKE and len(rows) == len(set(doc["expected_ids"])) == 74
         and [r["image_id"] for r in rows] == doc["expected_ids"]
         and all(r["original_hw"] == [512,512] for r in rows), "Exact original reserve image manifest")
    a.capture = a.capture_root/"capture"
    a.images = path/"images"
    upstream = module(a.code/"PSEG_TEST232_DFINE_FROZEN_REPLAY_20261121/replay.py",
                      REPLAY, "_dfine_accepted_cross_carrier")
    m = upstream.modules(a.code,a)
    m.author.cpu_runtime()
    return m, rows, cap


def old_masks_and_points(m, adapted, probability, models, coco, audit, planes, image, counts):
    """Collect the original writer's actual posteriors without changing its code."""
    pair, _, free, learner, head, center, scale = models
    eligible = iter(j for j,row in enumerate(audit,1) if 32 <= row["mask_area"] < 1024)
    original_points = adapted.paired_points
    original_predict = learner.probabilities
    pending, trace = [None,None], {}

    def points(*args):
        ordinal=next(eligible)
        data=original_points(*args)
        pending[:] = ordinal, data[0] if data is not None else None
        return data

    def predict(*args):
        q=original_predict(*args)
        ordinal, coords=pending
        need(coords is not None and ordinal not in trace and q.dtype == np.float32
             and q.shape == coords.shape, "Original binary32 posterior for the exact support")
        trace[ordinal] = coords, q
        return q

    adapted.paired_points = points
    learner.probabilities = predict
    try:
        original = m.old.image_edits(coco,audit,planes,pair.image_features(image),
                                     (adapted,probability,free,learner,head,center,scale),counts)
    finally:
        adapted.paired_points = original_points
        learner.probabilities = original_predict
    need(next(eligible,None) is None, "Every original eligible candidate visited")
    return original, trace


def capacities(coords, width):
    ay,ax=np.divmod(coords[:,0].astype(np.int64),width)
    dy,dx=np.divmod(coords[:,1].astype(np.int64),width)
    return (ay[:,None]-dy[None,:])**2+(ax[:,None]-dx[None,:])**2


def signed_record(original, coordinates, q, area, stock, fixed, free, old, solver):
    quota = min(8,area//10,len(q))
    f = np.asarray(fixed.choose_pairs(q[:,0]-q[:,1],quota),dtype=np.int64)
    k = len(f)
    need(k == original["fixed"]["swaps"] == original["free_matched_count"]["swaps"] and k > 0,
         "Inherited positive F budget")
    i = np.asarray(free.choose_free(q,quota,k),dtype=np.int64)
    cap = int(capacities(coordinates,512)[f,f].max())
    ai,di,info = solver.bounded_endpoints(coordinates, q.astype(np.float64),k,512,f,f,i[:,0],i[:,1])
    l = np.column_stack((ai,di))
    f_pairs = np.column_stack((f,f))
    for mode, pairs in (("fixed",f_pairs),("free_matched_count",i)):
        need(old.edit(stock,coordinates,pairs.tolist()) == original[mode]["rle"],
             "Archived writer F and I native writes recovered exactly from original endpoint choices")
    bounded=old.edit(stock,coordinates,l.tolist())
    need(info["cap_squared_sum"] == cap and np.all(capacities(coordinates,512)[ai,di] <= cap),
         "Exactly inherited maximum displacement and exact solver edge feasibility")
    return dict(
        coordinates=coordinates.tolist(), fixed_pairs=f_pairs.tolist(), bounded_pairs=l.tolist(),
        independent_pairs=i.tolist(), cap_squared=cap,
        fixed=original["fixed"], free_matched_count=original["free_matched_count"],
        bounded=dict(swaps=k,rle=bounded)
    ),info


def geometry_ledger_check(sparse, stock, entry):
    need(set(entry)=={"coordinates","fixed_pairs","bounded_pairs","independent_pairs","cap_squared",*ARMS[1:]},
         "Signed action, support and frozen mask schema")
    coords=np.asarray(entry["coordinates"])
    k=entry["fixed"]["swaps"]
    need(type(k) is int and 0 < k <= 8 and coords.ndim==2 and coords.shape[1]==2
         and coords.dtype.kind in "iu" and k <= len(coords) <= 32 and len(np.unique(coords))==coords.size
         and np.all((0<=coords)&(coords<512*512)),"Unique signed native support coordinates")
    ends=np.cumsum(sparse.decode_runs(stock["counts"]),dtype=np.int64)
    membership=np.searchsorted(ends,(coords%512)*512+coords//512,side="right")%2
    need(not membership[:,0].any() and membership[:,1].all(),"Support signs against exact native mask")
    d=capacities(coords,512)
    for arm,key in zip(ARMS[1:],("fixed_pairs","bounded_pairs","independent_pairs"),strict=True):
        values=np.asarray(entry[key])
        need(values.dtype.kind in "iu" and values.shape==(k,2)
             and np.all((0<=values)&(values<len(coords)))
             and all(np.unique(values[:,s]).size==k for s in range(2)),
             "Exact disjoint endpoint counts and pairs")
        row=entry[arm]
        need(row["swaps"]==k and sparse.xor_rle(stock,coords[values,np.arange(2)].ravel())==row["rle"],
             "Whole native RLE equals independent reconstruction under each policy")
        lengths=d[values[:,0],values[:,1]]
        if arm=="fixed":
            need(np.array_equal(values[:,0],values[:,1]) and int(lengths.max())==entry["cap_squared"],
                 "Fixed ordered row pairs define common cap")
        if arm=="bounded":
            need(np.all(lengths<=entry["cap_squared"]),"Bounded realized pairing respects inherited cap")


def observe_replay(a, verify):
    started=perf_counter()
    m,roster,capture=load_capture(a)
    import torch
    from pycocotools import mask as codec
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    need(not torch.cuda.is_initialized(),"CPU replay never initializes CUDA")
    solve=module(a.code/"PHOENIX_BOUNDED_MATCH_20261123/matching.py",MATCH,"_bounded_dfine")
    sparse=module(a.code/"RFDETR_SPARSE_RLE_EQUIVALENCE_20261120/sparse_rle.py",SPARSE,"_signed_dfine")
    a.refit_result=a.refit_root/"ACTUAL/RESULT_SAFE.json"
    a.refit_acceptance=a.refit_root/"INDEPENDENT/RESULT_SAFE.json"
    a.weights=a.refit_root/"ACTUAL/POINT_HEAD_PRIVATE.npz"
    pin(a.refit_acceptance,REFIT_ACCEPT)
    pin(a.weights,HEAD)
    models=m.old.restore(a)
    models[0].raw=m.legacy.pin(a.pair_math,m.native.PAIR)
    adapted,probability=m.native.make_native(models[0],models[1],512,512)
    counters,solver_stats=Counter(),Counter()
    stream=sha256()
    if verify:
        earlier=closed(a.work_root/"render/RESULT_SAFE.json",a.work_root/"observed_render/TERMINAL.json",
                        a.render_sha,a.render_terminal_sha,"prepare-edits",RENDER_STATUS)
        qualify_phase(a,earlier)
        output=a.work_root/"render/EDITS_PRIVATE.jsonl"
    else:
        a.output.mkdir()
        output=a.output/"EDITS_PRIVATE.jsonl"
    with output.open("rb" if verify else "xb") as tape:
        for j,(item,coco,audit,planes) in enumerate(m.legacy.blocks(a,m.author,roster,capture,True),1):
            rgb=m.legacy.rgb(a,item)
            need(digest(rgb.tobytes(order="C"))==item["rgb_sha256"],"Exact original decoded reserve RGB")
            baseline,trace=old_masks_and_points(m,adapted,probability,models,coco,audit,planes,rgb,counters)
            changes={}
            for pos,arms in baseline.items():
                if "fixed" not in arms:
                    need("free_matched_count" not in arms,"Zero-budget matched I passes through")
                    continue
                rank=int(pos)
                stock=codec.decode(coco[rank-1]["segmentation"]).astype(bool,order="C")
                coords,q=trace[rank]
                rec,num=signed_record(arms,coords,q,audit[rank-1]["mask_area"],stock,adapted,models[2],m.old,solve)
                geometry_ledger_check(sparse,coco[rank-1]["segmentation"],rec)
                changes[pos]=rec
                solver_stats.update(num)
                counters.update(positive=1,swaps=rec["fixed"]["swaps"])
            counters.update(images=1,candidates=len(coco))
            raw=encoded(dict(image_id=item["image_id"],rgb_sha256=item["rgb_sha256"],changes=changes))
            stream.update(raw)
            if verify:
                need(tape.readline()==raw,"Exact repeat of all support, actions, caps and native mask bytes")
            else:
                need(tape.write(raw)==len(raw),"Complete signed output stream")
        if verify:
            need(not tape.read(1) and earlier["edits_sha256"]==stream.hexdigest()
                 and earlier["counts"]==dict(counters) and earlier["solver_totals"]==dict(solver_stats),
                 "Whole label-free stream and counts exact independently")
    need(counters["images"]==74 and counters["candidates"]==capture["rows_at_100"]
         and counters["positive"]==counters["fixed_edited"]==counters["free_matched_count_edited"] and
         counters["swaps"]==counters["fixed_swaps"]==counters["free_matched_count_swaps"],
         "Preserved original candidate pool, F/I and signed budgets")
    finish(a, VERIFY_STATUS if verify else RENDER_STATUS, started, counts=dict(counters),
           solver_totals=dict(solver_stats), edits_sha256=stream.hexdigest(), label_files_opened=0,
           original_f_i_reconstructed=True, exact_signed_cap_readback=True, independent_replay=verify,
           solver_sha256=MATCH, weight_sha256=HEAD,
           render_result_sha256=a.render_sha if verify else None,
           render_terminal_sha256=a.render_terminal_sha if verify else None)


def qualify_phase(a,result):
    need(result["worker_sha256"]==digest(Path(__file__).read_bytes()) and
         result["protocol_sha256"]==a.protocol_sha and result["prelock_commit"]==a.prelock_commit and
         result["capture_result_sha256"]==a.capture_sha and
         result["capture_readback_sha256"]==a.capture_readback_sha and
         result["label_files_opened"]==0 and result["exact_signed_cap_readback"] and
         result["weight_sha256"]==HEAD and result["solver_sha256"]==MATCH,
         "Same locked output and no labels before independent readback")


def finish(a,status,started,**details):
    data=dict(status=status,phase=a.phase,worker_sha256=digest(Path(__file__).read_bytes()),
              protocol_sha256=a.protocol_sha,prelock_commit=a.prelock_commit,
              capture_result_sha256=a.capture_sha,capture_readback_sha256=a.capture_readback_sha,
              capture_terminal_sha256=a.capture_terminal_sha,
              capture_readback_terminal_sha256=a.capture_readback_terminal_sha,
              elapsed_seconds=perf_counter()-started,detector_forwards=0,optimizer_updates=0,**details)
    raw=encoded(data)
    if a.phase!="render":
        a.output.mkdir()
    with (a.output/"RESULT_SAFE.json").open("xb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(dict(status=status,result_sha256=digest(raw))),flush=True)



def records_for_score(a):
    m,roster,cap=load_capture(a)
    render=closed(a.work_root/"render/RESULT_SAFE.json",a.work_root/"observed_render/TERMINAL.json",
                  a.render_sha,a.render_terminal_sha,"prepare-edits",RENDER_STATUS)
    readback=closed(a.work_root/"verify/RESULT_SAFE.json",a.work_root/"observed_verify/TERMINAL.json",
                    a.verify_sha,a.verify_terminal_sha,"verify-edits",VERIFY_STATUS)
    qualify_phase(a,render)
    qualify_phase(a,readback)
    need(readback["independent_replay"] is True and readback["render_result_sha256"]==a.render_sha
         and readback["render_terminal_sha256"]==a.render_terminal_sha
         and readback["edits_sha256"]==render["edits_sha256"] and readback["counts"]==render["counts"],
         "Exact independently replayed edits before label opening")
    sparse=module(a.code/"RFDETR_SPARSE_RLE_EQUIVALENCE_20261120/sparse_rle.py",SPARSE,"_signed_dfine_score")
    edits=pin(a.work_root/"render/EDITS_PRIVATE.jsonl",readback["edits_sha256"])
    rows=[json.loads(raw) for raw in edits.splitlines()]
    blocks={}
    tally=Counter()
    for (item,coco,_,_),row in zip(m.legacy.blocks(a,m.author,roster,cap,False), rows, strict=True):
        iid=item["image_id"]
        need(row["image_id"]==iid and item["rgb_sha256"]==row["rgb_sha256"] and iid not in blocks,
             "Exact edit and original admitted image order")
        changes={}
        for pos,entry in row["changes"].items():
            count=int(pos)
            need(pos==str(count) and 1<=count<=len(coco),"Preserved original native candidate position")
            geometry_ledger_check(sparse,coco[count-1]["segmentation"],entry)
            changes[count]={arm:entry[arm] for arm in ARMS[1:]}
            tally.update(positive=1,swaps=entry["fixed"]["swaps"])
        blocks[iid]=dict(stock=coco,ranks=list(range(1,len(coco)+1)),changes=changes,
                         rgb=item["rgb_sha256"], group=item["acquisition_group"],
                         sensor=item["sensor"],file=item["source_image_name"])
        tally.update(images=1,candidates=len(coco))
    need(all(readback["counts"][k]==v for k,v in tally.items()) and tally["images"]==74
         and tally["candidates"]==cap["rows_at_100"],"Exact all-slot prelabel independent readback")
    return blocks,dict(tally)


def geometry(subset, blocks, anchors, core, boundary, shapely):
    matched, full=defaultdict(list),defaultdict(list)
    changed={name:(0,set()) for name,_,_ in CUTS}
    coverage=Counter()
    for ann in subset["annotations"]:
        iid=ann["image_id"]
        group=blocks[iid]["group"]
        gid=ann["id"]
        record=anchors.get(gid)
        if record is None:
            coverage["native_unmatched"]+=1
            full[group].append(np.zeros((len(ARMS),len(METRICS))))
            continue
        registered,rank,native=record
        need(registered==iid,"Original matched shot and image identity")
        stock=native["segmentation"]
        area=int(core.codec.area(stock))
        polygon=shapely.Polygon(np.asarray(ann["segmentation"][0]).reshape(-1,2))
        if not polygon.is_valid or polygon.area<=0 or not 32<=area<1024:
            reason="polygon_ineligible" if not polygon.is_valid or polygon.area<=0 else "native_area_ineligible"
            coverage[reason]+=1
            full[group].append(np.zeros((len(ARMS),len(METRICS))))
            continue
        variants=blocks[iid]["changes"].get(rank,{})
        coverage["eligible_positive_budget" if variants else "eligible_no_edit"]+=1
        policies=[variants.get(arm,{}).get("rle",stock) for arm in ARMS]
        ref=core.codec.frPyObjects(ann["segmentation"],512,512)
        raster=core.codec.decode(core.codec.merge(ref))
        rings=[boundary.boundary_ring(raster,width) for width in (1,2)]
        accepts={}
        for rle in policies:
            key=rle["counts"]
            if key in accepts:
                continue
            io=core.geometric_iou(rle,polygon,shapely)
            hit=io*(area+polygon.area)/(1+io)
            decoded=core.codec.decode(rle)
            bio=[boundary.ring_iou(boundary.boundary_ring(decoded,width),ring)
                 for width,ring in enumerate(rings,1)]
            need(np.isfinite(bio).all(),"Native Boundary IoU on same original valid polygon matches")
            accepts[key]=[100*io,100*(area-hit)/polygon.area,100*(polygon.area-hit)/polygon.area,
                          *[100*b for b in bio]]
        raw=np.asarray([accepts[p["counts"]] for p in policies])
        matched[group].append(raw)
        full[group].append(raw)
        for name,high,low in CUTS:
            if policies[high]!=policies[low]:
                n,collection=changed[name]
                collection.add(group)
                changed[name]=n+1,collection
    return matched,full,changed,coverage


def group_summary(grouped):
    parents=sorted(grouped)
    values=np.asarray([np.mean(grouped[p],axis=0) for p in parents])
    return parents,values


def differences(data, parents, high, low, dates, bootstrap):
    delta=data[:,high]-data[:,low]
    intervals=bootstrap(delta,[dates[p] for p in parents])
    sensitivity=bootstrap(delta,parents)
    return {name:dict(delta=float(mean),interval_95=ci,parent_interval_95=pi)
            for name,mean,ci,pi in zip(METRICS,delta.mean(axis=0),intervals,sensitivity,strict=True)}


def effect_decision(ap, interval, hulls, blocks):
    covered=hulls>=20 and blocks>=8
    if covered and ap>=.30 and interval[0]>0:
        return "positive"
    if covered and ap<=-.30 and interval[1]<0:
        return "negative"
    return "inconclusive"


def measures(gt, blocks, dates, core, evaluator, boundary, shapely, bootstrap):
    ids=set(blocks)
    subset=dict(gt,images=[x for x in gt["images"] if x["id"] in ids],
                annotations=[x for x in gt["annotations"] if x["image_id"] in ids])
    native,anchors=evaluator.score_one(subset,blocks,"raw",anchors=True)
    ap_metrics={"raw":native,**{arm:evaluator.score_one(subset,blocks,arm)[0] for arm in ARMS[1:]}}
    matched, full, changed, coverage=geometry(subset,blocks,anchors,core,boundary,shapely)
    parents,values=group_summary(matched)
    all_parents,all_values=group_summary(full)
    need(sum(map(len,full.values()))==len(subset["annotations"])==sum(coverage.values()),
         "All fixed author hulls preserved in zero-treatment sensitivity")
    estimates={}
    for name,high,low in CUTS:
        geo=differences(values,parents,high,low,dates,bootstrap) if parents else None
        complete=differences(all_values,all_parents,high,low,dates,bootstrap)
        ap=100*(ap_metrics[ARMS[high]]["AP"]-ap_metrics[ARMS[low]]["AP"])
        n,exposed=changed[name]
        day=len({dates[p] for p in exposed})
        status=effect_decision(ap,geo["continuous_iou_points"]["interval_95"],n,day) if geo else "inconclusive"
        estimates[name]=dict(ap_points=ap,continuous_and_boundary=geo,all_hull_zero_effect=complete,
                             changed_hulls=n,changed_parent_groups=len(exposed),
                             changed_analysis_blocks=day,decision=status)
    return dict(images=len(ids),author_hulls=len(subset["annotations"]),original_native_matches=len(anchors),
                eligible_matches=sum(map(len,matched.values())),eligible_parent_groups=len(parents),
                eligible_sensor_date_blocks=len({dates[p] for p in parents}),
                all_hull_parent_groups=len(all_parents),count_unique_native_candidates=sum(len(r["stock"]) for r in blocks.values()),
                mask_metrics_fraction=ap_metrics,contrasts=estimates,coverage=dict(coverage))


def score(a):
    started=perf_counter()
    blocks,tally=records_for_score(a)
    past=module(a.code/"LEARNED_BOUNDED_PAIRING_20261126/rsdd/score.py",HISTORICAL_SCORE,"_rsdd_bound_scores")
    original=past.module(a.code/"INSTANCE_RSDD_FROZEN_HRSID_20261122/score_rsdd.py",past.ORIGINAL_SCORE_SHA)
    evaluator=original.module(a.code/"PSEG_RFDETR_FROZEN_SOURCE_HEAD_20261122/score.py",original.SCORE_SOURCE)
    adapter=original.module(a.code/"UEIKAP_RFDETR_CARRIER_20261122/score_rf.py",original.GEOMETRY_SOURCE)
    core=adapter.load_geometry(a.code/"UEIKAP_RESERVED_HULL_FROZEN_20261122/score.py")
    reserve=past.module(a.code/"INSTANCE_RSDD_ACQUISITION_RESERVE_20261123/score_reserve.py",past.RESERVE_SCORE_SHA)
    auth=json.loads(pin(a.archive/"AUTHORITY_SCORE_PRIVATE.json",past.RESERVE_AUTHORITY_SHA))
    need(Path(auth["author_train"])==a.author_labels,"Exact original reserve authority reference path")
    census=json.loads(pin(Path(auth["census"]),reserve.CENSUS))
    dates={r["paper_scene"]:(r["sensor"],r["calendar_date"]) for r in census["per_parent"]}
    nested=set(census["date_separated_sensitivity"]["parent_scene_ids"])
    reduced={iid for iid,row in blocks.items() if row["group"] in nested}
    need(len(reduced)==42 and len(nested)==8 and set(dates)=={r["group"] for r in blocks.values()},
         "Same original nested 42-chip sensor-date view")
    if a.shapely_site is not None:
        sys.path.append(str(a.shapely_site))
    import shapely
    need(shapely.__version__=="2.0.2" and shapely.geos_version_string=="3.11.2",
         "Same runtime for valid polygon area intersection")
    boundary=module(a.code/"INSTANCE_RSDD_NATIVE_BOUNDARY_IOU_20261126/boundary_metric.py",BOUNDARY,"_dfine_biou")
    gt,_=original.coco_annotation_ids(reserve.select_labels(pin(a.author_labels,reserve.TRAIN),blocks))
    views={}
    for name,scoped,expected in (("full",blocks,(74,81)),("date_separated",{i:blocks[i] for i in blocks if i in reduced},(42,47))):
        r=measures(gt,scoped,dates,core,evaluator,boundary,shapely,reserve.bootstrap)
        need((r["images"],r["author_hulls"])==expected,"Both complete canonical author denominators")
        views[name]=r
    finish(a,SCORE_STATUS,started,views=views,counts=tally,label_files_opened=1,
           author_reference_sha256=reserve.TRAIN,verify_result_sha256=a.verify_sha,
           verify_terminal_sha256=a.verify_terminal_sha,render_result_sha256=a.render_sha,
           render_terminal_sha256=a.render_terminal_sha,historical_reserve=True,
           original_d_fine_matches_not_rf_detr=True)


def main():
    p=ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("phase",choices=("render","verify","score"))
    for key in ("code","work-root","output","protocol","capture-root","intake-root"):
        p.add_argument("--"+key,type=Path,required=True)
    for key in ("refit-root","archive","author-labels","numpy-site","codec-site","shapely-site"):
        p.add_argument("--"+key,type=Path)
    for key in ("capture-sha","capture-terminal-sha","capture-readback-sha","capture-readback-terminal-sha",
                "capture-protocol-sha","capture-commit","protocol-sha","prelock-commit"):
        p.add_argument("--"+key,required=True)
    for key in ("render-sha","render-terminal-sha","verify-sha","verify-terminal-sha"):
        p.add_argument("--"+key)
    a=p.parse_args()
    need(sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES")=="-1"
         and a.work_root.is_dir() and not a.output.exists() and a.output.resolve()==a.work_root.resolve()/a.phase
         and all(re.fullmatch("[0-9a-f]{40}",v) for v in (a.prelock_commit,a.capture_commit)),
         "Fresh isolated CPU execution with published protocol commitments")
    pin(a.protocol,a.protocol_sha)
    if a.phase!="render":
        need(a.render_sha and a.render_terminal_sha,"Qualified observed frozen editing receipt")
    if a.phase=="score":
        need(all((a.verify_sha,a.verify_terminal_sha,a.archive,a.author_labels)),
             "Exact independently verified edits and original archive labels")
    else:
        need(a.refit_root is not None and a.author_labels is None,"Frozen source head and label-free replay")
    for path in (a.numpy_site,a.codec_site):
        if path is not None:
            sys.path.insert(0,str(path))
    global np
    import numpy as np
    need(np.__version__=="1.26.4" and version("pycocotools")=="2.0.11",
         "Match original NumPy, RLE decoder and approved runtime")
    if a.phase=="score":
        score(a)
    else:
        observe_replay(a,a.phase=="verify")


if __name__=="__main__":
    main()

