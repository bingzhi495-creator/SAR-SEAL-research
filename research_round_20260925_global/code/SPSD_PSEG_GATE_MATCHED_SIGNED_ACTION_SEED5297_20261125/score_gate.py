"""Score a closed gate replay and inspect selected signed margins."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from copy import deepcopy
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
from time import perf_counter
from types import ModuleType

import numpy as np
from pycocotools import mask as codec
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scipy.optimize import linear_sum_assignment

LOCK = "789fe67f634f2fc6d4de977e5c480f25966205cafd5ba7108bfb934b4ee8465b"
GROUP = "4030a3cb5684a6dac0ea1e7765a48d05caec113c5c6e0e6de2c82b991e9d5f15"
RENDER = "c4841925513aeee64078064d483c73557b48c352594dd0d1edbe67a15be4b5b8"
LABELS = ("pseg","spsd")
TOL = 1e-12


def need(ok,message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return sha256(raw).hexdigest()


def pin(path,checksum):
    raw = path.read_bytes()
    need(digest(raw)==checksum,"Exact input: "+path.name)
    return raw


def accepted(a,root):
    renderer = ModuleType("paired_static_margin_render")
    path = root/"render_gate.py"
    renderer.__file__ = str(path)
    exec(compile(pin(path,RENDER),str(path),"exec"),renderer.__dict__)
    term_raw = (root/"RENDER_TERMINAL.json").read_bytes()
    term = json.loads(term_raw)
    need(term["phase"]=="render" and term["actual_exit"]==0 and term["output_footer_verified"] is True
         and term["worker_sha256"]==RENDER,"Observed label-free replay closure")
    result = json.loads(pin(root/"render/RESULT_SAFE.json",term["result_sha256"]))
    need(result["status"]=="SPSD_MATCHED_SIGNED_ACTION_GATE_RENDER_COMPLETE"
         and result["worker_sha256"]==RENDER and result["protocol_sha256"]==LOCK
         and result["fit_protocol_sha256"]==renderer.FIT_LOCK
         and result["source_gate_labels_opened"] is result["fit_labels_opened"] is result["historical_test_opened"] is False,
         "Sealed original prelabel scope")
    _,reader,_,stock,original,groups,previous,_ = renderer.prior(a.gate)
    new = [json.loads(x) for x in pin(root/"render/GATE_RENDER_PRIVATE.jsonl",
                                     result["render_stream_sha256"]).splitlines()]
    need(len(new)==183 and len({x["image_id"] for x in new})==len(groups)==183
         and len(set(groups.values()))==180,"Full source-gate roster")
    native, row_map, totals = {}, {}, Counter()
    for row in new:
        iid = row["image_id"]
        orig = original[iid]
        need(row["source_group"]==groups[iid] and row["filename"]==orig["filename"]
             and row["original_hw"]==orig["original_hw"],"Previous source identity")
        current = row["candidates"]
        need(len(current)==len(stock[iid])==100,"All candidate slots retained")
        for p,q in zip(current,stock[iid],strict=True):
            k = p["quota"]
            n = p["native"]
            need(q["native"]==n and q["score"]==p["score"] and k==q["quota"]
                 and set(p["changes"])==(set(renderer.ARMS[1:]) if k else set()),
                 "Exact anchor slots, scores and signed budget")
            if k:
                need(q["changes"]["static"]==p["changes"]["static"]
                     and q["changes"]["dynamic"]==p["changes"]["dynamic"],
                     "Previous dynamic and static masks exact")
                area = int(codec.area(n))
                for arm,rle in p["changes"].items():
                    inter=int(codec.area(codec.merge([n,rle],intersect=True)))
                    need(rle["size"]==n["size"] and int(codec.area(rle))==area and area-inter==k,
                         "Exact equal realized signed counts independently read back")
                margins=p["margin_estimates"]
                need(set(margins)==set(renderer.HEADS.values())
                     and all(set(x)==set(renderer.HEADS.values()) for x in margins.values())
                     and np.asarray([m[y] for m in margins.values() for y in renderer.HEADS.values()]).shape==(4,2),
                     "Both prediction heads on each chosen action")
                totals.update(positive=1,exchanges=k)
        row_map[iid]=row
        native[iid]=current
    need(totals==dict(positive=8785,exchanges=57692) and set(native)==set(stock)==set(groups),
         "Complete gate budget")
    print("All 18,300 candidates and prior dynamic/static RLE parity verified before opening gate labels.",flush=True)
    return renderer.ARMS,reader,previous,term["result_sha256"],digest(term_raw),result,native,row_map,groups


def coco_score(truth,native,arm):
    indexed, results = {}, []
    for iid, rows in native.items():
        for k,pos in enumerate(rows,1):
            indexed[len(results)+1]=(iid,k)
            results.append(dict(image_id=iid,category_id=0,score=pos["score"],
                                segmentation=pos["changes"].get(arm,pos["native"])))
    with redirect_stdout(StringIO()):
        gt=COCO()
        gt.dataset=deepcopy(truth)
        gt.createIndex()
        ev=COCOeval(gt,gt.loadRes(deepcopy(results)),"segm")
        ev.params.catIds,ev.params.imgIds,ev.params.maxDets=[0],sorted(native),[1,10,100]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    t90,=np.flatnonzero(np.isclose(ev.params.iouThrs,.90))
    pr=ev.eval["precision"][t90,:,0,0,2]
    pr=pr[pr>=0]
    summary={k:float(ev.stats[j])*100 if ev.stats[j]>=0 else None for k,j in
             (("AP",0),("AP50",1),("AP75",2),("APs",3),("AR100",8))}
    summary["AP90"]=float(pr.mean()*100) if len(pr) else None
    matched={}
    if arm=="native":
        t50,=np.flatnonzero(np.isclose(ev.params.iouThrs,.5))
        rows=[x for x in ev.evalImgs if x and x["aRng"]==ev.params.areaRng[0] and x["maxDet"]==100]
        need(sum(len(x["gtIds"]) for x in rows)==len(truth["annotations"]),"Native retains all original labels")
        for row in rows:
            for aid,did,ignore in zip(row["gtIds"],row["gtMatches"][t50],row["gtIgnore"],strict=True):
                if did and not ignore:
                    slot=indexed[int(did)]
                    need(slot[0]==row["image_id"] and aid not in matched,"Original fixed native association")
                    matched[int(aid)]=slot
    return summary,matched


def geometry(arms,truth,native,matches,groups,ordered):
    per_img=defaultdict(list)
    for entry in truth["annotations"]:
        iid=entry["image_id"]
        slot=matches.get(entry["id"])
        if slot is None:
            per_img[iid].append(np.zeros(len(arms)))
            continue
        need(slot[0]==iid,"Native match owner")
        row=native[iid][slot[1]-1]
        rles=[row["changes"].get(arm,row["native"]) for arm in arms]
        iou=codec.iou(rles,[entry["segmentation"]],[0])[:,0]
        need(np.isfinite(iou).all() and np.all((iou>=0)&(iou<=1)),"Valid exact raster IoU vector")
        per_img[iid].append(iou)
    group_values=defaultdict(list)
    for iid,g in groups.items():
        image=per_img[iid]
        group_values[g].append(np.mean(image,axis=0) if image else np.zeros(len(arms)))
    return np.stack([np.mean(group_values[g],axis=0) for g in ordered])*100


def contrasts(arms,values,metrics,draws):
    out={}
    for j,arm in enumerate(arms):
        if arm=="action_robust":
            continue
        d=values[:,arms.index("action_robust")]-values[:,j]
        out[arm]=dict(iou_points=float(d.mean()),
                      percentile95_points=np.percentile(d[draws].mean(axis=1),[2.5,97.5]).tolist(),
                      ap_points=metrics["action_robust"]["AP"]-metrics[arm]["AP"])
    return out


def label_pairs(truths,matches,native,groups):
    by_img=defaultdict(lambda:[[],[]])
    for j,k in enumerate(LABELS):
        for entry in truths[k]["annotations"]:
            by_img[entry["image_id"]][j].append(entry)
    counts, changed_groups, calibration = Counter(), set(), defaultdict(list)
    for iid,g in groups.items():
        pseg,spsd=by_img[iid]
        p,s=[[x["segmentation"] for x in v] for v in (pseg,spsd)]
        if not p or not s:
            continue
        overlap=codec.iou(p,s,[0]*len(s))
        ia,ib=linear_sum_assignment(overlap,maximize=True)
        counts["reference_pairs"]+=len(ia)
        for x,y in zip(ia,ib,strict=True):
            if overlap[x,y]<.5:
                continue
            counts["reference_pairs_ge050"]+=1
            a=matches["pseg"].get(pseg[x]["id"])
            b=matches["spsd"].get(spsd[y]["id"])
            if a is None or a!=b:
                continue
            counts["shared_native_matches"]+=1
            c=native[iid][a[1]-1]
            k=c["quota"]
            if k==0:
                continue
            counts["positive_quota_common_hulls"]+=1
            masks=c["changes"]
            if masks["action_robust"]!=masks["bce_robust"]:
                counts["changed_common_hulls"]+=1
                changed_groups.add(g)
            labels=(p[x],s[y])
            static=np.array([int(codec.area(codec.merge([masks["static"],l],intersect=True))) for l in labels])
            for writer in ("action","bce"):
                raw=masks[writer+"_robust"]
                outcome=np.array([int(codec.area(codec.merge([raw,l],intersect=True))) for l in labels])-static
                for estimator,estim in c["margin_estimates"][writer].items():
                    calibration[writer+"__estimated_by__"+estimator].append((k,np.asarray(estim),outcome))
    need((counts["reference_pairs"],counts["reference_pairs_ge050"],counts["shared_native_matches"],
          counts["positive_quota_common_hulls"])==(410,370,276,206),"Original common-match cohort and margin scope")
    return dict(counts,changed_common_groups=len(changed_groups)),calibration


def sign(value):
    return "positive" if value>TOL else "negative" if value<-TOL else "zero"


def margin_stats(sets):
    out={}
    for key,rows in sets.items():
        k=np.array([v[0] for v in rows])
        estimate=np.stack([v[1] for v in rows])
        actual=np.stack([v[2] for v in rows])
        errors=estimate-actual
        normalized=errors/(2*k[:,None])
        tables={}
        for tag,a,b in (("pseg",estimate[:,0],actual[:,0]),("spsd",estimate[:,1],actual[:,1]),
                        ("minimum",estimate.min(axis=1),actual.min(axis=1))):
            tables[tag]=Counter(sign(x)+"_pred__"+sign(y)+"_true" for x,y in zip(a,b,strict=True))
        out[key]=dict(common_hulls=len(k),raw_estimated_mean=estimate.mean(axis=0).tolist(),
                      raw_true_mean=actual.mean(axis=0).tolist(),raw_bias=errors.mean(axis=0).tolist(),
                      raw_mae=abs(errors).mean(axis=0).tolist(),normalized_bias=normalized.mean(axis=0).tolist(),
                      normalized_mae=abs(normalized).mean(axis=0).tolist(),sign_confusion=tables)
    return out


def gate(stats,exposure):
    lookup=lambda arm:[v["action_robust_minus"][arm] for v in stats.values()]
    base=lookup("bce_robust")
    mean=min(v["iou_points"] for v in lookup("action_mean"))
    pess=min(v["iou_points"] for v in lookup("action_cell_pessimistic"))
    checks=dict(iou_magnitude=all(v["iou_points"]>=.15 for v in base),
                iou_lower=all(v["percentile95_points"][0]>0 for v in base),
                ap_guard=all(v["ap_points"]>=-.10 for v in base),
                exposure=exposure["changed_common_hulls"]>=20 and exposure["changed_common_groups"]>=8,
                vs_action_mean=mean>0,vs_action_cell_pessimistic=pess>0)
    return dict(advances=all(checks.values()),criteria=checks,
                worst_ref_iou_vs_action_mean_points=mean,worst_ref_iou_vs_action_cell_pessimistic_points=pess)


def main():
    p=ArgumentParser(description=__doc__)
    for x in ("gate","labels","protocol","group-protocol","output"):
        p.add_argument("--"+x,type=Path,required=True)
    a=p.parse_args()
    start=perf_counter()
    pin(a.protocol,LOCK)
    pin(a.group_protocol,GROUP)
    root=Path(__file__).parent
    arms,reader,previous,rs,ts,sealed,native,records,groups=accepted(a,root)
    truths=reader.labels(a,records)
    order=sorted(set(groups.values()))
    rng=np.random.Generator(np.random.PCG64(5297))
    draws=rng.integers(180,size=(4096,180))
    refs,matches={},{}
    for key,truth in truths.items():
        metrics={}
        prior=previous["references"][key]
        for name in arms:
            metrics[name],matching=coco_score(truth,native,name)
            if name in ("native","static","dynamic"):
                need(all(abs(metrics[name][k]-v) <= 1e-12 for k,v in prior["metrics"][name].items()),
                     "Exact original score parity on the native/static/dynamic controls")
            if name=="native":
                matches[key]=matching
                need(len(matching)==prior["original_matches"],"Exact prior native mask association count")
        values=geometry(arms,truth,native,matches[key],groups,order)
        need(all(abs(values[:,arms.index(name)].mean()-prior["fixed_iou_points"][name]) <= 1e-12
                 for name in ("native","static","dynamic")), "Exact previous fixed geometry parity")
        refs[key]=dict(metrics=metrics,fixed_iou_points=dict(zip(arms,values.mean(axis=0).tolist(),strict=True)),
                       action_robust_minus=contrasts(arms,values,metrics,draws),
                       original_matches=len(matches[key]),author_hulls=len(truth["annotations"]))
        print("Scored both matched losses under",key,flush=True)
    exposed,calibration=label_pairs(truths,matches,native,groups)
    result=dict(status="SPSD_MATCHED_SIGNED_ACTION_GATE_SCORE_COMPLETE", protocol_sha256=LOCK,
                worker_sha256=digest(Path(__file__).read_bytes()),render_result_sha256=rs,
                render_terminal_sha256=ts,render_stream_sha256=sealed["render_stream_sha256"],
                fit_result_sha256=sealed["fit_result_sha256"],prior_score_sha256=sealed["prior_score_sha256"],
                candidate_slots=18300,images=183,decoded_groups=180,bootstrap_draws=4096,
                seed=5297,exposure=exposed,references=refs,margin_diagnostic=margin_stats(calibration),
                primary=gate(refs,exposed),fit_labels_opened=False,historical_test_opened=False,
                optimizer_updates=0,detector_forwards=0,elapsed_seconds=perf_counter()-start)
    a.output.mkdir()
    raw=(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    with (a.output/"RESULT_SAFE.json").open("xb") as out:
        out.write(raw)
    print(json.dumps(dict(status=result["status"],result_sha256=digest(raw))),flush=True)


if __name__=="__main__":
    main()

