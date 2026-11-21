"""Post-hoc native .50 match geometry and raw COCO precision/recall accounting for closed D-FINE PSeg."""
import argparse
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from copy import deepcopy
from hashlib import sha256
from io import StringIO
import json
import os
from pathlib import Path
from types import ModuleType
import sys
import numpy as np
from pycocotools import mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ARMS = ("raw","fixed","free_matched_count","free_natural","native_matched_count")
PAIRS = (("free_matched_count","fixed"),("free_natural","native_matched_count"),("free_natural","raw"),("free_natural","fixed"))
FIELDS = ("continuous_iou_points","fp_pixels_per_gt_area_pct","fn_pixels_per_gt_area_pct")
SCORE = "9bc6411cf35ee355f142a5fa5588c8e5820127b8101acf4da0e653e70553d3d1"
GEOM = "7cad8b51a9230101e2233db8455f84c28163abfb2ee8ddd8cbe8718a2e719bd0"
OLD = "77159ea493a95973fe422dac1d303eaeba603e6d8af7ae179c72aecc2e4301b0"
OMIT = {1,11,229,389,501,631,639,641,739,1121}


def need(v,s):
    if not v:
        raise ValueError(s)


def pinned(path,sha):
    raw=path.read_bytes()
    need(sha256(raw).hexdigest()==sha,"Exact source/input: "+path.name)
    return raw


def load(path,sha):
    m=ModuleType(path.stem)
    m.__file__=str(path)
    exec(compile(pinned(path,sha),str(path),"exec"),m.__dict__)
    return m


def read_data(a):
    score=json.loads(pinned(a.score,SCORE))
    gt=json.loads(pinned(a.gt,score["gt_sha256"]))
    roster=json.loads(pinned(a.manifest,score["manifest_sha256"]))["items"]
    pred,audit,edit=[[json.loads(line) for line in pinned(p,score[key]).splitlines()] for p,key in (
        (a.native,"input_predictions_sha256"),(a.audit,"candidate_audit_sha256"),(a.edits,"edits_sha256"))]
    images={im["file_name"]:im for im in gt["images"]}
    records={}
    for r,p,k,e in zip(roster,pred,audit,edit,strict=True):
        need(r["image_id"]==p["image_id"]==k["image_id"]==e["image_id"],"Native block alignment")
        name=r["image_ref"]["path"]
        original=images[name]
        cand=p["coco"][:100]
        aux=k["candidates"][:100]
        need(r["original_hw"]==[original["height"],original["width"]] and len(cand)==len(aux)==100
             and all(v["score"]==d["score"] and d["native_rank"]==i for i,(v,d) in enumerate(zip(cand,aux),1)),
             "Original canvas, ranked candidates and metadata")
        records[original["id"]]=dict(name=name,rgb=e["rgb_sha256"],native=cand,audit=aux,edit=e["changes"])
    need(len(records)==232 and len(gt["annotations"])==546,"Original scope")
    return score,gt,records


def evaluate_raw(gt,records,ids,expected):
    stats={}
    detections=[]; ranks={}
    for i in ids:
        for rank,p in enumerate(records[i]["native"],1):
            detections.append(dict(image_id=i,category_id=0,score=p["score"],segmentation=deepcopy(p["segmentation"])))
            ranks[len(detections)]=(i,rank)
    results={}
    with redirect_stdout(StringIO()):
        ref=COCO(); ref.dataset=deepcopy(gt); ref.createIndex()
        det=ref.loadRes(detections)
        ev=COCOeval(ref,det,"segm")
        ev.params.catIds=[0];ev.params.imgIds=ids
        ev.evaluate();ev.accumulate();ev.summarize()
    need(float(ev.stats[0])==expected["AP"] and float(ev.stats[8])==expected["AR100"],"Exact raw official score parity")
    rows=[e for e in ev.evalImgs if e and list(e["aRng"])==list(ev.params.areaRng[0]) and e["maxDet"]==100]
    need(len(rows)==len(ids),"All image matching records")
    denom=sum(int((~np.asarray(e["gtIgnore"],dtype=bool)).sum()) for e in rows)
    precision=ev.eval["precision"][:,:,0,0,-1]; recall=ev.eval["recall"][:,0,0,-1]
    rec_ceiling=(ev.params.recThrs[None,:] <= recall[:,None]).mean(axis=1)
    # With the observed TP assignments/counts, placing TPs before FPs saturates precision up to attained recall.
    stats.update(gt_instances=denom,raw_mask_AP_points=100*float(ev.stats[0]),raw_AR100_points=100*float(ev.stats[8]),
                 ap_grid_ceiling_given_observed_tp_counts_points=100*float(rec_ceiling.mean()),
                 ap_grid_loss_from_attained_recall_points=100*(1-float(rec_ceiling.mean())),
                 residual_precision_gap_at_attained_recalls_points=100*(float(rec_ceiling.mean())-float(ev.stats[0])),
                 per_threshold={})
    for t,x in enumerate(ev.params.iouThrs):
        hit=sum(int(((e["dtMatches"][t]>0)&~e["dtIgnore"][t]).sum()) for e in rows)
        need(abs(hit/denom-recall[t])<1e-10,"GT coverage equals COCO achieved recall")
        stats["per_threshold"][format(x,".2f")]=dict(TP=hit,unmatched_gt=denom-hit,AP_points=100*float(precision[t].mean()),
                                                     max_recall_points=100*float(recall[t]))
    for e in rows:
        for aid,did,ig in zip(e["gtIds"],e["gtMatches"][0],e["gtIgnore"],strict=True):
            if did and not ig:
                iid,rank=ranks[int(did)]
                need(iid==e["image_id"] and aid not in results,"Original COCO pair")
                results[int(aid)]=(iid,rank)
    need(len(results)==stats["per_threshold"]["0.50"]["TP"],"Frozen .50 matches")
    return results,stats


def summarize(rows):
    groups=defaultdict(list)
    for rgb,measures in rows:
        groups[rgb].append(measures)
    cluster=np.asarray([np.mean(groups[k],axis=0) for k in sorted(groups)])
    pairs=np.asarray([v for _,v in rows])
    draws=np.random.Generator(np.random.PCG64(5217)).integers(len(cluster),size=(4096,len(cluster)))
    output=dict(eligible_pairs=len(pairs),decoded_image_groups_with_pairs=len(cluster),arms={},pairwise={})
    for j,arm in enumerate(ARMS):
        output["arms"][arm]=dict(zip(FIELDS,map(float,cluster[:,j].mean(axis=0))))
        output["arms"][arm].update(pair_equal_continuous_iou_points=float(pairs[:,j,0].mean()),
                strict75_count=int((pairs[:,j,0]>=75).sum()),strict90_count=int((pairs[:,j,0]>=90).sum()))
    for a,b in PAIRS:
        ia,ib=ARMS.index(a),ARMS.index(b)
        d=cluster[:,ia]-cluster[:,ib]
        boot=d[draws].mean(axis=1)
        block={k:dict(delta=float(d[:,j].mean()),percentile95=np.percentile(boot[:,j],[2.5,97.5]).tolist()) for j,k in enumerate(FIELDS)}
        for t in (75,90):
            before=pairs[:,ib,0]>=t
            after=pairs[:,ia,0]>=t
            block["strict"+str(t)]=dict(gained=int((after&~before).sum()),lost=int((~after&before).sum()),
                                        retained=int((after&before).sum()),both_below=int((~after&~before).sum()))
        output["pairwise"][a+" minus "+b]=block
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("score","gt","manifest","native","audit","edits","code","shapely-site","output"):
        p.add_argument("--"+k,type=Path,required=True)
    a=p.parse_args()
    sys.path.append(str(a.shapely_site))
    import shapely
    need(shapely.__version__=="2.0.2" and shapely.geos_version_string=="3.11.2","Prior geometry engine versions")
    old=load(a.code/"DFINE_SEG_EXTERNAL_BASELINE_FIT32_E24_20261112/e24/dev705_free_endpoint/score_endpoint.py",OLD)
    geom=load(a.code/"MASK_RTDETR_DEV705_CONTINUOUS_AP_20261112/continuous_endpoint.py",GEOM)
    modules=geom.geometry_modules(a.code)
    score,gt,records=read_data(a)
    truth,neutral,support=modules.geometry.polygon_support(gt,modules.vector)
    need(len(truth)==546 and len(neutral)==0,"All original float polygons supported")
    views={"official":sorted(records), "exclude_exact_fit_overlap":sorted(i for i,r in records.items() if int(Path(r["name"]).stem) not in OMIT)}
    safe={}; original_matches=None; measured={}
    for view,ids in views.items():
        matches,diagnostic=evaluate_raw(gt,records,ids,score["views"][view]["metrics_fraction"]["raw"])
        if original_matches is None:
            original_matches=matches
        else:
            need(matches=={k:v for k,v in original_matches.items() if v[0] in ids},"Identical anchors across common images")
        rows=[];counts=Counter(); edited=Counter();eligible_slots=0
        for iid in ids:
            eligible_slots+=sum(32<=c["mask_area"]<1024 for c in records[iid]["audit"])
        for gid,(iid,rank) in matches.items():
            rec=records[iid]; area=rec["audit"][rank-1]["mask_area"]
            if not 32<=area<1024:
                counts["native_match_outside_frozen_area"]+=1
                continue
            arm_edits=rec["edit"].get(str(rank),{})
            if gid not in measured:
                base=rec["native"][rank-1]["segmentation"]
                original=old.continuous(base,area,truth[gid][0],modules)
                vals=[]
                for arm in ARMS:
                    edit=arm_edits.get(arm)
                    if edit:
                        need(int(mask.area(edit["rle"]))==area,"Realized equal-area edits")
                        vals.append(old.continuous(edit["rle"],area,truth[gid][0],modules))
                    else:
                        vals.append(original)
                measured[gid]=np.array(vals)
            vals=measured[gid]
            rows.append((rec["rgb"],vals))
            for arm in arm_edits:
                edited[arm]+=1
        safe[view]=summarize(rows)
        safe[view].update(images=len(ids),decoded_image_groups=len({records[i]["rgb"] for i in ids}),native_TOP100_slots=100*len(ids),
                          eligible_native_slots=eligible_slots,stock_matches_at_50=len(matches),raw_matches_outside_editable_area=counts["native_match_outside_frozen_area"],
                          edited_fixed_matches={a:edited[a] for a in ARMS},raw_mask_score_diagnostics=diagnostic)
        print(json.dumps(dict(view=view,raw_matches=len(matches),eligible=len(rows),groups=safe[view]["decoded_image_groups_with_pairs"])),flush=True)
    result=dict(status="PSEG_TEST232_DFINE_FROZEN_POSTHOC_PAIRED_GEOMETRY_COMPLETE",posthoc_historical_test=True,
                score_sha256=SCORE,source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),prior_geometry_source_sha256=GEOM,
                prior_continuous_function_source_sha256=OLD,bootstrap_seed=5217,bootstrap_draws=4096,
                group_weight="Equal decoded RGB groups with at least one native .50 match of predicted area 32..1023; within-group mean over original pairs.",
                intervals="Percentile paired decoded-image bootstrap conditional on this population; acquisition independence unknown.",
                shape_errors="100 times realized excess or missing mask footprint area divided by continuous GT polygon area.",
                scope="Fixed post-NMS top100, source model/head; original match IDs and endpoint only. No new selection decision.",
                diagnostics="Recall ceiling assumes perfect front-ranking of the observed matched detections independently at each threshold. It is an arithmetic accounting, not an achievable universal re-ranker.",
                model_forwards=0,optimizer_updates=0,views=safe)
    raw=(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+"\n").encode()
    with a.output.open("xb") as out:
        out.write(raw);out.flush();os.fsync(out.fileno())
    print(json.dumps(dict(status=result["status"],sha256=sha256(raw).hexdigest())),flush=True)


if __name__=="__main__":
    main()
