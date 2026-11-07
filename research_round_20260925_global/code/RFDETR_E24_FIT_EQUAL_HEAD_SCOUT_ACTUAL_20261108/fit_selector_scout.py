"""Execute the single preregistered equal-head FIT pilot after materialization."""
from contextlib import redirect_stdout
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import time

from fit_selector_materialize import captured_item,checked,digest,doc,load,need,original_gt,put,runtime

BRANCHES=("uncertainty","mean","rank")
ARMS=("no_edit","stock",*BRANCHES)


def apply_edits(np,codec,rle,edit):
    if not len(edit):
        return rle
    need(len(edit)<=16 and edit.ndim==1 and edit.dtype.kind in "iu" and len(set(edit.tolist()))==len(edit),
         "Up to 16 unique original pixel edits")
    source=np.ascontiguousarray(codec.decode(dict(size=rle["size"],counts=rle["counts"].encode("ascii"))),
                                dtype=np.uint8)
    pixels=source.ravel()
    need(int(edit.min())>=0 and int(edit.max())<pixels.size and bool(((pixels==0)|(pixels==1)).all()),
         "Binary original-grid edit bounds")
    new=source.copy()
    new.ravel()[edit]^=1
    need(np.array_equal(np.flatnonzero(new.ravel()!=pixels),np.sort(edit)),"Exact C-order pixel action")
    result=codec.encode(np.asfortranarray(new))
    need(result["size"]==rle["size"] and np.array_equal(codec.decode(result),new),"Edited RLE exact roundtrip")
    return dict(size=result["size"],counts=result["counts"].decode("ascii"))


def load_image(np,meta):
    match=doc(meta["match_ref"])
    raw=checked(meta["arrays_ref"],16*1024**2)
    with np.load(BytesIO(raw),allow_pickle=False) as f:
        arrays={k:f[k] for k in f.files}
    p=arrays["positions"]
    feat=arrays["features"]
    y=arrays["y"]
    need(feat.dtype==np.float32 and feat.shape==(len(p),3) and y.dtype==np.bool_ and y.ndim==1
         and len(match["rows"])+1==len(arrays["offsets"])==len(arrays["label_offsets"]),
         "Complete accepted equal-support image feature shape")
    return match["rows"],arrays


def features(a):
    return dict(abs_logit=a["features"][:,0],mean_contrast=a["features"][:,1],rank=a["features"][:,2],
                edit_sign=a["edit_sign"],valid=a["valid"])


def detector_scores(np,rows,a):
    counts=np.diff(a["offsets"])
    need(len(counts)==len(rows),"Unchanged score repeats for all native rows")
    return np.repeat(np.array([r["score"] for r in rows],dtype=np.float64),counts)


def train_inputs(np,records):
    parts={k:[] for k in ("abs_logit","mean_contrast","rank","edit_sign","valid","score","y","weight")}
    matched=0
    for meta in records:
        rows,a=load_image(np,meta)
        f=features(a)
        for j,row in enumerate(rows):
            start,stop=row["start"],row["stop"]
            if row["status"]!="matched" or start==stop:
                continue
            labels=a["y"][int(a["label_offsets"][j]):int(a["label_offsets"][j+1])]
            n=stop-start
            need(len(labels)==n,"Matched original label support")
            for name in f:
                parts[name].append(f[name][start:stop])
            parts["score"].append(np.full(n,row["score"],dtype=np.float64))
            parts["y"].append(labels)
            parts["weight"].append(np.full(n,1/n,dtype=np.float64))
            matched+=1
    a={k:np.concatenate(v) for k,v in parts.items()}
    need(matched>0 and np.isclose(a["weight"].sum(),matched,rtol=1e-12,atol=1e-8)
         and 0<int(a["y"].sum())<len(a["y"]),"Exactly equal per-instance matched training mass")
    return {k:a[k] for k in ("abs_logit","mean_contrast","rank","edit_sign","valid")},a,matched


def calibration_images(np,solver,model,records):
    for meta in records:
        rows,a=load_image(np,meta)
        yield dict(match_rows=rows,positions=a["positions"],label_offsets=a["label_offsets"],y=a["y"],
                   scores=solver.predict_head(model,features(a),detector_scores(np,rows,a)))


def identities(image_id,capture,rows):
    need(len(capture["records"])==len(rows),"Original candidate row population")
    out=[]
    for original,meta in zip(capture["records"],rows,strict=True):
        need(all(original[k]==meta[k] for k in ("native_rank","query_id","score")),
             "Same original order, query, rank and score")
        out.append(dict(image_id=image_id,category_id=1,score=original["score"],segmentation=original["segmentation"]))
    return out


def run(path,expected_sha):
    start=time.monotonic()
    raw=path.read_bytes()
    need(digest(raw)==expected_sha,"Exact scout execution request")
    q=json.loads(raw)
    need(q["schema"]=="Root-FIT741-equal-head-holdout-scout-v1" and q["DEV"] is q["TEST"] is False
         and os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU-only FIT pilot")
    np,coco,cocoeval,codec=runtime(q)
    solver=load(q["sources"]["solver"],"solver")
    policy=load(q["sources"]["policy"],"_fit_selector_policy")
    evaluator=load(q["sources"]["holdout_eval"],"_fit_selector_official_holdout")
    m=doc(q["material_acceptance"])
    need(m["status"]=="ACTUAL_FULL_FIT741_FIXED_MATCH_FEATURES_ACCEPTED"
         and m["capture_acceptance_ref"]==q["capture_acceptance"] and m["material_index_ref"]==q["material_index"],
         "Accepted original full FIT feature and label materialization")
    fit=doc(q["image_manifest"])
    original=doc(q["capture_index"])
    material=doc(q["material_index"])
    need(len(fit["items"])==len(original)==len(material)==741,"Fixed FIT images across every stage")
    splits={k:[] for k in ("head_fit","calibration","head_holdout")}
    for item,index,meta in zip(fit["items"],original,material,strict=True):
        need(all(item[k]==index[k]==meta[k] for k in ("dataset_index","partition","image_id")),
             "Exact predeclared image partition in every source")
        splits[item["partition"]].append(meta)
    need({k:len(v) for k,v in splits.items()}=={"head_fit":516,"calibration":114,"head_holdout":111},
         "Predeclared FIT partitions")

    out=path.parent
    fit_start=time.monotonic()
    f,data,matched=train_inputs(np,splits["head_fit"])
    heads={branch:solver.fit_head(f,data["score"],data["y"],branch=branch,sample_weight=data["weight"])
           for branch in BRANCHES}
    fit_seconds=time.monotonic()-fit_start
    train_summary=dict(matched_with_support=matched,training_pixels=len(data["y"]),
                       correct_flip_labels=int(data["y"].sum()),iterations_each=128)
    del f,data
    head_ref=put(out/"HEAD_MODELS_PRIVATE.json",dict(models=heads,training=train_summary))
    print(json.dumps(dict(event="FIT_EQUAL_HEADS_FITTED",seconds=fit_seconds)),flush=True)
    cal_start=time.monotonic()
    cal={branch:policy.calibrate(calibration_images(np,solver,heads[branch],splits["calibration"]))
         for branch in BRANCHES}
    thresholds={branch:cal[branch]["threshold"] for branch in BRANCHES}
    cal_seconds=time.monotonic()-cal_start
    cal_ref=put(out/"CALIBRATION_PRIVATE.json",dict(calibrations=cal,thresholds=thresholds))
    print(json.dumps(dict(event="FIT_CALIBRATION_THRESHOLDS_FROZEN",seconds=cal_seconds)),flush=True)

    hold_start=time.monotonic()
    preds={key:{} for key in ARMS}
    forced={key:[] for key in ("stock",*BRANCHES)}
    applied={key:[] for key in ARMS}
    decisions=[]
    ids=[]
    for meta in splits["head_holdout"]:
        index=meta["dataset_index"]
        item=fit["items"][index]
        rows,a=load_image(np,meta)
        captured,_=captured_item(Path(q["capture_root"]),item,original[index])
        img=meta["image_id"]
        ids.append(img)
        raw=identities(img,captured,rows)
        f=features(a)
        d=detector_scores(np,rows,a)
        scores={branch:solver.predict_head(heads[branch],f,d) for branch in BRANCHES}
        scores["stock"]=solver.stock_uncertainty(f)
        p=a["positions"]
        all_forced={arm:policy.select_image(rows,p,s,threshold=None) for arm,s in scores.items()}
        actual={branch:policy.select_image(rows,p,scores[branch],threshold=thresholds[branch])
                for branch in BRANCHES}
        actual["no_edit"]=tuple(np.empty(0,dtype=p.dtype) for _ in rows)
        actual["stock"]=all_forced["stock"]
        for arm,edits in all_forced.items():
            forced[arm].append(policy.account_image(rows,p,a["label_offsets"],a["y"],edits))
        for arm,edits in actual.items():
            applied[arm].append(policy.account_image(rows,p,a["label_offsets"],a["y"],edits))
            preds[arm][img]=[dict(r,segmentation=apply_edits(np,codec,r["segmentation"],edit))
                             for r,edit in zip(raw,edits,strict=True)]
        decisions.append(dict(dataset_index=index,image_id=img,forced={k:[x.tolist() for x in v] for k,v in all_forced.items()},
                              actual={k:[x.tolist() for x in v] for k,v in actual.items()}))
    dref=put(out/"HOLDOUT_DECISIONS_PRIVATE.json",decisions)
    pred_ref=put(out/"HOLDOUT_PREDICTIONS_PRIVATE.json",preds)
    gt=original_gt(q,fit["items"])
    metric={}
    with (out/"OFFICIAL_HOLDOUT_COCO.stdout").open("x",encoding="utf-8") as stream,redirect_stdout(stream):
        for arm in ARMS:
            ev=evaluator.evaluate_holdout(gt,preds["no_edit"],preds[arm],ids,coco_class=coco.COCO,cocoeval_class=cocoeval.COCOeval)
            if "no_edit" in metric:
                need(metric["no_edit"]==ev["original"],"Same official unedited endpoint across arms")
            metric[arm]=ev["edited"]
    net={k:policy.merge_accounts(v) for k,v in forced.items()}
    deploy={k:policy.merge_accounts(v) for k,v in applied.items()}
    gate=policy.check_dev_gate({k:net[k] for k in BRANCHES},{k:metric[k] for k in ("no_edit",*BRANCHES)})
    put(out/"RESULT.json",dict(status="ACTUAL_FIT_EQUAL_HEAD_SCOUT_COMPLETE_PENDING_ROOT",image_counts={k:len(v) for k,v in splits.items()},
        training=train_summary,thresholds=thresholds,forced=net,applied=deploy,official_holdout_metrics=metric,gate=gate,
        model_ref=head_ref,calibration_ref=cal_ref,decisions_ref=dref,predictions_ref=pred_ref,
        selector_only=True,carrier_frozen=True,DEV=False,TEST=False,
        elapsed_seconds=time.monotonic()-start,fit_seconds=fit_seconds,calibration_seconds=cal_seconds,
        holdout_seconds=time.monotonic()-hold_start))


if __name__=="__main__":
    need(len(sys.argv)==3,"Exact original scout spec")
    run(Path(sys.argv[1]),sys.argv[2])
