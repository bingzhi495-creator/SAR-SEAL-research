"""Post-hoc FIT109 B/J COCO matching, confidence order, and per-candidate area direction."""
from argparse import ArgumentParser
from collections import Counter
from contextlib import redirect_stdout
from copy import deepcopy
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
import numpy as np
from pycocotools import mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


QUERIES = "62737cefda8d04fe44069e0e9e515fb1081038ebea78ccbfcbad491c80e3da15"
LABELS = "e95cf26cc60a2c6fb651e907541380a256f2df7b1e153244a6c0f803b0b19f8f"
READBACK = "c791b3eb637acf5e30341768a53fa1acb8a3c8db5d6153717474b0b6958fc319"


def check(c, why):
    if not c:
        raise ValueError(why)


def pinned(file, digest):
    raw=file.read_bytes()
    check(sha256(raw).hexdigest()==digest,"Pinned closed FIT artifact")
    return raw


def bitmap_edit(native, idx):
    out=np.array(native,order="C",copy=True)
    out.reshape(-1)[idx]^=1
    encoded=mask.encode(np.asfortranarray(out))
    return dict(size=encoded["size"],counts=encoded["counts"].decode())


def evaluate(gt, rows):
    coco=COCO()
    coco.dataset=deepcopy(gt)
    with redirect_stdout(StringIO()):
        coco.createIndex()
        preds=coco.loadRes(deepcopy(rows))
        ev=COCOeval(coco,preds,"segm")
        ev.params.imgIds=[i["id"] for i in gt["images"]]
        ev.params.catIds=[1]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return ev


def ranking(ev):
    # COCO accumulation concatenates by native sorted image ID then uses stable score sorting.
    items=[x for x in ev.evalImgs if x is not None and x["aRng"]==ev.params.areaRng[0]]
    ids=np.concatenate([x["dtIds"] for x in items])
    scores=np.concatenate([x["dtScores"] for x in items])
    matched=np.concatenate([x["dtMatches"] for x in items],axis=1)
    ignored=np.concatenate([x["dtIgnore"] for x in items],axis=1)
    order=np.argsort(-scores,kind="mergesort")
    return ids[order],scores[order],matched[:,order],ignored[:,order]


def thresholds(ev_b,ev_j,strata):
    ids,scores,bmatch,bign=ranking(ev_b)
    idj,sj,jmatch,jign=ranking(ev_j)
    check(np.array_equal(ids,idj) and np.array_equal(scores,sj),"Exact common confidence order including ties")
    check(not bign.any() and not jign.any(),"All-area detections unignored for exact TP transitions")
    levels=ev_b.params.recThrs
    rows=[]
    for ix,threshold in enumerate(ev_b.params.iouThrs):
        bs,jstat=[],[]
        for ev,store in ((ev_b,bs),(ev_j,jstat)):
            prec=ev.eval["precision"][ix,:,0,0,-1]
            recall=ev.eval["recall"][ix,0,0,-1]
            reachable=levels<=recall
            check(np.all(prec[~reachable]==0) and (prec>=0).all(),"COCO 101-recall grid including unreached zero precision")
            span=float(reachable.mean())
            quality=float(prec[reachable].mean())
            ap=float(prec.mean())
            check(abs(span*quality-ap)<1e-12,"Exact recall-grid decomposition")
            store.extend((span,quality,ap))
        s_b,q_b,ap_b=bs
        s_j,q_j,ap_j=jstat
        gained=(bmatch[ix]>0)&(jmatch[ix]==0)
        lost=(jmatch[ix]>0)&(bmatch[ix]==0)
        reach=50*(s_b-s_j)*(q_b+q_j)
        quality=50*(q_b-q_j)*(s_b+s_j)
        check(abs(reach+quality-100*(ap_b-ap_j))<1e-9,"Exact two-factor sum")
        changed_both=(bmatch[ix]>0)&(jmatch[ix]>0)&(bmatch[ix]!=jmatch[ix])
        groups=lambda where:dict(Counter(strata[int(d)-1] for d in ids[where]))
        def rank_summary(where):
            ranks=np.flatnonzero(where)+1
            return dict(count=len(ranks),median_global_rank=float(np.median(ranks)) if len(ranks) else None,
                        rank_1_to_100=int((ranks<=100).sum()),rank_101_to_500=int(((ranks>100)&(ranks<=500)).sum()),
                        rank_501_to_1000=int(((ranks>500)&(ranks<=1000)).sum()),rank_beyond_1000=int((ranks>1000).sum()))
        rows.append(dict(iou=round(float(threshold),2),B_AP=ap_b,J_AP=ap_j,AP_gap_points=100*(ap_b-ap_j),
                         reachable_grid_contribution_points=reach,conditional_precision_contribution_points=quality,
                         B_true_positives=int((bmatch[ix]>0).sum()),J_true_positives=int((jmatch[ix]>0).sum()),
                         gained_B=rank_summary(gained),lost_B=rank_summary(lost),gained_by_mask_group=groups(gained),
                         lost_by_mask_group=groups(lost),TP_in_both_but_GT_changed=int(changed_both.sum())))
    check(abs(np.mean([r["AP_gap_points"] for r in rows])-100*(ev_b.stats[0]-ev_j.stats[0]))<1e-9,"Average is official overall AP gap")
    return rows


def main():
    p=ArgumentParser()
    for name in ("old-queries","fit-labels","witness","readback","output"):
        p.add_argument("--"+name,required=True,type=Path)
    args=p.parse_args()
    check(not args.output.exists(),"Fresh diagnostic output")
    previous=json.loads(pinned(args.readback,READBACK))
    records={tuple(v["key"]):v for v in map(json.loads,pinned(args.witness,previous["witness_sha256"]).splitlines())}
    full=json.loads(pinned(args.fit_labels,LABELS))
    queries=list(map(json.loads,pinned(args.old_queries,QUERIES).splitlines()))
    images=[q["image_id"] for q in queries]
    idset=set(images)
    by_id={im["id"]:im for im in full["images"]}
    gt=dict(info={},categories=full["categories"],images=[by_id[i] for i in images],annotations=[v for v in full["annotations"] if v["image_id"] in idset])
    check(len(records)==6738 and len(images)==len(idset)==109 and len(gt["annotations"])==496,"Complete FIT109")
    allrows={s:[] for s in ("B","J","B_on_J_shrink","B_on_J_expand")}
    populations={}
    strata=[]
    seen=set()
    for q in queries:
        iid=q["image_id"]
        for query,rank in zip(q["native_ids"],q["native_ranks"]):
            source=q["all_rles"][query]
            rles=dict.fromkeys(allrows.keys(),source)
            key=(iid,rank,query)
            role="outside_eligible"
            if key in records:
                row=records[key]
                seen.add(key)
                idx_b=np.asarray(row["B"],dtype=np.int64)
                idx_j=np.asarray(row["J"],dtype=np.int64)
                check(len(idx_b)==len(idx_j) and len(idx_b)==len(set(idx_b)) and len(idx_j)==len(set(idx_j)),"Matched budget")
                k=len(idx_b)//2
                native=np.asarray(mask.decode(source),dtype=np.uint8,order="C")
                flat=native.ravel()
                adds_b=set(idx_b[flat[idx_b]==0])
                drops_b=set(idx_b[flat[idx_b]!=0])
                adds_j=set(idx_j[flat[idx_j]==0])
                drops_j=set(idx_j[flat[idx_j]!=0])
                delta=len(adds_j)-len(drops_j)
                check(len(adds_b)==len(drops_b)==k and delta==row["diagnostics"]["J"]["area_delta"],"Actual sign counts against masks")
                role="J_shrinks" if delta<0 else "J_expands" if delta>0 else "same_composition"
                # Under a shared ranking, the larger same-sign selection must contain the smaller set.
                check((adds_b<=adds_j or adds_j<=adds_b) and (drops_b<=drops_j or drops_j<=drops_b),"Shared within-sign selection nesting")
                if delta==0:
                    check(adds_b==adds_j and drops_b==drops_j,"Same composition gives identical selected masks")
                counts=populations.setdefault(role,Counter())
                counts.update(candidates=1,zero_budget=int(k==0),B_add=len(adds_b),B_drop=len(drops_b),J_add=len(adds_j),J_drop=len(drops_j),
                              B_J_changed_pixels=len(adds_b^adds_j)+len(drops_b^drops_j))
                if k:
                    rles["B"]=bitmap_edit(native,idx_b)
                    rles["J"]=bitmap_edit(native,idx_j) if delta else rles["B"]
                    rles["B_on_J_shrink"]=rles["B"] if delta<0 else rles["J"]
                    rles["B_on_J_expand"]=rles["B"] if delta>0 else rles["J"]
            for arm in allrows:
                allrows[arm].append(dict(image_id=iid,category_id=1,score=q["ship_scores"][query],segmentation=rles[arm]))
            strata.append(role)
    check(seen==records.keys() and len(strata)==10488,"Full unchanged score pool")
    evs={a:evaluate(gt,v) for a,v in allrows.items()}
    for a in ("B","J"):
        check(list(evs[a].stats)==previous["independent_arms"][a]["all12"],"Reproduces published six-arm metric: "+a)
    J,B,S,E=(evs[a].stats[0] for a in ("J","B","B_on_J_shrink","B_on_J_expand"))
    gap=100*(B-J)
    shrink=50*((S-J)+(B-E))
    expand=50*((E-J)+(B-S))
    check(abs(gap-shrink-expand)<1e-10,"Exact two-mask-group decomposition")
    effects=thresholds(evs["B"],evs["J"],strata)
    result=dict(status="FIT109_POSTHOC_B_J_CONFIDENCE_AND_AREA_DIAGNOSTIC",FIT=True,DEV=False,TEST=False,
                provenance=READBACK,diagnostic_source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),images=109,annotations=496,
                mapped_rows=10488,eligible_rows=6738,per_mask_group=populations,native_rank_and_score_identical=True,
                equal_composition_all_edit_sets_identical=True,every_changed_sign_subselection_nested=True,
                four_counterfactual_AP={k:float(v.stats[0]) for k,v in evs.items()},B_minus_J_AP_points=gap,
                two_group_order_averaged_contributions_points=dict(J_shrinks=shrink,J_expands=expand),
                nonadditivity_interaction_points=100*(B+J-S-E),by_iou=effects,
                mean_reachable_grid_contribution_points=float(np.mean([r["reachable_grid_contribution_points"] for r in effects])),
                mean_conditional_precision_contribution_points=float(np.mean([r["conditional_precision_contribution_points"] for r in effects])),
                limits="Post-hoc decomposition within this FIT seed. Reachable-grid and conditional precision terms are arithmetic, not causal effects. Confidence scores never changed. Switching a mask changes direction counts and pixel locations together.")
    with args.output.open("x",encoding="utf-8",newline="\n") as out:
        json.dump(result,out,indent=2,sort_keys=True,allow_nan=False)
        out.write("\n")
    print(json.dumps(dict(status=result["status"],sha256=sha256(args.output.read_bytes()).hexdigest(),area_contributions=result["two_group_order_averaged_contributions_points"])),flush=True)


if __name__=="__main__":
    main()
