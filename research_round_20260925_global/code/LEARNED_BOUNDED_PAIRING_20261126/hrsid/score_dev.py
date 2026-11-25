"""Score a naturally closed DEV705 learned-pair cap replay and verify old metrics."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from pycocotools import mask as codec

sys.path.insert(0,str(Path(__file__).resolve().parent))
from bound_common import (B5, CAPTURE, CAP_TERMINAL, CAPTURE_CODE, REPLAY, REP_TERMINAL,
    REPLAY_CODE, SCORE, SCORE_TERMINAL, SCORE_CODE, MATCH, OLD_EDITS, SPARSE, STOCK, MODES,
    SEEDS, TOTAL, RENDER_STATUS, SCORE_STATUS, require, digest, pin, load, closed,
    verified_lines, write_result)


RENDER_CODE = "c17557fd9eb57e20e36b0674066dd1bffb62b7913ef6bc4339968c4fe7505f1d"
REFERENCE_COUNTS = dict(stock_matches=1664,valid=1461,neutral=15,unmatched=40849,supervised_pairs=46713)
CONTRASTS = (("I_minus_L","SU=F","L"),("L_minus_F","L","SF"),("I_minus_F","SU=F","SF"))
GEOMETRY_FIELDS = ("iou_points","false_positive_pixels","false_negative_pixels")
CTX = None


def score_operators(a):
    import torch

    require(not torch.cuda.is_initialized(),"Scoring stays on CPU")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    source=a.code/B5
    pin(source/"portable_dev705_capture.py",CAPTURE_CODE)
    pin(source/"portable_dev705_replay.py",REPLAY_CODE)
    old=load(source/"portable_dev705_score.py",SCORE_CODE,"_learned_cap_dev_score")
    sparse=load(a.code/"RFDETR_SPARSE_RLE_EQUIVALENCE_20261120/sparse_rle.py",
                SPARSE,"_learned_cap_sparse_xor")
    return old,old.operators(a),sparse


def original_edit(ops,hist,seed,rank,arm):
    edit=ops.score.get_edit(hist,seed,rank,arm)
    flips=edit["flipped"]
    k=edit["adds"]
    require(type(k) is int and 0<=k<=8 and edit["drops"]==k and edit["area_delta"]==0 and
            isinstance(flips,(list,tuple)) and len(flips)==edit["edits"]==2*k and
            all(type(i) is int for i in flips) and len(set(flips))==2*k,
            "Exact archived realized signed budget")
    return list(flips),k


def orientations(coords,rle,sparse):
    runs=np.cumsum(sparse.decode_runs(rle["counts"]),dtype=np.int64)
    h,w=rle["size"]
    require(runs[-1]==h*w,"Native mask covers full source canvas")
    locations=(coords%w)*h+coords//w
    return np.searchsorted(runs,locations,side="right")%2


def bounded_flip(coords,record,fixed,k,width):
    pairs=np.asarray(record["pairs"])
    require(pairs.dtype.kind in "iu" and pairs.shape==(k,2) and
            np.all((0<=pairs)&(pairs<len(coords))) and
            len(set(pairs[:,0]))==len(set(pairs[:,1]))==k,
            "Exact bounded k and unit capacity by signed side")
    target=np.column_stack((coords[pairs[:,0],0],coords[pairs[:,1],1]))
    original=np.asarray(fixed,dtype=np.int64).reshape(k,2)
    require(np.isin(original[:,0],coords[:,0]).all() and np.isin(original[:,1],coords[:,1]).all(),
            "Archived fixed interleaved pair orientation")

    def distance_squared(native_pairs):
        row,column=np.divmod(native_pairs,width)
        return np.square(row[:,0]-row[:,1])+np.square(column[:,0]-column[:,1])

    cap=int(distance_squared(original).max())
    require(type(record["max_fixed_distance_squared"]) is int and
            cap==record["max_fixed_distance_squared"] and np.all(distance_squared(target)<=cap),
            "Every selected edge obeys the maximum squared distance of original selected F edges")
    return target.reshape(-1).tolist()


def read_archive(a,old,ops,sparse,render):
    manifest=json.loads(pin(a.intake/"DEV_IMAGE_ONLY_MANIFEST_PRIVATE.json",old.cap.INPUT_MANIFEST))
    qa=verified_lines(a.capture/old.cap.QUERIES,old.cap.QUERIES_SHA)
    sa=verified_lines(a.capture/old.cap.STOCK,STOCK)
    pa=verified_lines(a.replay/"SPARSE_EDITS_PRIVATE.jsonl",OLD_EDITS)
    ra=verified_lines(a.render/"BOUNDED_PRIVATE.jsonl",render["bounded_stream_sha256"])
    prepared=[]
    totals=Counter()
    audits={s:Counter() for s in SEEDS}
    for ordinal,(item,qline,sline,pline,rline) in enumerate(zip(manifest,qa,sa,pa,ra,strict=True),1):
        query,native,hist,caprec=map(json.loads,(qline,sline,pline,rline))
        image=item["image_id"]
        group=Path(item["file_name"]).name.split("_",1)[0]
        require(image==query["image_id"]==native["image_id"]==hist["image_id"]==caprec["image_id"] and
                ordinal==hist["ordinal"]==caprec["ordinal"] and hist["original_hw"]==[800,800] and
                group==hist["group"]==query["source_prefix"] and
                item["encoded_sha256"]==query["encoded_sha256"] and
                item["decoded_rgb_sha256"]==query["image_content_sha256"] and
                set(caprec["seeds"])==set(hist["seeds"])=={str(s) for s in SEEDS},
                "Exact original image, group, rank order and seed identities")
        pool=ops.spot.native_candidate_rows(query)
        require(ops.readout.legacy_stock_bytes(image,pool)==sline,"Original native archive byte parity")
        for candidate in pool:
            candidate["area"]=int(codec.area(candidate["rle"]))
        eligible={p["rank"]:p for p in pool if 32<=p["area"]<1024}
        require(hist["mapped_identity_sha256"]==ops.readout.mapped_identity_sha256(pool) and
                [r["rank"] for r in hist["points"]]==list(eligible),
                "Original complete candidate and eligible point rosters")
        points={}
        edits={s:{} for s in SEEDS}
        for point in hist["points"]:
            rank=point["rank"]
            candidate=eligible[rank]
            x=np.asarray(point["coordinates"],dtype=np.int32).reshape(-1,2)
            require(point["query_id"]==candidate["query_id"] and len(x)<=32 and
                    np.all((0<=x)&(x<640000)) and np.unique(x).size==x.size,
                    "Original candidate point identities and disjoint valid indices")
            points[rank]=x
            if len(x):
                sign=orientations(x,candidate["rle"],sparse)
                require(not sign[:,0].any() and sign[:,1].all(),"Exact native foreground/background write support")
            for seed in SEEDS:
                f,k=original_edit(ops,hist,seed,rank,"SF")
                i,ki=original_edit(ops,hist,seed,rank,"SU=F")
                row=caprec["seeds"][str(seed)].get(str(rank))
                require(k==ki<=min(8,candidate["area"]//10,len(x)) and (row is not None)==bool(k),
                        "Identical per-candidate realized count and zero-budget native fallback")
                if not k:
                    continue
                require(np.isin(i,x).all() and np.isin(x[:,0],i).sum()==np.isin(x[:,1],i).sum()==k,
                        "Archived independent intervention is exactly k signed native edits")
                l=bounded_flip(x,row,f,k,800)
                edits[seed][rank]=(f,i,l)
                audits[seed].update(positive=1,swaps_per_arm=k,
                                    differs_fixed=int(set(l)!=set(f)),differs_independent=int(set(l)!=set(i)))
        for seed in SEEDS:
            require(set(caprec["seeds"][str(seed)])=={str(r) for r in edits[seed]},
                    "Bounded rows cannot introduce or omit any positive archived budget")
        prepared.append((native,pool,points,group,edits))
        totals.update(images=1,mapped=len(pool),eligible=len(points),supported=sum(bool(len(v)) for v in points.values()))
    require(dict(totals)==dict(images=705,mapped=67386,eligible=42325,supported=40897) and
            len({r[0]["image_id"] for r in prepared})==705 and
            all((r["positive"],r["swaps_per_arm"])==TOTAL[s] and
                all(v==render["census_by_seed"][str(s)][k] for k,v in r.items())
                for s,r in audits.items()),"Exhaustive independent render closure census")
    return prepared,totals


def make_predictions(prepared,seed,arm,sparse):
    predictions=[]
    for native,pool,_,_,edits in prepared:
        for record,candidate in zip(native["coco"],pool,strict=True):
            variants=edits[seed].get(candidate["rank"])
            if not variants or arm=="N":
                predictions.append(record)
                continue
            indices=variants[MODES.index(arm)-1]
            rle=sparse.xor_rle(record["segmentation"],indices)
            require(rle["size"]==record["segmentation"]["size"] and
                    int(codec.area(rle))==candidate["area"] and rle!=record["segmentation"],
                    "Exact reconstructed signed-count mask area; same score, class and image")
            predictions.append(dict(record,segmentation=rle))
    require(len(predictions)==67386,"Unchanged total native confidence-ranked candidate stream")
    return predictions


def init_coco(a,prepared,gt,empty_ids):
    global CTX
    _,ops,sparse=score_operators(a)
    CTX=ops,sparse,prepared,gt,empty_ids


def score_coco(pair):
    seed,arm=pair
    ops,sparse,prepared,gt,empties=CTX
    predictions=make_predictions(prepared,seed,arm,sparse)
    return seed,arm,ops.score.score_official(ops.native,gt,predictions,empties)


def sparse_edit(flips):
    k=len(flips)//2
    return dict(flipped=flips,edits=2*k,adds=k,drops=k,area_delta=0)


def fixed_geometry(ops,prepared,gt):
    annotations=defaultdict(list)
    for ann in gt["annotations"]:
        annotations[ann["image_id"]].append(ann)
    totals=Counter()
    groups=[]
    values=[]
    changed=[]
    match_ids=set()
    for image,record in zip(gt["images"],prepared,strict=True):
        native,pool,points,group,edits=record
        require(image["id"]==native["image_id"] and [image["height"],image["width"]]==[800,800],
                "Original author label order and native canvas")
        sealed=[dict(image_id=image["id"],rank=c["rank"],query_id=c["query_id"],
                     group=group,coordinates=points[c["rank"]]) for c in pool if c["rank"] in points]
        attached,counts=ops.fit.attach_truth(image,annotations[image["id"]],pool,sealed,ops.helpers)
        totals.update(counts)
        require(len(attached)==len(sealed),"Preserved author .50 matches and all original eligible rows")
        for row,original in zip(attached,sealed,strict=True):
            require(all(row[k]==original[k] for k in ("image_id","rank","query_id","group")) and
                    np.array_equal(row["coordinates"],original["coordinates"]),"Original untouched point identity")
            if row["role"]!="valid":
                continue
            key=(row["image_id"],row["matched_gt_id"])
            require(key not in match_ids and key[1] is not None,"Unique original matched nonignored author hull")
            match_ids.add(key)
            groups.append(group)
            hull_results=[]
            hull_change=[]
            for seed in SEEDS:
                variants=edits[seed].get(row["rank"],([],[],[]))
                modes=([],*variants)
                metric=[ops.geometry.candidate_geometry(row,sparse_edit(f)) for f in modes]
                hull_results.append([[100*m["iou"],m["P"]-m["I"],m["T"]-m["I"]] for m in metric])
                hull_change.append([set(modes[MODES.index(new)])!=set(modes[MODES.index(old)])
                                    for _,new,old in CONTRASTS])
            values.append(hull_results)
            changed.append(hull_change)
    unique=sorted(set(groups))
    require(dict(totals)==REFERENCE_COUNTS and len(match_ids)==1461 and len(unique)==27,
            "Original exact continuous polygon match counts and source-prefix groups")
    values=np.asarray(values)
    group_idx=[np.flatnonzero(np.asarray(groups)==group) for group in unique]
    return values,np.asarray(changed),group_idx


def summarize(coco,values,changed,groups):
    draws=np.random.Generator(np.random.PCG64(5217)).integers(len(groups),size=(4096,len(groups)))

    def geometry(g):
        means=np.array([g[idx].mean(axis=0) for idx in groups],dtype=np.float64)
        return {metric:float(means[:,i].mean()) for i,metric in enumerate(GEOMETRY_FIELDS)}

    def paired(values,exposure):
        means=np.array([values[idx].mean(axis=0) for idx in groups],dtype=np.float64)
        samples=means[draws].mean(axis=1)
        return dict(metrics={metric:dict(delta=float(means[:,i].mean()),
                                        interval95=np.quantile(samples[:,i],[.025,.975]).tolist())
                             for i,metric in enumerate(GEOMETRY_FIELDS)},
                    changed_hulls=int(exposure.sum()),
                    changed_groups=sum(bool(exposure[idx].any()) for idx in groups))

    individual={}
    for si,seed in enumerate(SEEDS):
        individual[seed]=dict(arms={arm:geometry(values[:,si,ai]) for ai,arm in enumerate(MODES)},
            contrasts={name:paired(values[:,si,MODES.index(new)]-values[:,si,MODES.index(old)],changed[:,si,ci])
                       for ci,(name,new,old) in enumerate(CONTRASTS)})
    mean_ar={arm:{key:float(np.mean([coco[s][arm]["primary"][key] for s in SEEDS])*100)
                    for key in ("AP","AP50","AP75","AP90","APs","AR100")} for arm in MODES}
    combo={}
    for ci,(name,new,old) in enumerate(CONTRASTS):
        deltas=values[:,:,MODES.index(new)]-values[:,:,MODES.index(old)]
        row=paired(deltas.mean(axis=1),changed[:,:,ci].any(axis=1))
        row["changed_hulls_all_seeds"]=int(changed[:,:,ci].all(axis=1).sum())
        row["AP_points"]=float(np.mean([(coco[s][new]["primary"]["AP"]-coco[s][old]["primary"]["AP"])*100
                                         for s in SEEDS]))
        low,high=row["metrics"]["iou_points"]["interval95"]
        ap=row["AP_points"]
        exposure=row["changed_hulls"]>=20 and row["changed_groups"]>=8
        row["directional_gate"]="positive" if exposure and ap>=.30 and low>0 else (
                                "negative" if exposure and ap<=-.30 and high<0 else "unqualified")
        combo[name]=row
    p=combo["I_minus_L"]
    low,high=p["metrics"]["iou_points"]["interval95"]
    recovery=combo["L_minus_F"]["directional_gate"]=="positive" and abs(p["AP_points"])<=.15 and -.15<=low<=high<=.15
    return individual,mean_ar,combo,recovery


def main():
    parser=ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ("code","source","render","render-terminal","protocol","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    for name in ("render-result-sha256","render-terminal-sha256","protocol-sha256"):
        parser.add_argument("--"+name,required=True)
    parser.add_argument("--workers",type=int,choices=range(1,7),required=True)
    a=parser.parse_args()
    start=perf_counter()
    require(sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES")=="-1"
            and not a.output.exists(),"Fresh isolated score-only CPU process")
    dependencies={"numpy":"1.26.4","torch":"2.10.0+cu130","pycocotools":"2.0.11","shapely":"2.0.2"}
    require(sys.version_info[:3]==(3,10,13) and all(version(k)==v for k,v in dependencies.items()),
            "Exact historical CPU scoring runtime")
    pin(a.protocol,a.protocol_sha256)
    render=json.loads(pin(a.render/"RESULT_SAFE.json",a.render_result_sha256))
    terminal=json.loads(pin(a.render_terminal,a.render_terminal_sha256))
    require(terminal["actual_exit"]==0 and terminal["phase"]=="replay" and terminal["output_footer_verified"] is True and
            terminal["result_sha256"]==a.render_result_sha256 and terminal["worker_sha256"]==render["worker_sha256"]==RENDER_CODE and
            render["status"]==RENDER_STATUS and render["protocol_sha256"]==a.protocol_sha256 and
            render["matching_worker_sha256"]==MATCH and render["prior_capture_sha256"]==CAPTURE and
            render["prior_replay_sha256"]==REPLAY and render["original_edits_sha256"]==OLD_EDITS and
            render["exact_native_coordinates"] is render["exact_old_selected_mask_indices_and_probability_gains"] is True and
            render["labels_opened"] is False and render["detector_forwards"]==render["optimizer_updates"]==0,
            "Exact natural observed label-free replay must qualify before any author reference read")
    closed(a.source,"capture",CAPTURE,CAP_TERMINAL,CAPTURE_CODE)
    closed(a.source,"replay",REPLAY,REP_TERMINAL,REPLAY_CODE)
    reference=closed(a.source,"score",SCORE,SCORE_TERMINAL,SCORE_CODE)
    a.capture,a.replay,a.intake=(a.source/k for k in ("capture","replay","intake"))
    old,ops,sparse=score_operators(a)
    old.admission(a,ops)
    a.output.mkdir()
    prepared,totals=read_archive(a,old,ops,sparse,render)
    prelabel=dict(status="DEV705_LEARNED_CAP_PRESCORE_QUALIFIED",render_result_sha256=a.render_result_sha256,
                  render_terminal_sha256=a.render_terminal_sha256,native_scores_masks_ranks_exact=True,
                  signed_budget_and_cap_all_rows_exact=True,labels_opened=False,totals=dict(totals))
    write_result(a.output/"PRELABEL_SAFE.json",prelabel)
    gt=ops.native.dev_file("DEV_ORIGINAL_GT_PRIVATE.json")
    empty_rows=ops.native.dev_file("DEV_EMPTY_RASTERS_PRIVATE.json")
    empties={r["annotation_id"] for r in empty_rows if r["kind"]=="EMPTY_COCO_RASTER"}
    require(len(empties)==len(empty_rows)==4 and len(gt["annotations"])==len({v["id"] for v in gt["annotations"]})==1782
            and [c["id"] for c in gt["categories"]]==[1] and
            [r["id"] for r in gt["images"]]==[r[0]["image_id"] for r in prepared],
            "Exact archived historical HRSID DEV-only author reference")
    coco={s:{} for s in SEEDS}
    jobs=[(SEEDS[0],"N")]+[(s,arm) for s in SEEDS for arm in MODES if arm!="N"]
    with ProcessPoolExecutor(max_workers=a.workers,initializer=init_coco,initargs=(a,prepared,gt,empties)) as pool:
        tasks={pool.submit(score_coco,job):job for job in jobs}
        values,changed,groups=fixed_geometry(ops,prepared,gt)
        for future in tasks:
            seed,arm,result=future.result()
            coco[seed][arm]=result
            print(json.dumps(dict(event="COCO_SCORE_COMPLETE",head_seed=seed,arm=arm)),flush=True)
    for s in SEEDS:
        coco[s]["N"]=coco[SEEDS[0]]["N"]
    per_seed,means,contrasts,recovery=summarize(coco,values,changed,groups)
    for seed in SEEDS:
        for arm in ("N","SF","SU=F"):
            require(coco[seed][arm]==reference["coco"][str(seed)][arm],"Exact prior primary and sensitivity COCO parity")
            past=reference["geometry_summary"]["per_seed"][str(seed)]["arms"][arm]["group_equal_iou_points"]
            require(abs(per_seed[seed]["arms"][arm]["iou_points"]-past)<=1e-12,"Exact prior fixed-match geometry parity")
    result=dict(status=SCORE_STATUS,phase="score",worker_sha256=digest(Path(__file__).read_bytes()),
        protocol_sha256=a.protocol_sha256,render_result_sha256=a.render_result_sha256,
        render_terminal_sha256=a.render_terminal_sha256,original_score_sha256=SCORE,
        render_census=render["census_by_seed"],native_scores_masks_ranks_exact=True,
        exact_old_coco_all_metrics_and_geometry=True,signed_budget_and_cap_all_rows_exact=True,
        seeds=list(SEEDS),arms=list(MODES),totals=dict(totals),primary_annotations=1782,sensitivity_annotations=1778,
        fixed_matching_counts=REFERENCE_COUNTS,fixed_matching_groups=len(groups),bootstrap_draws=4096,
        bootstrap_seed=5217,bootstrap_unit="source prefix; common seeds averaged on each fixed hull",
        coco=coco,mean_primary_coco_points=means,per_seed_fixed_geometry=per_seed,
        contrasts=contrasts,practical_recovery=recovery,DEV=True,FIT=False,TEST=False,
        optimizer_updates=0,detector_forwards=0,elapsed_seconds=perf_counter()-start)
    write_result(a.output/"RESULT_SAFE.json",result)


if __name__=="__main__":
    main()
