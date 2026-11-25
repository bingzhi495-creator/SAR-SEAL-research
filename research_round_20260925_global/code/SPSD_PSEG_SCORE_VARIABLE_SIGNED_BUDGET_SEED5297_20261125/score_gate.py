"""Score archived and variable signed-mask edit counts after exact geometric readback."""

from argparse import ArgumentParser
from collections import Counter, defaultdict
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from time import perf_counter
from types import ModuleType

import numpy as np
from pycocotools import mask as codec
from scipy.optimize import linear_sum_assignment


LOCK = "5866e98058edf8728dfb072062ad5d2ca1c8ef6b2f17609e9c54854b0182ad89"
MECHANISM = "d8b13f9c740e61732cf32173daf10ce94ee8433ab97f1a08c3f1dfc133ce6f66"
GROUP = "4030a3cb5684a6dac0ea1e7765a48d05caec113c5c6e0e6de2c82b991e9d5f15"
ARCH_AUTHORITY = "25a0837e5147e95421483d73b6e3146e28f8f95059c922e21514dc9e36f3a0e3"
CUES = ("fixed_iou","variable_iou","variable_mean_iou","variable_raw_min","near_iou")
REFERENCES = ("pseg","spsd")
CONTROLS = ("fixed_iou","bce_robust","bce_mean","bce_cell_pessimistic","static",
            "variable_mean_iou","variable_raw_min","near_iou")
ARCH_RENDER = "7aab5753eafe1e18464d0740a6b440fe09997dca2020c64a0d48180c228728fd"
ARCH_SCORE = "b93e0f4d355f538ae751b8622bf359541dd5705f25d1dfe6eca836984521b052"
ARCH_BCE = "a2c169f895b1435e689480836ebf4f04fd6e05d84652f51d74e0448ed099faff"


def need(ok,reason):
    if not ok:
        raise ValueError(reason)


def digest(raw):
    return sha256(raw).hexdigest()


def pin(path,expected):
    raw=path.read_bytes()
    need(digest(raw)==expected,"Exact immutable artifact: "+path.name)
    return raw


def load(path,checksum):
    module=ModuleType(path.stem)
    module.__file__=str(path)
    exec(compile(pin(path,checksum),str(path),"exec"),module.__dict__)
    return module


def read_receipt(a, old, base):
    pin(a.render_root/"render_gate.py",a.renderer_sha)
    pin(a.render_root/"PRELOCK.md",a.renderer_lock_sha)
    raw=pin(a.render_root/"RENDER_TERMINAL.json",a.renderer_terminal_sha)
    term=json.loads(raw)
    need(term["phase"]=="render" and term["actual_exit"]==0 and term["output_footer_verified"] is True
         and term["worker_sha256"]==a.renderer_sha and term["result_sha256"]==a.render_sha,
         "Naturally closed label-free render and exact immutable process identity")
    result=json.loads(pin(a.render_root/"render/RESULT_SAFE.json",a.render_sha))
    need(result["status"]=="SPSD_PSEG_VARIABLE_SIGNED_BUDGET_GATE_RENDER_COMPLETE"
         and result["worker_sha256"]==a.renderer_sha and result["protocol_sha256"]==a.renderer_lock_sha
         and result["archive_render_sha256"]==ARCH_RENDER and result["archive_score_sha256"]==ARCH_SCORE
         and result["archive_bce_tensor_sha256"]==ARCH_BCE and result["exact_bce_archive_rle_parity"] is True
         and result["point_terminal_sha256"]==base["point_terminal_sha256"]
         and result["detector_forwards"]==result["optimizer_updates"]==0 and result["decoded_groups"]==180
         and result["source_gate_labels_opened"] is result["fit_labels_opened"] is result["historical_test_opened"] is False,
         "Exact archived BCE driver and no test, labels, fitting or detector execution")
    point=json.loads(pin(a.points/"actual_retry/RESULT_SAFE.json",old.POINTS))
    need(point["worker_sha256"]==old.POINT_WORKER and
         point["train_labels_opened"] is point["test_labels_opened"] is point["test_images_opened"] is False,
         "Previously admitted original gate support")
    return result, point


def signed_support(a, point, replay, archived, groups, result, archived_arms, mechanism):
    root=a.points/"actual_retry/source_gate"
    idx=[json.loads(row) for row in pin(root/"POINT_INDEX_PRIVATE.jsonl",
           point["partitions"]["source_gate"]["point_index_sha256"]).splitlines()]
    need(result["point_index_sha256"]==point["partitions"]["source_gate"]["point_index_sha256"],
         "Exact same label-free original point-index lineage")
    need(len(idx)==len(replay)==len(archived)==len(groups)==183 and set(replay)==set(archived)==set(groups)
         and set(groups)=={r["image_id"] for r in idx},"Exact original 183-image roster")
    totals=Counter()
    arm_stats={arm:Counter() for arm in CUES}
    net_area={arm:Counter() for arm in CUES}
    for item in idx:
        iid=item["image_id"]
        image=replay[iid]
        source=archived[iid]
        need(image["source_group"]==source["source_group"]==item["source_group"]==groups[iid]
             and image["original_hw"]==source["original_hw"]==item["original_hw"]
             and image["filename"]==source["filename"] and len(image["candidates"])==len(source["candidates"])==100,
             "Exact original native image and 100 ranked slots")
        raw=pin(root/item["filename"],item["sha256"])
        need(len(raw)==item["bytes"],"Exact original point packet")
        with np.load(BytesIO(raw),allow_pickle=False) as packet:
            ranks,offset,coords,quotas=(packet[k] for k in ("native_ranks","point_offsets",
                                      "signed_coordinates","static_offsets"))
        n=len(ranks)
        need(n==len(set(ranks))==item["supported_candidates"] and np.all((ranks>=1)&(ranks<=100))
             and offset.shape==quotas.shape==(n+1,) and offset[0]==quotas[0]==0 and offset[-1]==len(coords)
             and len(coords)==item["paired_sites"] and quotas[-1]==item["realized_static_exchanges"],
             "Exact original support and per-candidate positive quota")
        span={int(rank):(coords[offset[j]:offset[j+1]],int(quotas[j+1]-quotas[j])) for j,rank in enumerate(ranks)}
        for rank,(p,q) in enumerate(zip(image["candidates"],source["candidates"],strict=True),1):
            need(all(key in p and p[key]==v for key,v in q.items() if key!="changes"),
                 "Exact native mask, score, quota and archived margins for each original slot")
            k=p["quota"]
            if not k:
                need(p["changes"]==q["changes"]=={} and not p.get("variable_signed_counts")
                     and (rank not in span or span[rank][1]==0),"Native zero-quota pass-through")
                continue
            y,k_old=span[rank]
            new=p["changes"]
            archived_changes=q["changes"]
            need(k==k_old and set(new)==set(archived_arms[1:])|set(CUES)
                 and all(new[arm]==value for arm,value in archived_changes.items())
                 and set(p["variable_signed_counts"])==set(p["variable_predicted_iou_gain"])
                 ==set(p["variable_predicted_intersection_gain"])==set(CUES),
                 "Frozen original dynamic/static/BCE/action masks and complete new finite writers")
            need(isinstance(p["variable_menu_actions"],int) and p["variable_menu_actions"]>0,
                 "Original positive candidate has nonempty label-free legal menu")
            old=codec.decode(p["native"]).astype(bool,order="C")
            size=int(codec.area(p["native"]))
            pred={arm:np.asarray(estimate) for arm,estimate in p["variable_predicted_iou_gain"].items()}
            raw={arm:np.asarray(estimate) for arm,estimate in p["variable_predicted_intersection_gain"].items()}
            need(all(v.shape==(2,) and np.isfinite(v).all() and np.max(np.abs(v))<=1 for v in pred.values())
                 and raw["variable_iou"].min()<=raw["variable_raw_min"].min()+1e-12
                 and pred["fixed_iou"].min()<=pred["near_iou"].min()+1e-12
                 and pred["near_iou"].min()<=pred["variable_iou"].min()+1e-12
                 and pred["variable_iou"].mean()<=pred["variable_mean_iou"].mean()+1e-12,
                 "Prespecified nested support and held estimated robust/mean objectives")
            static=archived_changes["static"]
            static_area=int(codec.area(static))
            for arm in CUES:
                rle=new[arm]
                need(rle["size"]==p["native"]["size"],"Original per-mask raster frame")
                corrected=codec.decode(rle).astype(bool,order="C")
                a,d=mechanism.signed_changes(old,corrected,y,k)
                mechanism.validate_pairwise_gain(raw[arm],int(codec.area(rle)),static_area,overlap(rle,static))
                need([a,d]==p["variable_signed_counts"][arm] and int(codec.area(rle))==size+a-d,
                     "Exact independently read back actual signed count and native size change")
                need((arm!="fixed_iou" or a==d==k) and (arm!="near_iou" or abs(a-k)<=1),
                     "Fixed and near signed-budget design restrictions")
                if arm in ("variable_iou","near_iou") and a==d:
                    need(new["fixed_iou"]==rle,"Equal split retains the exact fixed-area action")
                net_area[arm][str(a-d)]+=1
                arm_stats[arm].update(adds=a,drops=d,total=a+d,grew=int(a>d),shrank=int(a<d),
                                     unchanged_area=int(a==d),same_as_fixed=int(rle==new["fixed_iou"]),
                                     absolute_area_change=abs(a-d))
            totals.update(positive=1,exchanges=k)
        totals.update(images=1,candidates=100,supported=n)
    need(dict(totals)==dict(images=183,candidates=18300,supported=8821,positive=8785,exchanges=57692)
         and all(result["counts"][key]==v for key,v in totals.items())
         and set(result["arm_totals"])==set(CUES)
         and all(v["total"]==115384 and all(v[q]==result["arm_totals"][arm][q] for q in ("adds","drops","total"))
                 for arm,v in arm_stats.items()),"Exact source and full independent signed total parity")
    need({arm:dict(counts) for arm,counts in net_area.items()}==result["net_area_change_histograms"],
         "Independently exact signed area histogram of every new writer")
    return arm_stats,net_area


def contrast(arms, values, metrics, draws, main, rival):
    d=values[:,arms.index(main)]-values[:,arms.index(rival)]
    return dict(iou_points=float(d.mean()),percentile95_points=np.percentile(d[draws].mean(axis=1),[2.5,97.5]).tolist(),
                ap_points=metrics[main]["AP"]-metrics[rival]["AP"])


def overlap(mask, truth):
    return int(codec.area(codec.merge([mask,truth],intersect=True)))


def decomposition(truth,native,matches,groups,order,rival,draws,mech):
    by_image=defaultdict(list)
    counts=Counter()
    for hull in truth["annotations"]:
        image=hull["image_id"]
        association=matches.get(hull["id"])
        if association is None:
            counts["native_misses"]+=1
            by_image[image].append(np.zeros(6))
            continue
        need(association[0]==image,"Original native association owner")
        p=native[image][association[1]-1]
        g=hull["segmentation"]
        old=p["changes"].get(rival,p["native"])
        new=p["changes"].get("variable_iou",p["native"])
        m0,m1,area=(int(codec.area(rle)) for rle in (old,new,g))
        i0,i1=(overlap(mask,g) for mask in (old,new))
        terms=mech.iou_terms(m0,i0,m1,i1,area)
        by_image[image].append(np.asarray((*[q*100 for q in terms], m1-m0, i1-i0, abs(m1-area)-abs(m0-area))))
        counts.update(native_matches=1, changed_masks=int(old!=new), shrank=int(m1<m0), grew=int(m1>m0),
                      unchanged_area=int(m1==m0), intersection_increased=int(i1>i0),
                      intersection_decreased=int(i1<i0),intersection_unchanged=int(i1==i0))
    boxes=defaultdict(list)
    for image,tag in groups.items():
        v=by_image[image]
        boxes[tag].append(np.mean(v,axis=0) if v else np.zeros(6))
    tab=np.stack([np.mean(boxes[g],axis=0) for g in order])
    ci=np.percentile(tab[draws].mean(axis=1),[2.5,97.5],axis=0)
    keys=("total_iou_points","overlap_iou_points","area_iou_points","mask_area_cells",
          "intersection_cells","absolute_area_error_cells")
    return tab[:,0],dict(counts=counts,group_means=dict(zip(keys,tab.mean(axis=0).tolist(),strict=True)),
                percentile95=dict(zip(keys,ci.T.tolist(),strict=True)))


def matched_pairs(truths,matches,native,groups):
    source=defaultdict(lambda:[[],[]])
    for j,ref in enumerate(REFERENCES):
        for a in truths[ref]["annotations"]:
            source[a["image_id"]][j].append(a)
    totals=Counter()
    changed={arm:Counter() for arm in CONTROLS}
    coverage={arm:set() for arm in CONTROLS}
    mechanism={arm:Counter() for arm in ("fixed_iou","bce_robust")}
    for iid,g in groups.items():
        p,s=source[iid]
        if not p or not s:
            continue
        left,right=[[v["segmentation"] for v in a] for a in (p,s)]
        pair=codec.iou(left,right,[0]*len(right))
        ix,iy=linear_sum_assignment(pair,maximize=True)
        totals["reference_pairs"]+=len(ix)
        for j,k in zip(ix,iy,strict=True):
            if pair[j,k]<.5:
                continue
            totals["reference_pairs_ge050"]+=1
            x,y=matches["pseg"].get(p[j]["id"]),matches["spsd"].get(s[k]["id"])
            if x is None or x!=y:
                continue
            totals["shared_native_matches"]+=1
            c=native[iid][x[1]-1]
            if not c["quota"]:
                continue
            totals["positive_common_quota"]+=1
            masks=c["changes"]
            new=masks["variable_iou"]
            for arm in CONTROLS:
                old=masks[arm]
                if old!=new:
                    changed[arm]["hulls"]+=1
                    coverage[arm].add(g)
                if arm in mechanism:
                    delta=int(codec.area(new))-int(codec.area(old))
                    gains=[overlap(new,ref)-overlap(old,ref) for ref in (left[j],right[k])]
                    mechanism[arm].update(shrink=int(delta<0),grow=int(delta>0),no_area=int(delta==0),
                         minimum_intersection_gain_positive=int(min(gains)>0),
                         minimum_intersection_gain_negative=int(min(gains)<0),
                         minimum_intersection_gain_zero=int(min(gains)==0),
                         area_delta_cells=delta,pseg_intersection_delta_cells=gains[0],spsd_intersection_delta_cells=gains[1])
    need(tuple(totals[k] for k in ("reference_pairs","reference_pairs_ge050","shared_native_matches","positive_common_quota"))
         ==(410,370,276,206),"Exact previously defined common native association geometry")
    report={arm:dict(hulls=changed[arm]["hulls"],groups=len(coverage[arm])) for arm in CONTROLS}
    exposure={arm:(v["hulls"],v["groups"]) for arm,v in report.items()}
    return exposure,dict(counts=totals,exposure=report,paired_reference_signed_facts=mechanism)


def main():
    parser=ArgumentParser(description=__doc__)
    for k in ("gate","archive","authority","render-root","points","labels","protocol","group-protocol","output"):
        parser.add_argument("--"+k,type=Path,required=True)
    for k in ("renderer-sha","renderer-lock-sha","renderer-terminal-sha","render-sha"):
        parser.add_argument("--"+k,required=True)
    a=parser.parse_args()
    start=perf_counter()
    pin(a.protocol,LOCK)
    pin(a.group_protocol,GROUP)
    mech=load(Path(__file__).with_name("mechanism.py"),MECHANISM)
    auth=load(a.authority,ARCH_AUTHORITY)
    old,_,scorer,reader,base,archive,groups,previous,archive_term=auth.archive(a)
    result,point=read_receipt(a,old,base)
    rows=[json.loads(line) for line in pin(a.render_root/"render/GATE_RENDER_PRIVATE.jsonl",result["render_stream_sha256"]).splitlines()]
    records={item["image_id"]:item for item in rows}
    need(len(rows)==len(records)==183,"Complete independent prelabel replay cardinality")
    arm_stats,net_area=signed_support(a,point,records,archive,groups,result,auth.REFERENCE_ARMS,mech)
    print("Variable edit totals and every signed endpoint verified on the frozen archive before labels.",flush=True)
    truths=reader.labels(a,records)
    native={i:r["candidates"] for i,r in records.items()}
    arms=(*auth.REFERENCE_ARMS,*CUES)
    order=sorted(set(groups.values()))
    draws=np.random.Generator(np.random.PCG64(5297)).integers(180,size=(4096,180))
    refs,matches,parts={},{},{}
    for key,truth in truths.items():
        metrics={}
        prior=previous["references"][key]
        for arm in arms:
            metrics[arm],m=scorer.coco_score(truth,native,arm)
            if arm in auth.REFERENCE_ARMS:
                need(all(abs(metrics[arm][field]-v)<=1e-12 for field,v in prior["metrics"][arm].items()),
                     "All exact original archived COCO values")
            if arm=="native":
                matches[key]=m
                need(len(m)==prior["original_matches"],"Exact original native assignment")
        values=scorer.geometry(arms,truth,native,matches[key],groups,order)
        fixed=dict(zip(arms,values.mean(axis=0).tolist(),strict=True))
        need(all(abs(fixed[arm]-prior["fixed_iou_points"][arm])<=1e-12 for arm in auth.REFERENCE_ARMS),
             "All exact original archived fixed-match IoU values")
        for arm in ("fixed_iou","bce_robust"):
            composite,metrics_decomp=decomposition(truth,native,matches[key],groups,order,arm,draws,mech)
            need(np.allclose(composite,values[:,arms.index("variable_iou")]-values[:,arms.index(arm)],rtol=0,atol=1e-12),
                 "Exact additive overlap/area decomposition equals ordinary group IoU")
            parts.setdefault(key,{})[arm]=metrics_decomp
        compare={arm:contrast(arms,values,metrics,draws,"variable_iou",arm) for arm in arms if arm!="variable_iou"}
        refs[key]=dict(metrics=metrics,fixed_iou_points=fixed,variable_iou_minus=compare,
                       original_matches=len(matches[key]),author_hulls=len(truth["annotations"]))
        print("Scored all variable and archived methods under",key,flush=True)
    exposed,shared=matched_pairs(truths,matches,native,groups)
    doc=dict(status="SPSD_PSEG_VARIABLE_SIGNED_BUDGET_GATE_SCORE_COMPLETE",protocol_sha256=LOCK,mechanism_sha256=MECHANISM,
             worker_sha256=digest(Path(__file__).read_bytes()),render_result_sha256=a.render_sha,
             render_terminal_sha256=a.renderer_terminal_sha,render_stream_sha256=result["render_stream_sha256"],
             render_protocol_sha256=a.renderer_lock_sha,archive_render_sha256=ARCH_RENDER,archive_score_sha256=ARCH_SCORE,
             archive_score_terminal_sha256=archive_term,archive_bce_tensor_sha256=ARCH_BCE,
             exact_all_archive_score_parity=True,exact_original_support_and_actual_total_budget=True,
             cases=result["counts"],realized_signed_usage=arm_stats,net_area_change_histograms=net_area,
             decoded_groups=180,bootstrap_draws=4096,seed=5297,
             references=refs,shared_hull_diagnostic=shared,geometry_decomposition=parts,primary=mech.gate(refs,exposed),
             fit_labels_opened=False,historical_test_opened=False,optimizer_updates=0,detector_forwards=0,
             repeated_source_development=True,external_acquisition_evaluation=False,elapsed_seconds=perf_counter()-start)
    a.output.mkdir()
    blob=(json.dumps(doc,indent=2,sort_keys=True,allow_nan=False)+"\n").encode()
    with (a.output/"RESULT_SAFE.json").open("xb") as stream:
        stream.write(blob)
    print(json.dumps(dict(status=doc["status"],result_sha256=digest(blob))),flush=True)


if __name__=="__main__":
    main()
