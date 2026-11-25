"""Exact author dual-reference scoring after closure of the label-free selective replay."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from types import ModuleType

import numpy as np
from pycocotools import mask as codec
from scipy.optimize import linear_sum_assignment


ARMS_MECHANISM = ("static","bce_robust","margin_ranked","selective")
REFERENCES = ("pseg","spsd")


def need(ok,reason):
    if not ok:
        raise ValueError(reason)


def seal(path, checksum):
    raw=path.read_bytes()
    need(sha256(raw).hexdigest()==checksum,"Exact reviewed artifact: "+path.name)
    return raw


def accepted(a,root):
    path=root/"render_gate.py"
    renderer=ModuleType(path.stem)
    renderer.__file__=str(path)
    exec(compile(seal(path,a.renderer_sha),str(path),"exec"),renderer.__dict__)
    seal(a.protocol,renderer.LOCK)
    seal(a.group_protocol,renderer.GROUP)
    term_raw=(root/"RENDER_TERMINAL.json").read_bytes()
    term=json.loads(term_raw)
    need(term["phase"]=="render" and term["actual_exit"]==0 and term["output_footer_verified"] is True
         and term["worker_sha256"]==a.renderer_sha,"Observed natural prelabel renderer closure")
    result=json.loads(seal(root/"render/RESULT_SAFE.json",term["result_sha256"]))
    need(result["status"]=="SPSD_PSEG_SELECTIVE_BCE_GATE_RENDER_COMPLETE"
         and result["worker_sha256"]==a.renderer_sha and result["protocol_sha256"]==renderer.LOCK
         and result["policy_sha256"]==renderer.POLICY and result["fit_result_sha256"]==a.fit_sha
         and result["fit_readback_sha256"]==a.fit_readback_sha and result["fit_helper_sha256"]==a.helper_sha
         and result["archive_score_sha256"]==renderer.ARCH_SCORE and result["archive_render_sha256"]==renderer.ARCH_RENDER
         and result["archive_bce_tensor_sha256"]==renderer.ARCH_BCE and result["exact_bce_archive_rle_parity"] is True
         and result["detector_forwards"]==result["optimizer_updates"]==0
         and result["source_gate_labels_opened"] is result["fit_labels_opened"] is result["historical_test_opened"] is False,
         "Exact prelabel replay, closure, original archive and fit scope")
    _,_,old_scorer,reader,_,archive_rows,groups,previous,archive_term=renderer.archive(a)
    need(result["archive_score_terminal_sha256"]==archive_term,"Same exact older authority at scoring")
    policy=renderer.load(root/"policy.py",renderer.POLICY)
    replay=[json.loads(line) for line in seal(root/"render/GATE_RENDER_PRIVATE.jsonl",
                    result["render_stream_sha256"]).splitlines()]
    need(len(replay)==183 and len({x["image_id"] for x in replay})==len(groups)==183,"All exact gate images retained")
    counts,quota_hist,records=Counter(),Counter(),{}
    for row in replay:
        iid=row["image_id"]
        expected=archive_rows[iid]
        need(all(row[k]==expected[k] for k in ("source_group","filename","original_hw"))
             and len(row["candidates"])==100,"Exact previous order, geometry and candidate count")
        scores=row["eligible_scores"]
        elig={r:p for r,p in enumerate(expected["candidates"],1) if renderer.eligible(p)}
        need(len(scores)==len(elig) and set(elig)=={s[0] for s in scores},"Complete and exact original proposal eligibility")
        for rank,k,learned,margin in scores:
            p=elig[rank]
            need(p["quota"]==k and np.isfinite(learned) and margin==min(p["margin_estimates"]["bce"]["bce"])/(2*k),
                 "Same old exact BCE posterior margin and rank")
        chosen,quotas=renderer.selection(policy,scores)
        quota_hist.update(quotas)
        counts.update(images=1,candidates=100,eligible=len(scores),adopted=len(chosen["selective"]),
                      adopted_overlap=len(chosen["selective"]&chosen["margin_ranked"]),
                      images_with_adoptions=int(bool(chosen["selective"])))
        for rank,(p,q) in enumerate(zip(row["candidates"],expected["candidates"],strict=True),1):
            k=p["quota"]
            need(set(p)==set(q) and all(p[t]==q[t] for t in p if t!="changes"),
                 "Exact original native mask, score, quota and posterior margins")
            if not k:
                need(p["changes"]==q["changes"]=={},"Original zero-quota native passthrough")
                continue
            need(set(p["changes"])==set(renderer.ARMS[1:]) and
                 all(p["changes"][arm]==q["changes"][arm] for arm in renderer.REFERENCE_ARMS[1:]),
                 "Byte-identical original RLEs for every archive arm")
            area=int(codec.area(p["native"]))
            for arm,members in chosen.items():
                rle=p["changes"][arm]
                owner="bce_robust" if rank in members else "static"
                need(rle==q["changes"][owner] and int(codec.area(rle))==area
                     and area-int(codec.area(codec.merge([p["native"],rle],intersect=True)))==k,
                     "Exact fixed selection and actual signed native area for each renderer")
            counts.update(positive=1,exchanges=k)
        records[iid]=row
    need(dict(counts)==result["counts"] and quota_hist==Counter({int(k):n for k,n in result["adopted_same_quota_histogram"].items()})
         and (counts["positive"],counts["exchanges"],counts["candidates"])==(8785,57692,18300),
         "Independent exact policy replay totals and per-image quota structure")
    print("Exact archived BCE masks, same image adoption counts and original signed quotas verified before labels.",flush=True)
    return renderer,reader,old_scorer,policy,result,term["result_sha256"],sha256(term_raw).hexdigest(),records,groups,previous


def contrast(arms, table, metrics, draws, first, second):
    delta=table[:,arms.index(first)]-table[:,arms.index(second)]
    return dict(iou_points=float(delta.mean()),percentile95_points=np.percentile(delta[draws].mean(axis=1),[2.5,97.5]).tolist(),
                ap_points=metrics[first]["AP"]-metrics[second]["AP"])


def by_group(rows, key):
    groups=defaultdict(lambda:defaultdict(list))
    for iid,group,raw,normalized in rows:
        groups[group][iid].append(raw if key=="raw" else normalized)
    return np.stack([np.mean([np.mean(v,axis=0) for v in groups[g].values()],axis=0) for g in sorted(groups)])


def risk(truths, matching, native, groups):
    by_image=defaultdict(lambda:[[],[]])
    for j,ref in enumerate(REFERENCES):
        for item in truths[ref]["annotations"]:
            by_image[item["image_id"]][j].append(item)
    counts=Counter()
    comparison=("bce_robust","margin_ranked","bce_mean","static")
    exposure={arm:set() for arm in comparison}
    exposure_count=Counter()
    harms,zeros,benefits,adopts=(Counter() for _ in range(4))
    rows=[]
    for iid,g in groups.items():
        left,right=by_image[iid]
        if not left or not right:
            continue
        p,s=[[entry["segmentation"] for entry in cohort] for cohort in (left,right)]
        iou=codec.iou(p,s,[0]*len(s))
        ia,ib=linear_sum_assignment(iou,maximize=True)
        counts["reference_pairs"]+=len(ia)
        for l,r in zip(ia,ib,strict=True):
            if iou[l,r]<.5:
                continue
            counts["reference_pairs_ge050"]+=1
            a=matching["pseg"].get(left[l]["id"])
            b=matching["spsd"].get(right[r]["id"])
            if a is None or a!=b:
                continue
            counts["shared_native_matches"]+=1
            row=native[iid][a[1]-1]
            k=row["quota"]
            if not k:
                continue
            counts["positive_common_quota"]+=1
            masks=row["changes"]
            for arm in comparison:
                if masks["selective"]!=masks[arm]:
                    exposure[arm].add(g)
                    exposure_count[arm]+=1
            if masks["bce_robust"]==masks["static"]:
                continue
            counts["eligible_common_quota"]+=1
            baseline,robust=masks["static"],masks["bce_robust"]
            gains=[int(codec.area(codec.merge([robust,y],intersect=True)))-
                   int(codec.area(codec.merge([baseline,y],intersect=True))) for y in (p[l],s[r])]
            min_gain=min(gains)
            adopted=[masks[arm]==robust for arm in ARMS_MECHANISM]
            values=np.asarray(adopted,dtype=np.int64)*min_gain
            rows.append((iid,g,values,values/(2*k)))
            for arm,t in zip(ARMS_MECHANISM,adopted,strict=True):
                adopts[arm]+=int(t)
                harms[arm]+=int(t and min_gain<0)
                zeros[arm]+=int(t and min_gain==0)
                benefits[arm]+=int(t and min_gain>0)
    need([counts[x] for x in ("reference_pairs","reference_pairs_ge050","shared_native_matches","positive_common_quota")]
         ==[410,370,276,206] and rows,"Original both-reference target hull cohort")
    coverage={arm:(exposure_count[arm],len(exposure[arm])) for arm in comparison}
    size=len({g for _,g,_,_ in rows})
    draws=np.random.Generator(np.random.PCG64(5297)).integers(size,size=(4096,size))
    means={}
    for key in ("raw","normalized"):
        table=by_group(rows,key)
        means[key]=dict(group_means=dict(zip(ARMS_MECHANISM,table.mean(axis=0).tolist(),strict=True)),
                        selective_minus={})
        for j,arm in enumerate(ARMS_MECHANISM[:-1]):
            delta=table[:,-1]-table[:,j]
            means[key]["selective_minus"][arm]=dict(value=float(delta.mean()),
                 percentile95=np.percentile(delta[draws].mean(axis=1),[2.5,97.5]).tolist())
    stats=dict(counts,eligible_groups=size,eligible_images=len({i for i,_,_,_ in rows}),
         changed_common_hulls={arm:dict(hulls=n,groups=g) for arm,(n,g) in coverage.items()},
         adopted=adopts,harmful_adoptions=harms,zero_gain_adoptions=zeros,beneficial_adoptions=benefits,estimates=means)
    return coverage,stats


def main():
    parser=ArgumentParser(description=__doc__)
    for x in ("gate","archive","labels","protocol","group-protocol","output"):
        parser.add_argument("--"+x,type=Path,required=True)
    for x in ("fit-sha","fit-readback-sha","helper-sha","renderer-sha"):
        parser.add_argument("--"+x,required=True)
    a=parser.parse_args()
    start=perf_counter()
    root=Path(__file__).parent
    renderer,reader,old,policy,proof,render_sha,terminal_sha,records,groups,previous=accepted(a,root)
    truths=reader.labels(a,records)
    native={i:r["candidates"] for i,r in records.items()}
    order=sorted(set(groups.values()))
    draws=np.random.Generator(np.random.PCG64(5297)).integers(180,size=(4096,180))
    refs,matches={},{}
    for key,truth in truths.items():
        metrics={}
        prior=previous["references"][key]
        for name in renderer.ARMS:
            metrics[name],m=old.coco_score(truth,native,name)
            if name in renderer.REFERENCE_ARMS:
                need(all(abs(metrics[name][k]-v)<=1e-12 for k,v in prior["metrics"][name].items()),
                     "Exact all old native and learned archived metrics")
            if name=="native":
                matches[key]=m
                need(len(m)==prior["original_matches"],"Exact original native assignment count")
        values=old.geometry(renderer.ARMS,truth,native,matches[key],groups,order)
        fixed=dict(zip(renderer.ARMS,values.mean(axis=0).tolist(),strict=True))
        need(all(abs(fixed[name]-prior["fixed_iou_points"][name])<=1e-12 for name in renderer.REFERENCE_ARMS),
             "Exact all original fixed-IoU geometry")
        compare={arm:contrast(renderer.ARMS,values,metrics,draws,"selective",arm)
                 for arm in renderer.ARMS if arm!="selective"}
        refs[key]=dict(metrics=metrics,fixed_iou_points=fixed,selective_minus=compare,
                       original_matches=len(matches[key]),author_hulls=len(truth["annotations"]))
        print("Scored exact controls and selective policies under",key,flush=True)
    coverage,risk_report=risk(truths,matches,native,groups)
    doc=dict(status="SPSD_PSEG_SELECTIVE_BCE_GATE_SCORE_COMPLETE",protocol_sha256=renderer.LOCK,
         worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),policy_sha256=renderer.POLICY,
         render_result_sha256=render_sha,render_terminal_sha256=terminal_sha,render_stream_sha256=proof["render_stream_sha256"],
         fit_result_sha256=proof["fit_result_sha256"],fit_readback_sha256=proof["fit_readback_sha256"],
         archive_render_sha256=proof["archive_render_sha256"],archive_score_sha256=proof["archive_score_sha256"],
         archive_bce_tensor_sha256=proof["archive_bce_tensor_sha256"],exact_legacy_metric_parity=True,
         exact_matched_image_quota_histogram=True,counts=proof["counts"],decoded_groups=180,
         bootstrap_draws=4096,seed=5297,repeated_source_development=True,external_acquisition_evaluation=False,
         references=refs,risk=risk_report,primary=policy.gate(refs,coverage),detector_forwards=0,optimizer_updates=0,
         fit_labels_opened=False,historical_test_opened=False,elapsed_seconds=perf_counter()-start)
    a.output.mkdir()
    raw=(json.dumps(doc,indent=2,sort_keys=True,allow_nan=False)+"\n").encode()
    with (a.output/"RESULT_SAFE.json").open("xb") as stream:
        stream.write(raw)
    print(json.dumps(dict(status=doc["status"],result_sha256=sha256(raw).hexdigest())),flush=True)


if __name__=="__main__":
    main()
