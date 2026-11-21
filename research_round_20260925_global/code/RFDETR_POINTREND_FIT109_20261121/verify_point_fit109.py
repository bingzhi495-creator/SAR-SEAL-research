"""Post-exit, CPU-only FIT109 readback of the closed PointRend candidate witness."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from copy import deepcopy
from hashlib import sha256
from io import StringIO
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace

import numpy as np
from pycocotools import mask as codec


ARMS = ("N","C","P","F","B","J")
MATCHED = ("F","B","J")
OWNER = "POINTREND_FIT109_BENCHMARK_OWNER_TERMINAL_OBSERVED"


def need(ok, message):
    if not ok:
        raise ValueError(message)


def pinned(path, digest, limit):
    with path.open("rb") as f:
        need(0 < os.fstat(f.fileno()).st_size <= limit, "Expected bounded closed artifact")
        raw = f.read(limit+1)
    need(sha256(raw).hexdigest()==digest and len(raw)<=limit, "Independent closed byte identity: "+path.name)
    return raw


def read_doc(path, digest, limit=2**20):
    return json.loads(pinned(path,digest,limit))


def checked_module(path, digest):
    src = pinned(path,digest,2**20)
    module = ModuleType("_postexit_"+path.stem)
    module.__file__ = str(path)
    exec(compile(src,str(path),"exec"),module.__dict__)
    return module


def checked_terminal(path, digest):
    owner = read_doc(path,digest)
    need(owner.get("status")==OWNER and owner.get("original_process_terminal") is True
         and type(owner.get("original_exit")) is int and owner["original_exit"]==0
         and str(owner.get("original_tool_handle"))=="11801","Genuine original terminal record and zero exit")
    need(all(isinstance(owner.get(k),str) and len(owner[k])==64 for k in ("result_sha256","module_manifest_sha256")),
         "Original result and source manifest binding")
    return owner


def rebuild_candidate(stock_rle, witness, coordinates, budget):
    """Build geometry independently by flipping native bitmap cells in C order."""
    native = np.asarray(codec.decode(stock_rle),dtype=np.uint8,order="C")
    flat = native.reshape(-1)
    coords = np.asarray(coordinates)
    if not coords.size:
        coords = np.empty((0,2),dtype=np.int64)
    area = int(flat.sum())
    need(coords.ndim==2 and coords.shape[1]==2 and coords.dtype.kind in "iu" and len(coords)<=32
         and np.unique(coords).size==coords.size and (coords>=0).all() and (coords<flat.size).all()
         and not flat[coords[:,0]].any() and flat[coords[:,1]].all(), "Unique native endpoint pairs on their original sides")
    need(type(budget) is int and 0<=budget<=min(8,area//10,len(coords)), "Prelocked per-candidate edit cap")
    allowed = set(coords.reshape(-1).tolist())
    arms = dict(N=stock_rle)
    counts = {"N":dict(adds=0,drops=0,area_delta=0)}
    for name in ("C","P"):
        rle = witness[name]
        need(isinstance(rle,dict) and rle["size"]==stock_rle["size"] and isinstance(rle["counts"],str), "Native render RLE")
        value = codec.decode(rle)
        common = int(np.count_nonzero(value & native))
        adds, drops = int(np.count_nonzero(value))-common, area-common
        arms[name] = rle
        counts[name] = dict(adds=adds,drops=drops,area_delta=adds-drops)
    for name in MATCHED:
        changes = witness[name]
        need(type(changes) is list and len(changes)==len(set(changes))==2*budget
             and all(type(i) is int and i in allowed for i in changes), "Exact sparse native endpoint budget and unique support")
        index = np.asarray(changes,dtype=np.int64)
        drops = int(flat[index].sum())
        adds = len(changes)-drops
        need(name=="J" or adds==drops==budget, "Balanced addition/removal counts")
        if name=="F":
            original_pairs={tuple(map(int,pair)) for pair in coords}
            need(all(tuple(changes[j:j+2]) in original_pairs for j in range(0,len(changes),2)), "Original fixed pair identities")
        expected = dict(adds=adds,drops=drops,area_delta=adds-drops)
        diag = witness["diagnostics"][name]
        need(all(type(diag[k]) is int and diag[k]==v for k,v in expected.items()), "Exact original-side edit diagnostics")
        need(type(diag["nonpositive_flips"]) is int and 0<=diag["nonpositive_flips"]<=len(changes),
             "Feasible nonpositive-flip count")
        if len(index):
            changed = native.copy()
            changed.reshape(-1)[index] ^= 1
            encoded = codec.encode(np.asfortranarray(changed))
            arms[name] = dict(size=encoded["size"],counts=encoded["counts"].decode("ascii"))
        else:
            arms[name] = stock_rle
        counts[name] = expected
    return arms,counts


def footprint(rle, reference, helpers):
    area = int(codec.area(rle))
    if area == 0:
        return 0.
    mod = helpers.truth.modules
    rect = mod.geometry.boxes_from_counts(mod.shift.runs(rle["counts"]),*rle["size"])
    overlap = min(float(mod.geometry.overlap_area(rect,reference)),reference.area,float(area))
    return overlap/(reference.area+area-overlap)


def native_coco(gt, rows):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    reference=COCO()
    reference.dataset=deepcopy(gt)
    reference.dataset.setdefault("info",{})
    with redirect_stdout(StringIO()):
        reference.createIndex()
        # loadRes may mutate the supplied prediction dicts.
        dt=reference.loadRes(deepcopy(rows))
        ev=COCOeval(reference,dt,"segm")
        ev.params.imgIds=[im["id"] for im in gt["images"]]
        ev.params.catIds=[1]
        ev.params.maxDets=[1,10,100]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return [float(v) for v in ev.stats]


def recount(gt,pools,attached,budgets,witness,helpers):
    recs={(r["image_id"],r["rank"],r["query_id"]):r for r in attached}
    need(len(recs)==len(attached)==6738 and recs.keys()==witness.keys()==budgets.keys(), "Complete held candidate census")
    truth=helpers.truth.modules
    polys,_,_=truth.geometry.polygon_support(gt,truth.vector)
    streams={a:[] for a in ARMS}
    totals={a:Counter() for a in ARMS}
    groups=defaultdict(list)
    transitions={a:Counter() for a in ARMS}
    seen=set()
    for im in gt["images"]:
        iid=im["id"]
        for cand in pools[iid]:
            key=(iid,cand["rank"],cand["query_id"])
            source=cand["rle"]
            arms=dict.fromkeys(ARMS,source)
            if key in recs:
                seen.add(key)
                rec=recs[key]
                budget=budgets[key]
                k=int(budget["k"]) if isinstance(budget,dict) else int(budget)
                arms,counts=rebuild_candidate(source,witness[key],rec["coordinates"],k)
                for arm,t in counts.items():
                    totals[arm].update(t)
                if rec["role"]=="valid":
                    reference=polys[rec["matched_gt_id"]][0]
                    row=[footprint(arms[a],reference,helpers) for a in ARMS]
                    groups[rec["group"]].append(row)
                    for a,v in zip(ARMS,row):
                        before,after=row[0]>.90,v>.90
                        transitions[a]["lost"]+=int(before and not after)
                        transitions[a]["new"]+=int(not before and after)
            for a in ARMS:
                streams[a].append(dict(image_id=iid,category_id=1,score=cand["score"],segmentation=arms[a]))
    need(seen==set(recs) and all(len(v)==10488 for v in streams.values()),"Original mapped candidates and scores for all arms")
    means=100*np.array([np.mean(groups[g],axis=0) for g in sorted(groups)])
    need(len(groups)==92 and sum(map(len,groups.values()))==392,"Independent full fixed geometry population")
    rng=np.random.Generator(np.random.PCG64(5217))
    picks=rng.integers(len(groups),size=(4096,len(groups)))
    arms={}
    for ix,a in enumerate(ARMS):
        values=native_coco(gt,streams.pop(a))
        arms[a]=dict(all12=values,AP=values[0],AP75=values[2],source_equal_iou_points=float(means[:,ix].mean()),
                     strict90=dict(transitions[a]),edit_counts=dict(totals[a]))
    comparisons={}
    tests={}
    for lhs,arm in (("B","F"),("B","J"),("B","N"),("B","P"),("P","C"),("P","N")):
        delta=means[:,ARMS.index(lhs)]-means[:,ARMS.index(arm)]
        interval=np.quantile(delta[picks].mean(axis=1),[.025,.975]).tolist()
        ap=100*(arms[lhs]["AP"]-arms[arm]["AP"])
        comparisons[lhs+"_minus_"+arm]=dict(AP_delta_points=ap,geometry_delta_points=float(delta.mean()),geometry_ci95_points=interval)
        if lhs=="B" and arm in ("F","J"):
            tests[arm]=dict(AP=ap>=.30,geometry=bool(delta.mean()>=.20),geometry_lower=interval[0]>0,
                            AP75=arms["B"]["AP75"]>=arms[arm]["AP75"],strict=transitions["B"]["lost"]<=transitions[arm]["lost"])
    return arms,comparisons,tests


def schedules(out, audits, folds, attached, budgets, initial):
    import torch
    train=defaultdict(list)
    for r in attached:
        if r["role"]=="valid" and r["truth"] is not None and len(r["old14"]):
            train[r["group"]].append((r["group"],r["image_id"],r["rank"],r["query_id"],r["matched_gt_id"]))
    for rows in train.values():
        rows.sort(key=lambda x:x[1:4])
    need(sum(map(len,train.values()))==392,"Original supervised positives")
    hashes=[]
    for fold in range(5):
        audit=audits[str(fold)]
        groups=sorted(g for g in train if folds[g]!=fold)
        gen=torch.Generator().manual_seed(152170+fold)
        steps=[]
        for _ in range(200):
            g=groups[torch.randint(len(groups),(),generator=gen).item()]
            rows=train[g]
            steps.append(rows[torch.randint(len(rows),(),generator=gen).item()])
        need(steps==list(map(tuple,audit["private_schedule"])) and all(folds[s[0]]!=fold for s in steps),"Complete seeded positive draw and held-group exclusion")
        need(audit["updates"]==len(audit["losses"])==200 and audit["supervised_points"]==39200
             and audit["held_sources"]==sum(v==fold for v in folds.values())
             and audit["train_positive_rows"]==sum(len(train[g]) for g in groups)
             and audit["train_positive_sources"]==len(groups) and audit["used_sources"]==len({s[0] for s in steps})
             and audit["used_positive_rows"]==len(set(steps))
             and audit["zero_budget_updates"]==sum(budgets[tuple(s[1:4])]["k"]==0 for s in steps)
             and audit["trainable_parameters"]==198402 and audit["optimizer"]=="AdamW"
             and audit["lr"]==.002 and audit["weight_decay"]==.01 and np.isfinite(audit["losses"]).all()
             and audit["gradient_norm_min"]>0 and audit["gradient_norm_max"]>=audit["gradient_norm_min"],"Recorded optimizer budget and distinct supervised sources")
        file=out/f"HEAD_FOLD_{fold}_PRIVATE.npz"
        raw=file.read_bytes()
        hashes.append(sha256(raw).hexdigest())
        from io import BytesIO
        with np.load(BytesIO(raw),allow_pickle=False) as z:
            need(set(z.files)==set(initial) and len(z.files)==len(initial),"Complete original point-head parameter roster")
            delta=[]
            for name,old in initial.items():
                val=z[name]
                need(val.dtype==np.float32 and tuple(val.shape)==tuple(old.shape) and np.isfinite(val).all(),"Finite fold weights with original shapes")
                delta.append((torch.from_numpy(val)-old).norm())
        distance=torch.stack(delta).norm().item()
        need(distance>0 and math.isclose(distance,audit["parameter_update_l2"],rel_tol=3e-6),"Original endpoint movement from matched initialization")
    need(len(set(hashes))==5,"Separately trained fold weights")
    return hashes


def main():
    p=ArgumentParser(allow_abbrev=False)
    for name in ("owner","work","run","code","old-support","old-score","raw-capture","raw-acceptance","source-cohort","fit-gt","shapely-site","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--owner-sha256",required=True)
    args=p.parse_args()
    owner=checked_terminal(args.owner,args.owner_sha256)
    need(not args.output.exists(),"Fresh independent output")
    start=perf_counter()
    report=read_doc(args.run/"RESULT_SAFE.json",owner["result_sha256"])
    manifest=read_doc(args.work/"POINTREND_FIT109_MODULE_PINS.json",owner["module_manifest_sha256"])
    need(report["status"]=="RFDETR_POINTREND_FIT109_OOF_EVALUATION_COMPLETE" and report["FIT"] is True and report["AP"] is True
         and report["DEV"] is False and report["TEST"] is False and report["source_manifest_sha256"]==owner["module_manifest_sha256"],"Complete FIT-only endpoint")
    for name,digest in manifest.items():
        need(Path(name).name==name,"Original source filename")
        pinned(args.work/name,digest,2**20)
    runner=checked_module(args.work/"run_point_fit109.py",manifest["run_point_fit109.py"])
    protocol=read_doc(args.work/"POINTREND_FIT109_PRELOCK.json",runner.PROTOCOL_SHA)
    pre=read_doc(args.run/"PRELABEL_SAFE.json",report["prelabel_sha256"])
    need(report["protocol_sha256"]==runner.PROTOCOL_SHA and pre["native_budget_sha256"]==report["native_budget_sha256"]==protocol["prior_oof_budget_sha256"]
         and pre["capture_readback_sha256"]==report["capture_readback_sha256"]=="19393564375cde464d6ab3a51966b8861f28f02f2dc33a08fde14de38f13dd38"
         and pre["GT_opened"] is False and pre["swaps"]==48982,"Before-label feature admission and frozen matched budget")
    previous=args.code/"RFDETR_QUERY_FEATURE_FIT109_20261121"
    old=checked_module(previous/"run_fit109_query.py","97fdc3987462aabbcd7d4fbe672c9788a3b969a221243e13d82473093170a15a")
    spot=old.load(args.code/"RFDETR_CORE36_MATCHED_POINT_SPOT_20261120/run_spot.py")
    former=args.code/"RFDETR_MATCHED_POINT_FORMAL_20261120"
    prior,plan=(old.load(former/n) for n in ("run_fit109.py","fit_plan.py"))
    args.acceptance,args.capture,args.acceptance_sha256=args.raw_acceptance,args.raw_capture,old.OLD_RAW_ACCEPTANCE
    roster,_=prior.read_population(args,spot)
    helpers=old.load(previous/"fit109_score.py").load_helpers(args.code,args.shapely_site)
    pools,records=runner.collect_old_support(args,SimpleNamespace(old=old,spot=spot),roster,protocol)
    budgets,folds=runner.read_budgets(args.old_score,protocol,records,plan,roster,helpers.geometry)
    gt,attached=old.attach_fit(args,roster,records,pools,helpers,spot)
    need(len(gt["images"])==109 and len(gt["annotations"])==496 and len(folds)==109,"Complete source population")
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    need(os.environ.get("CUDA_VISIBLE_DEVICES")=="-1" and not torch.cuda.is_initialized(),"CPU-only independent check")
    adapter=checked_module(args.code/"RFDETR_POINTREND_SUBDIVISION_20261121/roi_point_head.py","672bdd2aed8c0056f29c816f829bae9ceae5f9a685309beebf107aa59e547274")
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(5217)
        initial=adapter.BinaryPointHead().state_dict()
    audits=read_doc(args.run/"HEAD_AUDITS_PRIVATE.json",report["train_audit_sha256"])
    weights=schedules(args.run,audits,folds,attached,budgets,initial)
    need(report["point_head_updates"]==1000 and report["supervised_points"]==196000,"Reported actual optimizer totals")
    witness_raw=pinned(args.run/"PREDICTION_WITNESS_PRIVATE.jsonl",report["prediction_witness_sha256"],80*2**20)
    entries=[json.loads(i) for i in witness_raw.splitlines()]
    witness={tuple(r["key"]):r for r in entries}
    need([tuple(r["key"]) for r in entries]==[(r["image_id"],r["rank"],r["query_id"]) for r in records]
         and all(len(k)==3 and all(type(i) is int for i in k) for k in witness),"Exactly ordered held witness for every original eligible candidate")
    print(json.dumps(dict(event="POINTREND_INDEPENDENT_SCHEDULE_AND_WITNESS_VERIFIED",rows=len(witness))),flush=True)
    arms,contrasts,checks=recount(gt,pools,attached,budgets,witness,helpers)
    metric=report["metrics"]
    maximum=0.
    for a,values in arms.items():
        oldmetric=metric["arms"][a]
        need(values["all12"]==oldmetric["all12"] and values["AP"]==oldmetric["AP"] and values["AP75"]==oldmetric["AP75"],
             "Exact independent stock COCO output: "+a)
        for k in ("lost","new"):
            need(values["strict90"][k]==oldmetric["strict90"][k],"Independent strict IoU transitions")
        err=abs(values["source_equal_iou_points"]-oldmetric["source_equal_iou_points"])
        maximum=max(maximum,err)
        need(err<2e-8,"Independently reconstructed group-equal geometry: "+a)
        edit=values["edit_counts"]
        need(all(edit[k]==oldmetric["edits"][k] for k in ("adds","drops")) and edit["adds"]+edit["drops"]==oldmetric["edits"]["edits"],"Actual realized edits: "+a)
    for name,d in contrasts.items():
        for k,v in d.items():
            error=float(np.max(np.abs(np.asarray(metric["contrasts"][name][k])-np.asarray(v))))
            maximum=max(maximum,error)
            need(error<2e-8,"Recomputed contrast and paired source interval: "+name)
    advance=all(all(v.values()) for v in checks.values())
    need(checks==metric["checks"] and advance is metric["advance"],"Independent progression checks")
    safe=dict(status="RFDETR_POINTREND_FIT109_OOF_INDEPENDENTLY_ACCEPTED",original_tool_handle="11801",original_exit=0,
              owner_record_sha256=args.owner_sha256,result_sha256=owner["result_sha256"],module_manifest_sha256=owner["module_manifest_sha256"],
              verifier_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),witness_sha256=report["prediction_witness_sha256"],
              head_audits_sha256=report["train_audit_sha256"],fold_weight_file_sha256=weights,source_groups=109,images=109,
              annotations=496,mapped_candidates=10488,eligible_candidates=6738,valid_fixed_matches=392,contributing_groups=92,
              all_seeded_optimizer_draws_and_source_holdouts_verified=True,original_optimizer_steps=1000,original_supervised_points=196000,
              all_fold_endpoint_weight_movements_verified=True,COCO12_bit_exact_all_six_arms=True,all_native_binary_edit_counts_exact=True,
              continuous_geometry_max_abs_disagreement_points=maximum,bootstrap_draws=4096,bootstrap_seed=5217,
              independent_arms=arms,independent_contrasts=contrasts,checks=checks,advance=advance,
              independent_detector_forwards=0,independent_head_updates=0,FIT=True,DEV=False,TEST=False,AP=True,
              limits="Witness readback; candidate probabilities and every intermediate optimizer state were not independently replayed.",
              seconds=round(perf_counter()-start,3))
    raw=(json.dumps(safe,indent=2,sort_keys=True,allow_nan=False)+"\n").encode()
    with args.output.open("xb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(dict(status=safe["status"],sha256=sha256(raw).hexdigest(),advance=advance)),flush=True)


if __name__=="__main__":
    main()
