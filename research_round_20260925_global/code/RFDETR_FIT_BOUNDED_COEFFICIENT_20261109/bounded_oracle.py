"""Closed FIT-only bounded-coefficient experiment with native BF16 replay."""
import argparse
from collections import Counter
from hashlib import sha256
from importlib import import_module
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import time
import types

import numpy as np

NATIVE_SHA = "37ed6cc581434012c9f1d65adf0a4264857673411022a7279eaf450a284279c3"
SOLVER_SHA = "04016d572bb2675278554ea54ffa4b6171815cc40831999e2615e08b1bf1d115"
ARM_NAMES = ("stock", "full_a_scalar", "bounded_offset", "bounded_one_step", "bounded_iterative")
CAP_TOLERANCE = 1.02


def load_module(path, expected, name):
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != expected:
        raise ValueError("Exact published dependency required")
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(raw,str(path),"exec"),module.__dict__)
    return module


def local_guard(logits, side):
    y,x = np.divmod(np.arange(side*side),side)
    mask = logits > 0
    return ((x>=x[mask].min()-8) & (x<=x[mask].max()+8)
            & (y>=y[mask].min()-8) & (y<=y[mask].max()+8))


def phase_drift(original, modified, local, cap, coefficient_ratio):
    d = modified.astype(np.float64)-original
    local_rms = float(np.sqrt(np.mean(d[local]**2)))
    frame_rms = float(np.sqrt(np.mean(d**2)))
    trust = float(np.sqrt(.5*((local_rms/cap)**2+(frame_rms/cap)**2)+coefficient_ratio**2))
    return dict(local_rms=local_rms,frame_rms=frame_rms,trust=trust,
                maximum=float(np.abs(d).max()),coefficient_ratio=coefficient_ratio)


def metrics(native, a, b, row):
    truth, held, local = row["truth"],row["held"],row["local"]
    m = native.case_metrics(a,b,truth,truth,held)
    m["a_outside_fp"] = int(((a>0) & ~truth & ~local).sum())
    m["b_outside_fp"] = int(((b>0) & ~truth & ~local).sum())
    m["normalized_error"] = (10*(m["a_fn"]+m["b_fn"])+9*(m["a_fp"]+m["b_fp"]))/(2*int(truth.sum()))
    return m


def arm_pair(views, left, right, bootstrap, metric):
    pairs = [(float(np.mean([row[left][metric] for row in view])),
              float(np.mean([row[right][metric] for row in view]))) for view in views if view]
    j = bootstrap(pairs)
    return dict(images=j["images"], first=j["iid"], second=j["jitter"],
                first_minus_second=j["iid_minus_jitter"], percentile95=j["percentile95"])


def run(args):
    started=time.perf_counter()
    native=load_module(args.native_source,NATIVE_SHA,"_pinned_native_coefficient_replay")
    solver=load_module(Path(__file__).with_name("bounded_fit.py"),SOLVER_SHA,"_bounded_fit")
    phase=native.module(args.phase_source,native.PHASE_SHA,"_pinned_phase_geometry")
    replay=native.module(args.replay_runtime,phase.REPLAY_SHA,"_closed_matcher_replay")
    need=replay.need
    need(sys.version_info[:3]==(3,12,14) and os.environ.get("CUDA_VISIBLE_DEVICES")=="0"
         and os.environ.get("HF_HUB_OFFLINE")==os.environ.get("TRANSFORMERS_OFFLINE")=="1"
         and version("scipy")=="1.15.3","Exact isolated DGX numeric runtime")
    replay.canonical(args.capture_dir,directory=True)
    need(args.output.is_absolute() and replay.canonical(args.output.parent,directory=True)==args.output.parent
         and not args.output.exists() and not args.output.is_symlink()
         and not args.output.is_relative_to(args.capture_dir),"Fresh output outside archived capture")
    cap=replay.module(args.capture_source,replay.CAPTURE_SHA,"_pinned_capture_matcher_cost")
    math=replay.module(args.replay_runtime.with_name("matcher_cost_math.py"),replay.MATH_SHA,"_pinned_matcher_math")
    report=json.loads(replay.checked_bytes(args.capture_dir/"CAPTURE_SAFE.json",args.capture_report_sha256))
    receipt=json.loads(replay.checked_bytes(args.capture_closure,args.capture_closure_sha256))
    rows=replay.validate_closed(report,receipt,args.capture_protocol,args.capture_report_sha256,cap)
    items=json.loads(replay.checked_bytes(args.fit_index,replay.FIT_MANIFEST_SHA))["items"]
    plan=cap.view_plan(items,report["tranche"])
    need(report["tranche"]=="cohort32" and len(items)==741 and len({v["image_sha256"] for v,_ in plan})==32,
         "Exact locked FIT scope")
    sys.addaudithook(cap.offline)
    import torch
    need(all(version(k)==v for k,v in cap.VERSIONS.items())
         and all(replay.digest(Path(import_module(k).__file__).read_bytes())==v for k,v in cap.NATIVE_SOURCE.items()),
         "Exact stock native source and runtime")
    need(torch.cuda.is_available() and torch.cuda.device_count()==1,"One DGX CUDA device")
    torch.cuda.set_per_process_memory_fraction(min(.35,cap.CAP/torch.cuda.get_device_properties(0).total_memory))
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("high")
    phases=math.DENSE_PHASES
    matrix=native.sampling_matrix((144,144),phase.SIDE,phases[0])
    views=[]
    scope=Counter()
    rejected=Counter()
    diagnostics={k:[] for k in ARM_NAMES[2:]}
    solver_stats=[]
    for view in rows:
        if view["status"]!="STOCK_TRACE_EXACT":
            continue
        packet=replay.load_packet(torch,args.capture_dir,view,plan,args.capture_protocol,cap)
        assigned,a,b,eligible,missing=native.phase_cases(torch,packet,phases,phase,math,cap)
        scope.update(targets=len(assigned),missing=missing,eligible=len(eligible))
        sparse=packet["sparse"]
        _,channels,h,w=sparse["spatial_features"].shape
        need((h,w,channels)==(144,144,256) and np.array_equal(a[3],b[3]),"Exact prior geometry and phase labels")
        basis=sparse["spatial_features"][0].float().numpy().reshape(channels,-1).T.astype(np.float64)
        q_rows=sparse["query_features"][0].float().numpy().astype(np.float64)
        lattice_gram=basis.T@basis/len(basis) if len(eligible) else None
        adjusted=[]
        for i in eligible:
            q=q_rows[assigned[i]]
            native_a,native_b=a[2][i],b[2][i]
            truth=a[3][i]
            fit,held=native.support(truth,native_a,phase.SIDE)
            guard=local_guard(native_a,phase.SIDE)
            radius=.5*float(np.clip(np.median(np.abs(native_a[guard])),1.,4.))
            q_scale=max(float(np.linalg.norm(q)),1e-12)
            guard_features=matrix[guard]@basis
            guard_gram=guard_features.T@guard_features/len(guard_features)
            metric=(guard_gram+lattice_gram)*(.5/radius**2)+np.eye(channels)/q_scale**2
            design=matrix[fit]@basis
            y=truth[fit]
            need(y.any() and not y.all(),"Exact two-class fit support")
            weights=np.where(y,.5/y.sum(),.5/(~y).sum())
            start=time.perf_counter()
            full=solver.solve_bounded(design,metric,native_a[fit].astype(np.float64),y.astype(np.float64),weights)
            offset=solver.solve_bounded(np.ones((len(y),1)),np.array([[1/radius**2]]),
                                        native_a[fit].astype(np.float64),y.astype(np.float64),weights)
            solver_stats.append(dict(seconds=time.perf_counter()-start,projected_gradient=full.projected_gradient,
                                     one_loss=full.one_loss,iterative_loss=full.final_loss,
                                     one_backtracks=full.backtracks,offset_loss=offset.final_loss,
                                     one_seconds=full.one_seconds,iterative_extra_seconds=full.iterative_extra_seconds,
                                     offset_seconds=offset.one_seconds+offset.iterative_extra_seconds))
            for delta in (full.one_step,full.iterative):
                need(float(delta@metric@delta)<=1+1e-9,"Continuous fit is within the common budget")
            need(abs(offset.iterative[0])<=radius+1e-9,"Offset uses the same continuous cap")
            adjusted.append(dict(query=int(assigned[i]),joint_q=q+full.iterative,only_q=q+full.one_step,
                                 offset=float(offset.iterative[0]),q=q,q_scale=q_scale,cap=radius,
                                 anchor_a=native_a,anchor_b=native_b,truth=truth,held=held,local=guard))
            scope.update(small=int(packet["is_small"][i]),fit_points=int(fit.sum()))
        result=[]
        if adjusted:
            ra,rb=native.stock_adjusted_scores(torch,packet,phases,phase.SIDE,adjusted)
            for idx,row in enumerate(adjusted):
                aa,bb=row["anchor_a"],row["anchor_b"]
                need(np.array_equal(ra[0,idx],aa) and np.array_equal(rb[0,idx],bb),"Full stock BF16 phase anchor")
                threshold=phase.best_threshold(aa,row["truth"])
                arms=dict(stock=metrics(native,aa,bb,row),
                          full_a_scalar=metrics(native,aa-threshold,bb-threshold,row))
                qdtype=sparse["query_features"].dtype
                candidates=(("bounded_offset",aa+row["offset"],bb+row["offset"],None),
                            ("bounded_one_step",ra[2,idx],rb[2,idx],"only_q"),
                            ("bounded_iterative",ra[1,idx],rb[1,idx],"joint_q"))
                for name,pa,pb,q_name in candidates:
                    dq=0.
                    if q_name:
                        cast=torch.as_tensor(row[q_name],dtype=qdtype).float().numpy()
                        dq=float(np.linalg.norm(cast-row["q"])/row["q_scale"])
                    drift=[phase_drift(z,p,row["local"],row["cap"],dq) for z,p in ((aa,pa),(bb,pb))]
                    diagnostics[name].append({k:max(z[k] for z in drift) for k in drift[0]})
                    if max(x["trust"] for x in drift)>CAP_TOLERANCE:
                        rejected[name]+=1
                        pa,pb=aa,bb
                    arms[name]=metrics(native,pa,pb,row)
                result.append(arms)
        views.append(result)
        print(json.dumps(dict(event="BOUNDED_COEFFICIENT_VIEW_COMPLETE",ordinal=view["ordinal"],
                              eligible=len(eligible))),flush=True)
    cases=[row for v in views for row in v]
    need(len(views)==30 and sum(bool(v) for v in views)==21
         and (scope["targets"],scope["missing"],len(cases))==(55,42,42)
         and scope["fit_points"]==174132
         and sum(r["stock"]["both"] for r in cases)==0
         and sum(r["full_a_scalar"]["both"] for r in cases)==3,"Exact prior locked cases and full scalar reference")
    arms={k:{p:sum(x[k][p] for x in cases) for p in cases[0][k] if p!="normalized_error"} for k in ARM_NAMES}
    paired={}
    for name,left,right in (("one_vs_offset","bounded_one_step","bounded_offset"),
                            ("iterative_vs_offset","bounded_iterative","bounded_offset"),
                            ("iterative_vs_one","bounded_iterative","bounded_one_step")):
        paired[name]={m:arm_pair(views,left,right,replay.image_pair,m) for m in ("both","held_a","normalized_error")}
    def passes(g):
        return (g["both"]["first_minus_second"]>=.10 and g["both"]["percentile95"][0]>0
                and g["held_a"]["first_minus_second"]>=0 and g["normalized_error"]["first_minus_second"]<=0)
    gates={k:passes(v) for k,v in paired.items()}
    chosen=("bounded_iterative" if gates["iterative_vs_offset"] and
            (not gates["one_vs_offset"] or gates["iterative_vs_one"]) else
            "bounded_one_step" if gates["one_vs_offset"] else "stop_bounded_readout")
    stats={arm:{k:np.quantile([v[k] for v in per_case],[.5,.95,1]).tolist() for k in per_case[0]}
           for arm,per_case in diagnostics.items()}
    fit_summary={k:float(np.mean([x[k] for x in solver_stats])) for k in solver_stats[0]}
    safe=dict(status="SSDD_FIT_BOUNDED_COEFFICIENT_ORACLE_COMPLETE",protocol=args.protocol,
              FIT_only=True,DEV=False,TEST=False,AP=False,optimizer_updates=0,
              traced_views=len(views),empty_views=len(rows)-len(views),scope=dict(scope),eligible_images=21,
              paired=paired,arms=arms,gates=gates,decision=chosen,attempted_bf16_drift_quantiles=stats,
              rejected={arm:rejected[arm] for arm in ARM_NAMES[2:]},solver=fit_summary,
              realized_trust_tolerance=CAP_TOLERANCE,iterations=solver.ITERATIONS,
              capture_protocol=args.capture_protocol,cuda_peak_bytes=torch.cuda.max_memory_allocated(),
              seconds=time.perf_counter()-started)
    args.output.mkdir()
    with (args.output/"RESULT.json").open("x",encoding="utf-8") as f:
        json.dump(safe,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(safe,sort_keys=True),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("native-source","phase-source","replay-runtime","capture-dir","capture-source","capture-closure","fit-index","output"):
        p.add_argument("--"+n,type=Path,required=True)
    for n in ("protocol","capture-protocol","capture-report-sha256","capture-closure-sha256"):
        p.add_argument("--"+n,required=True)
    run(p.parse_args())
