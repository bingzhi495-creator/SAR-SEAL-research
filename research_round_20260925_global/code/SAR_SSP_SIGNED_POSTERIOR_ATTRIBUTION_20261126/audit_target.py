"""Replay the closed SAR-SSP regression and attribute its posterior margin error."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType

HERE=Path(__file__).resolve().parent
LOCK="a1b2fdbdef368ac4887f4c0371e1a1edefc3270e07184b91a53a4540ff8aa32f"
CORE="9bbf19f7dc3cc42a8dcc05c51f8f04a4c95c7b9ba4b861ee0a532866803a5772"
PRIOR_WORKER="dd1b56f42d4358e356f7127f0c46ad8b4f4489502ccb54a2fbe3992361b9c4b6"
PRIOR_MATH="792ffcba9e5ecfc527dad3c4dd74e1f7fbb50c7397d8076fe9ad817bc49dfb7c"
PRIOR_RESULT="69091fcc597ce134cac91b6038bffdd718f8d7ed5e165496ee4b7cb1632677f8"
PRIOR_TERMINAL="b0957ec28fba2dd766e331a0f29bb6a408b08e562a2670db9e9c7b129e3b9493"
DIRECTIONS=("outside_plus","outside_minus","inside_plus","inside_minus")


def need(ok,message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return sha256(raw).hexdigest()


def pin(file,sha):
    raw=file.read_bytes()
    need(digest(raw)==sha,"Exact closed input "+file.name)
    return raw


def module(source,name,path):
    m=ModuleType(name)
    m.__file__=str(path)
    exec(compile(source,str(path),"exec"),vars(m))
    return m


def instrument(source):
    zero='        per_parent[groups[iid]].append(np.full(len(ARMS), baseline))'
    positive='            result = audit(policy, probability_rows, incumbent, k, area, annot["area"], native_hit,'
    need(source.count(zero)==source.count(positive)==1,"Exact accepted readback capture points")
    mark='            record_packet(groups[iid], probability_rows, gt[coords], observed, area, annot["area"], native_hit)'
    return source.replace(zero,zero+'\n        record_zero(groups[iid])').replace(positive,mark+'\n'+positive)


def group_stats(np, rows):
    order=sorted(rows)
    fields=sorted({f for group in order for item,_ in rows[group] for f in item}.difference(("native_area",)))
    def convert(value,key,area):
        return 100*value if key.endswith("_iou_delta") else 100*value/area
    matrix=np.array([[np.mean([convert(item.get(key,0.), key, area) for item,area in rows[g]]) for key in fields]
                     for g in order])
    draws=np.random.Generator(np.random.PCG64(5217)).integers(len(order),size=(4096,len(order)))
    upper_lower=np.percentile(matrix[draws].mean(axis=1),[2.5,97.5],axis=0).T
    mean=matrix.mean(axis=0)
    data={key:dict(points=float(value),interval95=interval.tolist())
          for key,value,interval in zip(fields,mean,upper_lower,strict=True)}
    directions={}
    for label in DIRECTIONS:
        values=np.array([[np.mean([item.get(label+"_"+suffix,0.) for item,_ in rows[g]])
                          for suffix in ("count","p_sum","y_sum")] for g in order])
        global_count,p,y=values.sum(axis=0)
        raw_sums=[sum(item.get(label+"_"+key,0.) for g in order for item,_ in rows[g])
                  for key in ("count","p_sum","y_sum")]
        if global_count:
            resampled=values[draws].sum(axis=1)
            valid=resampled[:,0]>0
            error=(resampled[valid,1]-resampled[valid,2])/resampled[valid,0]
            directions[label]=dict(raw_count=int(raw_sums[0]),raw_probability_sum=float(raw_sums[1]),
                         raw_occupancy_count=int(raw_sums[2]),predicted_rate=float(p/global_count),
                         observed_rate=float(y/global_count),prediction_minus_observation=float((p-y)/global_count),
                         paired_error_interval95=np.percentile(error,[2.5,97.5]).tolist(),valid_draws=int(valid.sum()))
        else:
            directions[label]=dict(raw_count=0)
    return data,directions


def run(args):
    start=perf_counter()
    previous=json.loads(pin(args.regret/"RESULT_SAFE.json",PRIOR_RESULT))
    end=json.loads(pin(args.regret/"observer_02/TERMINAL.json",PRIOR_TERMINAL))
    need(end["actual_exit"]==0 and end["output_footer_verified"] is True and end["worker_sha256"]==PRIOR_WORKER
         and end["result_sha256"]==PRIOR_RESULT and previous["fixed_variable_exact_parity"],"Observed original closed regret diagnostic")
    pin(args.regret/"frontier.py",PRIOR_MATH)
    source=pin(args.regret/"diagnose.py",PRIOR_WORKER)
    old=module(instrument(source.decode("utf-8-sig")),"source_reference_replay",args.regret/"diagnose.py")
    outcomes={k:defaultdict(list) for k in ("pseg_fit_column","spsd_fit_column")}
    changed=Counter()
    math=None

    def record_zero(g):
        for rows in outcomes.values():
            rows[g].append(({},1))

    def record_packet(g,posteriors,y,original,area,truth_area,native_hit):
        nonlocal math
        if math is None:
            math=module(pin(HERE/"signed_ledger.py",CORE),"signed_posterior_ledger",HERE/"signed_ledger.py")
        fixed=original["fixed_iou"]
        variable=original["variable_iou"]
        changed.update(positive=1,changed=int(fixed!=variable))
        for column,rows in enumerate(outcomes.values()):
            data=math.audit(posteriors[:,:,column],y,fixed,variable,area,truth_area,native_hit)
            for side in ("outside","inside"):
                plus,minus=(side+"_"+sense for sense in ("plus","minus"))
                data[side+"_signed_error"]=data[plus+"_p_sum"]-data[plus+"_y_sum"]-data[minus+"_p_sum"]+data[minus+"_y_sum"]
            need(abs(data["outside_signed_error"]+data["inside_signed_error"]-data["overlay_error"])<1e-10,
                 "Signed endpoint regions add up to intersection overcredit")
            data["overlay_minus_slope"]=data["overlay_error"]-data["slope_error"]
            data["outside_minus_inside_error"]=data["outside_signed_error"]-data["inside_signed_error"]
            data["true_fp_change"]=data["area_delta"]-data["true_intersection_gain"]
            rows[g].append((data,truth_area))
    old.record_zero=record_zero
    old.record_packet=record_packet
    repeated=old.core_run(args.transfer)
    need(repeated["paired"]==previous["paired"] and repeated["iou_group_equal_points"]==previous["iou_group_equal_points"],
         "Exact earlier readback and geometry parity")
    need(dict(changed)==dict(positive=50,changed=48) and all(len(z)==23 and sum(map(len,z.values()))==139 for z in outcomes.values()),
         "Predeclared target cohort and exposed-hull counts")
    import numpy as np
    results={}
    for key,rows in outcomes.items():
        geometry,directions=group_stats(np,rows)
        need(abs(geometry["true_iou_delta"]["points"]-previous["paired"]["variable_vs_fixed"]["delta"])<1e-10,
             "Exact historical fixed-geometry IoU effect")
        results[key]=dict(group_equal_geometry=geometry,exposed_endpoint_rates=directions)
    positive=lambda metric: all(d["group_equal_geometry"][metric]["interval95"][0]>0 for d in results.values())
    return dict(status="SAR_SSP_SIGNED_POSTERIOR_ATTRIBUTION_COMPLETE",protocol_sha256=LOCK,
                worker_sha256=digest(Path(__file__).read_bytes()),audit_core_sha256=CORE,
                prior_diagnostic_sha256=PRIOR_RESULT,exact_original_geometry_parity=True,
                no_detector_forwards_or_optimizer_updates=True,target_reference_count=1,target_prediction_columns=2,
                historically_exposed_target=True,fixed_native_hulls=139,fixed_native_matched=88,mother_groups=23,
                positive_matches=50,changed_matches=48,paired_draws=4096,
                both_columns_intersection_overcredit=positive("overlay_error"),
                both_columns_intersection_error_greater_than_slope_error=positive("overlay_minus_slope"),
                results=results,elapsed_seconds=perf_counter()-start)


def main():
    p=ArgumentParser(description=__doc__)
    for field in ("transfer","regret","output"):
        p.add_argument("--"+field,type=Path,required=True)
    a=p.parse_args()
    need(sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES")=="-1"
         and not a.output.exists(),"Isolated source frozen replay and a new output")
    pin(HERE/"PRELOCK.md",LOCK)
    def offline(event,_):
        if event.startswith("socket."):
            raise PermissionError("Closed diagnostic is offline")
    sys.addaudithook(offline)
    out=run(a)
    content=(json.dumps(out,sort_keys=True,allow_nan=False,indent=2)+"\n").encode()
    with a.output.open("xb") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(dict(status=out["status"],result_sha256=digest(content))),flush=True)

if __name__=="__main__":
    main()
