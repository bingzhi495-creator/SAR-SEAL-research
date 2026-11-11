"""Fixed-area pixel-swap bounds at the best rigid translation's native footprint budget."""

from argparse import ArgumentParser
from contextlib import redirect_stdout
from hashlib import sha256
from io import StringIO
import json
from math import floor
import os
from pathlib import Path
from statistics import mean
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace

import numpy as np

PROTOCOL = "4cf283b1947d2a3f9c870d889f801e10e1ce86c788f974534b20ec3100b1bc16"
RISK_PRELOCK = "48a4ffcae9779b84c7e19b4262be6bd14e97973346dfd26439fb449fe3d5f6aa"
AXIS_SOURCE = "3f3003c4b25bb6d0b55a78314b44407d912b2dbee1c6b27abee3abf90487e204"
AXIS_TOOL = "3fad654aaf4fa677421939de7dd65aa4f8a222f881431c7f50350dc25386b69e"
AXIS_WAIT = "36a1f943484ef43db902fadace2d7212cf1bff9e39700778a7f7e57a41465e34"
AXIS_RESULT = "b5b445b6845d59cab31bfcdf48ed35f9eb2541b7496a2558168f2613369bb4fe"
MODES = ("raw", "rigid", "constrained", "unrestricted")


def need(ok, why):
    if not ok:
        raise ValueError(why)


def exact(path, digest):
    need(path == path.resolve(strict=True) and 0 < path.stat().st_size <= 2**20, "Bounded canonical input")
    raw = path.read_bytes()
    need(sha256(raw).hexdigest() == digest, "Exact closed input: " + path.name)
    return raw


def load_axis(args):
    root = args.request.parent.parent
    actual = root / "HRSID_FIT_CENTERED_TWO_FLANK_AXIS_ORACLE_ACTUAL_20261112_01"
    draft = root / "HRSID_FIT_CENTERED_TWO_FLANK_AXIS_ORACLE_DRAFT_20261112_01"
    outer = json.loads(exact(actual / "TOOL_TERMINAL_PRIVATE.json", AXIS_TOOL))
    wait = json.loads(exact(actual / "ORIGINAL_WAIT_PRIVATE.json", AXIS_WAIT))
    last = json.loads(outer["original_exec"]["output"].splitlines()[-1])
    child = json.loads(exact(actual / "ORIGINAL.stdout", wait["original_stdout_sha256"]).splitlines()[-1])
    result = json.loads(exact(actual / "RESULT_SAFE.json", AXIS_RESULT))
    need(outer["terminal_observed"] and outer["original_exec"]["exit_code"] == wait["original_os_exit"] == 0
         and wait["original_os_wait"] and wait["job_assigned"] and wait["process_group_closed"] and wait["stop_reason"] is None
         and last["wait_sha256"] == AXIS_WAIT and last["result_sha256"] == child["result_sha256"] == AXIS_RESULT
         and result["FIT"] and result["matched"] == 109 and result["original_replayed"]
         and not any(result[k] for k in ("DEV", "TEST", "AP", "optimizer_updates", "model_forward_calls"))
         and wait["score_source_sha256"] == AXIS_SOURCE, "Exact closed fixed axis oracle")
    path = draft / "axis_oracle.py"
    mod = ModuleType("_closed_axis_oracle")
    mod.__file__ = str(path)
    exec(compile(exact(path,AXIS_SOURCE),str(path),"exec"),mod.__dict__)
    return mod, result


def integer_overlap(prediction, x_shift, y_shift):
    h, w = prediction.shape
    x0, x1 = max(0,x_shift), min(w,w+x_shift)
    y0, y1 = max(0,y_shift), min(h,h+y_shift)
    if x0 >= x1 or y0 >= y1:
        return 0
    return int(np.count_nonzero(prediction[y0:y1,x0:x1] & prediction[y0-y_shift:y1-y_shift,x0-x_shift:x1-x_shift]))


def one_sided_budget(prediction, displacement):
    dx, dy = map(float, displacement)
    ix, iy = floor(dx), floor(dy)
    fx, fy = dx-ix, dy-iy
    overlap = sum(wx*wy*integer_overlap(prediction,x,y)
                  for x,wx in ((ix,1-fx),(ix+1,fx)) for y,wy in ((iy,1-fy),(iy+1,fy)))
    area = int(np.count_nonzero(prediction))
    need(-1e-9 <= area-overlap <= area+1e-9, "Continuous one-sided displacement support")
    return min(float(area),max(0.,float(area-overlap)))


def select_swaps(adds, removals, cap):
    add_indices = np.flatnonzero(adds>0.)
    remove_indices = np.flatnonzero(removals<1.)
    count = min(cap,len(add_indices),len(remove_indices))
    add_order = add_indices[np.argsort(-adds[add_indices],kind="stable")][:count]
    remove_order = remove_indices[np.argsort(removals[remove_indices],kind="stable")][:count]
    gains = adds[add_order]-removals[remove_order]
    useful = int(np.count_nonzero(gains>0.))
    need(not np.any(gains[useful:]>0.), "Monotonic independent swap benefits")
    return add_order[:useful],remove_order[:useful],float(gains[:useful].sum())


def coverage(shapely, xy, truth):
    if not len(xy):
        return np.zeros(0,dtype=float)
    x,y = xy.T
    values = shapely.area(shapely.intersection(shapely.box(x,y,x+1,y+1),truth))
    need(np.isfinite(values).all() and np.min(values)>=-1e-10 and np.max(values)<=1+1e-10,
         "Exact continuous unit-cell polygon occupancy")
    return np.clip(values,0.,1.)


def metrics(intersection, area, target):
    need(-1e-9 <= intersection <= min(area,target)+1e-8, "Valid fixed-area overlap")
    intersection=min(max(0.,intersection),area,target)
    return dict(iou=100*intersection/(area+target-intersection),fp=100*(area-intersection)/target,
                fn=100*(target-intersection)/target)


def risk(edited, cropped, upper, original_iou):
    from pycocotools import mask
    from scipy.ndimage import label
    structure=np.ones((3,3),dtype=bool)
    before,after=label(cropped[0],structure)[1],label(cropped[1],structure)[1]
    after_iou=float(np.max(mask.iou([mask.encode(np.asfortranarray(edited.astype(np.uint8)))],upper,[0]*len(upper)))) if upper else 0.
    return dict(components_added=after>before,components_removed=after<before,
                neighbor_crossed=original_iou<.20<=after_iou,neighbor_increased=after_iou>original_iou+1e-10)


def candidate_analysis(candidate, upper, truth, axis_calls, dense_spot):
    from pycocotools import mask
    import shapely
    from shapely.affinity import translate
    from scipy.ndimage import binary_dilation, binary_erosion

    direction, best=min(axis_calls,key=lambda value:(-value[1]["iou"],abs(value[1]["offset"]),value[1]["offset"]))
    displacement=direction*best["offset"]
    rle=candidate["rle"]
    prediction=mask.decode(rle).astype(bool)
    y,x=np.nonzero(prediction)
    area=int(len(x))
    need(area==candidate["area"] and prediction.shape==(800,800),"Exact frozen original mask and native grid")
    tight=prediction[y.min():y.max()+1,x.min():x.max()+1]
    budget=one_sided_budget(tight,displacement)
    cap=floor(budget)
    need(cap<=budget and cap<=area and ((best["offset"]!=0) or cap==budget==0.),"Conservative native swap budget")
    tx0,ty0,tx1,ty1=truth.bounds
    x0,y0=max(0,min(int(x.min()),floor(tx0))-2),max(0,min(int(y.min()),floor(ty0))-2)
    x1,y1=min(800,max(int(x.max())+1,int(np.ceil(tx1)))+2),min(800,max(int(y.max())+1,int(np.ceil(ty1)))+2)
    roi=prediction[y0:y1,x0:x1]
    all_foreground=np.column_stack((x,y))
    bg_y,bg_x=np.nonzero(~roi)
    background=np.column_stack((bg_x+x0,bg_y+y0))
    inside_bounds=(background[:,0]<tx1)&(background[:,0]+1>tx0)&(background[:,1]<ty1)&(background[:,1]+1>ty0)
    background=background[inside_bounds]
    a_weight,b_weight=coverage(shapely,background,truth),coverage(shapely,all_foreground,truth)
    raw_intersection=float(b_weight.sum())
    raw=metrics(raw_intersection,area,truth.area)
    structure=np.ones((3,3),dtype=bool)
    band_add=binary_dilation(roi,structure,iterations=2)&~roi
    band_remove=roi&~binary_erosion(roi,structure,iterations=2)
    allowed_add=np.flatnonzero(band_add[background[:,1]-y0,background[:,0]-x0])
    allowed_remove=np.flatnonzero(band_remove[y-y0,x-x0])
    original_iou=float(np.max(mask.iou([rle],upper,[0]*len(upper)))) if upper else 0.
    result=dict(raw=raw,rigid=best,budget=budget,cap=cap,dense_error=0.)
    if dense_spot:
        dense=shapely.union_all(shapely.box(x,y,x+1,y+1))
        moved=translate(dense,xoff=float(displacement[0]),yoff=float(displacement[1]))
        dense_budget=(dense.area-moved.intersection(dense).area)
        error=abs(dense_budget-budget)
        need(error<1e-8,"Independent Shapely rigid symmetric-difference budget")
        result["dense_error"]=error
    support=(("constrained",allowed_add,allowed_remove),
             ("unrestricted",np.arange(len(background)),np.arange(len(all_foreground))))
    for name,adds,removes in support:
        ai,ri,gain=select_swaps(a_weight[adds],b_weight[removes],cap)
        added,removed=background[adds[ai]],all_foreground[removes[ri]]
        edited=prediction.copy()
        edited[added[:,1],added[:,0]]=True
        edited[removed[:,1],removed[:,0]]=False
        swaps=len(ai)
        need(swaps==len(ri)<=cap and np.count_nonzero(edited)==area
             and np.count_nonzero(edited!=prediction)==2*swaps<=2*budget,
             "Attainable integer fixed-area actions within rigid footprint budget")
        value=metrics(raw_intersection+gain,area,truth.area)
        diagnostic=risk(edited,(roi,edited[y0:y1,x0:x1]),upper,original_iou) if swaps else dict(
            components_added=False,components_removed=False,neighbor_crossed=False,neighbor_increased=False)
        result[name]=dict(**value,swaps=swaps,available_add=int((a_weight[adds]>0.).sum()),
                          available_remove=int((b_weight[removes]<1.).sum()),**diagnostic)
        if dense_spot:
            ey,ex=np.nonzero(edited)
            edited_union=shapely.union_all(shapely.box(ex,ey,ex+1,ey+1))
            observed=100*edited_union.intersection(truth).area/edited_union.union(truth).area
            error=abs(observed-value["iou"])
            need(error<1e-8,"Independent dense union attained swap IoU")
            result["dense_error"]=max(error,result["dense_error"])
    need(result["unrestricted"]["iou"]+1e-10>=result["constrained"]["iou"]>=raw["iou"]-1e-10,
         "Exact unrestricted, constrained and zero action partial order")
    return result


def summarize(rows, footprint, draws):
    if not rows:
        return dict(pairs=0)
    result=dict(pairs=len(rows),chips=len({r["chip"] for r in rows}),continuous_one_sided_mean=mean(r["budget"] for r in rows),
                sum_integer_caps=sum(r["cap"] for r in rows),zero_rigid_budget=sum(r["budget"]==0 for r in rows),
                subunit_positive_budget=sum(0<r["budget"]<1 for r in rows),nonzero_integer_budget=sum(r["cap"]>0 for r in rows))
    for arm in MODES:
        result[arm]=dict(**{key:mean(r[arm][key] for r in rows) for key in ("iou","fp","fn")},
                        strict=sum(r[arm]["iou"]>=90 for r in rows),
                        strict_added=sum(r["raw"]["iou"]<90<=r[arm]["iou"] for r in rows),
                        gain_ge_one=sum(r[arm]["iou"]-r["raw"]["iou"]>=1 for r in rows),
                        gain_gt_quarter=sum(r[arm]["iou"]-r["raw"]["iou"]>.25 for r in rows),
                        gain_vs_raw=footprint.paired_cluster_stats([(r["chip"],r[arm]["iou"]-r["raw"]["iou"]) for r in rows],draws))
        if arm in ("constrained","unrestricted"):
            result[arm].update(swaps_total=sum(r[arm]["swaps"] for r in rows),
                acted=sum(r[arm]["swaps"]>0 for r in rows),positive_budget_noop=sum(r["cap"]>0 and r[arm]["swaps"]==0 for r in rows),
                full_cap=sum(r["cap"]>0 and r[arm]["swaps"]==r["cap"] for r in rows),
                **{key:sum(r[arm][key] for r in rows) for key in
                   ("components_added","components_removed","neighbor_crossed","neighbor_increased")})
    for a,b in (("constrained","rigid"),("unrestricted","rigid"),("unrestricted","constrained")):
        result[f"{a}_minus_{b}"]=footprint.paired_cluster_stats([(r["chip"],r[a]["iou"]-r[b]["iou"]) for r in rows],draws)
    return result


def run(args):
    start=perf_counter()
    exact(Path(__file__).resolve().with_name("PROTOCOL.md"),PROTOCOL)
    exact(Path(__file__).resolve().with_name("RISK_PRELOCK.md"),RISK_PRELOCK)
    axis, prior=load_axis(args)
    load_original, axis_bound, summarize_axis=axis.load_original,axis.axis_bound,axis.summarize
    neighborhoods, completed=[],[]
    by_candidate={}
    state={"current":None,"axis_calls":[],"aggregates":None}

    def observed_scout(a):
        scout, expected=load_original(a)
        route,measure=scout.route,scout.measure
        def observe_route(*values):
            answer=route(*values)
            pool, decisions=answer[:2]
            neighborhoods.append(pool)
            for row in decisions.values():
                candidate=row[0]
                by_candidate[id(candidate)]=[q["rle"] for q in pool if q["rank"]<candidate["rank"]]
            return answer
        def observe_measure(row, truth, modules, hull):
            observed=measure(row,truth,modules,hull)
            need(not state["axis_calls"],"Previous candidate's two axes completed")
            state["current"]=(row[0],truth)
            return observed
        scout.route,scout.measure=observe_route,observe_measure
        return scout,expected

    def observe_axis(rectangles, truth, area, direction, offsets, overlap):
        winner=axis_bound(rectangles,truth,area,direction,offsets,overlap)
        state["axis_calls"].append((direction,winner))
        if len(state["axis_calls"])==2:
            candidate,expected=state["current"]
            need(expected is truth,"Original fixed matches align with both axes")
            completed.append(candidate_analysis(candidate,by_candidate[id(candidate)],truth,state["axis_calls"],len(completed)<2))
            state["axis_calls"].clear()
        return winner

    def observe_summary(rows, footprint, draws):
        if state["aggregates"] is None:
            need(len(rows)==len(completed)==109 and len(draws)==4096,"Exact original 109-match chip cohort")
            for r,found in zip(completed,rows,strict=True):
                need(abs(r["raw"]["iou"]-found["raw"]["iou"])<1e-8
                     and abs(r["rigid"]["iou"]-max(found["short"]["iou"],found["long"]["iou"]))<1e-10,
                     "Raw continuous geometry and original rigid winners must agree")
                r.update(chip=found["chip"], label=found["label"])
            groups=dict(all=completed,neighbor_vetoed=[v for v in completed if v["label"]=="neighbor_veto"],
                        allowed=[v for v in completed if v["label"]!="neighbor_veto"],
                        allowed_abstained=[v for v in completed if v["label"] not in ("neighbor_veto","acted")])
            state["aggregates"]={k:summarize(v,footprint,draws) for k,v in groups.items()}
        return summarize_axis(rows,footprint,draws)

    axis.load_original,axis.axis_bound,axis.summarize=observed_scout,observe_axis,observe_summary
    replay=args.output.with_name("AXIS_REPLAY_SAFE.json")
    with redirect_stdout(StringIO()):
        axis.run(SimpleNamespace(**dict(vars(args),output=replay)))
    obtained=json.loads(replay.read_bytes())
    need({k:v for k,v in obtained.items() if k!="wall_seconds"} == {k:v for k,v in prior.items() if k!="wall_seconds"}
         and len(neighborhoods)==36 and not state["axis_calls"], "Full original axis replay, excluding wall time")
    summaries=state["aggregates"]
    main=summaries["all"]
    def passes(mode):
        delta=main[f"{mode}_minus_rigid"]
        return (delta["row_weighted_mean_delta"]>=.25 and delta["row_weighted_image_cluster_percentile95"][0]>0.
                and main[mode]["strict_added"]>=main["rigid"]["strict_added"])
    decision="CONSTRAINED_ORACLE_LEAD" if passes("constrained") else "UNRESTRICTED_CAPACITY_ONLY" if passes("unrestricted") else "STOP_INTEGER_FIXED_AREA_AT_RIGID_BUDGET"
    report=dict(status="HRSID_FIT_CENTERED_SHIFT_BUDGET_SWAP_ORACLE_ACTUAL",FIT=True,DEV=False,TEST=False,AP=False,
                GT_oracle=True,images=36,matched=109,model_forward_calls=0,optimizer_updates=0,prior_replayed=True,
                prior_axis_sha256=AXIS_RESULT,protocol_sha256=PROTOCOL,risk_prelock_sha256=RISK_PRELOCK,
                bootstrap_draws=4096,independent_dense_candidates=2,independent_dense_max_abs_error=max(r["dense_error"] for r in completed),
                original_gate_reasons=prior["gate_reasons"],groups=summaries,gate=decision,wall_seconds=perf_counter()-start)
    payload=(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n").encode()
    with args.output.open("xb") as stream:
        need(stream.write(payload)==len(payload),"Fresh complete swap oracle aggregate")
    print(json.dumps(dict(status=report["status"],result_sha256=sha256(payload).hexdigest(),gate=decision)),flush=True)


def main():
    parser=ArgumentParser(allow_abbrev=False)
    for name in ("request","fit-images","fit-index","code-root","match-source","shapely-site","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--request-sha",required=True)
    args=parser.parse_args()
    need(sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES")=="-1"
         and os.environ.get("OMP_NUM_THREADS")=="1", "Exact isolated one-thread CPU")
    def offline(event,_):
        if event.startswith("socket."):
            raise PermissionError("Offline closed FIT swap oracle")
    sys.addaudithook(offline)
    run(args)


if __name__=="__main__":
    main()
