"""Compare equal-menu short/long translation ceilings after exact scout replay."""

from argparse import ArgumentParser
from collections import Counter
from contextlib import redirect_stdout
from hashlib import sha256
from io import StringIO
import json
import os
from pathlib import Path
from statistics import mean
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace

import numpy as np

PROTOCOL_SHA = "3b4e08b3da745499ddd50c1128d6905c12d6b44211c116f1d8e13bb070b27478"
REQUEST_SHA = "877dfbe0db56ac1087d236f0547101d59a724027061ec271c2e345161a381135"
SOURCE_SHA = "4af4c1ffe90b2af091498ca1225b5d3602c9ad774a6298b8c34aed004c1d24d8"
RESULT_SHA = "5c63e31d63f8e103b9406dc99b2facc229365bc0c3d6ef4eb338aa5168655dfe"
TOOL_SHA = "a1c2c5b72d4fcd67716cbbc3fa92a076d891158b78475e14d77eb7d4495cb038"
WAIT_SHA = "fbb50f4f40764f0cd3a08fdec4b31f71ae4b049059588a31265189639541d4fb"
OFFSETS = np.array([-2., -1., -.5, 0., .5, 1., 2.])


def need(ok, why):
    if not ok:
        raise ValueError(why)


def exact(path, digest):
    need(path == path.resolve(strict=True) and 0 < path.stat().st_size <= 2**20, "Bounded canonical input")
    raw = path.read_bytes()
    need(sha256(raw).hexdigest() == digest, "Exact closed source: " + path.name)
    return raw


def load_original(args):
    draft = args.request.parent
    actual = draft.parent / "HRSID_FIT_CENTERED_TWO_FLANK_SCOUT_ACTUAL_20261112_01"
    outer = json.loads(exact(draft / "TOOL_TERMINAL_PRIVATE.json", TOOL_SHA))
    wait = json.loads(exact(actual / "ORIGINAL_WAIT_PRIVATE.json", WAIT_SHA))
    end = json.loads(outer["original_exec"]["output"].splitlines()[-1])
    tail = json.loads(exact(actual / "ORIGINAL.stdout", wait["original_stdout_sha256"]).splitlines()[-1])
    exact_path = actual / "ORIGINAL.stderr"
    need(sha256(exact_path.read_bytes()).hexdigest() == wait["original_stderr_sha256"], "Exact original stderr digest")
    result = json.loads(exact(actual / "RESULT_SAFE.json", RESULT_SHA))
    need(outer["terminal_observed"] and outer["original_exec"]["exit_code"] == wait["original_os_exit"] == 0
         and wait["original_os_wait"] and wait["job_assigned"] and wait["process_group_closed"] and wait["stop_reason"] is None
         and end["wait_sha256"] == WAIT_SHA and end["result_sha256"] == tail["result_sha256"] == RESULT_SHA
         and result["FIT"] and not any(result[k] for k in ("DEV", "TEST", "AP", "optimizer_updates", "model_forward_calls"))
         and result["counts"]["matched_eligible"] == 109 and result["request_sha256"] == args.request_sha == REQUEST_SHA
         and wait["score_source_sha256"] == SOURCE_SHA, "Exact original closed 109-target scout")
    exact(args.request, REQUEST_SHA)
    path = draft / "score_flank.py"
    mod = ModuleType("_closed_two_flank_scout")
    mod.__file__ = str(path)
    exec(compile(exact(path, SOURCE_SHA), str(path), "exec"), mod.__dict__)
    return mod, result


def axis_bound(rectangles, truth, area, axis, offsets, overlap):
    need(np.isclose(np.linalg.norm(axis),1.,rtol=0.,atol=1e-12), "Frozen unit axis required")
    ref = float(truth.area)
    candidates = []
    for offset in offsets:
        boxes = rectangles + np.tile(float(offset)*axis,2)
        moved_area = float((boxes[:,2:] - boxes[:,:2]).prod(axis=1).sum())
        need((boxes[:,:2]>=-1e-10).all() and (boxes[:,2:]<=800+1e-10).all()
             and abs(moved_area-area)<1e-8, "Original prediction support and foreground area preserved")
        inter = overlap(boxes,truth)
        need(0.<=inter<=min(area,ref)+1e-8, "Valid continuous footprint intersection")
        inter = min(inter,area,ref)
        candidates.append(dict(offset=float(offset),iou=100*inter/(area+ref-inter),
                               fp=100*(area-inter)/ref,fn=100*(ref-inter)/ref))
    winner = min(candidates,key=lambda row:(-row["iou"],abs(row["offset"]),row["offset"]))
    need(winner["iou"]>=candidates[3]["iou"], "Oracle must dominate original zero action")
    return winner


def dense_verify(rle, truth, direction, winner):
    from pycocotools import mask
    import shapely
    from shapely.affinity import translate

    y, x = np.nonzero(mask.decode(rle))
    original = shapely.union_all(shapely.box(x,y,x+1,y+1))
    dx,dy = winner["offset"]*direction
    moved = translate(original,xoff=float(dx),yoff=float(dy))
    expected = 100*moved.intersection(truth).area/moved.union(truth).area
    error = abs(expected-winner["iou"])
    need(error<1e-8, "Independent dense pixel union translated IoU")
    return error


def reason(flank, values):
    paired = flank.paired(values)
    halves = np.stack((paired[:3].mean(axis=0),paired[3:].mean(axis=0)))
    a,b = map(flank.best_offset,halves)
    if a != b:
        return "halves_disagree"
    if a == flank.ZERO:
        return "both_select_zero"
    if np.min(halves[:,a])<1.:
        return "gain_below_one"
    if values[:,:,a].mean()<=0.:
        return "nonpositive_raw_average"
    return "acted"


def summarize(records, footprint, draws):
    if not records:
        return dict(pairs=0)
    gains = lambda axis: [(r["chip"],r[axis]["iou"]-r["raw"]["iou"]) for r in records]
    paired = [(r["chip"],r["short"]["iou"]-r["long"]["iou"]) for r in records]
    out = dict(pairs=len(records),logical_chips=len({r["chip"] for r in records}),
               raw={key:mean(r["raw"][key] for r in records) for key in ("iou","fp","fn","short")},
               short_minus_long=footprint.paired_cluster_stats(paired,draws))
    for axis in ("short","long"):
        out[axis]=dict(**{key:mean(r[axis][key] for r in records) for key in ("iou","fp","fn")},
            gain=footprint.paired_cluster_stats(gains(axis),draws),
            nonzero_winners=sum(r[axis]["offset"]!=0 for r in records),
            gains_gt_quarter=sum(v>.25 for _,v in gains(axis)),
            gains_ge_one=sum(v>=1 for _,v in gains(axis)),
            new_strict=sum(r["raw"]["iou"]<90<=r[axis]["iou"] for r in records))
    return out


def run(args):
    start = perf_counter()
    exact(Path(__file__).resolve().parent/"PROTOCOL.md",PROTOCOL_SHA)
    original, prior = load_original(args)
    route, measure = original.route, original.measure
    cohort, records = {}, []
    state = dict(routes=0,dense_checks=0,dense_error=0.,footprint=None)

    def observed_route(image, block, logits, queries, match, flank):
        chip = state["routes"]
        state["routes"] += 1
        decisions = []
        previous = flank.selected
        need(np.array_equal(flank.OFFSETS, OFFSETS), "Exact original offset menu")
        def traced(values):
            answer = previous(values)
            failed = reason(flank,values)
            need((answer!=flank.ZERO)==(failed=="acted"), "Exact original flank decision outcome")
            decisions.append(failed)
            return answer
        flank.selected = traced
        try:
            out = route(image,block,logits,queries,match,flank)
        finally:
            flank.selected = previous
        labels = iter(decisions)
        for row in out[1].values():
            label = "neighbor_veto" if row[4] else next(labels)
            cohort[id(row)] = (chip,label)
        need(next(labels,None) is None,"One original agreement gate per nonvetoed candidate")
        return out

    def observed_measure(row, truth, modules, hull):
        metrics = measure(row,truth,modules,hull)
        state["footprint"] = modules.footprint
        chip,label = cohort[id(row)]
        candidate, _, normal, _, _ = row
        tangent = np.array([normal[1],-normal[0]])
        rle = candidate["rle"]
        rectangles = modules.geometry.boxes_from_counts(modules.shift.runs(rle["counts"]),*rle["size"])
        area = float(candidate["area"])
        record = dict(chip=chip,label=label,raw=metrics["zero"])
        for axis,direction in (("short",normal),("long",tangent)):
            winner = axis_bound(rectangles,truth,area,direction,OFFSETS,modules.geometry.overlap_area)
            need(winner["iou"]+1e-10>=metrics["zero"]["iou"],"Width-preserving oracle cannot lose vs zero")
            record[axis] = winner
            if len(records)<2:
                state["dense_error"] = max(state["dense_error"],dense_verify(rle,truth,direction,winner))
                state["dense_checks"] += 1
        records.append(record)
        return metrics

    replay_path = args.output.with_name("PARENT_REPLAY_SAFE.json")
    original.route, original.measure = observed_route, observed_measure
    with redirect_stdout(StringIO()):
        original.run(SimpleNamespace(**dict(vars(args),output=replay_path)))
    replay = json.loads(replay_path.read_bytes())
    need({k:v for k,v in replay.items() if k!="wall_seconds"} == {k:v for k,v in prior.items() if k!="wall_seconds"},
         "All parent scout evidence except wall time must replay exactly")
    need(len(records)==109 and state["routes"]==36 and state["dense_checks"]==4,
         "Exact fixed closed scout cohorts")
    counts = Counter(record["label"] for record in records)
    need(counts["acted"] == prior["counts"]["matched_acted"] == 2,"Retain exact original matched actions")
    groups = dict(all=records,
                  neighbor_vetoed=[r for r in records if r["label"]=="neighbor_veto"],
                  allowed=[r for r in records if r["label"]!="neighbor_veto"],
                  allowed_abstained=[r for r in records if r["label"] not in ("neighbor_veto","acted")],
                  allowed_acted=[r for r in records if r["label"]=="acted"])
    footprint = state["footprint"]
    draws = footprint.bootstrap_draws(state["routes"])
    result = dict(status="HRSID_FIT_CENTERED_TWO_FLANK_AXIS_ORACLE_ACTUAL",FIT=True,DEV=False,TEST=False,AP=False,
                  model_forward_calls=0,optimizer_updates=0,GT_oracle=True,images=36,matched=109,
                  prior_scouter_result_sha256=RESULT_SHA,protocol_sha256=PROTOCOL_SHA,original_replayed=True,
                  gate_reasons=dict(counts),original_counts=prior["counts"],matched_cohorts={k:summarize(v,footprint,draws) for k,v in groups.items()},
                  dense_checks=state["dense_checks"],dense_max_error=state["dense_error"],wall_seconds=perf_counter()-start)
    data = (json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    with args.output.open("xb") as stream:
        need(stream.write(data)==len(data),"Complete fresh aggregate")
    print(json.dumps(dict(status=result["status"],result_sha256=sha256(data).hexdigest())),flush=True)


def main():
    parser=ArgumentParser(allow_abbrev=False)
    for name in ("request","fit-images","fit-index","code-root","match-source","shapely-site","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--request-sha",required=True)
    args=parser.parse_args()
    need(sys.flags.isolated and os.environ.get("CUDA_VISIBLE_DEVICES")=="-1"
         and os.environ.get("OMP_NUM_THREADS")=="1", "Exact isolated one-thread CPU")
    def offline(event,_):
        if event.startswith("socket."):
            raise PermissionError("Offline closed FIT oracle")
    sys.addaudithook(offline)
    run(args)


if __name__=="__main__":
    main()
