"""Measure candidate association structure on one already closed HRSID FIT carrier."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
from statistics import median
import sys
from time import perf_counter
from types import ModuleType

import numpy as np

CAPTURE_SHA = "582d11542c3ec777fb2309493459045d51b51edc500a888970ccb03d2b57b3ac"
PROTOCOL_SHA = "55c6294f6efc1a0aa5a9b2ec677276a55f5aae1dba2c9ae56a2cf0fe4addc2ea"
FLOW_SHA = "397e8b7f663c6c1e26d8140664a7adeb7fa8554e6532b5514ed7d7d8cfb7948b"
MATCH_SHA = "caf79d33d7897999215c04c01dd6d0b55fc6fb22a1de6168dd0c9da647f39e6b"
SEEDS = (5201,5202,5203)
DONORS = (5201,5203)


def need(condition, message):
    if not condition:
        raise ValueError(message)


def pinned(path, digest, limit=32*2**20):
    need(path.resolve(strict=True)==path and 0<path.stat().st_size<=limit, "Exact bounded file")
    raw=path.read_bytes()
    need(sha256(raw).hexdigest()==digest,"Exact closed digest: "+path.name)
    return raw


def module(path, digest, name):
    m=ModuleType(name)
    m.__file__=str(path)
    sys.modules[name]=m
    exec(compile(pinned(path,digest),str(path),"exec"),m.__dict__)
    return m


def first_exclusive(left, right):
    out={}
    used_l=set()
    used_r=set()
    for i,(row_l,row_r) in enumerate(zip(left,right,strict=True)):
        candidates_l=[int(j) for j in np.argsort(-row_l,kind="stable") if row_l[j]>=.50 and int(j) not in used_l]
        candidates_r=[int(j) for j in np.argsort(-row_r,kind="stable") if row_r[j]>=.50 and int(j) not in used_r]
        if candidates_l and candidates_r:
            l,r=candidates_l[0],candidates_r[0]
            out[i]=l,r
            used_l.add(l)
            used_r.add(r)
    return out


def greedy_representatives(overlap, threshold):
    kept=[]
    lookup={}
    for row in range(overlap.shape[0]):
        rep=next((i for i in kept if overlap[i,row]>=threshold),None)
        if rep is None:
            rep=row
            kept.append(row)
        lookup[row]=rep
    return kept,lookup


def iou(a,b,codec):
    if not a or not b:
        return np.empty((len(a),len(b)),dtype=np.float64)
    return codec.iou([x["rle"] for x in a],[x["rle"] for x in b],[0]*len(b))


def lifted(local, anchor, left, right):
    return {anchor[i]:(left[l],right[r]) for i,(l,r) in local.items()}


def reciprocal(left,right,matcher,margin):
    a,_=matcher.reciprocal_matches(left,margin=margin)
    b,_=matcher.reciprocal_matches(right,margin=margin)
    return {i:(j,b[i]) for i,j in a.items() if i in b}


def select(mats, anchor_ids, donor_ids, eligible, matcher, solver):
    left,right=(mats[s][np.ix_(anchor_ids,donor_ids[s])] for s in DONORS)
    rec=reciprocal(left,right,matcher,.10)
    no_margin=reciprocal(left,right,matcher,0.)
    rec=lifted(rec,anchor_ids,donor_ids[5201],donor_ids[5203])
    no_margin=lifted(no_margin,anchor_ids,donor_ids[5201],donor_ids[5203])
    kept=[i for i,id_ in enumerate(anchor_ids) if id_ in eligible]
    a_ids=[anchor_ids[i] for i in kept]
    l,r=left[kept],right[kept]
    both=(l>=.50).any(axis=1)&(r>=.50).any(axis=1)
    flow=solver.common_flow(l>=.50,r>=.50)
    first=first_exclusive(l,r)
    need(len(first)<=len(flow)<=int(both.sum()),"Exact flow nested capacity")
    return a_ids,int(both.sum()),{
        "reciprocal":{i:p for i,p in rec.items() if i in eligible},
        "no_margin":{i:p for i,p in no_margin.items() if i in eligible},
        "first":lifted(first,a_ids,donor_ids[5201],donor_ids[5203]),
        "flow":lifted(flow,a_ids,donor_ids[5201],donor_ids[5203]),
    }


def one(blocks,matcher,solver,codec):
    first=blocks[5202]
    need(all((b["ordinal"],b["basename"],b["rgb_sha256"]) ==
             (first["ordinal"],first["basename"],first["rgb_sha256"]) for b in blocks.values()),
         "Exact same image")
    rows={s:b["candidates"] for s,b in blocks.items()}
    for data in rows.values():
        need(all(p["area"]>0 for p in data) and [p["rank"] for p in data]==sorted({p["rank"] for p in data}),
             "Native rank and nonempty support")
    mats={s:iou(rows[5202],rows[s],codec) for s in DONORS}
    anchor_ids=list(range(len(rows[5202])))
    donor_ids={s:list(range(len(rows[s]))) for s in DONORS}
    eligible={i for i,p in enumerate(rows[5202]) if 32<=p["area"]<1024}
    _,edge_count,orig=select(mats,anchor_ids,donor_ids,eligible,matcher,solver)
    methods={"raw_"+k:v for k,v in orig.items()}
    denom=dict(raw_eligible=len(eligible),raw_both_edges=edge_count)
    same={s:iou(rows[s],rows[s],codec) for s in SEEDS}
    reps={}
    for pct in (90,85):
        nms={s:greedy_representatives(same[s],pct/100) for s in SEEDS}
        a_count,cap,computed=select(mats,nms[5202][0],{s:nms[s][0] for s in DONORS},eligible,matcher,solver)
        tag="nms"+str(pct)
        denom[tag+"_eligible"]=len(a_count)
        denom[tag+"_both_edges"]=cap
        for s in SEEDS:
            denom[f"{tag}_suppressed_{s}"]=len(rows[s])-len(nms[s][0])
        roster=set(a_count)
        methods[tag+"_original_on_reps"]={i:p for i,p in orig["reciprocal"].items() if i in roster}
        methods.update({tag+"_"+k:v for k,v in computed.items()})
        reps[str(pct)]={s:{rows[s][i]["rank"]:rows[s][j]["rank"] for i,j in nms[s][1].items()} for s in SEEDS}
    witness={}
    shifts={}
    for key,pairs in methods.items():
        witness[key]={"pairs":{rows[5202][i]["rank"]:[rows[5201][l]["rank"],rows[5203][r]["rank"]]
                                for i,(l,r) in sorted(pairs.items())}}
        offsets=[]
        for i,(l,r) in pairs.items():
            a=rows[5202][i]["area"]
            offsets.append(median((a,rows[5201][l]["area"],rows[5203][r]["area"]))-a)
        shifts[key]=Counter("add" if x>0 else "drop" if x<0 else "zero" for x in offsets)
    return dict(ordinal=first["ordinal"],basename=first["basename"],methods=witness,reps=reps),denom,shifts


def encode(obj):
    return (json.dumps(obj,sort_keys=True,allow_nan=False,indent=None)+"\n").encode()


def main(args):
    from pycocotools import mask as codec
    import scipy

    start=perf_counter()
    here=Path(__file__).resolve().parent
    pinned(here/"PROTOCOL.md",PROTOCOL_SHA)
    solver=module(here/"dual_flow.py",FLOW_SHA,"_structure_dual_flow")
    matcher=module(args.match_source,MATCH_SHA,"_structure_original_match")
    need(scipy.__version__=="1.15.3","Fixed SciPy flow witness version")
    r=json.loads(pinned(args.capture/"RESULT_SAFE.json",CAPTURE_SHA))
    need(r["FIT"] and not (r["DEV"] or r["TEST"] or r["GT_opened"]) and r["coverage"]["gate"]=="COVERAGE_STOP",
         "Closed original no-label result")
    arms={a["seed"]:[json.loads(line) for line in pinned(args.capture/str(a["seed"])/"CANDIDATES_PRIVATE.jsonl",
                                                       a["prediction_sha256"]).splitlines()] for a in r["arms"]}
    need(set(arms)==set(SEEDS) and all(len(blocks)==36 for blocks in arms.values()),"Exact original row counts")
    totals=Counter()
    denom_groups=Counter()
    method_counts=defaultdict(Counter)
    groups=Counter()
    directional_groups=Counter()
    lines=[]
    for rows in zip(*(arms[s] for s in SEEDS),strict=True):
        witness,denom,shifts=one(dict(zip(SEEDS,rows,strict=True)),matcher,solver,codec)
        totals.update(denom)
        denom_groups.update({key:1 for key,count in denom.items() if count>0})
        for method,counts in shifts.items():
            method_counts[method].update(counts)
            groups[method]+=int(bool(counts))
            directional_groups[method]+=int(bool(counts["add"]+counts["drop"]))
        lines.append(encode(witness))
    summary={}
    for method,counts in method_counts.items():
        both=sum(counts.values())
        nonzero=counts["add"]+counts["drop"]
        g=groups[method]
        dg=directional_groups[method]
        summary[method]=dict(covered=both,covered_groups=g,nonzero=nonzero,nonzero_groups=dg,
                             offsets={k:counts[k] for k in ("add","drop","zero")},
                             old_gate_counts_met=both>=60 and g>=20 and nonzero>=30 and dg>=15)
    old=summary["raw_reciprocal"]
    need((totals["raw_eligible"],old["covered"],old["covered_groups"],old["offsets"]) ==
         (r["coverage"]["eligible"],r["coverage"]["covered_both"],r["coverage"]["covered_groups"],r["coverage"]["offsets"]),
         "Original headline and exact area offset counts")
    args.output.mkdir()
    data=b"".join(lines)
    (args.output/"WITNESS_PRIVATE.jsonl").write_bytes(data)
    result=dict(status="HRSID_FIT_ASSOC_STRUCTURE_ACTUAL",FIT=True,DEV=False,TEST=False,GT_opened=False,
                AP=False,model_forwards=0,optimizer_updates=0,association_only=True,images=36,
                capture_sha256=CAPTURE_SHA,protocol_sha256=PROTOCOL_SHA,witness_sha256=sha256(data).hexdigest(),
                totals=totals,denominator_groups=denom_groups,methods=summary,worker_seconds=perf_counter()-start)
    safe=encode(result)
    (args.output/"RESULT_SAFE.json").write_bytes(safe)
    print(json.dumps(dict(status=result["status"],result_sha256=sha256(safe).hexdigest())))


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    for field in ("capture","match-source","output"):
        p.add_argument("--"+field,type=Path,required=True)
    args=p.parse_args()
    need(sys.flags.isolated and os.environ.get("CUDA_VISIBLE_DEVICES")=="-1","Isolated CPU only")
    def offline(event,_args):
        if event.startswith("socket."):
            raise PermissionError("No network")
    sys.addaudithook(offline)
    main(args)
