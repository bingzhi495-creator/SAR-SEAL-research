"""Exact variable-sign truth envelope on the fixed, closed dual-reference gate."""
from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
from io import BytesIO
from itertools import combinations
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent
OLD: Path
POINTS: Path
ENUM: Path
LOCK = "48aae1fbe809492e29e4e49df5b48ecb8945548dfd6edcc586235aa94a79e5b1"
PRIOR_WORKER = "344fea17198fd8a70128aaabfcc9b13dbe188bda503a761e3222011aaf87fce0"
ENUM_SHA = "29bf58bb29ab825f7baa85a3276337ee6e2e93567fece6ff25cb140868788eda"
POINT_SHA = "87693897f873a2129764781611b0f7c35cadde51c0c4e830d7f8187c3b684049"
TERMINAL = "95f3d054cda720005f31203de62dd19e5b60b0c24a2486bba13cd4297549975a"
POINT_INDEX = "5f6f2c1897fae3307f6ef80caaa91e66bbdb07172644bb8211ba35dbabad3276"
POINT_WORKER = "926f87b220c4ba9f1a7074088a52641cd66c17d4016bac91fba20c84f450f766"
PLAN = "SPSD_GATE_VARIABLE_SIGN_SUPPORT_VERIFIED"
SCORE = "SPSD_GATE_VARIABLE_SIGN_ORACLE_COMPLETE"
WIDTHS = {"equal":0,"narrow":1,"full":8}
REFS = ("pseg","spsd")
MEMBERS = np.array([[0,0],[1,0],[0,1],[1,1]],dtype=int)
TOL = 1e-12


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(raw):
    return sha256(raw).hexdigest()


def pin(path, expected):
    raw = path.read_bytes()
    need(digest(raw)==expected,f"Pinned exact bytes: {path.name}")
    return raw


def load(path, expected, name):
    out = ModuleType(name)
    out.__file__ = str(path)
    exec(compile(pin(path,expected),str(path),"exec"),vars(out))
    return out


def point_index(prev, records):
    term = POINTS/"observed_points_retry/TERMINAL.json"
    need(prev.terminal(term,"prepare-edits",POINT_WORKER,POINT_SHA)==TERMINAL,"Exact point observer")
    safe = json.loads(pin(POINTS/"actual_retry/RESULT_SAFE.json",POINT_SHA))
    status = safe["partitions"]["source_gate"]
    directory = POINTS/"actual_retry/source_gate"
    need(status["point_index_sha256"]==POINT_INDEX and status["counts"]["images"]==183
         and status["counts"]["static_positive_candidates"]==8785 and status["counts"]["static_exchanges"]==57692
         and not any(safe[k] for k in ("train_labels_opened","test_labels_opened","test_images_opened")),
         "Prior source-gate point population")
    raw = pin(directory/"POINT_INDEX_PRIVATE.jsonl",POINT_INDEX)
    index = [json.loads(line) for line in raw.splitlines()]
    need(len(index)==183 and set(records)=={row["image_id"] for row in index},"Exact original point manifest pool")
    for row in index:
        origin = records[row["image_id"]]
        need(all(row[key]==origin[key] for key in ("original_hw","source_group"))
             and origin["filename"] is not None
             and row["filename"]==f"POINTS_PRIVATE/{row['native_archive_ordinal']:04d}.npz", "Exact point entry binding")
    return directory,index


def points_for(directory,row):
    data = pin(directory/row["filename"],row["sha256"])
    need(len(data)==row["bytes"],"Exact original point archive")
    with np.load(BytesIO(data),allow_pickle=False) as reader:
        ranks, offsets, co, static = [reader[key] for key in (
            "native_ranks","point_offsets","signed_coordinates","static_offsets")]
    need(ranks.dtype==np.int16 and offsets.dtype==co.dtype==static.dtype==np.int32
         and len(ranks)==row["supported_candidates"] and len(set(map(int,ranks)))==len(ranks)
         and len(offsets)==len(static)==len(ranks)+1 and offsets[0]==static[0]==0
         and co.shape==(offsets[-1],2) and len(co)==row["paired_sites"]
         and bool((np.diff(offsets)>0).all()),"Native row-major signed coordinate and quota rows")
    return {int(rank):(co[i:j],int(k)) for rank,i,j,k in
            zip(ranks,offsets[:-1],offsets[1:],np.diff(static),strict=True)}


def check_mask(codec, native, target, coords,k,stock):
    result = codec.decode(target).ravel(order="C").astype(bool)
    changed = result!=stock
    changed_adds = int(changed[coords[:,0]].sum())
    changed_drops = int(changed[coords[:,1]].sum())
    need(target["size"]==native["size"] and changed_adds==changed_drops==k
         and int(changed.sum())==2*k and int(result.sum())==int(stock.sum()),
         "Canonical fully supported archived equal signed mask")


def safe_write(prev,output,data):
    raw = prev.enc(data)
    prev.write(output/"RESULT_SAFE.json",raw)
    print(json.dumps(dict(status=data["status"],result_sha256=digest(raw))),flush=True)


def flow(prev,output,status,start,**values):
    output.mkdir()
    safe_write(prev,output,dict(status=status, protocol_sha256=LOCK,
          worker_sha256=digest(Path(__file__).read_bytes()), point_result_sha256=POINT_SHA,
          point_index_sha256=POINT_INDEX, prior_render_sha256=prev.OLD_RENDER, prior_score_sha256=prev.OLD_SCORE,
          detector_forwards=0, optimizer_updates=0, fit_labels_opened=False, historical_test_opened=False,
          AP=False, elapsed_seconds=perf_counter()-start,**values))


def prepare(prev,output):
    from pycocotools import mask as codec

    start = perf_counter()
    _,records,native,groups,_,_ = prev.inputs()
    directory,index = point_index(prev,records)
    counts = Counter()
    for metadata in index:
        iid = metadata["image_id"]
        entries = points_for(directory,metadata)
        for rank,pos in enumerate(native[iid],1):
            k = pos["quota"]
            if k==0:
                continue
            coordinates, stored_k = entries[rank]
            stock = codec.decode(pos["native"]).ravel(order="C").astype(bool)
            need(0<k==stored_k<=8 and k<=len(coordinates)<=32
                 and len(np.unique(coordinates))==2*len(coordinates)
                 and np.all((0<=coordinates)&(coordinates<len(stock)))
                 and not stock[coordinates[:,0]].any() and stock[coordinates[:,1]].all(),
                 "Exact original signed membership and feasible k")
            for field in ("static","bce_robust"):
                check_mask(codec,pos["native"],pos["changes"][field],coordinates,k,stock)
            counts.update(positive_masks=1,positive_signed_exchanges=k,supported_pairs=len(coordinates))
    need(counts["positive_masks"]==8785 and counts["positive_signed_exchanges"]==57692
         and len(set(groups.values()))==180,"Complete signed scope")
    flow(prev,output,PLAN,start,support=counts,images=183,decoded_groups=180,gate_labels_opened=False)


def frontier(codes, limit):
    # A state is reachable if picking exactly n sites yields x and y
    # intersections in the two references. Advance whole exchangeable
    # binary membership classes at a time, keeping all reachable counts.
    available = np.bincount(codes,minlength=4)
    size = limit+1
    reachable = np.zeros((size,size,size),dtype=bool)
    reachable[0,0,0] = True
    for (x,y),cap in zip(MEMBERS,available,strict=True):
        ahead = reachable.copy()
        for n in range(1,min(int(cap),limit)+1):
            a,b = x*n,y*n
            ahead[n:,a:,b:] |= reachable[:size-n,:size-a,:size-b]
        reachable = ahead
    out = []
    for plane in reachable:
        exact = np.argwhere(plane)
        kept,best = [],-1
        for x,y in exact[::-1]:
            if y>best:
                kept.append((x,y))
                best=y
        out.append(np.asarray(kept,dtype=int).reshape(-1,2))
    return out


def iou(area,gt,intersection):
    union = area+gt-intersection
    need(bool((union>0).all() and (intersection>=0).all()
              and (intersection<=gt).all() and (intersection<=area).all()),"Valid raster areas")
    return 100.*intersection/union


def rows_for_q(a,b,add,drop,area,gt,inter,baseline,denom):
    # Drops were represented by absent counts, so exactly b sites give
    # b-absent occupied reference cells.
    intersections = inter + add[:,None,:] + drop[None,:,:] - b
    delta = (iou(area+a-b,gt,intersections)-baseline)/denom
    return intersections,delta.reshape(-1,2)


def candidate(k,p,d,area,gt,inter,baseline,denom,baseline_i,exhaustive):
    limit=min(len(p),2*k)
    add,drop=frontier(p,limit),frontier(3-d,limit)
    by_count = {}
    for a in range(max(0,2*k-len(d)),limit+1):
        b=2*k-a
        _,delta=rows_for_q(a,b,add[a],drop[b],area,gt,inter,baseline,denom)
        minimum = np.min(delta,axis=1)
        best = float(minimum.max())
        maxima = np.flatnonzero(minimum==best)
        index = maxima[int(np.argmax(delta[maxima].sum(axis=1)))]
        by_count[a] = (delta[index], best, delta.max(axis=0))
    check_add = exhaustive.count_vectors(np.bincount(p,minlength=4),k)@MEMBERS
    check_drop = exhaustive.count_vectors(np.bincount(d,minlength=4),k)@MEMBERS
    exact = inter + check_add[:,None,:] - check_drop[None,:,:]
    need(bool(np.any(np.all(exact==baseline_i,axis=2))),"Archived BCE mask intersection feasible within balanced pool")
    check = float(np.min((iou(area,gt,exact)-baseline)/denom,axis=2).max())
    need(abs(check-by_count[k][1])<=TOL and check>=-TOL,"Independent previous balanced exhaustive solve parity")
    references, picks, values = {},{},{}
    for family,width in WIDTHS.items():
        possible = [a for a in by_count if abs(a-k)<=width]
        best = max(possible,key=lambda a:(by_count[a][1],-abs(a-k),-a))
        ref, minimum, _ = by_count[best]
        picks[family]=best
        values[family]=np.r_[ref,minimum]
        references[family]=np.stack([by_count[a][2] for a in possible]).max(axis=0)
    need(values["equal"][2]<=values["narrow"][2]+TOL<=values["full"][2]+2*TOL,"Nested per-candidate objective")
    return values,picks,references


def synthetic(exhaustive):
    codes = np.array([0,1,2,3,3,1])
    for n,states in enumerate(frontier(codes,6)):
        truth = {tuple(MEMBERS[list(combo)].sum(axis=0)) for combo in combinations(codes,n)}
        pset = {a for a in truth if not any(b!=a and b[0]>=a[0] and b[1]>=a[1] for b in truth)}
        need({tuple(r) for r in states}==pset,"Exact multi-reference Pareto counts against subsets")
    p = np.array([3,3,0])
    d = np.array([3,0,0])
    k = 1
    area,gt,inter = 7,np.array([10,10]),np.array([5,5])
    baseline_i = inter+1
    base = iou(area,gt,baseline_i)
    result,picks,_ = candidate(k,p,d,area,gt,inter,base,np.ones(2),baseline_i,exhaustive)
    need(picks=={"equal":1,"narrow":2,"full":2} and result["full"][2]>result["equal"][2],
         "Known area expansion wins with equal total flips")
    p = np.array([0,0,3])
    d = np.array([3,0,0])
    result,picks,_ = candidate(k,p,d,area,gt,inter,base,np.ones(2),baseline_i,exhaustive)
    need(picks["equal"]==1 and picks["full"]==1 and abs(result["equal"][2]-result["full"][2])<=TOL,
         "Balanced incumbent preferred under infeasible directional gain")


def from_rle(codec, n, labels):
    return np.asarray([int(codec.area(codec.merge([n,r],intersect=True))) for r in labels],dtype=int)


def solve(prev,output,exhaustive):
    from pycocotools import mask as codec

    start = perf_counter()
    terminal = ROOT/"observed_prelabel/TERMINAL.json"
    term = json.loads(terminal.read_bytes())
    receipt = term["result_sha256"]
    term_sha = prev.terminal(terminal,"capture",digest(Path(__file__).read_bytes()),receipt)
    proof = json.loads(pin(ROOT/"prelabel/RESULT_SAFE.json",receipt))
    need(proof["status"]==PLAN and proof["gate_labels_opened"] is False and proof["protocol_sha256"]==LOCK
         and proof["point_result_sha256"]==POINT_SHA and proof["point_index_sha256"]==POINT_INDEX,
         "Exact label-free signed feasibility proof")
    reader,records,native,groups,prior,_ = prev.inputs()
    directory,index = point_index(prev,records)
    labels = reader.labels(SimpleNamespace(labels=prev.LABELS),records)
    matches = {name:prev.native_050(labels[name],native) for name in REFS}
    need([len(labels[name]["annotations"]) for name in REFS]==[411,410]
         and [len(matches[name]) for name in REFS]==[311,308],"Whole fixed native match and hull denominator")
    pairs,denom,exposure = prev.paired_truth(codec,native,groups,labels,matches)
    count_group = Counter(groups.values())
    g_index = {g:i for i,g in enumerate(sorted(count_group))}
    agg = {key:np.zeros((180,3)) for key in WIDTHS}
    upper = {key:np.zeros((180,2)) for key in WIDTHS}
    choices,signed = {key:Counter() for key in WIDTHS},{key:Counter() for key in WIDTHS}
    changed = Counter()
    parity = 0
    shared_images,shared_groups=set(),set()
    for metadata in index:
        iid = metadata["image_id"]
        group = groups[iid]
        if not pairs[iid]:
            continue
        supported = points_for(directory,metadata)
        for rank,targets in pairs[iid].items():
            pos = native[iid][rank-1]
            k = pos["quota"]
            if k==0:
                continue
            xy,stored_k = supported[rank]
            need(k==stored_k and k<=8,"Original point quota")
            flat = [codec.decode(label).ravel(order="C") for label in targets]
            codes = flat[0][xy]+2*flat[1][xy]
            p,d = codes.T
            n,bce = pos["native"],pos["changes"]["bce_robust"]
            area = int(codec.area(n))
            target_areas=np.asarray([int(codec.area(label)) for label in targets],dtype=int)
            native_intersection=from_rle(codec,n,targets)
            bce_intersection=from_rle(codec,bce,targets)
            baseline=iou(area,target_areas,bce_intersection)
            need(np.allclose(baseline,100*codec.iou([bce],list(targets),[0,0])[0],rtol=0,atol=1e-12),
                 "Independent mask IoU parity")
            divisor = denom[iid]*count_group[group]
            delta,picks,refs = candidate(k,p,d,area,target_areas,native_intersection,baseline,divisor,
                                         bce_intersection,exhaustive)
            parity += 1
            shared_images.add(iid)
            shared_groups.add(group)
            for family in WIDTHS:
                agg[family][g_index[group]] += delta[family]
                upper[family][g_index[group]] += refs[family]
                shift = picks[family]-k
                choices[family][str(shift)]+=1
                signed[family].update(expands=int(shift>0),shrinks=int(shift<0),balanced=int(shift==0),
                                      adds=picks[family],drops=2*k-picks[family])
                changed[family]+=int(delta[family][2]>delta["equal"][2]+TOL)
    need(parity==206 and len(shared_groups)==107,"Exact common positive support")
    draws = np.random.Generator(np.random.PCG64(5297)).integers(180,size=(4096,180))

    def stat(rows):
        return dict(points=rows.mean(axis=0).tolist(),
                    bootstrap95_points=np.percentile(rows[draws].mean(axis=1),[2.5,97.5],axis=0).T.tolist())

    contrasts = {}
    for lhs,rhs in (("narrow","equal"),("full","equal"),("full","narrow")):
        contrasts[f"{lhs}_minus_{rhs}"]=dict(stat(agg[lhs]-agg[rhs]),
            independent_ref_ceiling_delta_points=(upper[lhs]-upper[rhs]).mean(axis=0).tolist())
    flow(prev,output,SCORE,start,images=183,decoded_groups=180,gate_labels_opened=True,
         author_hulls=[411,410],native_matches=[311,308],positive_common_hulls=parity,
         positive_common_images=len(shared_images),positive_common_groups=len(shared_groups),original_association=exposure,
         full_pool_equal_independent_parity=parity, group_bootstrap_draws=4096,bootstrap_seed=5297,
         label_free_result_sha256=receipt,label_free_terminal_sha256=term_sha,balanced_enumerator_sha256=ENUM_SHA,
         metrics_order=["pseg","spsd","per_hull_min_normalized"],synthetic_checks=3,
         nested_family_results={key:dict(stat(agg[key]),independent_reference_upper=stat(upper[key]),
              selected_signed_counts=signed[key],add_minus_k_histogram=choices[key],
              strictly_better_hulls_vs_equal=changed[key]) for key in WIDTHS},contrasts=contrasts,
         incremental_full_geometry_gt_015=contrasts["full_minus_equal"]["points"][2]>.15)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("prelabel","score","selftest"))
    parser.add_argument("--output",type=Path)
    parser.add_argument("--closed-adoption-oracle", type=Path, required=True)
    parser.add_argument("--point-source", type=Path, required=True)
    parser.add_argument("--balanced-enumerator", type=Path, required=True)
    a=parser.parse_args()
    if a.output is not None:
        a.output = a.output.resolve()
    global OLD, POINTS, ENUM
    OLD = a.closed_adoption_oracle.resolve()
    POINTS = a.point_source.resolve()
    ENUM = a.balanced_enumerator.resolve()
    need(sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES")=="-1",
         "Isolated CPU only")
    pin(ROOT/"PRELOCK.md",LOCK)
    prev=load(OLD/"selective_bound.py",PRIOR_WORKER,"_prior")
    sys.path.insert(0,str(prev.CODEC))
    exhaustive=load(ENUM,ENUM_SHA,"_prior_balanced_enumerator")
    synthetic(exhaustive)
    if a.phase=="selftest":
        print("Three locked analytic checks pass.")
        return
    need(a.output is not None and a.output.parent==ROOT and not a.output.exists(),"Fresh private output")
    if a.phase=="prelabel":
        prepare(prev,a.output)
    else:
        solve(prev,a.output,exhaustive)


if __name__=="__main__":
    main()

