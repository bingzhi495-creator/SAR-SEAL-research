"""Compare native and PointRend ordering, then measure the geometry ceiling on fixed FIT matches."""

from argparse import ArgumentParser
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
from pycocotools import mask as codec


PRIOR_VERIFY = "daa3f6dc7751fe546700b8297b89c0f7295238b3876631f073c8f863d72d449f"
PRIOR_PEER = "c791b3eb637acf5e30341768a53fa1acb8a3c8db5d6153717474b0b6958fc319"
PRIOR_RESULT = "31aaf615621911e82c8f32c5230ebb662b1de303f5c10a8f93283918ab3ea052"


def balanced(c, q, n_add, n_drop):
    add = np.lexsort((c[:,0], -q[:,0]))[:n_add]
    drop = np.lexsort((c[:,1], q[:,1]))[:n_drop]
    return dict(flipped=np.r_[c[add,0],c[drop,1]], adds=n_add, drops=n_drop, edits=n_add+n_drop,
                area_delta=n_add-n_drop)


def expected_geometry(q, c, k, row, calc):
    truth = balanced(c,q,k,k)
    # Same support and 2k native flips. The best positions are obtained by
    # separate ordering at each feasible pair of sign counts.
    size = len(c)
    need = range(max(0,2*k-size),min(size,2*k)+1)
    flexible = max(((calc(row,balanced(c,q,n,2*k-n))["iou"],-abs(n-k),-n) for n in need))[0]
    capped = max((calc(row,balanced(c,q,n,n))["iou"],-n) for n in range(k+1))
    return calc(row,truth), flexible, capped[0], -capped[1], truth


def analyze(gt,pools,rows,budgets,witness,helpers,base,reference, protocol):
    g = helpers.geometry
    recs = {g.identity(r):r for r in rows}
    assert set(recs) == set(budgets) == set(witness) and len(recs) == protocol["eligible"]
    groups, counts = defaultdict(list), Counter()
    streams = {k:[] for k in ("N","D","B")}
    names = ("N","D","B","O_B","O_J","O_cap")
    for image in gt["images"]:
        iid = image["id"]
        for pred in pools[iid]:
            ident = (iid,pred["rank"],pred["query_id"])
            masks = dict.fromkeys(streams,pred["rle"])
            if ident in recs:
                rec = recs[ident]
                c = g.point_geometry(rec)
                k = budgets[ident]["k"]
                q = rec["old14"][:,:,0]
                assert q.shape == c.shape and np.all((q>=0)&(q<=1))
                native = balanced(c,q,k,k)
                actual = dict(flipped=witness[ident]["B"],adds=k,drops=k,edits=2*k,area_delta=0)
                for a,ed in (("D",native),("B",actual)):
                    g.sparse_edit(rec,ed)
                    masks[a] = helpers.sparse.xor_rle(pred["rle"],np.asarray(ed["flipped"],dtype=np.int64))
                    assert int(codec.area(masks[a])) == pred["area"]
                    counts[a+"_edits"] += 2*k
                counts["eligible"] += 1
                counts["D_differs_from_B"] += int(set(native["flipped"]) != set(actual["flipped"]))
                if rec["role"] == "valid":
                    truth = np.asarray(rec["truth"],dtype=np.float64).reshape((-1,2))
                    b = g.candidate_geometry(rec,actual)
                    d = g.candidate_geometry(rec,native)
                    upper,free,capped,n_best,truth_ed = expected_geometry(truth,c,k,rec,g.candidate_geometry)
                    assert upper["iou"] >= max(b["iou"],d["iou"])-1e-12 and min(free,capped) >= upper["iou"]-1e-12
                    assert capped >= b["raw_iou"]-1e-12
                    xadd,xdrop,_,_=g.sparse_edit(rec,actual)
                    yadd,ydrop,_,_=g.sparse_edit(rec,truth_ed)
                    counts["valid_matches"] += 1
                    counts["valid_budget"] += k
                    counts["oracle_capped_unused_swaps"] += k-n_best
                    counts["oracle_ceiling_retains_original"] += int(capped <= b["raw_iou"]+1e-12)
                    a_reg = truth[yadd,0].sum()-truth[xadd,0].sum()
                    d_reg = truth[xdrop,1].sum()-truth[ydrop,1].sum()
                    groups[rec["group"]].append([b["raw_iou"],d["iou"],b["iou"],upper["iou"],free,capped,
                                                   float(a_reg),float(d_reg)])
                    for arm,v in zip(names,groups[rec["group"]][-1][:6],strict=True):
                        counts[arm+"_strict90"] += int(v>.90)
                        counts[arm+"_strict90_losses"] += int(b["raw_iou"]>.90 and v<=.90)
            for a in streams:
                streams[a].append(dict(image_id=iid,category_id=1,score=pred["score"],segmentation=masks[a]))

    assert counts["valid_matches"] == 392 and len(groups) == 92
    assert all(len(rows) == protocol["mapped"] for rows in streams.values())
    assert counts["D_edits"] == counts["B_edits"] == 2*protocol["swaps"]
    metrics = {a:base.pooled(gt,streams[a],helpers.official) for a in streams}
    assert all(metrics[a]["all12"] == reference["metrics"]["arms"][a]["all12"] for a in ("N","B"))
    values = np.stack([np.asarray(groups[g]).mean(axis=0) for g in sorted(groups)])
    mean = dict(zip(names,map(float,100*values[:,:6].mean(axis=0)),strict=True))
    rng = np.random.Generator(np.random.PCG64(5217))
    resample = rng.integers(len(values),size=(4096,len(values)))
    delta = {}
    for a,b in (("D","B"),("D","N"),("O_B","B"),("O_B","D"),("O_J","O_B"),("O_cap","O_B")):
        v=100*(values[:,names.index(a)]-values[:,names.index(b)])
        delta[a+"_minus_"+b] = dict(points=float(v.mean()),interval=np.quantile(v[resample].mean(axis=1),[.025,.975]).tolist())
    assert abs(mean["B"]-reference["metrics"]["arms"]["B"]["source_equal_iou_points"]) < 1e-10
    return dict(coco_nonoracle=metrics, geometry_source_equal_points=mean, contrasts=delta,
                original_92_source_groups=len(values), counts=dict(counts), draws=4096, seed=5217,
                oracle_ranking_regret_source_equal_native_pixels=dict(add=float(values[:,6].mean()),drop=float(values[:,7].mean())))


def main():
    p = ArgumentParser(allow_abbrev=False)
    for n in ("code","work","run","owner","accepted","old-support","old-score","raw-capture","raw-acceptance",
              "source-cohort","fit-gt","shapely-site","protocol","output"):
        p.add_argument("--"+n,type=Path,required=True)
    p.add_argument("--owner-sha256",required=True)
    p.add_argument("--protocol-sha256",required=True)
    a = p.parse_args()
    assert sys.flags.isolated and sys.dont_write_bytecode and not a.output.exists()
    verifier = a.work/"verify_point_fit109.py"
    raw = verifier.read_bytes()
    assert sha256(raw).hexdigest() == PRIOR_VERIFY
    v = ModuleType("_approved_fit_closure")
    v.__file__ = str(verifier)
    exec(compile(raw,str(verifier),"exec"),v.__dict__)
    load,read = v.checked_module,v.read_doc
    lock = read(a.protocol,a.protocol_sha256)
    assert lock["FIT"] and not lock["DEV"] and not lock["TEST"]
    end=v.checked_terminal(a.owner,a.owner_sha256)
    peer=read(a.accepted,PRIOR_PEER)
    assert peer["result_sha256"] == end["result_sha256"] == PRIOR_RESULT and peer["COCO12_bit_exact_all_six_arms"]
    report=read(a.run/"RESULT_SAFE.json",PRIOR_RESULT)
    pin=read(a.work/"POINTREND_FIT109_MODULE_PINS.json",end["module_manifest_sha256"])
    worker=load(a.work/"run_point_fit109.py",pin["run_point_fit109.py"])
    orig=read(a.work/"POINTREND_FIT109_PRELOCK.json",worker.PROTOCOL_SHA)
    old_dir=a.code/"RFDETR_QUERY_FEATURE_FIT109_20261121"
    old=load(old_dir/"run_fit109_query.py","97fdc3987462aabbcd7d4fbe672c9788a3b969a221243e13d82473093170a15a")
    spot=old.load(a.code/"RFDETR_CORE36_MATCHED_POINT_SPOT_20261120/run_spot.py")
    formal=a.code/"RFDETR_MATCHED_POINT_FORMAL_20261120"
    prior,plan=(old.load(formal/n) for n in ("run_fit109.py","fit_plan.py"))
    a.capture,a.acceptance,a.acceptance_sha256 = a.raw_capture,a.raw_acceptance,old.OLD_RAW_ACCEPTANCE
    roster,_=prior.read_population(a,spot)
    base=old.load(old_dir/"fit109_score.py")
    helpers=base.load_helpers(a.code,a.shapely_site)
    pools,rows=worker.collect_old_support(a,SimpleNamespace(old=old,spot=spot),roster,orig)
    budgets,_=worker.read_budgets(a.old_score,orig,rows,plan,roster,helpers.geometry)
    gt,attached=old.attach_fit(a,roster,rows,pools,helpers,spot)
    assert len(gt["images"])==109 and len(gt["annotations"])==496
    saved=v.pinned(a.run/"PREDICTION_WITNESS_PRIVATE.jsonl",report["prediction_witness_sha256"],80*2**20)
    witness={tuple(r["key"]):r for r in map(json.loads,saved.splitlines())}
    out=analyze(gt,pools,attached,budgets,witness,helpers,base,report,lock)
    out.update(status="POINTREND_FIT109_NATIVE_RANKING_AND_OCCUPANCY_CEILING_ACTUAL",
               FIT=True,DEV=False,TEST=False,point_head_updates=0,detector_forwards=0,
               oracle_AP_computed=False, original_sha256=PRIOR_RESULT,protocol_sha256=a.protocol_sha256,
               source_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    digest=worker.write(a.output,out)
    print(json.dumps(dict(status=out["status"],sha256=digest)),flush=True)


if __name__=="__main__":
    main()
