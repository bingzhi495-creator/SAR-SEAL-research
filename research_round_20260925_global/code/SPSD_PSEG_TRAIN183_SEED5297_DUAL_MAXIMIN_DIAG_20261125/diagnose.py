"""Decompose finite-family regret on the previously scored source gate."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from copy import deepcopy
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
from time import perf_counter
from types import ModuleType, SimpleNamespace

import numpy as np
from pycocotools import mask as codec
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scipy.optimize import linear_sum_assignment

LOCK = "d3e876ce1a253b7bd17826ecf78c252a0f2b31fd308fb03d3b310e7bfe155320"
GROUP_PROTOCOL = "4030a3cb5684a6dac0ea1e7765a48d05caec113c5c6e0e6de2c82b991e9d5f15"
FAMILY = "c2a95dcf5488ccfd1f322850f5c4a7755a1d5c536c09d8b59f9a7007ee67efab"
ORACLE = "29bf58bb29ab825f7baa85a3276337ee6e2e93567fece6ff25cb140868788eda"
TOL = 1e-12
OPTIONS = ("deployed_robust", "deployed_mean", "deployed_cell_pessimistic", "predicted_normalized_family",
           "true_raw_family", "true_normalized_family", "true_normalized_full")
LABELS = ("pseg", "spsd", "per_hull_minimum")


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(blob):
    return sha256(blob).hexdigest()


def pin(path, expected):
    raw = path.read_bytes()
    need(digest(raw) == expected, "Exact bound input: " + path.name)
    return raw


def load(path, expected):
    code = ModuleType(path.stem)
    code.__file__ = str(path)
    exec(compile(pin(path, expected), str(path), "exec"), code.__dict__)
    return code


def verify_prelabel(root):
    term_raw = (root/"FAMILY_TERMINAL.json").read_bytes()
    terminal = json.loads(term_raw)
    need(terminal["phase"] == "prepare" and terminal["actual_exit"] == 0
         and terminal["output_footer_verified"] is True and terminal["worker_sha256"] == FAMILY,
         "Returned label-free family observer termination")
    safe = json.loads(pin(root/"prepare/RESULT_SAFE.json", terminal["result_sha256"]))
    need(safe["status"] == "SPSD_DUAL_5297_ORACLE_FAMILY_SEALED"
         and safe["prelock_sha256"] == LOCK and safe["worker_sha256"] == FAMILY
         and safe["detector_forwards"] == safe["optimizer_updates"] == 0
         and safe["gate_labels_opened"] is safe["historical_test_opened"] is False,
         "Exact closed label-free preparation")
    rows = [json.loads(x) for x in pin(root/"prepare/FAMILY_PRIVATE.jsonl", safe["family_sha256"]).splitlines()]
    need(len(rows) == len({x["image_id"] for x in rows}) == safe["counts"]["images"] == 183,
         "Complete exact family image stream")
    return {x["image_id"]: x for x in rows}, safe, terminal["result_sha256"], digest(term_raw)


def stock_matches(truth, native):
    index, dt = {}, []
    for iid, rows in native.items():
        for i, row in enumerate(rows, 1):
            index[len(dt) + 1] = (iid, i)
            dt.append(dict(image_id=iid,category_id=0,score=row["score"],segmentation=row["native"]))
    with redirect_stdout(StringIO()):
        gt = COCO()
        gt.dataset = deepcopy(truth)
        gt.createIndex()
        ev = COCOeval(gt, gt.loadRes(deepcopy(dt)), "segm")
        ev.params.catIds, ev.params.imgIds, ev.params.maxDets = [0], sorted(native), [1,10,100]
        ev.params.iouThrs = np.array([.50])
        ev.evaluate()
    matches = {}
    scope = [x for x in ev.evalImgs if x is not None and x["aRng"] == ev.params.areaRng[0] and x["maxDet"] == 100]
    need(len(scope) == 183 and sum(len(x["gtIds"]) for x in scope) == len(truth["annotations"]),
         "Whole native match scope")
    for row in scope:
        for target, prediction, ignore in zip(row["gtIds"],row["gtMatches"][0],row["gtIgnore"],strict=True):
            if prediction and not ignore:
                key = int(target)
                slot = index[int(prediction)]
                need(key not in matches and slot[0] == row["image_id"], "Original candidate one-to-one association")
                matches[key] = slot
    return matches


def family_gains(p, d, adds, drops):
    q_a = p[np.asarray(adds, dtype=np.intp)].sum(axis=1)
    q_d = d[np.asarray(drops, dtype=np.intp)].sum(axis=1)
    return q_a[:,None,:] - q_d[None,:,:]


def best(table):
    return np.unravel_index(int(table.min(axis=2).argmax()), table.shape[:2])


def intersections(codec_native, rles):
    return np.array([int(codec.area(codec.merge([codec_native, x], intersect=True))) for x in rles])


def candidate(entry, pos, rr, denominators, oracle):
    coords = np.asarray(entry["coords"], dtype=np.int64)
    k = entry["k"]
    need(entry["n"] == len(coords) and coords.shape == (len(coords),2)
         and 0 < k == pos["quota"] <= min(len(coords),8), "Original fixed budget and full sampled support")
    y = np.stack([codec.decode(label).astype(bool).ravel() for label in rr])
    p, d = y[:,coords[:,0]].T, y[:,coords[:,1]].T
    q_a, q_d = np.asarray(entry["ap"]), np.asarray(entry["dp"])
    truth_raw = family_gains(p,d,entry["adds"],entry["drops"])
    pred_raw = q_a[:,None,:]-q_d[None,:,:]
    need(pred_raw.shape == truth_raw.shape and pred_raw.ndim == 3 and pred_raw.shape[2] == 2,
         "Exact prelabel family sum shape")
    raw_intersection = intersections(pos["native"],rr)
    static_intersection = intersections(pos["changes"]["static"],rr)
    total = y.sum(axis=1) + int(codec.area(pos["native"]))
    need(np.array_equal(static_intersection, raw_intersection + truth_raw[0,0]), "Finite incumbent exact intersection")
    true_intersection = raw_intersection + truth_raw
    true_iou = oracle.iou(true_intersection,total)-oracle.iou(static_intersection,total)
    orig_delta = truth_raw-truth_raw[0,0]
    pred_delta = pred_raw-pred_raw[0,0]
    actual = tuple(entry["robust_pos"])
    need(actual == best(pred_delta) and np.array_equal(
        intersections(pos["changes"]["dual_robust"],rr), true_intersection[actual]),
        "Predicted objective and existing rendered robust parity")
    plugin_intersection = np.clip(static_intersection + pred_delta, 0, np.minimum(
        int(codec.area(pos["native"])),y.sum(axis=1)))
    plugin = oracle.iou(plugin_intersection,total)-oracle.iou(static_intersection,total)

    comparing = {arm: oracle.iou(intersections(pos["changes"][previous],rr),total)-oracle.iou(
        static_intersection,total) for arm, previous in (("deployed_mean","dual_mean"),
                                                         ("deployed_cell_pessimistic","dual_cell_pessimistic"))}
    comparing.update(deployed_robust=true_iou[actual],
                     predicted_normalized_family=true_iou[best(plugin/denominators)],
                     true_raw_family=true_iou[best(orig_delta)],
                     true_normalized_family=true_iou[best(true_iou/denominators)])
    code = np.array([1,2], dtype=np.int64)
    ac, dc, result, lower = oracle.maximin(p@code, d@code, k, raw_intersection,
                                           total, static_intersection, denominators)
    comparing["true_normalized_full"] = result
    floor = comparing["true_normalized_family"] / denominators
    need(lower >= float(floor.min())-TOL and float(floor.min()) >= -TOL, "Nested finite and exact all-pool dominance")
    if ac is not None:
        raw = codec.decode(pos["native"]).astype(bool,order="C")
        flat = raw.ravel()
        aa = oracle.selected_sites(p@code, ac, coords[:,0])
        dd = oracle.selected_sites(d@code, dc, coords[:,1])
        need(len(aa) == len(dd) == k and not flat[aa].any() and flat[dd].all(), "Exact all-pool signed quota")
        flat[aa], flat[dd] = True, False
        measured = oracle.iou(y[:,flat].sum(axis=1),total)-oracle.iou(static_intersection,total)
        need(np.allclose(measured,result,atol=TOL,rtol=0), "Rerendered full-pool exact truth parity")
    return comparing, pred_delta[actual], orig_delta[actual]


def describe(vector, draws):
    sampler = vector[draws].mean(axis=1)
    return {label: dict(points=float(vector[:,j].mean()),
                        percentile95_points=np.percentile(sampler[:,j],[2.5,97.5]).tolist())
            for j,label in enumerate(LABELS)}


def sign(val):
    return "positive" if val > TOL else "negative" if val < -TOL else "zero"


def pair_stats(delta, pred, true):
    flags = Counter()
    for option in OPTIONS:
        p,q = delta[option]
        flags[option] = dict(both_positive=p>TOL and q>TOL,
                             both_nonnegative=p>=-TOL and q>=-TOL,
                             negative=min(p,q)<-TOL, mixed=min(p,q)<-TOL and max(p,q)>TOL,
                             neutral=max(abs(p),abs(q))<=TOL)
    signs = [[sign(a)+"_pred__"+sign(b)+"_true" for a,b in zip(pred,true,strict=True)] +
             [sign(pred.min())+"_pred__"+sign(true.min())+"_true"]]
    return flags,signs[0]


def score(a, root):
    started = perf_counter()
    family = load(root/"family.py", FAMILY)
    solver = load(a.oracle,ORACLE)
    reader, _, old_score, _, native, records, groups = family.closed(a)
    families, safe, prep_hash, terminal_hash = verify_prelabel(root)
    need(safe["prior_score_sha256"] == family.OLD_SCORE and set(families) == set(native)
         and set(native) == set(groups) and len(set(groups.values())) == 180,
         "Complete prelabel family and exact retained native roster")
    scope = Counter()
    for iid,row in families.items():
        need(row["group"]==groups[iid],"Bound exact decoded identity")
        positive = {str(i) for i,p in enumerate(native[iid],1) if p["quota"]}
        need(set(row["families"])==positive,"Full label-free family coverage on every positive k")
        for rank,f in row["families"].items():
            need(f["k"]==native[iid][int(rank)-1]["quota"],"Same prelabel k and candidate")
            scope.update(candidates=1,swaps=f["k"],combinations=len(f["adds"])*len(f["drops"]))
    need(scope["candidates"]==safe["counts"]["active_candidates"]==8785 and scope["swaps"]==57692
         and scope["combinations"]==safe["counts"]["combinations"],"Exact positive quotas and family admission")
    print("Complete prelabel family verified before opening either author source-gate reference.",flush=True)
    truths = reader.labels(a,records)
    matches = {key:stock_matches(t,native) for key,t in truths.items()}
    need([len(matches[key]) for key in ("pseg","spsd")] == [311,308],"Exact previous native matches")
    bind = defaultdict(lambda: [[],[]])
    for j,key in enumerate(("pseg","spsd")):
        for row in truths[key]["annotations"]:
            bind[row["image_id"]][j].append(row)
    group_counts = Counter(groups.values())
    group_ids = {key:i for i,key in enumerate(sorted(group_counts))}
    aggregates = {option:np.zeros((180,3)) for option in OPTIONS}
    events = Counter()
    signs = {key:Counter() for key in LABELS}
    wins = {option:Counter() for option in OPTIONS}
    changed, optimizable = set(), set()
    pred_errors = []
    for iid,group in groups.items():
        a1,a2 = bind[iid]
        need(len(a1)>0 and len(a2)>0,"Preserved author nonempty per-image denominators")
        h1,h2 = [x["segmentation"] for x in a1], [x["segmentation"] for x in a2]
        ious = codec.iou(h1,h2,[0]*len(h2))
        rr,cc = linear_sum_assignment(ious,maximize=True)
        for i,j in zip(rr,cc,strict=True):
            events["label_pairs"] += 1
            if ious[i,j] < .5:
                continue
            events["label_pairs_ge050"] += 1
            left, right = matches["pseg"].get(a1[i]["id"]), matches["spsd"].get(a2[j]["id"])
            if left is None or left != right:
                continue
            events["shared_matches"] += 1
            rank = left[1]
            pos = native[iid][rank-1]
            if not pos["quota"]:
                continue
            events["positive_quota_shared_matches"] += 1
            optimizable.add(group)
            if pos["changes"]["dual_robust"] != pos["changes"]["static"]:
                events["deployed_change_common_hulls"] += 1
                changed.add(group)
            denom = np.array([len(a1),len(a2)])*group_counts[group]
            delta,pred,truth = candidate(families[iid]["families"][str(rank)],pos,(h1[i],h2[j]),denom,solver)
            for option,value in delta.items():
                scaled = value / denom
                aggregates[option][group_ids[group]] += [scaled[0],scaled[1],scaled.min()]
            flags,names = pair_stats(delta,pred,truth)
            for option,record in flags.items():
                wins[option].update({k:int(v) for k,v in record.items()})
            for key,name in zip(LABELS,names,strict=True):
                signs[key][name] += 1
            pred_errors.append(pred-truth)
    need((events["label_pairs"],events["label_pairs_ge050"],events["shared_matches"],
          events["deployed_change_common_hulls"],len(changed)) == (410,370,276,206,107),
         "Original paired reference and changed-exposure identity")
    draws = np.random.Generator(np.random.PCG64(5297)).integers(180,size=(4096,180))
    contrasts = {}
    for title, aa, bb in (("posterior_signal_gap","true_raw_family","deployed_robust"),
                          ("true_geometry_normalization_gap","true_normalized_family","true_raw_family"),
                          ("finite_search_gap","true_normalized_full","true_normalized_family"),
                          ("total_diagnostic_headroom","true_normalized_full","deployed_robust"),
                          ("predicted_label_assisted_normalization_effect","predicted_normalized_family","deployed_robust")):
        contrasts[title] = describe(aggregates[aa]-aggregates[bb],draws)
    errors = np.stack(pred_errors)
    data = dict(status="SPSD_DUAL_5297_ORACLE_DIAGNOSTIC_COMPLETE", prelock_sha256=LOCK,
                worker_sha256=digest(Path(__file__).read_bytes()), original_score_sha256=family.OLD_SCORE,
                prelabel_result_sha256=prep_hash, prelabel_terminal_sha256=terminal_hash,
                prelabel_family_sha256=safe["family_sha256"], solver_sha256=ORACLE,
                image_groups=180, group_bootstrap_draws=4096, author_denominator_hulls=[411,410],
                native_matches=[311,308], exposure=dict(events,changed_common_groups=len(changed),
                                                         positive_quota_common_groups=len(optimizable)),
                ladder={k:describe(v,draws) for k,v in aggregates.items()},
                ladder_pairwise=contrasts, both_wins_and_tradeoff_counts=wins, predicted_true_sign=signs,
                deployed_signed_count_error={"mean":errors.mean(axis=0).tolist(),
                                             "mean_absolute":abs(errors).mean(axis=0).tolist()},
                scored_AP=False, detector_forwards=0, optimizer_updates=0, historical_test_opened=False,
                retrospective_truth_assisted=True, elapsed_seconds=perf_counter()-started)
    a.output.mkdir()
    payload = (json.dumps(data,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    with (a.output/"RESULT_SAFE.json").open("xb") as out:
        out.write(payload)
    print(json.dumps(dict(status=data["status"],result_sha256=digest(payload))),flush=True)


def main():
    p = ArgumentParser(description=__doc__)
    for x in ("gate","labels","protocol","group-protocol","oracle","output"):
        p.add_argument("--"+x,type=Path,required=True)
    a = p.parse_args()
    pin(a.protocol,LOCK)
    pin(a.group_protocol,GROUP_PROTOCOL)
    score(a,Path(__file__).parent)


if __name__ == "__main__":
    main()

