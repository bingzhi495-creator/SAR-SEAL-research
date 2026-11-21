"""Post-hoc localization of the FIT109 matched-writer AP and geometry differences."""

from argparse import ArgumentParser
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from copy import deepcopy
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


ARMS = ("N", "F", "B", "J")
PEER_SHA = "c791b3eb637acf5e30341768a53fa1acb8a3c8db5d6153717474b0b6958fc319"
VERIFIER_SHA = "daa3f6dc7751fe546700b8297b89c0f7295238b3876631f073c8f863d72d449f"


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def checked(path, digest, limit):
    with path.open("rb") as f:
        raw = f.read(limit + 1)
    need(0 < len(raw) <= limit and sha256(raw).hexdigest() == digest, "Pinned input " + path.name)
    return raw


def decomposition(Ib, Pb, Ij, Pj, T):
    """Exact symmetric algebra on two physically realized intersections/unions."""
    denom = (Pb+T-Ib)*(Pj+T-Ij)
    need(denom > 0, "Positive realized unions")
    return ((Ib-Ij)*(T+(Pb+Pj)/2)/denom, (Ib+Ij)*(Pj-Pb)/(2*denom))


def coco_curve(gt, rows, keys):
    ref = COCO()
    ref.dataset = deepcopy(gt)
    ref.dataset.setdefault("info", {})
    with redirect_stdout(StringIO()):
        ref.createIndex()
        dt = ref.loadRes(deepcopy(rows))
        ev = COCOeval(ref, dt, "segm")
        ev.params.imgIds = [im["id"] for im in gt["images"]]
        ev.params.catIds = [1]
        ev.params.maxDets = [1,10,100]
        ev.evaluate()
        ev.accumulate()
    statuses = [{ } for _ in ev.params.iouThrs]
    for row in ev.evalImgs:
        if row is None or list(row["aRng"]) != list(ev.params.areaRng[0]) or row["maxDet"] != 100:
            continue
        for index, status in enumerate(statuses):
            for did, gid, ignore in zip(row["dtIds"],row["dtMatches"][index],row["dtIgnore"][index],strict=True):
                key = keys[did-1]
                need(key[0] == row["image_id"] and key not in status, "Official result identity")
                status[key] = (-1 if ignore else int(gid))
    curve = ev.eval["precision"][:, :, 0, 0, -1]
    need(all(len(s)==len(keys) for s in statuses), "Preserved mapped candidate population")
    return [float(v[v>=0].mean()*100) for v in curve], statuses


def write_quality(geo, rec, edits, witness):
    for arm in ("F","B","J"):
        vals = witness["diagnostics"][arm]
        edits[arm] = geo.candidate_geometry(rec, dict(flipped=witness[arm], edits=len(witness[arm]),
                                                       adds=vals["adds"], drops=vals["drops"], area_delta=vals["area_delta"]))


def band(native_rank):
    if native_rank <= 10:
        return "01-10"
    if native_rank <= 30:
        return "11-30"
    return "31-100"


def analyze(gt, pools, attached, budgets, witness, helpers, report):
    recs = {helpers.geometry.identity(row):row for row in attached}
    need(len(recs) == 6738 and set(recs) == set(witness) == set(budgets), "Full eligible population")
    keys, streams = [], {a:[] for a in ARMS}
    native_order, diagnostics = defaultdict(list), Counter()
    for im in gt["images"]:
        iid = im["id"]
        for pred in pools[iid]:
            key = (iid,pred["rank"],pred["query_id"])
            keys.append(key)
            rle = pred["rle"]
            result = dict.fromkeys(ARMS,rle)
            if key in recs:
                row, saved, k = recs[key], witness[key], budgets[key]["k"]
                for arm in ARMS[1:]:
                    need(len(saved[arm]) == 2*k, "Same fixed per-candidate native edits")
                    result[arm] = helpers.sparse.xor_rle(rle,np.asarray(saved[arm],dtype=np.int64))
                    diagnostics[arm+"_nonpositive_flips"] += saved["diagnostics"][arm]["nonpositive_flips"]
                drift = saved["diagnostics"]["J"]["area_delta"]
                diagnostics["joint_"+("net_add" if drift>0 else "net_drop" if drift<0 else "balanced")] += 1
                if row["role"] == "valid":
                    data = {}
                    write_quality(helpers.geometry,row,data,saved)
                    b,j = data["B"],data["J"]
                    iv,pv = decomposition(b["I"],b["P"],j["I"],j["P"],b["T"])
                    need(abs(iv+pv-(b["iou"]-j["iou"])) < 1e-12,"Exact algebraic decomposition")
                    native_order[row["group"]].append((b["iou"]-j["iou"],iv,pv,(abs(j["P"]-j["T"])-abs(b["P"]-b["T"]))/b["T"]))
                    error = abs(j["P"]-j["T"])-abs(b["P"]-b["T"])
                    diagnostics["joint_area_error_"+("worse" if error>1e-12 else "better" if error< -1e-12 else "equal")] += 1
            for arm in ARMS:
                streams[arm].append(dict(image_id=iid,category_id=1,score=pred["score"],segmentation=result[arm]))
    need(len(keys) == 10488 == len(set(keys)), "Complete original rank list")
    curves = {arm:coco_curve(gt, streams[arm],keys) for arm in ARMS}
    for arm,(curve,_) in curves.items():
        need(abs(float(np.mean(curve))-100*report["metrics"]["arms"][arm]["AP"]) < 1e-10, "Per-threshold AP recovers original AP")
    thresholds = []
    for i in range(10):
        b,j = curves["B"][1][i],curves["J"][1][i]
        transitions = Counter()
        bands = Counter()
        for key in keys:
            y,x = b[key],j[key]
            if y>0 and x==0:
                transitions["FP_to_TP"] += 1
                bands[band(key[1])] += 1
            if y==0 and x>0:
                transitions["TP_to_FP"] += 1
                bands[band(key[1])] -= 1
            if min(y,x)>0 and y!=x:
                transitions["TP_identity_switch"] += 1
            if (y<0) != (x<0):
                transitions["ignore_state_change"] += 1
        row = {arm:curves[arm][0][i] for arm in ARMS}
        row.update(iou_threshold=round(.50+.05*i,2), B_minus_J=row["B"]-row["J"],
                   B_minus_F=row["B"]-row["F"], match_transitions_B_relative_J=dict(transitions),
                   net_FP_TP_changes_by_original_rank={a:bands[a] for a in ("01-10","11-30","31-100")})
        thresholds.append(row)
    need(len(native_order)==92 and sum(map(len,native_order.values()))==392, "Original valid geometry matches")
    means = 100*np.array([np.mean(native_order[g],axis=0) for g in sorted(native_order)])
    rng = np.random.Generator(np.random.PCG64(5217))
    idx = rng.integers(len(means),size=(4096,len(means)))
    est = {}
    for i,name in enumerate(("observed_IoU","intersection_term","predicted_area_term","J_minus_B_abs_area_error_per_truth_area")):
        values = means[:,i]
        est[name] = dict(points=float(values.mean()),ci95=np.quantile(values[idx].mean(axis=1),[.025,.975]).tolist())
    need(abs(est["observed_IoU"]["points"]-report["metrics"]["contrasts"]["B_minus_J"]["geometry_delta_points"])<1e-9,
         "Area/overlap decomposition recovers original contrast")
    return dict(thresholds=thresholds, source_group_equal_geometry=est, counts=dict(diagnostics),
                mapped_candidates=len(keys), original_fixed_matches=392, contributing_source_groups=92)


def main():
    p = ArgumentParser(allow_abbrev=False)
    for name in ("code","work","run","owner","accepted","old-support","old-score","raw-capture","raw-acceptance",
                 "source-cohort","fit-gt","shapely-site","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--owner-sha256",required=True)
    a = p.parse_args()
    helper_bytes = checked(a.work/"verify_point_fit109.py",VERIFIER_SHA,2**20)
    env = {"__file__":str(a.work/"verify_point_fit109.py"),"__name__":"_closed_fit109"}
    exec(compile(helper_bytes,env["__file__"],"exec"),env)
    load, read = env["checked_module"],env["read_doc"]
    owner = env["checked_terminal"](a.owner,a.owner_sha256)
    peer = read(a.accepted,PEER_SHA)
    need(peer["status"]=="RFDETR_POINTREND_FIT109_OOF_INDEPENDENTLY_ACCEPTED" and peer["COCO12_bit_exact_all_six_arms"]
         and peer["result_sha256"] == owner["result_sha256"] and peer["original_exit"] == 0, "Independent completed original")
    report = read(a.run/"RESULT_SAFE.json",owner["result_sha256"])
    manifest = read(a.work/"POINTREND_FIT109_MODULE_PINS.json",owner["module_manifest_sha256"])
    worker = load(a.work/"run_point_fit109.py",manifest["run_point_fit109.py"])
    protocol = read(a.work/"POINTREND_FIT109_PRELOCK.json",worker.PROTOCOL_SHA)
    previous = a.code/"RFDETR_QUERY_FEATURE_FIT109_20261121"
    old = load(previous/"run_fit109_query.py","97fdc3987462aabbcd7d4fbe672c9788a3b969a221243e13d82473093170a15a")
    spot = old.load(a.code/"RFDETR_CORE36_MATCHED_POINT_SPOT_20261120/run_spot.py")
    formal = a.code/"RFDETR_MATCHED_POINT_FORMAL_20261120"
    prior,plan = (old.load(formal/n) for n in ("run_fit109.py","fit_plan.py"))
    a.capture,a.acceptance,a.acceptance_sha256=a.raw_capture,a.raw_acceptance,old.OLD_RAW_ACCEPTANCE
    roster,_ = prior.read_population(a,spot)
    helpers = old.load(previous/"fit109_score.py").load_helpers(a.code,a.shapely_site)
    pools,rec = worker.collect_old_support(a,SimpleNamespace(old=old,spot=spot),roster,protocol)
    budgets,_ = worker.read_budgets(a.old_score,protocol,rec,plan,roster,helpers.geometry)
    gt,attached = old.attach_fit(a,roster,rec,pools,helpers,spot)
    rows = [json.loads(line) for line in checked(a.run/"PREDICTION_WITNESS_PRIVATE.jsonl",
                                        report["prediction_witness_sha256"],80*2**20).splitlines()]
    witness = {tuple(row["key"]):row for row in rows}
    need(len(witness) == len(rows),"Unique closed witness")
    result = analyze(gt,pools,attached,budgets,witness,helpers,report)
    result.update(status="FIT109_AP_GEOMETRY_POSTHOC_DIAGNOSTIC_COMPLETE",phase="FIT_ONLY",FIT=True,DEV=False,TEST=False,
                  AP=True,point_head_updates=0,model_forward_calls=0,
                  original_independent_report_sha256=PEER_SHA,original_result_sha256=owner["result_sha256"],
                  separation="The paired area and intersection terms sum exactly. They are algebraic, not causal, and AP threshold differences use original COCO matching.",
                  source_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    raw = (json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    with a.output.open("xb") as f:
        f.write(raw)
    print(json.dumps(dict(status=result["status"],sha256=sha256(raw).hexdigest())))


if __name__ == "__main__":
    main()
