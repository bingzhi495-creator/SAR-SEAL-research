"""Post-hoc attainable overlap on the already frozen candidate, endpoints and budget."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
from pycocotools import mask

SCORE_SOURCE = "77a3fbfaa04ac08287eff4ec7bb827ce3914e61aa20dd4f1e6f9c1e3b8ccc9fa"
SCORE = "b3252decceb28c9da468b0e86fc42568343385d79b27304ad071678c4ac8035b"
VERIFY = "828ed385ab8dff38840a0e179a0fdf4f4bc0a192088d062aeae193fad750656f"
DYNAMIC = "attraction_dynamic"
LEARNED = "learned_at_attraction_count"
ARMS = ("raw", "learned", "oracle_same_count", "oracle_up_to_count", "oracle_free_area_same_flips")
PAIRS = ((2,1),(2,0),(3,2),(4,2))


def iou(hit, pred, target):
    if not -1e-7 <= hit <= min(pred,target) + 1e-7:
        raise ValueError("Continuous unit-cell intersection outside valid area")
    hit = np.clip(hit, 0, min(pred,target))
    return 100 * hit / (pred + target - hit)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("code","stock","capture","edits","verify","score-result","score-terminal","gt","shapely-site","output"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    from types import ModuleType

    path = Path(__file__).with_name("score.py")
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != SCORE_SOURCE:
        raise ValueError("Exact scoring source")
    s = ModuleType("_closed_score")
    s.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), s.__dict__)
    prior = s.document(a.score_result, SCORE)
    obs = json.loads(a.score_terminal.read_bytes())
    s.require(obs["actual_exit"] == 0 and obs["output_footer_verified"] and obs["result_sha256"] == SCORE,
              "Closed previous score")
    verified = s.document(a.verify, VERIFY)
    s.require(prior["independent_edit_receipt_sha256"] == VERIFY, "Exact independently accepted edit input")
    preds = s.document(a.stock, verified["predictions_sha256"])
    capture = s.stream(a.capture / "CAPTURE_PRIVATE.jsonl", verified["capture_stream_sha256"])
    edits = s.stream(a.edits, verified["edits_sha256"])
    blocks,_ = s.cohort(preds,capture,edits)
    point_sets = {e["image_id"]:{r["rank"]:r["coordinates"] for r in e["points"]} for e in edits}
    gt = s.document(a.gt, "8d98e6f49200d5390c17bee66697de1f582133d08063259a561ed5f710be7b6c")
    raw_metrics, matches = s.score_one(gt, blocks, "raw", anchors=True)
    s.require(raw_metrics == prior["views"]["full"]["arms_fraction"]["raw"] and
              len(matches) == prior["geometry"]["fixed_matches"], "Original scoring and matching identities")
    sys.path.append(str(a.shapely_site))
    import shapely
    s.require(shapely.__version__=="2.0.2" and shapely.geos_version_string=="3.11.2","Geometry backend")
    suite = s.load(a.code / "MASK_RTDETR_DEV705_CONTINUOUS_AP_20261112/continuous_endpoint.py",s.GEOMETRY)
    old = s.load(a.code / "DFINE_SEG_EXTERNAL_BASELINE_FIT32_E24_20261112/e24/dev705_free_endpoint/score_endpoint.py",s.IOU)
    g = suite.geometry_modules(a.code)
    truth,neutral,_ = g.geometry.polygon_support(gt,g.vector)
    s.require(len(truth)==380 and not neutral,"Original author-float polygons")
    groups, rows, counts = defaultdict(list), [], Counter()
    for gt_id,(image_id,rank,stock) in matches.items():
        rle = stock["segmentation"]
        area = int(mask.area(rle))
        if not 32<=area<1024:
            continue
        record = blocks[image_id]
        plan = record["changes"].get(rank,{})
        k = plan.get(DYNAMIC, {}).get("swaps",0)
        raw_iou = old.continuous(rle,area,truth[gt_id][0],g)[0]
        learned = old.continuous(plan[LEARNED]["rle"],area,truth[gt_id][0],g)[0] if k else raw_iou
        if not k:
            values = [raw_iou,learned,raw_iou,raw_iou,raw_iou]
        else:
            target = truth[gt_id][0]
            h,w = rle["size"]
            coords = np.asarray(point_sets[image_id][rank],dtype=np.int64)
            s.require(coords.ndim==2 and coords.shape[1]==2 and len(coords)>=k
                      and (coords>=0).all() and (coords<h*w).all(), "Frozen endpoint locations")
            binary = mask.decode(rle).astype(bool).ravel()
            s.require(not binary[coords[:,0]].any() and binary[coords[:,1]].all()
                      and all(len(set(coords[:,i]))==len(coords) for i in (0,1)),
                      "Unique legitimate addition and removal sites")
            y,x = np.divmod(coords,w)
            q = shapely.area(shapely.intersection(shapely.box(x,y,x+1,y+1),target))
            add = np.r_[0.,np.cumsum(np.sort(q[:,0])[::-1])]
            drop = np.r_[0.,np.cumsum(np.sort(q[:,1]))]
            boxes = g.geometry.boxes_from_counts(g.shift.runs(rle["counts"]),h,w)
            hit = g.geometry.overlap_area(boxes,target)
            s.require(abs(iou(hit,area,target.area)-raw_iou)<1e-8,"Independent original polygon overlap")
            fixed = iou(hit+add[k]-drop[k],area,target.area)
            lower_count = max(iou(hit+add[t]-drop[t],area,target.area) for t in range(k+1))
            free_area = max(iou(hit+add[t]-drop[2*k-t],area+2*t-2*k,target.area)
                            for t in range(max(0,2*k-len(coords)),min(2*k,len(coords))+1))
            s.require(fixed+1e-8>=learned and lower_count+1e-8>=fixed and free_area+1e-8>=fixed,"Exact upper bound")
            counts.update(positive_budget=1,swaps=k,forced_count_cannot_equal_raw=int(fixed<raw_iou-1e-8))
            values = [raw_iou,learned,fixed,lower_count,free_area]
        rows.append(values)
        groups[record["rgb"]].append(values)
    mu = np.asarray([np.mean(groups[k],axis=0) for k in sorted(groups)])
    s.require(len(rows)==281 and len(groups)==119, "Complete area-eligible geometry scope")
    s.require(abs(mu[:,0].mean()-prior["geometry"]["group_equal_arms"]["raw"]["continuous_iou_points"])<1e-10
              and abs(mu[:,1].mean()-prior["geometry"]["group_equal_arms"][LEARNED]["continuous_iou_points"])<1e-10,
              "Exact already scored original and learned geometry")
    samples=np.random.Generator(np.random.PCG64(5217)).integers(len(mu),size=(4096,len(mu)))
    contrasts={}
    for i,j in PAIRS:
        diff=mu[:,i]-mu[:,j]
        contrasts[ARMS[i]+" minus "+ARMS[j]]=dict(delta_points=float(diff.mean()),
                         interval_95_points=np.percentile(diff[samples].mean(axis=1),[2.5,97.5]).tolist())
    result=dict(status="PSEG_TARGET_RF5201_POSTHOC_ORACLE_COMPLETE",posthoc=True,score_sha256=SCORE,
                worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                matched_pairs=len(rows),decoded_image_groups=len(groups),counts=dict(counts),
                continuous_iou_points=dict(zip(ARMS,mu.mean(axis=0).tolist())),
                strict_90_counts=dict(zip(ARMS,(np.array(rows)>=90).sum(axis=0).tolist())),
                contrasts_points=contrasts,model_forwards=0,optimizer_updates=0)
    a.output.mkdir()
    data=(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    (a.output/"RESULT_SAFE.json").write_bytes(data)
    print(json.dumps(dict(status=result["status"],result_sha256=sha256(data).hexdigest())),flush=True)


if __name__=="__main__":
    main()

