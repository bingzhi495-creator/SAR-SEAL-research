"""Direct stock COCO reconstruction of a closed five-arm, two-population historical PSeg score."""

import argparse
from contextlib import redirect_stdout
from hashlib import sha256
from importlib.metadata import version
from io import StringIO
import json
import math
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


RESULT = "9bc6411cf35ee355f142a5fa5588c8e5820127b8101acf4da0e653e70553d3d1"
GT = "021d52858ffa3fda9a2d28ee21f1ca114800554edf08007704f6d06a412d50bf"
ARMS = ("raw", "fixed", "free_matched_count", "free_natural", "native_matched_count")
OMITTED = {1, 11, 229, 389, 501, 631, 639, 641, 739, 1121}
DELTA = (("free_natural","raw"),("free_matched_count","fixed"),("free_natural","free_matched_count"),
         ("free_natural","native_matched_count"))


def need(ok, why):
    if not ok:
        raise ValueError(why)


def pinned(path, expected):
    data = path.read_bytes()
    need(sha256(data).hexdigest() == expected, "Exact closed " + path.name)
    return data


def from_tensors(ev):
    params = ev.params
    need(params.catIds == [0] and params.maxDets == [1, 10, 100] and len(params.recThrs) == 101
         and np.allclose(params.iouThrs, np.arange(.5,.951,.05)) and params.iouType == "segm", "Scoring grid")
    def aggregate(metric, area="all", limit=100, iou=None):
        array = ev.eval[metric][..., params.areaRngLbl.index(area), params.maxDets.index(limit)]
        if iou is not None:
            array = array[np.isclose(params.iouThrs, iou)]
        values = array[array > -1.]
        return float(values.mean()) if len(values) else -1.
    out = {"AP":aggregate("precision")}
    for cut in (50,75,90):
        out["AP"+str(cut)] = aggregate("precision", iou=cut/100.)
    for suffix, area in zip("sml", ("small","medium","large")):
        for prefix, key in (("AP","precision"),("AR","recall")):
            out[prefix+suffix] = aggregate(key, area)
    for cap in (1,10,100):
        out["AR"+str(cap)] = aggregate("recall", limit=cap)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("result","manifest","gt","predictions","candidate-audit","edits","output"):
        p.add_argument("--"+key,type=Path,required=True)
    args = p.parse_args()
    started = perf_counter()
    original = json.loads(pinned(args.result, RESULT))
    runtime = dict(python=platform.python_version(), numpy=np.__version__, pycocotools=version("pycocotools"))
    need(runtime == original["runtime"] == dict(python="3.10.13",numpy="1.26.4",pycocotools="2.0.11")
         and sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "Original CPU COCO runtime")
    need(original["status"] == "PSEG_TEST232_DFINE_FROZEN_POINT_SCORE_COMPLETE" and original["gt_sha256"] == GT
         and original["arms"] == list(ARMS) and original["source_only_frozen_head"] is True, "Frozen score specification")
    need(not args.output.exists(), "Fresh closure path")
    manifest = json.loads(pinned(args.manifest,original["manifest_sha256"]))
    ground_truth = json.loads(pinned(args.gt,GT))
    names = {x["file_name"]:x for x in ground_truth["images"]}
    pairs = {entry["image_id"]:names[entry["image_ref"]["path"]] for entry in manifest["items"]}
    for entry in manifest["items"]:
        im = pairs[entry["image_id"]]
        need([im["height"], im["width"]] == entry["original_hw"], "Source image and native canvas")
    need(len(names) == len(pairs) == 232 and len(ground_truth["annotations"]) == 546
         and ground_truth["categories"][0]["id"] == 0 and len(ground_truth["categories"]) == 1, "Original 232 / 546 float source GT")
    blobs = []
    for path, key in ((args.predictions,"input_predictions_sha256"),(args.candidate_audit,"candidate_audit_sha256"),(args.edits,"edits_sha256")):
        data = pinned(path,original[key])
        need(data.endswith(b"\n"), "Complete stream")
        by_id = {}
        for line in data.splitlines():
            record = json.loads(line)
            need(record["image_id"] not in by_id, "Unique image blocks")
            by_id[record["image_id"]] = record
        need(list(by_id) == manifest["expected_ids"] and set(by_id) == set(pairs), "One ordered record per manifest image")
        blobs.append(by_id)
    native, audits, sparse = blobs
    del data
    # One independently derived ordered candidate table. Each table entry retains all arm-specific masks by native rank.
    all_rows, counts = [], {}
    for source_id, image in pairs.items():
        base = native[source_id]["coco"]
        detail = audits[source_id]["candidates"]
        ops = sparse[source_id]["changes"]
        n = min(100,len(base))
        need(len(base) == len(detail) == audits[source_id]["post_nms"] and all(1 <= int(rank) <= n for rank in ops), "Original capture and edits")
        counts[image["id"]] = n
        for idx in range(n):
            row = base[idx]
            need(row["image_id"] == source_id and row["category_id"] == 1 and row["score"] == detail[idx]["score"]
                 and detail[idx]["native_rank"] == idx+1 and math.isfinite(row["score"]), "Native ranked confidence and class")
            edits = ops.get(str(idx+1), {})
            need(set(edits).issubset(ARMS[1:]), "Frozen treatment membership")
            masks = {a:edits[a]["rle"] for a in edits}
            all_rows.append((image["id"],row["score"],row["segmentation"],masks))
    full = sorted(i["id"] for i in names.values())
    filtered = sorted(i["id"] for name,i in names.items() if int(Path(name).stem) not in OMITTED)
    views = dict(official=full,exclude_exact_fit_overlap=filtered)
    need(len(set(full)-set(filtered)) == 10 and len(filtered) == 222, "Predeclared exact-image exclusion")
    obs = {name:{} for name in views}
    gt = COCO()
    gt.dataset = ground_truth
    with redirect_stdout(StringIO()):
        gt.createIndex()
    nc = 0
    for arm in ARMS:
        # Keep the entire dataset in the official API and restrict each evaluator to the prelocked image IDs.
        rows = [dict(image_id=i,category_id=0,score=p,segmentation=dict(mask.get(arm,base))) for i,p,base,mask in all_rows]
        with redirect_stdout(StringIO()):
            dt = gt.loadRes(rows)
        for name, ids in views.items():
            expected = original["views"][name]
            need((expected["images"],expected["annotations"],expected["candidates"]) ==
                 (len(ids),len(gt.getAnnIds(imgIds=ids)),sum(counts[i] for i in ids)), "Independent population accounting")
            ev = COCOeval(gt,dt,"segm")
            ev.params.catIds = [0]
            ev.params.imgIds = ids
            with redirect_stdout(StringIO()):
                ev.evaluate()
                ev.accumulate()
            vals = from_tensors(ev)
            need(vals == expected["metrics_fraction"][arm], "Exact all-threshold/AP90/area/recall statistics: "+name+"/"+arm)
            nc += len(vals)
            obs[name][arm] = vals
        print(json.dumps({"verified_arm":arm,"cumulative_metrics":nc}),flush=True)
    comparisons = 0
    for name, outcomes in obs.items():
        for a,b in DELTA:
            comp = {k:(100*(v-outcomes[b][k]) if min(v,outcomes[b][k]) >= 0 else None) for k,v in outcomes[a].items()}
            need(comp == original["views"][name]["paired_contrasts_points"][a+" minus "+b], "Independent paired effects")
            comparisons += len(comp)
    need(nc == 130 and comparisons == 104, "Complete comparison census")
    report = dict(status="PSEG_TEST232_DFINE_FROZEN_SCORE_INDEPENDENT_RECOMPUTATION_ACCEPTED", score_result_sha256=RESULT,
                  source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(), original_score_exit_reported_by_root=0,
                  independent_evaluations=10, reproduced_metrics=nc, reproduced_paired_contrasts=comparisons,
                  exact_metric_equality=True, exact_contrast_equality=True, maximum_absolute_difference=0.,
                  populations={name:{k:original["views"][name][k] for k in ("images","annotations","candidates")} for name in views},
                  GT_sha256=GT, runtime=runtime, own_model_forwards=0, own_optimizer_updates=0,
                  elapsed_seconds=round(perf_counter()-started,3), **{k:original[k] for k in
                  ("manifest_sha256","input_predictions_sha256","candidate_audit_sha256","edits_sha256","excluded_filenames_sha256")})
    raw = (json.dumps(report,sort_keys=True,indent=2)+"\n").encode()
    with args.output.open("xb") as dest:
        need(dest.write(raw) == len(raw), "Complete independent receipt")
        dest.flush()
        os.fsync(dest.fileno())
    print(json.dumps(dict(status=report["status"],readback_sha256=sha256(raw).hexdigest())),flush=True)


if __name__ == "__main__":
    main()
