"""Repeat the fixed original SSDD float hull co-centering diagnostic without AP."""

from argparse import ArgumentParser
from collections import Counter
from contextlib import redirect_stdout
from copy import deepcopy
from hashlib import sha256
from importlib.metadata import version
from io import StringIO
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType

import numpy as np
from pycocotools import mask


MIB = 1024**2
ARMS = ("native","center","historical")
GROUPS = ("small_elongated","thin")
MODES = ("raw","short","long","both")
METRICS = ("continuous_iou_points","fp_per_gt_area_points","fn_per_gt_area_points")
GT_SHA = "8d98e6f49200d5390c17bee66697de1f582133d08063259a561ed5f710be7b6c"
IMAGE_SHA = "18df007e15e3e838e607f178f14e7a26f2b37b992ee39bf091e0b74cf084fb13"
PRIMARY_SHA = "5741125a623dbbdab97040347897ba21430e56fa17de101a446d31e9ab270f06"
WAIT_SHA = "9db6716c0cf0d2be0865d902e8907c99599516c1d1ff7ed4eabe1e688bf05563"
CLOSED_SHA = "2d946939aa9f1860215418da105f6450c4cd3db35b9218376f64738a7c0c390e"
REQUEST_SHA = "de4f13234e52dba5d373d4e86ddb6b1425e76a411cd148570b48c095f95d4748"
PRIOR_SHA = "339b06cfb7bd01bf023b6b03f59f0844a809d9bb22d1bff3dacc29b243dd6261"
PRIOR_WAIT_SHA = "79a06839425c90c26e31fdd69f51cf3d8e219de75a3840877a62e780d4a2f20a"
STRONG_SHA = "480af387cc954b1bc9699ada0df6aa520add6c0e0155cb25499a3f86f02b5c72"
STRONG_ACCEPT_SHA = "97c53f1df007ddfc2cc0c200041a5b6246a40b4130c50e90151483e113d70a62"
STRONG_WAIT_SHA = "3cf920c352ee392692f96539d4b264b573f7a1845eb948c92ff5795db4670077"
REF_SHA = "9f273cab09b086d7f303cc98a2915349cf69cbfa5f36b03394e481b62f27354a"
REF_WAIT_SHA = "27645cf90aaaff98d7035d1a12f8eba46a581a32e8028116210cf40bb72b07a2"
COCENTER_SOURCE = "177950401f897ecc53a87976cc675cab571a91614ddf646b7bb8fae7e6f19b56"
GEOMETRY_SOURCE = "dedd47d5453b2442ff02585c9b4d216917e5ed175e01eae07a354a22710f3a94"
ADMISSION_SOURCE = "7cad8b51a9230101e2233db8455f84c28163abfb2ee8ddd8cbe8718a2e719bd0"


def need(ok, detail):
    if not ok:
        raise ValueError(detail)


def pinned(path, digest, limit=MIB, size=None):
    need(path.is_absolute() and path.resolve(strict=True) == path and path.is_file()
         and 0 < path.stat().st_size <= limit and (size is None or path.stat().st_size == size),
         "Canonical bounded exact input: "+path.name)
    raw = path.read_bytes()
    need(sha256(raw).hexdigest()==digest,"Closed input digest differs: "+path.name)
    return raw


def read(path, digest, limit=MIB, size=None):
    return json.loads(pinned(path,digest,limit,size))


def ref(ref_, digest, limit=MIB):
    need(ref_["sha256"]==digest,"Pinned original artifact reference")
    return read(Path(ref_["path"]),digest,limit,ref_["bytes"])


def load(path, digest, name):
    module = ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(pinned(path,digest),str(path),"exec"),module.__dict__)
    return module


def write(path, value):
    raw = (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    with path.open("xb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    return sha256(raw).hexdigest()


def lineage(args):
    root = args.score_dir
    accepted = read(root/"INDEPENDENT_FLOAT_PRIMARY_SCORE_SAFE.json",PRIMARY_SHA)
    closed = read(root/"SCORE"/"CLOSED_STAGE_PRIVATE.json",CLOSED_SHA)
    req = read(root/"SCORE"/"STAGE_REQUEST_PRIVATE.json",REQUEST_SHA)
    wait = read(args.score_wait,WAIT_SHA)
    need(accepted["status"]=="SSDD_FLOAT_COMMON_DEV187_SCORE_INDEPENDENTLY_ACCEPTED_V1"
         and accepted["seed"]==5202 and accepted["initialization_seed"]==5201
         and accepted["DEV"] is True and accepted["FIT"] is accepted["TEST"] is False
         and accepted["annotations"]==380 and accepted["images"]==187
         and accepted["floating_gt_sha256"]==GT_SHA and accepted["image_manifest_sha256"]==IMAGE_SHA
         and accepted["original_score_outer_tool_wait_sha256"]==WAIT_SHA
         and wait["stage"]=="SCORE"
         and wait["exact_tool_wait"]["exit_code"]==0 and wait["request_sha256"]==REQUEST_SHA
         and wait["score_report_sha256"]==accepted["original_score_report_sha256"]==closed["report_ref"]["sha256"]
         and wait["original_child_terminal_sha256"]==accepted["original_score_worker_terminal_sha256"]
            ==closed["terminal_ref"]["sha256"],"Independent 5202 primary scope and original tool closure")
    for arm in ARMS[:2]:
        a = accepted["arms"][arm]
        original = req[arm]
        c = original["closure"]
        need(a["prediction_sha256"]==original["predictions_ref"]["sha256"]
             and a["independent_export_acceptance_sha256"]==original["acceptance_ref"]["sha256"]
             and a["export_report_sha256"]==original["report_ref"]["sha256"]==c["report_sha256"]
             and c["terminal_sha256"]==a["original_export_terminal_sha256"]
             and c["original_os_wait"] is True and c["original_os_exit"]==0
             and c["resource_stop"] is False and c["worker_closed"] is True,
             "Exact independently accepted old export lineage: "+arm)
    prior = read(args.prior5201,PRIOR_SHA)
    prior_wait = read(args.prior5201_wait,PRIOR_WAIT_SHA)
    need(prior["seed"]==5201 and prior["match_iou"]==.5 and prior["original_all12_and_AP90_exact"]
         and prior_wait["original_os_wait"] and prior_wait["original_os_exit"]==0 and prior_wait["process_group_closed"],
         "Exact accepted seed-5201 geometry comparator")
    reference = read(args.reference_safe,REF_SHA)
    ref_wait = read(args.reference_wait,REF_WAIT_SHA)
    strong = read(args.strong_acceptance,STRONG_ACCEPT_SHA)
    strong_wait = read(args.strong_wait,STRONG_WAIT_SHA)
    hist = reference["seeds"]["5201"]
    need(reference["status"]=="P370_SSDD_TWO_SEED_FIXED_DEV187_FLOAT_SENSITIVITY_COMPLETE"
         and reference["DEV"] and not reference["TEST"] and reference["regenerated_float_equal"]
         and ref_wait["original_os_wait"] and ref_wait["original_os_exit"]==0 and ref_wait["process_group_closed"]
         and ref_wait["result"]["sha256"]==REF_SHA and ref_wait["stop_reason"] is None
         and hist["primary_exact"] and hist["acceptance_sha256"]==STRONG_ACCEPT_SHA
         and hist["original_wait_sha256"]==STRONG_WAIT_SHA
         and hist["candidate_float"]["native"]["AP"]==accepted["historical_integer_trained_native_float_score"]["AP"]
         and hist["prediction_sha256"]["native"]==strong["arms"]["native"]["prediction_sha256"]==STRONG_SHA
         and hist["candidate_counts"]["native"]==strong["arms"]["native"]["prediction_count"]==15007
         and strong["status"]=="P286_SSDD_DEV187_INDEPENDENTLY_ACCEPTED" and strong["DEV"] and not strong["TEST"]
         and strong["image_manifest_sha256"]==IMAGE_SHA and strong["coordinator_original_wait_sha256"]==STRONG_WAIT_SHA
         and strong_wait["original_os_wait"] and strong_wait["original_os_exit"]==0
         and strong_wait["coordinator_closed"] and not strong_wait["resource_stop"],
         "Exact historical strong-reference predictions on the same DEV187 roles")
    return req,accepted,prior


def load_unchanged_inputs(req, accepted, args):
    manifest = ref(req["image_manifest_ref"],IMAGE_SHA)
    ids = manifest["expected_ids"]
    items = manifest["items"]
    need(manifest["schema"]=="Root-RFDETR-DEV187-image-only-manifest-v1" and not manifest["TEST"]
         and len(ids)==len(items)==len(set(ids))==187 and ids==[r["image_id"] for r in items],
         "Exact fixed logical chip census")
    gt = ref(req["float_gt_ref"],GT_SHA)
    need(len(gt["images"])==187 and len(gt["annotations"])==len({r["id"] for r in gt["annotations"]})==380
         and [c["id"] for c in gt["categories"]]==[1]
         and all(row["id"]==item["image_id"] and [row["height"],row["width"]]==item["original_hw"]
                 for row,item in zip(gt["images"],items,strict=True)), "Exact original float GT canvas and order")
    streams = {arm:ref(req[arm]["predictions_ref"],accepted["arms"][arm]["prediction_sha256"],8*MIB) for arm in ARMS[:2]}
    streams["historical"] = read(args.strong_predictions,STRONG_SHA,8*MIB,2937349)
    expected = {arm:accepted["arms"][arm]["prediction_count"] for arm in ARMS[:2]}
    expected["historical"] = 15007
    hw = {r["image_id"]:r["original_hw"] for r in items}
    for arm,rows in streams.items():
        counts = Counter(r["image_id"] for r in rows)
        need(len(rows)==expected[arm] and max(counts.values())<=100 and set(counts)<=set(ids)
             and all(r["category_id"]==1 and r["segmentation"]["size"]==hw[r["image_id"]]
                     and 0<=r["score"]<=1 for r in rows), "Exact admitted original rows: "+arm)
    content_ids = [r["image_ref"]["sha256"] for r in items]
    content_order = {key:i for i,key in enumerate(dict.fromkeys(content_ids))}
    need(len(content_order)==183,"Exact original encoded chip groups")
    return gt,streams,[content_order[key] for key in content_ids]


def stock_matches(gt, streams, resolve):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    original = COCO()
    original.dataset = deepcopy(gt)
    annotation_images = {r["id"]:r["image_id"] for r in gt["annotations"]}
    rows = [{k:row[k] for k in ("image_id","category_id","score","segmentation")} for row in streams]
    with redirect_stdout(StringIO()):
        original.createIndex()
        detected = original.loadRes(rows)
        evaluation = COCOeval(original,detected,"segm")
        evaluation.params.imgIds = [r["id"] for r in gt["images"]]
        evaluation.params.catIds = [1]
        evaluation.params.maxDets = [1,10,100]
        evaluation.params.areaRng = [evaluation.params.areaRng[0]]
        evaluation.params.areaRngLbl = ["all"]
        evaluation.evaluate()
    need(len(evaluation.params.iouThrs)==10 and evaluation.params.iouThrs[0]==.5,"Exact original .50 match plane")
    return resolve(evaluation.evalImgs,streams,annotation_images)


def build_cases(gt, matched, geom, cocenter, modules):
    polygons,neutral,_ = modules.geometry.polygon_support(gt,modules.vector)
    need(len(polygons)==380 and not neutral,"Exact official float continuous GT support")
    images = {row["id"]:(i,row) for i,row in enumerate(gt["images"])}
    cases,dense_checks,dense_error = [],0,0.
    for ann in sorted(gt["annotations"],key=lambda r:r["id"]):
        aid = ann["id"]
        if not all(aid in matched[a] for a in ARMS[:2]):
            continue
        reference,_ = polygons[aid]
        aspect,width,tangent,normal = geom.hull(reference)
        groups = [g for g,ok in zip(GROUPS,(aspect>=2 and modules.geometry.stratum(ann["area"])=="small",
                                          aspect>=2 and width<=8),strict=True) if ok]
        if not groups:
            continue
        chip,image = images[ann["image_id"]]
        hw = (image["height"],image["width"])
        row = dict(chip=chip,groups=groups,values={},outside={})
        dense = {}
        for arm in ARMS:
            if aid not in matched[arm]:
                continue
            rle = matched[arm][aid]
            need(rle["size"]==list(hw),"Exact unchanged prediction mask dimensions")
            counts = rle["counts"]
            text = counts.decode("ascii") if isinstance(counts,bytes) else counts
            rectangles = modules.geometry.boxes_from_counts(modules.shift.runs(text),*hw)
            area = float(mask.area(rle))
            offsets = cocenter.translations(rectangles,reference,tangent,normal)
            need(not cocenter.outside(rectangles,offsets["raw"],*hw),"Raw mask domain")
            flags = {m:cocenter.outside(rectangles,offsets[m],*hw) for m in MODES[1:]}
            values = {mode:cocenter.readout(rectangles+np.tile(offsets[mode],2),area,reference,modules.geometry.overlap_area)
                      for mode in MODES if not flags.get(mode,False)}
            row["values"][arm],row["outside"][arm] = values,flags
            dense[arm] = rle,offsets,values
        if "small_elongated" in groups and dense_checks<2 and not any(any(row["outside"][a].values()) for a in ARMS[:2]):
            for arm in ARMS[:2]:
                rle,offsets,values = dense[arm]
                for mode in MODES:
                    dense_error = max(dense_error,cocenter.dense_witness(rle,reference,offsets[mode],values[mode]))
            dense_checks += 1
        cases.append(row)
    need(dense_checks==2,"Exact first-two retained primary dense witnesses")
    return cases,dense_error


def cohort(cases, group, support, left, right, geom, cocenter, footprint, draws, content, content_draws):
    full = [r for r in cases if group in r["groups"] and all(a in r["values"] for a in support)]
    boundaries = {a:dict.fromkeys(MODES[1:],0) for a in support}
    for row in full:
        for arm in support:
            for mode,value in row["outside"][arm].items():
                boundaries[arm][mode] += value
    retained = [r for r in full if not any(any(r["outside"][a].values()) for a in support)]
    def rows(items):
        return [(r["chip"],r["values"][left],r["values"][right]) for r in items]
    outcome = cocenter.cohort_summary(geom,footprint,draws,rows(full),rows(retained),boundaries)
    def interval(values):
        stats = footprint.paired_cluster_stats(values,content_draws)
        return stats["row_weighted_image_cluster_percentile95"]
    result = dict()
    for mode in MODES:
        result[mode] = {}
        for metric in METRICS:
            diffs = [(content[chip],b[mode][metric]-a[mode][metric]) for chip,a,b in rows(retained)]
            result[mode][metric] = dict(interval=interval(diffs))
            if mode!="raw":
                changes = [(content[chip],(b[mode][metric]-a[mode][metric])-(b["raw"][metric]-a["raw"][metric]))
                           for chip,a,b in rows(retained)]
                result[mode][metric]["change_from_raw_interval"] = interval(changes)
    outcome["exact_encoded_content_cluster_intervals"] = result
    outcome["left_arm"],outcome["right_arm"] = left,right
    return outcome


def comparison(prior, repeat):
    def selected(group):
        raw = group["retained_modes"]["raw"]["continuous_iou_points"]
        changes = {m:group["retained_change_from_raw"][m]["continuous_iou_points"] for m in ("short","long")}
        return dict(full_pairs=group["full_pairs"],retained_pairs=group["retained_pairs"],raw=raw,changes=changes,
                    short_change_more_negative_than_long=changes["short"]["delta_change_from_raw"]
                        < min(0.,changes["long"]["delta_change_from_raw"]))
    return {g:dict(seed5201=selected(prior["groups"][g]),seed5202=selected(repeat[g])) for g in GROUPS}


def run(args):
    started = perf_counter()
    pinned(Path(__file__).with_name("PROTOCOL.md"),args.protocol_sha)
    sys.path.append(str(args.shapely_site.resolve(strict=True)))
    import shapely
    need(version("numpy")=="1.26.4" and version("pycocotools")=="2.0.11" and shapely.__version__=="2.0.2"
         and shapely.geos_version_string=="3.11.2","Exact prior CPU geometry runtime")
    code = args.code_root
    cocenter = load(code/"SSDD_FLOAT_COMMON_DEV187_HULL_COCENTER_20261112"/"hull_cocenter.py",COCENTER_SOURCE,"_repeat_cocenter_kernel")
    geom = load(code/"MASK_RTDETR_DEV705_FIXED_MATCH_GEOMETRY_20261112"/"fixed_match_geometry.py",GEOMETRY_SOURCE,"_repeat_cocenter_hull")
    adm = load(code/"MASK_RTDETR_DEV705_CONTINUOUS_AP_20261112"/"continuous_endpoint.py",ADMISSION_SOURCE,"_repeat_cocenter_polygon")
    wrap = geom.load_module(code/geom.MATCH_WRAPPER,geom.MATCH_WRAPPER_SHA,"_repeat_cocenter_match_bridge")
    _,_,_,_,_,matching = wrap.load_helpers(code,code/geom.MATCH_GEOMETRY)
    req,accepted,prior = lineage(args)
    args.output.mkdir(exist_ok=False)
    pre = write(args.output/"PRELABEL_SAFE.json",dict(status="SSDD_FLOAT_COCENTER_REPEAT5202_PRELABEL",DEV=True,FIT=False,
        TEST=False,GT_opened=False,predictions_opened=False,AP=False,protocol_sha256=args.protocol_sha,
        primary_acceptance_sha256=PRIMARY_SHA,reference_acceptance_sha256=STRONG_ACCEPT_SHA,
        source_sha256=sha256(Path(__file__).read_bytes()).hexdigest()))
    gt,streams,content = load_unchanged_inputs(req,accepted,args)
    matched = {arm:stock_matches(gt,rows,matching.match_official) for arm,rows in streams.items()}
    modules = adm.geometry_modules(code)
    cases,dense_error = build_cases(gt,matched,geom,cocenter,modules)
    draws,content_draws = modules.footprint.bootstrap_draws(187),modules.footprint.bootstrap_draws(183)
    def summaries(support,left,right):
        return {g:cohort(cases,g,support,left,right,geom,cocenter,modules.footprint,draws,content,content_draws) for g in GROUPS}
    pair = summaries(ARMS[:2],"native","center")
    triple = {name:summaries(ARMS,left,right) for name,left,right in
              (("center_minus_native","native","center"),("center_minus_historical","historical","center"),
               ("native_minus_historical","historical","native"))}
    result = dict(status="SSDD_FLOAT_COMMON_DEV187_HULL_COCENTER_REPEAT5202_ACTUAL",DEV=True,FIT=False,TEST=False,
        GT_informed=True,AP=False,model_forward_calls=0,optimizer_updates=0,seed=5202,images=187,annotations=380,
        match_iou=.5,matched_by_arm={a:len(v) for a,v in matched.items()},match_accumulation=False,
        prediction_counts={a:len(v) for a,v in streams.items()},primary=pair,all_three=triple,versus_seed5201=comparison(prior,pair),
        protocol_sha256=args.protocol_sha,prelabel_receipt_sha256=pre,primary_acceptance_sha256=PRIMARY_SHA,
        prior5201_sha256=PRIOR_SHA,historical_prediction_sha256=STRONG_SHA,gt_sha256=GT_SHA,
        bootstrap_draws=len(draws),encoded_content_clusters=183,dense_checks=16,
        dense_max_abs_metric_error=dense_error,worker_seconds=perf_counter()-started)
    result_sha = write(args.output/"RESULT_SAFE.json",result)
    print(json.dumps(dict(status=result["status"],result_sha256=result_sha),sort_keys=True),flush=True)


def main():
    parser = ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ("code-root","score-dir","score-wait","shapely-site","output","prior5201","prior5201-wait",
                 "reference-safe","reference-wait","strong-acceptance","strong-wait","strong-predictions"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--protocol-sha",required=True)
    args = parser.parse_args()
    need(sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES")=="-1"
         and os.environ.get("OMP_NUM_THREADS")==os.environ.get("OPENBLAS_NUM_THREADS")=="1"
         and args.output.is_absolute() and args.output.parent.resolve(strict=True)==args.output.parent
         and not args.output.exists(),"Fresh isolated CPU-only scratch output")
    def offline(event,_):
        if event.startswith("socket."):
            raise PermissionError("Offline closed DEV-only geometry")
    sys.addaudithook(offline)
    run(args)


if __name__=="__main__":
    main()

