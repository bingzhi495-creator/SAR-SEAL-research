"""Exact post-primary official-float DEV187 geometry on original native matches."""
from argparse import ArgumentParser
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace

PRIMARY_SHA = "07bbf74198bec0e55f131a2aeeca2f7efa4c6008654dd3816e7b3ea57d8178f4"
SCORE_WAIT_SHA = "b5ea98ca81695cd7527589dd9be07615ded9d36067b6fabf4d63445c728fb3f0"
SCORE_REQUEST_SHA = "46b41a2261a746523049e74910cf7d417ecc595c583e80c57fd97de42c4759ce"
CLOSED_SHA = "2517a94ab0f8f20a98a5e90fb06fc4176a52b116b8ea3d445a04f3d392812acf"
WORKER_SHA = "5463f294191929bceaac3b04fb6c17179c05066eaa29b6d859ef4943cca686c7"
GEOMETRY_SHA = "dedd47d5453b2442ff02585c9b4d216917e5ed175e01eae07a354a22710f3a94"
ADMISSION_SHA = "7cad8b51a9230101e2233db8455f84c28163abfb2ee8ddd8cbe8718a2e719bd0"
ARMS = ("native", "center")


def require(ok, why):
    if not ok:
        raise ValueError(why)


def exact(path, expected):
    require(path == path.resolve(strict=True) and 0 < path.stat().st_size <= 2**20, "Canonical bounded authority")
    raw = path.read_bytes()
    require(sha256(raw).hexdigest() == expected, "Exact closed input digest: " + path.name)
    return raw


def module(path, digest, name):
    obj = ModuleType(name)
    obj.__file__ = str(path)
    exec(compile(exact(path,digest),str(path),"exec"),obj.__dict__)
    return obj


def load_score(args, worker, stage, original, scorer):
    root = args.score_dir
    accepted = json.loads(exact(root/"INDEPENDENT_FLOAT_PRIMARY_SCORE_SAFE.json",PRIMARY_SHA))
    wait = json.loads(exact(args.score_wait,SCORE_WAIT_SHA))
    closed = json.loads(exact(root/"SCORE/CLOSED_STAGE_PRIVATE.json",CLOSED_SHA))
    request = json.loads(exact(root/"SCORE/STAGE_REQUEST_PRIVATE.json",SCORE_REQUEST_SHA))
    require(accepted["status"]=="SSDD_FLOAT_COMMON_DEV187_SCORE_INDEPENDENTLY_ACCEPTED_V1"
            and accepted["DEV"] and not accepted["TEST"] and accepted["annotations"]==380
            and accepted["original_score_outer_tool_wait_sha256"]==SCORE_WAIT_SHA
            and wait["stage"]=="SCORE" and wait["original_session_id"]==99645
            and wait["exact_tool_wait"]["exit_code"]==0 and wait["request_sha256"]==SCORE_REQUEST_SHA
            and wait["score_report_sha256"]==accepted["original_score_report_sha256"]==closed["report_ref"]["sha256"]
            and wait["original_child_terminal_sha256"]==accepted["original_score_worker_terminal_sha256"]
            ==closed["terminal_ref"]["sha256"], "Prior independent score and original closed stage")
    train = stage.train_args(request)
    pair = request["training_pair_ref"]
    _, trained = worker.read_training_pair(original,pair,**train)
    evaluation = request["evaluation_protocol"]
    sessions = [worker.verify_export_acceptance(stage,original,request[arm],arm,pair_sha=pair["sha256"],
                   evaluation_protocol=evaluation) for arm in ARMS]
    require(len(set(sessions))==2 and pair["sha256"]==accepted["training_pair_sha256"], "Distinct accepted export owners")
    manifest = request["image_manifest_ref"]
    _, ids, items = scorer.fixed_images(scorer.Artifact(manifest,stage.read_artifact(manifest,2**20)))
    gt = worker.read_json(stage,request["float_gt_ref"])
    require(request["float_gt_ref"]["sha256"]==accepted["floating_gt_sha256"]==worker.FLOAT_GT_SHA,
            "Exact author-float GT digest")
    scorer.validate_gt(gt,ids,items)
    apis = worker.score_apis(scorer)
    streams = {}
    for arm in ARMS:
        index, rows, evidence = worker.load_export(stage,original,scorer,request[arm],arm,trained[arm],
                pair_sha=pair["sha256"],arm_sha=train[arm+"_arm_ref"]["sha256"],evaluation_protocol=evaluation)
        doc = apis.bridge.group_canonical_rows(index,rows,items)
        flat, _ = apis.adapter.normalize_export(doc,gt["images"])
        require(flat == rows and evidence["prediction_sha256"] == accepted["arms"][arm]["prediction_sha256"],
                "Exact accepted prediction array and stable ranks")
        streams[arm] = flat
    return gt, streams, accepted, apis.official


def run(args):
    started = perf_counter()
    exact(Path(__file__).resolve().parent/"PROTOCOL.md",args.protocol_sha)
    sys.path.append(str(args.shapely_site.resolve(strict=True)))
    import shapely
    require(version("numpy")=="1.26.4" and version("pycocotools")=="2.0.11"
            and shapely.__version__=="2.0.2" and shapely.geos_version_string=="3.11.2","Exact CPU geometry versions")
    worker = module(args.code_root/"SSDD_FLOAT_PRIMARY_DEV187_20261112/float_dev187.py",WORKER_SHA,"_closed_ssdd_worker")
    geometry = module(args.code_root/"MASK_RTDETR_DEV705_FIXED_MATCH_GEOMETRY_20261112/fixed_match_geometry.py",
                      GEOMETRY_SHA,"_closed_geometry_core")
    admission = module(args.code_root/"MASK_RTDETR_DEV705_CONTINUOUS_AP_20261112/continuous_endpoint.py",
                       ADMISSION_SHA,"_closed_geometry_module_admission")
    stage, mods = worker.load_stock(args.code_root)
    stage.admit_runtime("SCORE")
    gt, streams, accepted, official = load_score(args,worker,stage,mods["ssdd_phase_dev_admission"],mods["ssdd_phase_dev_score"])
    wrap = geometry.load_module(args.code_root / geometry.MATCH_WRAPPER,geometry.MATCH_WRAPPER_SHA,"_prior_ssdd_match_loader")
    _,_,template,_,_,matching = wrap.load_helpers(args.code_root,args.code_root/geometry.MATCH_GEOMETRY)
    helpers=SimpleNamespace(template=template,adapter=official,match=matching.match_official)
    matches=geometry.original_matches(gt,streams,accepted["primary"],helpers,admission)
    shapes=geometry.analyze(gt,matches,admission.geometry_modules(args.code_root),expected_support=(380,0),dense_spots=8)
    result=dict(status="SSDD_FLOAT_COMMON_DEV187_FIXED_NATIVE_MATCH_GEOMETRY_ACTUAL",
                DEV=True,FIT=False,TEST=False,training=False,model_forward_calls=0,optimizer_updates=0,
                images=len(gt["images"]),annotations=len(gt["annotations"]),seed=5201,
                original_AP_and_AP90_exact=True,match_iou=.5,prediction_counts={arm:len(streams[arm]) for arm in ARMS},
                score_acceptance_sha256=PRIMARY_SHA,protocol_sha256=args.protocol_sha,
                worker_seconds=perf_counter()-started,**shapes)
    data=(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n").encode()
    with args.output.open("xb") as stream:
        stream.write(data)
    print(json.dumps(dict(status=result["status"],result_sha256=sha256(data).hexdigest())),flush=True)


def main():
    parser=ArgumentParser()
    for name in ("code-root","score-dir","score-wait","shapely-site","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--protocol-sha",required=True)
    args=parser.parse_args()
    require(sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES")=="-1"
            and os.environ.get("OMP_NUM_THREADS")==os.environ.get("OPENBLAS_NUM_THREADS")=="1","Isolated offline CPU scoring")
    def offline(event,_):
        if event.startswith("socket."):
            raise PermissionError("Closed offline DEV geometry")
    sys.addaudithook(offline)
    run(args)


if __name__=="__main__":
    main()

