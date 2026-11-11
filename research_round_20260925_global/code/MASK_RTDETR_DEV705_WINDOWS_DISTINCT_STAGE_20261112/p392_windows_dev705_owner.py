"""Own the fixed Mask RT-DETR DEV705 CUDA export and independent CPU scoring."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from time import monotonic
from types import ModuleType, SimpleNamespace

PROTOCOL = "437fa55f14644eab15156dc96092903bfde147da"
DEV_MANIFEST_SHA = "39a40d089d4a54448fe6efca8ea5cc98ad1eb7c0587815f3dd9a3b097173e78f"
FIT_MANIFEST_SHA = "a7e4d2a7ff9e8ff238adbf94a1d0397553d8746e2558c316db4e417f86f8fec9"
STOCK_OWNER_SHA = "071716cc00c76d060681d4216bcb228e16cbca686912e089bde6a1c6f1c4ee9f"
STOCK_WORKER_SHA = "19df923ed4842c33e396d852ac6589f9d8a53f39df0296d5b35ff37b94277605"
P392_PINS = {
    "author_dev_export.py": "cbf2aef0bf4b3d80af021e0a10d852f0df368250c051a295cb0cb420c16c53d2",
    "author_dev_score.py": "b64b9c4d9ceb90a53fa8b84a05fc95690ab60b67214eb7ba5ab2cc42af5a4c43",
    "author_endpoint.py": "63b16ced6cb9bc4eb654939754c135c9a876a346e4fbfdbaafe480af3754493f",
    "author_stream.py": "55e730a51432b4a2973e7ad75c45c31480b763b3b45b0b93b1b9aa340b7228e7",
}
KEY = "MASK_RTDETR_S_HRSID_DEV705_P392_WINDOWS_DISTINCT_STAGE_RETRY_20261112_01"
STAGES = ("fit_smoke", "image_stage", "native", "center")
IMAGE_DATA_DIR = "DEV_IMAGES_STAGED"
ARMS = STAGES[2:]
FIT_ID = 3641
SMOKE = "P392_FIT_ONE_AUTHOR_CUDA_ENDPOINT_SMOKE_COMPLETE"
EXPORT = "MASK_RTDETR_DEV705_E24_AUTHOR_EXPORT_COMPLETE"
IMAGE_STAGE = "MASK_RTDETR_DEV705_IMAGE_STAGE_COMPLETE"
CLOSED = "P392_MASK_RTDETR_DEV705_ORIGINAL_CUDA_EXPORTS_CLOSED_PENDING_SCORE"
WAIT = "P392_MASK_RTDETR_DEV705_ORIGINAL_CUDA_OWNER_TOOL_WAIT"
SOURCE = Path(__file__).resolve()
MIB = 1024**2
GIB = 1024**3


def need(ok, message):
    if not ok:
        raise ValueError(message)


def digest(data):
    return sha256(data).hexdigest()


def is_digest(value):
    return type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None


def pinned(path, expected, limit=MIB):
    need(path.is_absolute() and path.resolve(strict=True) == path and path.is_file()
         and 0 < path.stat().st_size <= limit and is_digest(expected), "Canonical bounded original artifact required")
    raw = path.read_bytes()
    need(digest(raw) == expected, "Exact pinned bytes required: " + path.name)
    return raw


def load(path, sha, name):
    module = ModuleType(name)
    module.__file__ = str(path)
    exec(compile(pinned(path, sha), str(path), "exec"), module.__dict__)
    return module


def emit(value):
    print(json.dumps(value, sort_keys=True, allow_nan=False), flush=True)


def source_context(p392, stock_path, *, with_job=False):
    for name, sha in P392_PINS.items():
        pinned(p392 / name, sha)
    exporter = load(p392 / "author_dev_export.py", P392_PINS["author_dev_export.py"], "_p392_original_export")
    need(exporter.PROTOCOL == PROTOCOL and exporter.MANIFEST_SHA == DEV_MANIFEST_SHA
         and exporter.ARMS == ARMS and exporter.IMAGE_COUNT == 705 and exporter.CANDIDATES == 100,
         "Exact original P392 protocol, roster and 100 author candidates required")
    need(stock_path.parent / "cuda_worker.py" == exporter.CUDA_WORKER
         and exporter.CUDA_WORKER_SHA == STOCK_WORKER_SHA, "Exact stock CUDA owner/worker family required")
    pinned(exporter.CUDA_WORKER, STOCK_WORKER_SHA)
    sys.path.insert(0, str(stock_path.parent))
    stock = load(stock_path, STOCK_OWNER_SHA, "_p392_original_cuda_stock_owner")
    need(stock.PYTHON.is_file() and stock.JOB.is_file(), "Exact stock CUDA interpreter and owner helper required")
    pinned(stock.JOB, stock.JOB_SHA)
    job = load(stock.JOB, stock.JOB_SHA, "_p392_original_windows_job") if with_job else None
    return SimpleNamespace(exporter=exporter, stock=stock, job=job)


def exact_workspace(exporter, work):
    need(work.is_absolute() and work.parent.resolve(strict=True) == work.parent
         and work.parent == exporter.ROOT and work.name.startswith("P392_MASK_RTDETR_WINDOWS_DEV705_")
         and not work.is_symlink(), "Fresh isolated P392 sibling execution root required")
    need(IMAGE_DATA_DIR.casefold() not in {stage.casefold() for stage in STAGES},
         "Image data storage must never alias a job control directory")
    exporter.STAGE = work / IMAGE_DATA_DIR


def endpoint_args(args, command, arm=None, **more):
    return SimpleNamespace(
        native_receipt=args.native_receipt, native_receipt_sha256=args.native_receipt_sha256,
        center_receipt=args.center_receipt, center_receipt_sha256=args.center_receipt_sha256,
        paired_receipt_sha256=args.paired_receipt_sha256, command=command, arm=arm, **more)


def arguments(args):
    values = ["--p392", str(args.p392), "--stock-owner", str(args.stock_owner),
              "--work-root", str(args.work_root), "--native-receipt", str(args.native_receipt),
              "--native-receipt-sha256", args.native_receipt_sha256,
              "--center-receipt", str(args.center_receipt), "--center-receipt-sha256", args.center_receipt_sha256,
              "--paired-receipt-sha256", args.paired_receipt_sha256, "--fit-manifest", str(args.fit_manifest),
              "--fit-images", str(args.fit_images)]
    return values


def transport_cuda_batch(batch, paddle, stats, *, verify_bytes):
    image = batch["image"]
    gt = sorted(key for key in batch if key.startswith("gt_"))
    before = dict(image_shape=list(image.shape), image_dtype=str(image.dtype),
                  image_place=str(image.place), gt_fields=gt)
    need(tuple(image.shape) == (1, 3, 640, 640) and image.dtype == paddle.float32 and not gt,
         "Unexpected pretransport stock FP32 image-only reader batch: " + json.dumps(before, sort_keys=True))
    stats = stats.setdefault("batch_transport", dict(
        verified_cuda_batches=0, host_to_cuda_batches=0, host_to_cuda_tensors=0,
        stock_gpu_batches=0, fit_transfer_byte_checks=0, fit_transfer_bytes=0))
    converted, copied = {}, 0
    for key, value in batch.items():
        if not isinstance(value, paddle.Tensor):
            converted[key] = value
            continue
        old_place = value.place
        if old_place.is_gpu_place():
            got = value
        else:
            need(old_place.is_cpu_place() or old_place.is_cuda_pinned_place(), "Unexpected original reader tensor place")
            reference = value.numpy().tobytes() if verify_bytes else None
            got = value.cuda(blocking=True)
            copied += 1
            if verify_bytes:
                need(got.numpy().tobytes() == reference, "Bitwise FIT reader tensor bytes changed in CUDA transport")
                stats["fit_transfer_byte_checks"] += 1
                stats["fit_transfer_bytes"] += len(reference)
        need(str(got.place) == "Place(gpu:0)" and got.dtype == value.dtype and tuple(got.shape) == tuple(value.shape),
             "Exact original reader dtype and shape on gpu:0 required")
        converted[key] = got
    need(all(isinstance(converted[key], paddle.Tensor) and str(converted[key].place) == "Place(gpu:0)"
             for key in ("image", "im_shape", "scale_factor", "im_id")), "All stock model input tensors must use gpu:0")
    stats["verified_cuda_batches"] += 1
    stats["host_to_cuda_tensors"] += copied
    stats["host_to_cuda_batches"] += int(copied > 0)
    stats["stock_gpu_batches"] += int(copied == 0)
    if verify_bytes:
        need(stats["verified_cuda_batches"] == 1, "Only the fixed FIT smoke may record transfer tensor bytes")
        stats["fit_before"] = before
        stats["fit_before_image_host"] = bool(image.place.is_cpu_place() or image.place.is_cuda_pinned_place())
        stats["fit_after"] = dict(before, image_place=str(converted["image"].place))
    return converted


def transport_source(exporter):
    path = exporter.HERE / "author_endpoint.py"
    raw = pinned(path, P392_PINS[path.name]).decode("utf-8")
    anchor = '            require(tuple(batch["image"].shape) == INPUT_SHAPE and batch["image"].dtype == paddle.float32'
    addition = '            if use_gpu:\n                batch = _transport_cuda_reader_batch(batch, paddle, stats)\n'
    need(raw.count(anchor) == 1, "Exact one original author TestReader boundary required")
    return path, compile(raw.replace(anchor, addition + anchor), str(path), "exec")


def install_reader_transport(exporter, *, fit):
    path, code = transport_source(exporter)
    module = ModuleType("author_endpoint")
    module.__file__ = str(path)
    module._transport_cuda_reader_batch = lambda batch, paddle, stats: transport_cuda_batch(
        batch, paddle, stats, verify_bytes=fit)
    sys.path[:0] = [str(exporter.HERE), str(exporter.RELEASE_CODE)]
    exec(code, module.__dict__)
    sys.modules["author_endpoint"] = module


def verify_reader_transport(report, *, fit=False):
    witness = report["batch_transport"]
    count = report["images"]
    need(witness["verified_cuda_batches"] == count
         and witness["host_to_cuda_batches"] + witness["stock_gpu_batches"] == count
         and witness["host_to_cuda_tensors"] >= witness["host_to_cuda_batches"],
         "All author reader tensors must retain their type and shape on GPU")
    if fit:
        before, after = witness["fit_before"], witness["fit_after"]
        need(count == witness["host_to_cuda_batches"] == 1 and witness["fit_before_image_host"] is True
             and witness["fit_transfer_byte_checks"] == witness["host_to_cuda_tensors"] > 0
             and witness["fit_transfer_bytes"] >= 1 * 3 * 640 * 640 * 4
             and before["image_shape"] == [1, 3, 640, 640] and before["image_dtype"] == "paddle.float32"
             and before["gt_fields"] == [] and after == dict(before, image_place="Place(gpu:0)")
             and before["image_place"] != after["image_place"], "FIT smoke must prove exact host-to-CUDA device repair")
    else:
        need(witness["fit_transfer_byte_checks"] == witness["fit_transfer_bytes"] == 0,
             "DEV typed transfers must avoid FIT-only round trips")


def verify_export_pair(reports, endpoints, stage_sha, pair_sha):
    need(tuple(reports) == ARMS and len({e["checkpoint_sha256"] for e in endpoints.values()}) == 2
         and len({e["training_independent_receipt_sha256"] for e in endpoints.values()}) == 2,
         "Distinct exact E24 EMA and independent arm receipts required")
    for arm in ARMS:
        report = reports[arm]
        expected = dict(status=EXPORT, arm=arm, protocol_commit=PROTOCOL, image_manifest_sha256=DEV_MANIFEST_SHA,
                        image_stage_report_sha256=stage_sha, paired_receipt_sha256=pair_sha,
                        GPU=True, DEV=True, GT_opened=False, official_TEST_opened=False, AP=False,
                        images=705, rows=70500, candidates_per_image=100, device="gpu:0", CUDA="11.8",
                        attention_calls={"author_paddle_grid_sample": 4230})
        expected.update({key: value for key, value in endpoints[arm].items() if key != "path"})
        need(all(key in report and type(report[key]) is type(value) and report[key] == value
                 for key, value in expected.items()), "Exact original CUDA export and correct per-arm EMA receipt required")
        verify_reader_transport(report)


def process_closed(terminal, live):
    return (terminal["job_active_after_wait"] == 0 and
            (terminal["owner"] is None or not live(terminal["owner"])))


def require_clean_job(terminal, live):
    need(process_closed(terminal, live) and terminal["owner"] is not None
         and terminal["original_os_wait"] is True and type(terminal["original_os_exit"]) is int
         and terminal["original_os_exit"] == 0 and terminal["job_assigned"] is True
         and terminal["job_active_before_drain"] == 0 and terminal["stop_reason"] is None,
         "Exact original natural wait and empty JobObject required")


def fit_smoke(ctx, args, endpoint, output):
    exporter = ctx.exporter
    rows = json.loads(pinned(args.fit_manifest, FIT_MANIFEST_SHA, 4 * MIB))["images"]
    row, = (row for row in rows if row["image_id"] == FIT_ID)
    need(len(rows) == 2937 and args.fit_images.parent == args.fit_manifest.parent
         and Path(row["native_basename"]).name == row["native_basename"], "Exact fixed image-only FIT layout required")
    image = args.fit_images / row["native_basename"]
    need(len(pinned(image, row["encoded_sha256"], 4 * MIB)) == row["encoded_bytes"], "Exact fixed FIT PNG length")
    sys.path[:0] = [str(exporter.HERE), str(exporter.RELEASE_CODE)]
    worker, paddle, np, runtime = exporter.load_runtime()
    from author_endpoint import create_endpoint_model, export_native_endpoint
    from ppdet.modeling.transformers.utils import deformable_attention_core_func

    model = create_endpoint_model(exporter.CONFIG, endpoint["path"], paddle=paddle, np=np)
    calls = worker.witness_attention(model, deformable_attention_core_func)
    output.mkdir()
    stats = export_native_endpoint(output / "PREDICTIONS_PRIVATE.jsonl", [(FIT_ID, str(image))],
                                   {FIT_ID: (800, 800)}, model=model, paddle=paddle, np=np,
                                   synchronize=paddle.device.cuda.synchronize)
    need(stats["images"] == 1 and stats["rows"] == 100 and stats["candidates_per_image"] == 100
         and dict(calls) == {"author_paddle_grid_sample": 6}, "Exact one-image native CUDA FIT smoke")
    stats.update(status=SMOKE, protocol_commit=PROTOCOL, checkpoint_sha256=endpoint["checkpoint_sha256"],
                 FIT=True, DEV=False, TEST=False, AP=False, GPU=True,
                 CUDA=paddle.version.cuda(), device=paddle.device.get_device(), runtime=runtime,
                 attention_calls=dict(calls))
    report_sha = exporter.write_report(output / "EXPORT_SAFE.json", stats)
    emit(dict(status=SMOKE, report_sha256=report_sha, predictions_sha256=stats["predictions_sha256"]))


def child(ctx, args):
    stage = args.worker
    directory = args.work_root / stage
    need(Path.cwd() == directory / "CONTROL"
         and (Path.cwd() / "JOB_ATTACHED.txt").read_bytes() == digest(SOURCE.read_bytes()).encode("ascii"),
         "Exact original Windows JobObject attachment required before starting")
    need((args.image_stage_sha256 is not None) == (stage in ARMS), "Image-stage receipt only for DEV GPU export")
    exporter = ctx.exporter
    if stage != "image_stage":
        install_reader_transport(exporter, fit=stage == "fit_smoke")
    if stage == "fit_smoke":
        endpoints = exporter.admit(endpoint_args(args, "stage"))
        fit_smoke(ctx, args, endpoints["native"], directory / "OUTPUT")
    elif stage == "image_stage":
        need(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "Image staging must keep CUDA hidden")
        endpoints = exporter.admit(endpoint_args(args, "stage"))
        exporter.stage_images(exporter.frozen_manifest(), endpoints, args.paired_receipt_sha256)
    else:
        request = endpoint_args(args, "export", stage, image_stage_report_sha256=args.image_stage_sha256,
                                output=directory / "OUTPUT")
        endpoints = exporter.admit(request)
        exporter.export_arm(request, exporter.frozen_manifest(), endpoints)


def launch_stage(ctx, args, name, reserved, stage_sha, track_closure):
    job, stock = ctx.job, ctx.stock
    directory = args.work_root / name / "CONTROL"
    directory.mkdir(parents=True)
    scratch = directory / "scratch"
    scratch.mkdir()
    gpu = name != "image_stage"
    free, available = stock.gpu_resources() if gpu else (None, stock.psutil.virtual_memory().available)
    win = Path(os.environ["SystemRoot"])
    env = dict(os.environ, PATH=os.pathsep.join(map(str, (win / "System32", win, Path(sys.base_prefix), *stock.DLL_DIRS))),
               CUDA_VISIBLE_DEVICES="0" if gpu else "-1", FLAGS_allocator_strategy="auto_growth",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
               FLAGS_paddle_num_threads="1", PYTHONNOUSERSITE="1", TMP=str(scratch), TEMP=str(scratch))
    start, process, peak, host_floor, reason = monotonic(), None, 0, available, None

    def popen(*values, **kwargs):
        nonlocal process
        process = subprocess.Popen(*values, env=env, **kwargs)
        return process

    def authorized():
        nonlocal peak, host_floor, reason
        if job.STATE.read_bytes() != reserved:
            reason = "SHARED_GPU_RESERVATION_CHANGED"
        elif monotonic() - start >= 900:
            reason = "ORIGINAL_900_SECOND_RESOURCE_LIMIT"
        else:
            host_floor = min(host_floor, stock.psutil.virtual_memory().available)
            if host_floor < 18 * GIB:
                reason = "HOST_AVAILABLE_MEMORY_BELOW_18_GIB"
            if process is not None and process.poll() is None:
                try:
                    parent = stock.psutil.Process(process.pid)
                    peak = max(peak, sum(child.memory_info().rss for child in [parent, *parent.children(recursive=True)]
                                         if child.is_running()))
                except stock.psutil.NoSuchProcess:
                    pass
                if peak > 10 * GIB:
                    reason = "WORKER_TREE_RSS_ABOVE_10_GIB"
        return reason is None

    job.CONTROL, job.OWNER_PYTHON = directory, stock.PYTHON
    job.subprocess = SimpleNamespace(Popen=popen, CREATE_NO_WINDOW=subprocess.CREATE_NO_WINDOW,
                                     BELOW_NORMAL_PRIORITY_CLASS=subprocess.BELOW_NORMAL_PRIORITY_CLASS)
    argv = ["--worker", name, *arguments(args)]
    if name in ARMS:
        argv.extend(["--image-stage-sha256", stage_sha])
    track_closure(False)
    terminal = job.wait_owner(SOURCE, argv, authorized)
    track_closure(process_closed(terminal, job.live))
    job.write_json(directory / "RESOURCE_PRIVATE.json", dict(trigger=reason, minimum_available_host_bytes=host_floor,
                   peak_worker_tree_rss_bytes=peak, initial_cuda_free_mib=free, timeout_seconds=900,
                   GPU=gpu, expected_child_source_sha256=digest(SOURCE.read_bytes())))
    return terminal


def read_closed_stage(ctx, work, name, expected=None):
    job = ctx.job
    control = work / name / "CONTROL"
    paths = {role: control / filename for role, filename in (
        ("terminal", "OWNER_TERMINAL_PRIVATE.json"), ("resource", "RESOURCE_PRIVATE.json"),
        ("stdout", "OWNER.stdout"), ("stderr", "OWNER.stderr"), ("birth", "OWNER_BIRTH_PRIVATE.json"))}
    report = work / IMAGE_DATA_DIR / "STAGE_SAFE.json" if name == "image_stage" else work / name / "OUTPUT/EXPORT_SAFE.json"
    refs = {role: job.log_ref(path) for role, path in paths.items()}
    terminal = json.loads(pinned(paths["terminal"], refs["terminal"]["sha256"]))
    resource = json.loads(pinned(paths["resource"], refs["resource"]["sha256"]))
    need(type(terminal["original_os_exit"]) is int and terminal["original_os_exit"] == 0,
         f"Original {name} child exit={terminal['original_os_exit']}; see its immutable stderr and terminal")
    require_clean_job(terminal, job.live)
    refs["report"] = job.log_ref(report)
    if expected is not None:
        need(refs == expected, "Original child/resource/report references differ")
    need(terminal["owner"] == json.loads(pinned(paths["birth"], refs["birth"]["sha256"]))
         and terminal["stdout_ref"] == refs["stdout"] and terminal["stderr_ref"] == refs["stderr"]
         and resource["trigger"] is None and resource["minimum_available_host_bytes"] >= 18 * GIB
         and 0 <= resource["peak_worker_tree_rss_bytes"] <= 10 * GIB
         and resource["timeout_seconds"] == 900 and resource["GPU"] is (name != "image_stage")
         and (resource["initial_cuda_free_mib"] is None if name == "image_stage" else resource["initial_cuda_free_mib"] >= 12288)
         and (control / "JOB_ATTACHED.txt").read_bytes() == resource["expected_child_source_sha256"].encode("ascii")
         and resource["expected_child_source_sha256"] == digest(SOURCE.read_bytes()),
         "Exact original resource, source attachment and log closure required")
    stdout = pinned(paths["stdout"], refs["stdout"]["sha256"], 8 * MIB).decode("utf-8")
    marker = json.loads(stdout.splitlines()[-1])
    value = json.loads(pinned(report, refs["report"]["sha256"]))
    status, marker_key = {
        "fit_smoke": (SMOKE, "report_sha256"), "image_stage": (IMAGE_STAGE, "image_stage_report_sha256"),
        "native": (EXPORT, "export_report_sha256"), "center": (EXPORT, "export_report_sha256"),
    }[name]
    need(value["status"] == marker["status"] == status and value["protocol_commit"] == PROTOCOL
         and marker[marker_key] == refs["report"]["sha256"], "Original child stdout must bind its sealed report")
    if name in ARMS:
        need(marker["arm"] == value["arm"] == name, "Exact original export arm marker")
    if name != "image_stage":
        need(marker["predictions_sha256"] == value["predictions_sha256"], "Original predictions digest mismatch")
    return refs, value, terminal, stdout


def verify_fit_smoke(report, endpoint):
    verify_reader_transport(report, fit=True)
    expected = dict(status=SMOKE, protocol_commit=PROTOCOL, FIT=True, DEV=False, TEST=False, AP=False,
                    GT_opened=False, official_TEST_opened=False, GPU=True, CUDA="11.8", device="gpu:0",
                    images=1, rows=100, candidates_per_image=100, attention_calls={"author_paddle_grid_sample": 6},
                    checkpoint_sha256=endpoint["checkpoint_sha256"])
    need(is_digest(report["predictions_sha256"])
         and all(key in report and type(report[key]) is type(value) and report[key] == value
                 for key, value in expected.items()), "Exact FIT-only CUDA smoke and accepted native EMA required")


def run(ctx, args):
    work, job, stock = args.work_root, ctx.job, ctx.stock
    need(not work.exists(), "Fresh isolated P392 root required")
    raw = job.P189["stable"](job.STATE, stock.STATE_LIMIT)
    state = json.loads(raw)
    need(KEY not in state and state.get("GPU_owner_or_worker_active") is False
         and state.get("gpu_training_active") is False
         and all(value.get("active_attempt") is None for value in state.values() if type(value) is dict),
         "Exact HRSID original B and shared GPU authority handoff required before reading endpoint inputs")
    endpoints = ctx.exporter.admit(endpoint_args(args, "stage"))
    stock.gpu_resources()
    work.mkdir()
    birth = job.birth(os.getpid())
    state[KEY] = dict(status="P392_DEV705_RESERVED", active_attempt=KEY, owner_birth=birth, resource_closed=False,
                      paired_receipt_sha256=args.paired_receipt_sha256, official_TEST_opened=False)
    state.update(GPU_owner_or_worker_active=True, gpu_training_active=True)
    job.cas(raw, state, "P392_DEV705_CUDA_RESERVE")
    reserved = job.P189["stable"](job.STATE, stock.STATE_LIMIT)
    need(json.loads(reserved) == state, "Exact sole GPU owner reservation required")
    stages, reports, stage_sha, failure, closed = {}, {}, None, None, True

    def track_closure(value):
        nonlocal closed
        closed = value

    try:
        for name in STAGES:
            launch_stage(ctx, args, name, reserved, stage_sha, track_closure)
            refs, report, _, _ = read_closed_stage(ctx, work, name)
            stages[name], reports[name] = refs, report
            if name == "fit_smoke":
                verify_fit_smoke(report, endpoints["native"])
            if name == "image_stage":
                stage_sha = refs["report"]["sha256"]
            emit(dict(status="P392_ORIGINAL_CHILD_CLOSED", stage=name, report_sha256=refs["report"]["sha256"]))
        verify_export_pair({arm: reports[arm] for arm in ARMS}, endpoints, stage_sha, args.paired_receipt_sha256)
    except (Exception, KeyboardInterrupt) as error:
        failure = type(error).__name__ + ": " + str(error)
    summary = dict(status=CLOSED if failure is None else "P392_DEV705_ORIGINAL_CUDA_OWNER_FAILED",
                   owner_birth=birth, owner_source_sha256=digest(SOURCE.read_bytes()),
                   protocol_commit=PROTOCOL, paired_receipt_sha256=args.paired_receipt_sha256,
                   stages=stages, failure=failure, resource_closed=closed, official_TEST_opened=False)
    path = work / "ORIGINAL_OWNER_SUMMARY_PRIVATE.json"
    job.write_json(path, summary)
    summary_sha = job.ref(path)["sha256"]
    if closed:
        state[KEY].update(status=summary["status"], active_attempt=None, resource_closed=True,
                          summary_ref=job.ref(path), failure=failure)
        state.update(GPU_owner_or_worker_active=False, gpu_training_active=False)
        job.cas(reserved, state, "P392_DEV705_CUDA_CLOSE")
    emit(dict(status=summary["status"], owner_birth=birth, owner_summary_sha256=summary_sha,
              resource_closed=closed, failure=failure))
    return 0 if failure is None else 5


def score(ctx, args):
    job, work = ctx.job, args.work_root
    wait = json.loads(pinned(args.original_wait, args.original_wait_sha256))
    terminal = wait["exact_tool_wait"]
    need(wait["status"] == WAIT and type(wait["original_session_id"]) is int and wait["original_session_id"] > 0
         and type(terminal["exit_code"]) is int and terminal["exit_code"] == 0 and terminal.get("session_id") is None,
         "Exact original external owner tool wait required")
    marker = json.loads(terminal["output"].splitlines()[-1])
    combined = wait["combined_stdout"]
    need(type(combined) is str and len(combined) <= 16 * 1024,
         "Exact concatenated original owner stdout required")
    owner_summary_sha = marker["owner_summary_sha256"]
    summary = json.loads(pinned(work / "ORIGINAL_OWNER_SUMMARY_PRIVATE.json", owner_summary_sha))
    need(marker == dict(status=CLOSED, owner_birth=summary["owner_birth"], owner_summary_sha256=owner_summary_sha,
                       resource_closed=True, failure=None)
         and summary["status"] == CLOSED and summary["owner_source_sha256"] == digest(SOURCE.read_bytes())
         and summary["protocol_commit"] == PROTOCOL and summary["paired_receipt_sha256"] == args.paired_receipt_sha256
         and summary["resource_closed"] is True and summary["failure"] is None and not job.live(summary["owner_birth"])
         and set(summary["stages"]) == set(STAGES), "Exact closed original four-stage CUDA owner required")
    endpoints = ctx.exporter.admit(endpoint_args(args, "stage"))
    closed = {stage: read_closed_stage(ctx, work, stage, summary["stages"][stage]) for stage in STAGES}
    births = [closed[stage][2]["owner"] for stage in STAGES]
    chronology = [summary["owner_birth"], *births]
    need(len({row["pid"] for row in chronology}) == len(chronology)
         and all(a["creation_time"] < b["creation_time"] for a, b in zip(chronology, chronology[1:])),
         "Original owner and four sequential child births must be distinct")
    stage_markers = [dict(status="P392_ORIGINAL_CHILD_CLOSED", stage=stage,
                          report_sha256=closed[stage][0]["report"]["sha256"]) for stage in STAGES]
    need([json.loads(line) for line in combined.splitlines() if line] == [*stage_markers, marker],
         "Exact original owner stdout must bind four ordered child closures")
    verify_fit_smoke(closed["fit_smoke"][1], endpoints["native"])
    stage_sha = closed["image_stage"][0]["report"]["sha256"]
    verify_export_pair({arm: closed[arm][1] for arm in ARMS}, endpoints, stage_sha, args.paired_receipt_sha256)
    need(args.score_python.is_absolute() and args.score_python.resolve(strict=True) == args.score_python
         and args.score_output.is_absolute() and args.score_output.parent.resolve(strict=True) == args.score_output.parent
         and args.score_output.parent == work.parent and not args.score_output.exists(),
         "Separate canonical CPU interpreter and new score scope required")
    args.score_output.mkdir()
    result = args.score_output / "RESULT_SAFE.json"
    score_path = args.p392 / "author_dev_score.py"
    pinned(score_path, P392_PINS[score_path.name])
    cmd = [str(args.score_python), "-I", "-B", "-X", "utf8", str(score_path), "--output", str(result)]
    for arm in ARMS:
        refs, _, child_terminal, stdout = closed[arm]
        closure = args.score_output / (arm + "_ORIGINAL_CLOSED_PRIVATE.json")
        job.write_json(closure, dict(arm=arm, evidence="ORIGINAL_WINDOWS_OS_WAIT_AND_EXTERNAL_OWNER_TOOL_WAIT",
            original_child_terminal_ref=refs["terminal"], original_external_wait_sha256=args.original_wait_sha256,
            exact_tool_wait=dict(exit_code=child_terminal["original_os_exit"], output=stdout)))
        cmd.extend(["--" + arm + "-export", str(work / arm / "OUTPUT"), "--" + arm + "-closure", str(closure),
                    "--" + arm + "-closure-sha256", job.ref(closure)["sha256"]])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    stdout, stderr = args.score_output / "ORIGINAL_SCORE.stdout", args.score_output / "ORIGINAL_SCORE.stderr"
    with stdout.open("xb") as out, stderr.open("xb") as err:
        score_run = subprocess.run(cmd, env=env, stdout=out, stderr=err, creationflags=subprocess.CREATE_NO_WINDOW)
    job.write_json(args.score_output / "ORIGINAL_SCORE_TERMINAL_PRIVATE.json", dict(
        original_os_wait=True, original_os_exit=score_run.returncode, stdout_ref=job.log_ref(stdout),
        stderr_ref=job.log_ref(stderr), original_export_owner_wait_sha256=args.original_wait_sha256))
    need(score_run.returncode == 0, "Original independent CPU score failed")
    report_sha = job.ref(result)["sha256"]
    last = json.loads(stdout.read_bytes().splitlines()[-1])
    need(last["result_sha256"] == report_sha
         and last["status"] == "MASK_RTDETR_DEV705_E24_AUTHOR_PAIR_SCORE_COMPLETE",
         "Exact independent CPU score marker required")
    emit(last)
    return 0


def main():
    cli = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    action = cli.add_mutually_exclusive_group(required=True)
    action.add_argument("--check-sources", action="store_true")
    action.add_argument("--run", action="store_true")
    action.add_argument("--worker", choices=STAGES)
    action.add_argument("--score", action="store_true")
    for key in ("p392", "stock-owner"):
        cli.add_argument("--" + key, type=Path, required=True)
    for key in ("work-root", "native-receipt", "center-receipt", "fit-manifest", "fit-images",
                "original-wait", "score-python", "score-output"):
        cli.add_argument("--" + key, type=Path)
    for key in ("native-receipt-sha256", "center-receipt-sha256", "paired-receipt-sha256",
                "original-wait-sha256", "image-stage-sha256"):
        cli.add_argument("--" + key)
    args = cli.parse_args()
    ctx = source_context(args.p392, args.stock_owner, with_job=args.run or args.score)
    if args.check_sources:
        transport_source(ctx.exporter)
        emit(dict(status="P392_OWNER_SOURCE_CHECK_COMPLETE", protocol_commit=PROTOCOL,
                  exact_native_reader_boundary_compiles=True,
                  author_files=P392_PINS, launch_requested=False))
        return 0
    need(all((args.work_root, args.native_receipt, args.center_receipt)) and all(is_digest(value) for value in (
        args.native_receipt_sha256, args.center_receipt_sha256, args.paired_receipt_sha256)),
        "Exact closed A/B training receipts and paired acceptance required")
    need(all((value is not None) == args.score for value in (
        args.original_wait, args.original_wait_sha256, args.score_python, args.score_output)),
        "Independent CPU score requires exact external wait and a separate output")
    exact_workspace(ctx.exporter, args.work_root)
    if args.score:
        return score(ctx, args)
    need(args.fit_manifest is not None and args.fit_images is not None, "Exact FIT smoke inputs required")
    if args.worker:
        return child(ctx, args)
    return run(ctx, args)


if __name__ == "__main__":
    sys.exit(main())
