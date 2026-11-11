"""Run and independently accept the float-trained SSDD DEV187 export or score stage."""

from argparse import ArgumentParser
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace


STOCK_SHA = "cf78814f52adb9d774d719729879be098b5288133523418b1f8b14d519d7e85f"
WORKER_SHA = "5463f294191929bceaac3b04fb6c17179c05066eaa29b6d859ef4943cca686c7"
IMAGE_ATTESTATION_SHA = "83e7352236d3bbc37c37fcc9849a93fe8eb87a2495b2ad42aed87d6ef37593be"
IMAGE_ATTESTATION = "SSDD_WINDOWS_Q100_E24_ACCEPTANCE_AND_FIRST_DEV_EXPORT_FAILURE_SAFE_20261111.json"
IMAGE_DIRECTORY = "SSDD_WINDOWS_DEV187_IMAGE_ONLY_ACTUAL_PRIVATE_20261111_01"
CONFIG_STATUS = "SSDD_FLOAT_COMMON_DEV187_WINDOWS_OWNER_CONFIG_V1"
SCORE_WAIT_STATUS = "SSDD_FLOAT_COMMON_DEV187_SCORE_ORIGINAL_TOOL_WAIT_V1"
SCORE_OWNER_CLOSED = "SSDD_FLOAT_COMMON_DEV187_SCORE_ORIGINAL_OWNER_CLOSED_V1"
SCORE_ACCEPTED = "SSDD_FLOAT_COMMON_DEV187_SCORE_INDEPENDENTLY_ACCEPTED_V1"
PENDING = "SSDD_FLOAT_COMMON_DEV187_STAGE_NATURALLY_CLOSED_PENDING_EXTERNAL_WAIT_V1"
RESERVED = "SSDD_FLOAT_COMMON_DEV187_STAGE_RESERVED_V1"
FAILED = "SSDD_FLOAT_COMMON_DEV187_STAGE_CLOSED_FAILED_V1"
UNCERTAIN = "SSDD_FLOAT_COMMON_DEV187_STAGE_CONTAINMENT_UNCERTAIN_V1"
STAGES = ("EXPORT_NATIVE", "EXPORT_CENTER", "SCORE")
MODES = ("native", "center")
WAIT_KEYS = ("stage", "request_sha256", "score_report_sha256", "original_child_terminal_sha256")
LIMIT = 64 * 1024


def need(ok, detail):
    if not ok:
        raise ValueError(detail)


def load_source(path, expected, name):
    need(path.is_absolute() and path.resolve(strict=True) == path and path.is_file(),
         "Canonical pinned owner or worker source required")
    with path.open("rb") as stream:
        raw = stream.read(LIMIT + 1)
    need(0 < len(raw) <= LIMIT and sha256(raw).hexdigest() == expected, "Pinned source differs: " + path.name)
    module = ModuleType(name)
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def context(stock_path, worker_path, code_root, protocol):
    stock = load_source(stock_path, STOCK_SHA, "_float_exact_stock_owner")
    worker = load_source(worker_path, WORKER_SHA, "_float_exact_dev_worker")
    stage, modules = worker.load_stock(code_root)
    sources = stock.load_sources()
    need(stock.SECONDS == 1200 and stock.GPU_FREE_MIB == 6144
         and sources.runtime.RAM_FLOOR == 16 * 1024**3 and sources.runtime.RSS_CAP == 10 * 1024**3,
         "Original proven SSDD Windows stage resources must be retained")
    stage.read_artifact(worker.file_ref(protocol, worker.PROTOCOL_SHA), stage.MIB)
    evidence = code_root.parent / "results" / IMAGE_ATTESTATION
    doc = worker.read_json(stage, worker.file_ref(evidence, IMAGE_ATTESTATION_SHA))
    image_attestation = doc["dev_image_preparation"]
    need(stock.matches(image_attestation, dict(status="P287_SSDD_DEV187_IMAGE_ONLY_STAGE_READY_V1",
          manifest_sha256=stock.IMAGE_MANIFEST_SHA, manifest_bytes=stock.IMAGE_MANIFEST_BYTES, images=187,
          distinct_jpeg_sha256=183, image_aliases=4, unique_encoded_bytes=7118253, logical_encoded_bytes=7274234,
          FIT=False, DEV=True, TEST=False, GT_read=False, AP=False, model_execution=False), ()),
         "Exact published attestation for the existing image-only DEV stage required")
    stock.STAGE_REPORT = worker.REPORT_EVENT
    sources.short.SHORT_SECONDS = stock.SECONDS
    return SimpleNamespace(stock=stock, worker=worker, stage=stage, modules=modules, sources=sources,
                           worker_path=worker_path, worker_ref=worker.file_ref(worker_path, WORKER_SHA),
                           code_root=code_root, protocol=protocol, image_attestation=image_attestation)


def runtime_for(runtime, stage):
    def env(scratch):
        result = dict(runtime.windows_runtime(scratch), CUDA_VISIBLE_DEVICES="-1" if stage == "SCORE" else "0")
        if stage == "SCORE":
            result.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
        return result
    return SimpleNamespace(RUNTIME=runtime.RUNTIME, psutil=runtime.psutil, RAM_FLOOR=runtime.RAM_FLOOR,
                           RSS_CAP=runtime.RSS_CAP, windows_runtime=env)


def read_config(ctx, path, expected_sha):
    stage, worker, stock = ctx.stage, ctx.worker, ctx.stock
    ref = worker.file_ref(path, expected_sha)
    cfg = worker.read_json(stage, ref)
    refs = ("training_pair_ref", "native_arm_ref", "center_arm_ref", "native_checkpoint_ref", "center_checkpoint_ref",
            "image_manifest_ref", "float_gt_ref", "integer_gt_ref", "historical_ref")
    stage.keys(cfg, {"status", "evaluation_protocol", "work_root", "dev_root", *refs}, "Exact float owner config required")
    need(cfg["status"] == CONFIG_STATUS and stage.hex_digest(cfg["evaluation_protocol"], 40),
         "Exact float owner and evaluation protocol required")
    work, dev = (stage.path_value(cfg[key]) for key in ("work_root", "dev_root"))
    need(work.parent == stock.ROOT and work.parent.resolve(strict=True) == work.parent
         and work.name.startswith("SSDD_WINDOWS_FLOAT_PRIMARY_DEV187_ACTUAL_PRIVATE_")
         and work.resolve() == work and not work.is_symlink() and (not work.exists() or work.is_dir())
         and dev == stock.ROOT / IMAGE_DIRECTORY, "Fresh operation root and original P287 image-only location required")
    paths = []
    for key in refs:
        archive = key.endswith("_checkpoint_ref")
        paths.append(stage.ref_path(cfg[key], 1024 * stage.MIB if archive else stage.MIB,
                                   name="checkpoint_23.ckpt" if archive else None))
    need(len(set(paths)) == len(paths) and all(not p.is_relative_to(work) and not p.is_relative_to(dev) for p in paths),
         "Only separate canonical stage inputs allowed")
    for key, digest in (("image_manifest_ref", stock.IMAGE_MANIFEST_SHA), ("float_gt_ref", worker.FLOAT_GT_SHA),
                        ("integer_gt_ref", worker.INTEGER_GT_SHA), ("historical_ref", worker.REFERENCE_SHA)):
        need(cfg[key]["sha256"] == digest, "Exact prelocked float evaluation file reference required: " + key)
    need(cfg["image_manifest_ref"]["bytes"] == stock.IMAGE_MANIFEST_BYTES
         and cfg["integer_gt_ref"]["bytes"] == stock.GT_BYTES, "Frozen image and integer file sizes required")
    return SimpleNamespace(cfg=cfg, config_ref=ref, work=work, dev=dev)


def read_training(ctx, scope):
    cfg, worker, stage = scope.cfg, ctx.worker, ctx.stage
    original = ctx.modules["ssdd_phase_dev_admission"]
    _, accepted = worker.read_training_pair(
        original, cfg["training_pair_ref"], native_arm_ref=cfg["native_arm_ref"], center_arm_ref=cfg["center_arm_ref"],
        training_protocol=worker.TRAINING_PROTOCOL, evaluation_protocol=cfg["evaluation_protocol"])
    for mode in MODES:
        ref = cfg[mode + "_checkpoint_ref"]
        need(ref["sha256"] == accepted[mode]["checkpoint_sha256"]
             and ref["bytes"] == accepted[mode]["checkpoint_bytes"],
             "Exact independently accepted float arm archive identities required")
        original.reference(ref, max_bytes=1024 * stage.MIB, suffix=".ckpt")
    return accepted


def stage_key(scope, stage):
    return "SSDD_WINDOWS_FLOAT_PRIMARY_DEV187_" + stage + "_" + scope.config_ref["sha256"][:16]


def wait_status(worker, stage):
    return SCORE_WAIT_STATUS if stage == "SCORE" else worker.EXPORT_WAIT_STATUS


def closed_event(worker, stage, record):
    report = "score_report_sha256" if stage == "SCORE" else "export_report_sha256"
    return dict(status=SCORE_OWNER_CLOSED if stage == "SCORE" else worker.EXPORT_OWNER_CLOSED,
                stage=stage, request_sha256=record["request_ref"]["sha256"],
                original_child_terminal_sha256=record["terminal_ref"]["sha256"],
                **{report: record["report_ref"]["sha256"]})


def verify_original_wait(stock, worker, stage, record, receipt):
    event = closed_event(worker, stage, record)
    join = {key: value for key, value in event.items() if key != "status"}
    need(stock.matches(receipt, dict(status=wait_status(worker, stage), **join),
                       ("original_session_id", "exact_tool_wait")), "Exact original outer stage join required")
    result = receipt["exact_tool_wait"]
    need(type(receipt["original_session_id"]) is int and receipt["original_session_id"] > 0
         and type(result["exit_code"]) is int and result["exit_code"] == 0 and result.get("session_id") is None
         and json.loads(result["output"].splitlines()[-1]) == event,
         "Exact original outer stage exit 0 and complete footer required")
    return receipt["original_session_id"]


def prep_authority(ctx):
    sources = ctx.sources
    io, authority = sources.accepted.io, sources.accepted.authority
    raw = io.stable_read(authority.STATE, ctx.stock.STATE_LIMIT)
    state = sources.accepted.entry.document(raw)
    sources.accepted.owner.check_idle(state)
    return SimpleNamespace(io=io, authority=authority, raw=raw, state=state)


def previous_exports(ctx, scope, stage, state):
    exports = {}
    preceding = () if stage == "EXPORT_NATIVE" else (("native",) if stage == "EXPORT_CENTER" else MODES)
    for mode in preceding:
        row = state.get(stage_key(scope, "EXPORT_" + mode.upper()), {})
        need(ctx.stock.matches(row, dict(status=ctx.worker.EXPORT_ACCEPTED_STATUS, active_attempt=None,
             resource_closed=True, failure=None, config_ref=scope.config_ref)),
             "Earlier independently accepted float export and original outer wait required")
        exports[mode] = row["export_record"]
    return exports


def request_for(ctx, scope, stage, area, exports):
    cfg, worker = scope.cfg, ctx.worker
    request = dict(status=worker.REQUEST_STATUS, stage=stage, training_protocol=worker.TRAINING_PROTOCOL,
                   evaluation_protocol=cfg["evaluation_protocol"], output_directory=(area / "ACTUAL").as_posix(),
                   **{key:cfg[key] for key in ("training_pair_ref", "native_arm_ref", "center_arm_ref",
                                              "image_manifest_ref")})
    if stage != "SCORE":
        mode = stage.split("_")[-1].lower()
        return dict(request, archive_ref=cfg[mode + "_checkpoint_ref"], dev_root=scope.dev.as_posix())
    need(set(exports) == set(MODES), "Both original float exports must independently close before score")
    return dict(request, **{key:cfg[key] for key in ("float_gt_ref", "integer_gt_ref", "historical_ref")}, **exports)


def stage_owner(ctx, area, request_ref, stage, reserved, coordinator):
    stock, sources, legacy = ctx.stock, ctx.sources, ctx.sources.legacy
    arm = SimpleNamespace(control=area / "CONTROL", output=area / "ACTUAL", audit=area / "UNUSED_AUDIT",
                          arguments=["--code-root", str(ctx.code_root), "--protocol", str(ctx.protocol),
                                     "--request", request_ref["path"], "--request-sha256", request_ref["sha256"],
                                     "--stage", stage])
    original_ref = legacy.ref

    def verified_ref(path):
        value = original_ref(path)
        need(path != ctx.worker_path or value == ctx.worker_ref, "Exact float worker bytes required at JobObject attach")
        return value

    legacy.ref = verified_ref
    try:
        terminal, guard = sources.short.original(legacy, runtime_for(sources.runtime, stage), arm,
                                                 ctx.worker_path, reserved)
    finally:
        legacy.ref = original_ref
    term_ref = sources.short.closed_arm(legacy, arm, terminal, coordinator)
    return terminal, guard, term_ref


def stage_readback(ctx, stage, area, request_ref, terminal, guard, terminal_ref, coordinator):
    shim = SimpleNamespace(lock=dict(worker=ctx.worker_ref))
    seen = {ctx.sources.p283.birth_key(coordinator)}
    return ctx.stock.stage_readback(ctx.sources, shim, stage, area, request_ref, terminal, guard, terminal_ref, seen)


def emit(value):
    print(json.dumps(value, sort_keys=True, allow_nan=False), flush=True)


def run_original(ctx, scope, stage):
    stock, sources = ctx.stock, ctx.sources
    authority = prep_authority(ctx)
    key, area = stage_key(scope, stage), scope.work / stage
    need(key not in authority.state and not area.exists() and not area.is_symlink(), "Fresh original float stage required")
    need(scope.work.exists() == (stage != "EXPORT_NATIVE"), "Original evaluation stages must run in order")
    exports = previous_exports(ctx, scope, stage, authority.state)
    read_training(ctx, scope)
    stock.resources(sources.runtime, cuda=stage != "SCORE")
    need(authority.io.stable_read(authority.authority.STATE, stock.STATE_LIMIT) == authority.raw,
         "Exclusive authority changed before float stage reservation")
    coordinator = sources.legacy.birth(os.getpid())
    (area / "CONTROL/scratch").mkdir(parents=True)
    row = dict(status=RESERVED, active_attempt=key, owner_birth=coordinator, resource_closed=False,
               stage=stage, config_ref=scope.config_ref, worker_ref=ctx.worker_ref, TEST=False)
    authority.state[key] = row
    authority.state.update(GPU_owner_or_worker_active=True, gpu_training_active=False)
    authority.authority.cas(authority.raw, authority.state, "SSDD_FLOAT_DEV187_RESERVE_" + stage)
    reserved = authority.io.stable_read(authority.authority.STATE, stock.STATE_LIMIT)
    need(sources.accepted.entry.document(reserved) == authority.state, "Exact exclusive float stage reservation required")
    failure, uncertain, provenance = None, False, None
    emit(dict(status=RESERVED, stage=stage, owner_birth=coordinator, config_sha256=scope.config_ref["sha256"]))
    try:
        if stage != "SCORE":
            shim = SimpleNamespace(dev=scope.dev, lock={"image_manifest":scope.cfg["image_manifest_ref"]})
            stock.verify_image_root(shim, ctx.modules["ssdd_phase_dev_admission"])
        request_path = area / "STAGE_REQUEST_PRIVATE.json"
        sources.legacy.write_json(request_path, request_for(ctx, scope, stage, area, exports))
        request_ref = sources.legacy.ref(request_path)
        uncertain = True
        terminal, guard, terminal_ref = stage_owner(ctx, area, request_ref, stage, reserved, coordinator)
        uncertain = False
        _, provenance = stage_readback(ctx, stage, area, request_ref, terminal, guard, terminal_ref, coordinator)
        sources.legacy.write_json(area / "CLOSED_STAGE_PRIVATE.json", provenance)
    except (Exception, KeyboardInterrupt) as error:
        failure = type(error).__name__ + ": " + str(error)
        sources.legacy.write_json(area / "ORIGINAL_FAILURE_PRIVATE.json", dict(failure=failure, closure_uncertain=uncertain))
    if not uncertain:
        row.update(status=PENDING if failure is None else FAILED, active_attempt=None, resource_closed=True,
                   failure=failure, provenance=provenance)
        authority.state.update(GPU_owner_or_worker_active=False, gpu_training_active=False)
        authority.authority.cas(reserved, authority.state, "SSDD_FLOAT_DEV187_CLOSE_" + stage)
    if failure is not None:
        emit(dict(status=UNCERTAIN if uncertain else FAILED, stage=stage, failure=failure, resource_closed=not uncertain))
        return 5
    emit(closed_event(ctx.worker, stage, provenance))
    return 0


def existing_or_write(legacy, path, document):
    if path.exists() or path.is_symlink():
        need(path.resolve(strict=True) == path and legacy.P189["read"](path) == document,
             "Any existing independent receipt must match the exact admitted evidence")
    else:
        legacy.write_json(path, document)
    return legacy.ref(path)


def checked_export(ctx, scope, stage, report, record, trained, wait_ref):
    worker, cfg = ctx.worker, scope.cfg
    mode = stage.split("_")[-1].lower()
    output = Path(record["report_ref"]["path"]).parent
    result = report["result"]
    pred, index = result["predictions_ref"], result["image_manifest_ref"]
    need(ctx.stock.ref_value(pred, 64 * ctx.stage.MIB) == output / "ARTIFACTS/PREDICTIONS.json"
         and ctx.stock.ref_value(index, 4 * ctx.stage.MIB) == output / "ARTIFACTS/IMAGE_MANIFEST.json",
         "Exact closed export report and artifact paths required")
    accepted = dict(status=worker.EXPORT_ACCEPTED_STATUS, mode=mode, evaluation_protocol=cfg["evaluation_protocol"],
                    training_pair_sha256=cfg["training_pair_ref"]["sha256"],
                    export_report_sha256=record["report_ref"]["sha256"], request_sha256=record["request_ref"]["sha256"],
                    original_child_terminal_ref=record["terminal_ref"], original_outer_wait_ref=wait_ref)
    path = scope.work / stage / "INDEPENDENT_ACCEPTANCE_PRIVATE.json"
    raw = (json.dumps(accepted, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    expected = dict(path=path.as_posix(), bytes=len(raw), sha256=sha256(raw).hexdigest())
    export = dict(closure=record["closure"], report_ref=record["report_ref"], predictions_ref=pred,
                  export_manifest_ref=index, checkpoint_ref=cfg[mode + "_checkpoint_ref"], acceptance_ref=expected)
    original, scorer = (ctx.modules["ssdd_phase_dev_" + suffix] for suffix in ("admission", "score"))
    worker.load_export(ctx.stage, original, scorer, export, mode, trained[mode],
                       pair_sha=cfg["training_pair_ref"]["sha256"], arm_sha=cfg[mode + "_arm_ref"]["sha256"],
                       evaluation_protocol=cfg["evaluation_protocol"])
    need(existing_or_write(ctx.sources.legacy, path, accepted) == expected, "Exact independent export receipt required")
    worker.verify_export_acceptance(ctx.stage, original, export, mode,
                                   pair_sha=cfg["training_pair_ref"]["sha256"],
                                   evaluation_protocol=cfg["evaluation_protocol"])
    return export


def valid_cell(worker, cell):
    indices = dict(AP=0, AP50=1, AP75=2, AP90=None, APs=3, APm=4, APl=5, AR100=8)
    need(set(cell) == {*MODES, "center_minus_native_points"}, "Both original score arms required")
    for mode in MODES:
        values = cell[mode]
        need(set(values) == {*indices, "all12"} and type(values["all12"]) is list
             and len(values["all12"]) == 12 and all(type(x) in (int,float) and math.isfinite(x) and -1 <= x <= 1
                                                 for x in [*values["all12"], *(values[k] for k in indices)])
             and all(index is None or values[key] == values["all12"][index] for key,index in indices.items()),
             "Exact finite original COCO12/AP90 metrics required")
    need(cell["center_minus_native_points"] == worker.delta(cell["native"], cell["center"]),
         "Original paired score arithmetic mismatch")


def checked_score(ctx, scope, report, record, exports, wait_ref):
    worker, cfg = ctx.worker, scope.cfg
    expected = dict(status=worker.SCORE_STATUS, training_protocol=worker.TRAINING_PROTOCOL,
                    evaluation_protocol=cfg["evaluation_protocol"], protocol_sha256=worker.PROTOCOL_SHA,
                    training_pair_sha256=cfg["training_pair_ref"]["sha256"], image_manifest_sha256=ctx.stock.IMAGE_MANIFEST_SHA,
                    integer_gt_sha256=worker.INTEGER_GT_SHA, floating_gt_sha256=worker.FLOAT_GT_SHA,
                    historical_reference_sha256=worker.REFERENCE_SHA, images=187, annotations=380, max_dets=[1,10,100],
                    seed=5201, primary_GT="OFFICIAL_SOURCE_FLOAT_PRIMARY", sensitivity_GT="LEGACY_INTEGER_SENSITIVITY",
                    FIT=False, DEV=True, TEST=False, AP=True, training=False, model_forward_calls=0)
    need(ctx.stock.matches(report, expected) and set(report["arms"]) == set(MODES),
         "Exact original float-primary score scope required")
    for mode, export in exports.items():
        arm = report["arms"][mode]
        need(ctx.stock.matches(arm, dict(checkpoint_sha256=export["checkpoint_ref"]["sha256"],
                training_arm_sha256=cfg[mode + "_arm_ref"]["sha256"], export_report_sha256=export["report_ref"]["sha256"],
                prediction_sha256=export["predictions_ref"]["sha256"], export_manifest_sha256=export["export_manifest_ref"]["sha256"],
                independent_export_acceptance_sha256=export["acceptance_ref"]["sha256"],
                original_export_terminal_sha256=export["closure"]["terminal_sha256"], image_count=187, native_count_total=18700)),
             "Score must bind both exact original inference outputs and independent acceptance")
    for key in ("primary", "sensitivity"):
        valid_cell(worker, report[key])
    old = worker.historical_native(ctx.stage, cfg["historical_ref"])
    a, b = (report["primary"][mode] for mode in MODES)
    need(report["historical_integer_trained_native_float_score"] == old
         and report["first_seed_gate"] == worker.gate(a, b, old)
         and report["center_minus_historical_native_float_points"] == worker.delta(old, b)
         and report["native_minus_historical_native_float_points"] == worker.delta(old, a),
         "Exact historical float reference and fixed gate arithmetic required")
    safe = dict(report, status=SCORE_ACCEPTED, original_worker_score_status=worker.SCORE_STATUS,
                original_score_report_sha256=record["report_ref"]["sha256"],
                original_score_worker_terminal_sha256=record["terminal_ref"]["sha256"],
                original_score_outer_tool_wait_sha256=wait_ref["sha256"])
    return existing_or_write(ctx.sources.legacy, scope.work / "INDEPENDENT_FLOAT_PRIMARY_SCORE_SAFE.json", safe)


def accept_original(ctx, scope, stage, wait_path, wait_sha):
    stock, sources, worker = ctx.stock, ctx.sources, ctx.worker
    state = prep_authority(ctx)
    key, area = stage_key(scope, stage), scope.work / stage
    row = state.state.get(key, {})
    need(stock.matches(row, dict(status=PENDING, active_attempt=None, resource_closed=True, failure=None, stage=stage,
             config_ref=scope.config_ref, worker_ref=ctx.worker_ref, TEST=False))
         and not state.authority.live(row["owner_birth"]), "Natural original float stage closure required")
    trained = read_training(ctx, scope)
    exports = previous_exports(ctx, scope, stage, state.state)
    recorded = row["provenance"]
    wait_ref = worker.file_ref(wait_path, wait_sha)
    need(not wait_path.is_relative_to(area), "Retain external original tool wait outside the original stage")
    wait = worker.read_json(ctx.stage, wait_ref, stock.LOG_LIMIT)
    session = verify_original_wait(stock, worker, stage, recorded, wait)
    original_request = worker.read_json(ctx.stage, recorded["request_ref"])
    need(Path(recorded["request_ref"]["path"]) == area / "STAGE_REQUEST_PRIVATE.json"
         and original_request == request_for(ctx, scope, stage, area, exports),
         "Exact unchanged original SSDD float request required")
    terminal = worker.read_json(ctx.stage, recorded["terminal_ref"])
    arm = SimpleNamespace(control=area / "CONTROL", output=area / "ACTUAL", audit=area / "UNUSED_AUDIT")
    need(sources.short.closed_arm(sources.legacy, arm, terminal, row["owner_birth"]) == recorded["terminal_ref"],
         "Independent original Windows JobObject readback required")
    guard = worker.read_json(ctx.stage, recorded["guard_ref"])
    report, reread = stage_readback(ctx, stage, area, recorded["request_ref"], terminal,
                                   guard["guard"], recorded["terminal_ref"], row["owner_birth"])
    need(recorded == reread == sources.legacy.P189["read"](area / "CLOSED_STAGE_PRIVATE.json")
         and state.io.stable_read(state.authority.STATE, stock.STATE_LIMIT) == state.raw,
         "Original contained readback and shared authority must remain unchanged")
    if stage == "SCORE":
        accepted_ref = checked_score(ctx, scope, report, reread, exports, wait_ref)
        status, extra = SCORE_ACCEPTED, {}
    else:
        export = checked_export(ctx, scope, stage, report, reread, trained, wait_ref)
        accepted_ref = export["acceptance_ref"]
        status, extra = worker.EXPORT_ACCEPTED_STATUS, {"export_record": export}
    row.update(status=status, original_tool_wait_ref=wait_ref, original_tool_session=session,
               independent_acceptance_ref=accepted_ref, **extra)
    state.authority.cas(state.raw, state.state, "SSDD_FLOAT_DEV187_ACCEPT_" + stage)
    need(sources.accepted.entry.document(state.io.stable_read(state.authority.STATE, stock.STATE_LIMIT)) == state.state,
         "Exact accepted float evaluation authority readback required")
    emit(dict(status=status, stage=stage, independent_acceptance_sha256=accepted_ref["sha256"],
              original_report_sha256=reread["report_ref"]["sha256"]))
    return 0


def main(argv=None):
    cli = ArgumentParser(description=__doc__, allow_abbrev=False)
    action = cli.add_mutually_exclusive_group(required=True)
    for name in ("check", "run", "accept"):
        action.add_argument("--" + name, action="store_true")
    for name in ("stock-owner", "worker", "protocol", "code-root"):
        cli.add_argument("--" + name, type=Path, required=True)
    cli.add_argument("--stage", choices=STAGES)
    cli.add_argument("--config", type=Path)
    cli.add_argument("--config-sha256")
    cli.add_argument("--wait", type=Path)
    cli.add_argument("--wait-sha256")
    args = cli.parse_args(argv)
    need(os.name == "nt" and sys.version_info[:3] == (3, 10, 13) and sys.flags.isolated == sys.flags.utf8_mode == 1
         and sys.dont_write_bytecode, "Original isolated Windows owner runtime required")
    ctx = context(args.stock_owner, args.worker, args.code_root, args.protocol)
    need(Path(sys.executable).resolve(strict=True) == ctx.stock.OWNER_PY.resolve(strict=True),
         "Exact original SSDD Windows outer interpreter required")
    if args.check:
        need(all(x is None for x in (args.stage, args.config, args.config_sha256, args.wait, args.wait_sha256)),
             "Source check does not accept actual inputs")
        emit(dict(status="SSDD_FLOAT_COMMON_DEV187_WINDOWS_OWNER_SOURCE_CHECK_PASSED_V1",
                  worker_sha256=WORKER_SHA, training_protocol=ctx.worker.TRAINING_PROTOCOL, phase_data_opened=False,
                  image_attestation_sha256=IMAGE_ATTESTATION_SHA, host_floor_GiB=16, cuda_export_free_GiB=6,
                  child_RSS_cap_GiB=10, stage_wall_seconds=1200, GPU=False, DEV=False, TEST=False))
        return 0
    need(args.stage is not None and args.config is not None and ctx.stage.hex_digest(args.config_sha256)
         and (args.wait is not None and ctx.stage.hex_digest(args.wait_sha256) if args.accept
              else args.wait is args.wait_sha256 is None),
         "Original stage/config pins and independent accept wait role required")
    scope = read_config(ctx, args.config, args.config_sha256)
    if args.accept:
        return accept_original(ctx, scope, args.stage, args.wait, args.wait_sha256)
    return run_original(ctx, scope, args.stage)


if __name__ == "__main__":
    raise SystemExit(main())
