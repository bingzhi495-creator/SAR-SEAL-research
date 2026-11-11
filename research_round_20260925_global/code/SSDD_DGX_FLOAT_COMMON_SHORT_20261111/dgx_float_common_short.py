"""Run and independently verify two DGX float-polygon SSDD fixed-common FIT short arms."""

import argparse
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from hashlib import sha1, sha256
from io import StringIO
import json
import math
import os
from pathlib import Path
import re
import signal
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


GIB = 1024**3
MIB = 1024**2
MODES = ("native", "center")
SCOPE = dict(FIT=True, DEV=False, TEST=False, AP=False)
FIT_IDS = 741
INTEGER_FIT_SHA = "62966857c78cea1e3b060700c62a7b896942686c2e674d3eb09f516a92250bc1"
FLOAT_FIT_SHA = "97162d23a33b9387fe0b47a117f7e5b1f9dde7471033f244ab336cee1ca9039b"
FLOAT_ADMISSION_SHA = "c5841463f15a3062d8d99899c2034dc8a1c9b97ce05961a20403e61c8776d285"
BASE_LIFECYCLE_SHA = "5ad5d3398666fc64f7463646cde8b26a6f4b603893b07061aa05131c763ee1e7"
PROTOCOL_SHA1 = "0cf70c678b8095c6256cd678e03994408a86e56b"
PROTOCOL_SHA256 = "0ce3681c184cd1c36e777fa74bb1740583a8527e3b28e2ced77294ff374dac8c"
LABEL_BINDING = dict(policy="official_source_float_polygon_only_v1",
                     stock_integer_fit_sha256=INTEGER_FIT_SHA, fit_float_sha256=FLOAT_FIT_SHA,
                     fixed_view_input_admission_sha256=FLOAT_ADMISSION_SHA)
REQUEST_STATUS = "SSDD_DGX_FLOAT_COMMON_SHORT_REQUEST_V1"
AUTHORITY = "SSDD_DGX_FLOAT_COMMON_SHORT_5201"
POLICY = "intersection_of_ordered_nonempty_native_center_target_rows_v1"
WAIT_STATUS = "SSDD_DGX_FLOAT_ORIGINAL_OWNER_WAIT_V1"
PENDING = "SSDD_DGX_FLOAT_COMMON_ORIGINALS_CLOSED_PENDING_INDEPENDENT_ACCEPTANCE"
ARM_STATUS = "SSDD_DGX_FLOAT_COMMON_ARM_CLOSED_PENDING_ACCEPTANCE"
ACCEPTED = "SSDD_DGX_FLOAT_COMMON_SHORT_INDEPENDENTLY_ACCEPTED_V1"
STOCK_PROTOCOL = "5d69d04342e241e32d56c219d99213aed0242e62"
STOCK_MANIFEST = "6218d274eaff924ae3ea05903b19517a31a3717efae6f7bb6a006ffe3b516221"
STOCK_OWNER_SHA = "9ff57344d680a700958227b837d50f771114f5d4983f62c05d607086eee06893"
GUARD_SHA = "b2bbba923870efb4dd31db6fe24bd47db1e29118e0c01f5d03640dcb798297c0"
COMMON_WORKER_SHA = "8aa1785f6727656d564177c6850120ce65d89f1ed67132e5a72ce331b97e564b"
QUALIFIER_SHA = "8b76b42d59fff857dfb0c0fc9b67664817c72a494966f5f0ec808156776cdaec"
P330_REPORT_SHA = "bd6b77baac3900e7b6536adb9fc08dc409bc37f93a2b50209c65776e757acfdb"
P330_SOURCE_SHA = "5c1aa7441375692c0ca724fdab7b54e74659af481e42d82b268deb1dd50ceb51"
LEASES = ("P250_HRSID_TO_SSDD_FROZEN_TRANSFER_GPU_OWNER.lock", "SSDD_PHASE_DGX_GPU_OWNER_20261110.lock")
AUDIT_NAME = "CONSUMED_COMMON_PRIVATE.jsonl"
AUDIT_REPORT = "COMMON_SHORT_SAFE.json"
SUMMARY = "ORIGINAL_OWNER_SUMMARY_PRIVATE.json"
STATS = ("fetched_visits", "consumed_visits", "prefetched_unconsumed", "unconsumed_prefetch_sha256",
         "common_targets_total", "common_empty_visits", "changed_mask_visits")


def need(ok, why):
    if not ok:
        raise ValueError(why)


def digest(data):
    return sha256(data).hexdigest()


def encode(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")


def unique(pairs):
    result = dict(pairs)
    need(len(result) == len(pairs), "Duplicate document keys")
    return result


def reject_constant(_):
    raise ValueError("Nonfinite document value")


def decode(raw):
    return json.loads(raw, object_pairs_hook=unique, parse_constant=reject_constant)


def exact(value, expected, extra=()):
    return (type(value) is dict and set(value) == set(expected) | set(extra)
            and all(type(value[k]) is type(v) and value[k] == v for k, v in expected.items()))


def is_hash(value, length=64):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{%d}" % length, value) is not None


def stable(path, limit, *, empty=False):
    need(path.is_absolute() and path.resolve(strict=True) == path and path.is_file(), "Canonical regular input required")
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        raw = stream.read(limit+1)
        after = os.fstat(stream.fileno())
    def sig(stat):
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns
    need((0 if empty else 1) <= len(raw) == before.st_size <= limit
         and sig(before) == sig(after) == sig(path.stat()), "Unstable or out-of-scope closed input")
    return raw


def read_ref(value, expected):
    need(type(value) is dict and set(value) == {"path", "bytes", "sha256"}
         and type(value["path"]) is str and type(value["bytes"]) is int and 0 < value["bytes"] <= MIB
         and is_hash(expected) and value["sha256"] == expected, "Exact request source reference required")
    path = Path(value["path"])
    raw = stable(path, MIB)
    need(value["path"] == path.as_posix() and len(raw) == value["bytes"] and digest(raw) == expected,
         "Exact request source bytes required")
    return path, raw


def module(path, raw, name):
    result = ModuleType(name)
    result.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), result.__dict__)
    return result


def write(path, raw):
    with path.open("xb") as stream:
        need(stream.write(raw) == len(raw), "Incomplete closed artifact")
        stream.flush()
        os.fsync(stream.fileno())


def paired_annotations(integer, floating, fit_ids, *, rows=1661):
    """Return ordered annotation pairs for the exact polygon-only FIT overlay."""
    need(type(integer) is dict and type(floating) is dict, "Expected two FIT COCO documents")
    baseline = {k: v for k, v in integer.items() if k != "annotations"}
    alternate = {k: v for k, v in floating.items() if k != "annotations"}
    need(encode(baseline) == encode(alternate), "Only FIT polygon labels may change")
    ids = [row["id"] for row in baseline["images"]]
    need(all(type(i) is int for i in ids) and sorted(ids) == list(fit_ids)
         and len(set(ids)) == len(ids), "Exact FIT image roster required")
    source, overlay = integer["annotations"], floating["annotations"]
    need(type(source) is list and type(overlay) is list and len(source) == len(overlay) == rows,
         "Exact FIT annotation counts required")
    grouped = {i: [] for i in ids}
    seen = set()
    for a, b in zip(source, overlay, strict=True):
        need(type(a) is dict and type(b) is dict
             and encode({k: v for k, v in a.items() if k != "segmentation"})
                 == encode({k: v for k, v in b.items() if k != "segmentation"}),
             "Only FIT segmentation may change; ordered annotation metadata must match")
        image_id, annotation_id = a["image_id"], a["id"]
        need(type(image_id) is int and image_id in grouped and type(annotation_id) is int
             and annotation_id not in seen and type(a["category_id"]) is int and a["category_id"] == 1,
             "Exact unique ship annotation identities required")
        raw, restored = a["segmentation"], b["segmentation"]
        need(type(raw) is list and type(restored) is list and len(raw) == len(restored) > 0,
             "Exact ordered FIT polygon rings required")
        for old, new in zip(raw, restored, strict=True):
            need(type(old) is list and type(new) is list and len(old) == len(new) >= 6 and len(old) % 2 == 0
                 and all(type(x) in (int, float) and math.isfinite(x) and int(x) == x for x in old)
                 and all(type(x) in (int, float) and math.isfinite(x) for x in new)
                 and [int(x) for x in new] == old, "Exact per-vertex integer truncation required")
        seen.add(annotation_id)
        grouped[image_id].append((a, b))
    return grouped


@contextmanager
def float_dataset_overlay(module_data, detection_type, fit_root, fit_ids, grouped):
    """Overlay polygon records before the stock dataset scales or rasterizes them."""
    original_build = module_data.build_dataset
    dataset = None

    def build(split, *args, **kwargs):
        nonlocal dataset
        need(split == "train" and dataset is None, "Only one original FIT dataset may be overlaid")
        candidate = original_build(split, *args, **kwargs)
        need(type(candidate) is detection_type and Path(candidate.root).resolve(strict=True) == fit_root
             and list(candidate.ids) == list(fit_ids) and "_load_target" not in vars(candidate)
             and callable(candidate._load_target), "Exact unmodified stock FIT dataset required")
        load_original = candidate._load_target

        def load_target(image_id):
            need(type(image_id) is int and image_id in grouped, "Only admitted FIT image targets may be loaded")
            original = load_original(image_id)
            expected = grouped[image_id]
            need(type(original) is list and len(original) == len(expected), "Stock FIT annotation count differs")
            records = []
            for row, (integer, floating) in zip(original, expected, strict=True):
                need(encode(row) == encode(integer), "Stock FIT annotation bytes or order differ")
                records.append({**row, "segmentation": deepcopy(floating["segmentation"])})
            return records

        dataset = candidate
        dataset._load_target = load_target
        return dataset

    try:
        with patch.object(module_data, "build_dataset", build):
            yield
        need(dataset is not None, "Expected exactly one original FIT dataset")
    finally:
        if dataset is not None:
            del dataset._load_target


def get_request(path, expected):
    raw = stable(path, 64*1024)
    need(is_hash(expected) and digest(raw) == expected, "Exact prospective request SHA required")
    cfg = decode(raw)
    need(exact(cfg, {"status": REQUEST_STATUS}, ("training_protocol", "implementation_commit", "python", "code_root",
            "work_root", "lease_directory", "worker", "stock_owner", "stock_binding_sha256", "linux_guard",
            "common_worker", "p330_report", "float_overlay", "p396_report", "protocol_document")),
         "Exact new DGX float fixed-common short request required")
    train, impl = cfg["training_protocol"], cfg["implementation_commit"]
    need(train == PROTOCOL_SHA1 and is_hash(impl, 40) and len({train, impl, STOCK_PROTOCOL,
         "22e4976a4536c908921b4756d928c5ba9b2f1ffd"}) == 4 and is_hash(cfg["stock_binding_sha256"]),
         "Separate new protocol and implementation required")
    need(type(cfg["python"]) is str and Path(cfg["python"]).is_absolute()
         and cfg["python"] == sys.executable, "Exact original isolated DGX interpreter required")
    for key in ("code_root", "lease_directory", "work_root"):
        need(type(cfg[key]) is str and Path(cfg[key]).is_absolute(), "Canonical DGX workspace paths required")
    root, leases, code = (Path(cfg[k]) for k in ("work_root", "lease_directory", "code_root"))
    need(root.parent.resolve(strict=True) == root.parent == leases and not root.is_symlink()
         and code.resolve(strict=True) == code, "Fresh separate work root under the canonical lease directory required")
    self_path, _ = read_ref(cfg["worker"], cfg["worker"]["sha256"])
    need(self_path == Path(__file__).resolve(strict=True), "Exact executing DGX float owner source required")
    return cfg, root, leases, code


def dependencies(cfg, code):
    stock = module(*read_ref(cfg["stock_owner"], STOCK_OWNER_SHA), "_ssdd_float_p242_stock")
    guard = module(*read_ref(cfg["linux_guard"], GUARD_SHA), "_ssdd_float_p298_guard")
    common = module(*read_ref(cfg["common_worker"], COMMON_WORKER_SHA), "_ssdd_float_p333_common")
    _, protocol = read_ref(cfg["protocol_document"], PROTOCOL_SHA256)
    need(sha1(protocol).hexdigest() == cfg["training_protocol"], "Exact frozen float protocol SHA-1 required")
    overlay_path, _ = read_ref(cfg["float_overlay"], FLOAT_FIT_SHA)
    _, admission_raw = read_ref(cfg["p396_report"], FLOAT_ADMISSION_SHA)
    admission = decode(admission_raw)
    need(overlay_path != Path(cfg["work_root"]) and Path(cfg["work_root"]) not in overlay_path.parents
         and admission["status"] == "P396_SSDD_RFDETR_FLOAT_SAME_VIEW_ADMISSION_COMPLETE"
         and admission["overlay_sha256"] == FLOAT_FIT_SHA and admission["images"] == 32
         and admission["counts"]["equal_nonmask_rng_and_rows_views"] == 32
         and admission["model_forward_calls"] == admission["optimizer_updates"] == 0
         and admission["actual_float_training_started"] is False
         and all(admission[key] is value for key, value in SCOPE.items()),
         "Exact separate FIT-only float input admission required")
    _, preflight = read_ref(cfg["p330_report"], P330_REPORT_SHA)
    proof = decode(preflight)
    p330_path = code / common.P330_SOURCE[0]
    p330_raw = stable(p330_path, 64*1024)
    need(digest(p330_raw) == P330_SOURCE_SHA == common.P330_SOURCE[1]
         and proof["status"] == "P330_SSDD_FIXED_COMMON_CPU_PREFLIGHT_COMPLETE"
         and proof["ordered_common_rows_nonmask_image_rng_trace_exact"] is True
         and proof["model_forward_calls"] == proof["optimizer_updates"] == 0
         and common.POLICY == POLICY and stock.MODES == common.MODES == MODES
         and tuple(guard.LEASE_NAMES) == LEASES and (stock.HOST_FLOOR, stock.RSS_CAP, stock.WALL_CAP, stock.GPU_CAP)
         == (40*GIB, 8*GIB, 300, 6*GIB), "Exact independently verified helper and resource provenance required")
    p330 = module(p330_path, p330_raw, "_ssdd_float_p330_validation")
    return SimpleNamespace(stock=stock, guard=guard, common=common, p330=p330)


def check_runtime(cfg, device):
    import platform
    need(platform.system() == "Linux" and platform.machine() == "aarch64" and sys.version_info[:3] == (3, 12, 14)
         and sys.flags.isolated == sys.flags.utf8_mode == 1 and sys.dont_write_bytecode
         and sys.executable == cfg["python"]
         and os.environ.get("CUDA_VISIBLE_DEVICES") == device
         and os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1",
         "Exact isolated DGX runtime, device and offline environment required")


def stock_binding(ctx, cfg):
    frozen = ctx.stock.verify_sources(cfg["stock_binding_sha256"])
    need(frozen["training_protocol"] == STOCK_PROTOCOL
         and frozen["effective_source_manifest_sha256"] == STOCK_MANIFEST, "Exact prior DGX runtime and FIT closure")
    return frozen


def float_inputs(cfg, layout, ids):
    _, floating = read_ref(cfg["float_overlay"], FLOAT_FIT_SHA)
    integer = stable(layout / "train/_annotations.coco.json", MIB)
    need(len(ids) == FIT_IDS and digest(integer) == INTEGER_FIT_SHA,
         "Exact original integer FIT source is required and remains immutable")
    return paired_annotations(decode(integer), decode(floating), ids)


def manifest(cfg):
    return digest(encode(dict(authority=AUTHORITY, policy=POLICY, source=cfg["worker"]["sha256"],
                 stock_manifest_sha256=STOCK_MANIFEST, stock_binding_sha256=cfg["stock_binding_sha256"],
                 p298_guard_sha256=GUARD_SHA, p333_binding_sha256=COMMON_WORKER_SHA, p330_source_sha256=P330_SOURCE_SHA,
                 p330_report_sha256=P330_REPORT_SHA, qualifier_sha256=QUALIFIER_SHA,
                 base_lifecycle_sha256=BASE_LIFECYCLE_SHA, protocol_sha256=PROTOCOL_SHA256,
                 label_binding=LABEL_BINDING)))


def arm_start(mode, request_sha):
    return dict(status="SSDD_DGX_FLOAT_COMMON_ORIGINAL_ARM_START", mode=mode, request_sha256=request_sha)


def pending_event(request_sha, summary_sha):
    return dict(status=PENDING, request_sha256=request_sha, summary_sha256=summary_sha)


def output(value):
    print(encode(value).decode("ascii"), end="", flush=True)


def arm_fixed(cfg, mode, request_sha, common_raw, visit_raw, report_raw):
    return dict(status=ARM_STATUS, authority=AUTHORITY, policy=POLICY, mode=mode,
        training_protocol=cfg["training_protocol"], implementation_commit=cfg["implementation_commit"],
        source_manifest_sha256=manifest(cfg), request_sha256=request_sha, label_binding=LABEL_BINDING,
        common_ledger_sha256=digest(common_raw),
        stock_visits_sha256=digest(visit_raw), stock_report_sha256=digest(report_raw), visits=32, optimizer_updates=8,
        ema_updates=8, original_empty_filter_removals=0, checkpoint_saved=False, **SCOPE)


def stop_on_parent_death(parent_pid):
    """Keep an isolated child from outliving its Linux owner on SSH loss."""
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.prctl(1, ctypes.c_ulong(signal.SIGKILL), ctypes.c_ulong(0), ctypes.c_ulong(0), ctypes.c_ulong(0))
    if result != 0:
        raise OSError(ctypes.get_errno(), "Setting Linux parent-death signal failed")
    need(os.getppid() == parent_pid, "Original DGX owner exited during child attachment")


def run_arm(ctx, cfg, root, code, mode, request_sha):
    check_runtime(cfg, "0")
    stage = root / mode
    owner = decode(stable(root / "OWNER_BIRTH_PRIVATE.json", 4096))
    need(mode in MODES and Path.cwd() == stage and os.getppid() == owner["pid"] and ctx.stock.alive(owner)
         and os.sched_getaffinity(0) == {0, 1}, "Exact live original DGX owner required before FIT")
    stop_on_parent_death(owner["pid"])
    if mode == MODES[1]:
        terminal, _ = closed_terminal(ctx, root, MODES[0])
        need(terminal["original_os_exit"] == 0, "Native original must close before center arm")
    lock = stock_binding(ctx, cfg)
    train, deps = ctx.stock.entry(lock)
    ids = ctx.stock.fit_ids(lock)
    layout = Path(lock["layout"])
    grouped = float_inputs(cfg, layout, ids)
    torch, free = ctx.stock.runtime(lock, "0")
    from rfdetr.datasets import coco
    from rfdetr.training import module_data
    actual, audit = stage / "ACTUAL", stage / "COMMON_AUDIT"
    need(not actual.exists() and not audit.exists(), "Fresh actual arm and audit roots required")
    output(dict(event="SSDD_DGX_FLOAT_COMMON_GPU_ADMITTED", mode=mode, cuda_free_bytes=free))
    stream = StringIO()
    helpers = {key: ctx.p330.pinned(code, key) for key in ("trace_resize", "common_mask_supervision")}
    with patch.dict(sys.modules, helpers):
        paired = ctx.p330.pinned(code, "paired_fit_transform")
        binding = ctx.p330.pinned(code, "common_fit_binding")
        with patch.dict(sys.modules, {"paired_fit_transform": paired}):
            ledger = binding.ConsumedFitLedger(stream)
            wrapped, chosen = ctx.common.bind_dependencies(deps, torch, ctx.p330, paired, binding, layout, mode, ledger)
            with float_dataset_overlay(module_data, coco.CocoDetection, layout / "train", ids, grouped):
                report = train(stage="qualification", mode=mode, output=actual, fit_layout=layout,
                    coco_path=Path(lock["coco"]), expected_fit_ids=ids, expected_initial_sha256=ctx.stock.INITIAL_SHA,
                    protocol=cfg["training_protocol"], recipe_sha256=ctx.stock.RECIPE_SHA,
                    source_manifest_sha256=manifest(cfg), dependencies=wrapped, event_sink=output)
    need(grouped == float_inputs(cfg, layout, ids), "Original FIT sources changed across the actual arm")
    stats = ledger.finish(32)
    need(stats["changed_mask_visits"] > 0 and stats["common_targets_total"] > 0
         and report["visits"] == 32 and report["optimizer_updates"] == report["ema_updates"] == 8
         and report["removed_empty_rows"] == 0 and report["checkpoint_saved"] is False,
         "Exact original actual 32/8/8 FIT-only short closure required")
    common_raw = stream.getvalue().encode("utf-8")
    visit_raw = stable(actual / "QUALIFICATION_VISITS_PRIVATE.json", 256*1024)
    report_raw = stable(actual / "SSDD_PHASE_TRAIN_SAFE.json", 64*1024)
    need(ctx.common.verify_consumption([decode(line) for line in common_raw.splitlines()], decode(visit_raw), mode)
         == chosen.hexdigest(), "Actual selected mask stream differs from common and stock visit evidence")
    arm = dict(arm_fixed(cfg, mode, request_sha, common_raw, visit_raw, report_raw),
               selected_mask_stream_sha256=chosen.hexdigest(), **stats)
    audit.mkdir(mode=0o700)
    write(audit / AUDIT_NAME, common_raw)
    write(audit / AUDIT_REPORT, encode(arm))
    output(dict(status=ARM_STATUS, report_sha256=digest(encode(arm)), common_ledger_sha256=digest(common_raw)))


def read_birth(value):
    need(type(value) is dict and set(value) == {"pid", "created", "boot_id"}
         and type(value["pid"]) is int and value["pid"] > 0 and type(value["created"]) is float
         and math.isfinite(value["created"]) and value["created"] > 0
         and type(value["boot_id"]) is str and bool(value["boot_id"]), "Exact original Linux process birth required")
    return value


def closed_terminal(ctx, root, mode):
    stage = root / mode
    raw = stable(stage / "TERMINAL_PRIVATE.json", 16*1024)
    terminal = decode(raw)
    fixed = dict(original_os_wait=True, original_os_exit=0, worker_closed=True, resource_stop=False, stop_reason=None)
    need(exact(terminal, fixed, ("worker", "peak_rss_bytes", "seconds", "stdout_sha256", "stderr_sha256")),
         "Exact original clean GPU worker wait required")
    birth = read_birth(decode(stable(stage / "WORKER_BIRTH_PRIVATE.json", 4096)))
    need(birth == terminal["worker"] and not ctx.stock.alive(birth) and not ctx.guard.group_live(birth["pid"])
         and type(terminal["peak_rss_bytes"]) is int and 0 < terminal["peak_rss_bytes"] <= 8*GIB
         and type(terminal["seconds"]) in (int, float) and math.isfinite(terminal["seconds"]) and terminal["seconds"] > 0,
         "Exact original worker birth, closure and resource scope required")
    return terminal, digest(raw)


def run_owner(ctx, cfg, root, leases, request, request_sha):
    import fcntl
    import psutil

    check_runtime(cfg, "0")
    need(not root.exists() and psutil.virtual_memory().available >= 40*GIB,
         "Fresh DGX scope and at least 40 GiB host memory required")
    os.sched_setaffinity(0, {0, 1})

    def interrupted(signum, _frame):
        raise KeyboardInterrupt(signal.Signals(signum).name)

    for stop_signal in (signal.SIGHUP, signal.SIGTERM):
        signal.signal(stop_signal, interrupted)
    with ExitStack() as stack:
        for name in LEASES:
            path = leases / name
            need(path.resolve(strict=True) == path and path.is_file(), "Existing canonical project GPU lease required")
            lock = stack.enter_context(path.open("r+b"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        root.mkdir(mode=0o700)
        owner = ctx.guard.birth(psutil.Process())
        write(root / "OWNER_BIRTH_PRIVATE.json", encode(owner))
        terminals, failure = {}, None
        try:
            for mode in MODES:
                stage = root / mode
                stage.mkdir(mode=0o700)
                scratch = stage / "SCRATCH"
                scratch.mkdir(mode=0o700)
                env = dict(os.environ, HF_HOME=str(scratch), TORCH_HOME=str(scratch), TMPDIR=str(scratch),
                           OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
                args = [sys.executable, "-I", "-B", "-X", "utf8", str(Path(__file__).resolve(strict=True)),
                        "--request", str(request), "--request-sha256", request_sha, "--worker", mode]
                output(arm_start(mode, request_sha))
                ctx.guard.supervise_child(stage, args, env, host_floor=40*GIB, rss_cap=8*GIB, seconds=300)
                terminal, terminal_sha = closed_terminal(ctx, root, mode)
                terminals[mode] = terminal_sha
                need(terminal["worker"]["pid"] != owner["pid"] and terminal["worker"]["created"] >= owner["created"],
                     "Actual worker must follow original owner birth")
        except (Exception, KeyboardInterrupt) as exc:
            failure = type(exc).__name__ + ": " + str(exc)
        summary = dict(status=PENDING if failure is None else "SSDD_DGX_FLOAT_COMMON_ORIGINALS_FAILED",
                       authority=AUTHORITY, owner=owner, request_sha256=request_sha, source_manifest_sha256=manifest(cfg),
                       terminal_sha256s=terminals, failure=failure, **SCOPE)
        raw = encode(summary)
        write(root / SUMMARY, raw)
        if failure is not None:
            output(dict(status=summary["status"], failure=failure, summary_sha256=digest(raw)))
            return 5
        output(pending_event(request_sha, digest(raw)))
    return 0


def verify_external_wait(observed, owner, request_sha, summary_sha, is_live):
    fixed = dict(status=WAIT_STATUS, owner=read_birth(owner), request_sha256=request_sha,
                 original_os_wait=True, original_os_exit=0, process_group_closed=True)
    need(exact(observed, fixed, ("combined_stdout",))
         and type(observed["combined_stdout"]) is str and 0 < len(observed["combined_stdout"]) <= 16*1024,
         "Exact original owner wait captured by the external transport required")
    need(is_live(owner) is False, "Original DGX owner still live")
    actual = [decode(line) for line in observed["combined_stdout"].splitlines() if line.strip()]
    expected = [arm_start(mode, request_sha) for mode in MODES] + [pending_event(request_sha, summary_sha)]
    need(actual == expected, "Exact ordered original DGX owner stdout required")


def closed_arm(ctx, qualifier, cfg, root, mode, request_sha, expected_terminal):
    terminal, terminal_sha = closed_terminal(ctx, root, mode)
    need(terminal_sha == expected_terminal, "Original terminal differs from coordinator closure")
    stage = root / mode
    stdout = stable(stage / "WORKER.stdout", 2*MIB)
    stderr = stable(stage / "WORKER.stderr", 2*MIB, empty=True)
    need(digest(stdout) == terminal["stdout_sha256"] and digest(stderr) == terminal["stderr_sha256"],
         "Exact original worker stdout/stderr required")
    actual, audit = stage / "ACTUAL", stage / "COMMON_AUDIT"
    files = sorted(item.name for item in actual.iterdir())
    need(files == sorted((qualifier.REPORT, qualifier.VISITS))
         and {item.name for item in audit.iterdir()} == {AUDIT_NAME, AUDIT_REPORT}, "Exact original short and audit roster")
    report = stable(actual / qualifier.REPORT, 64*1024)
    visits = stable(actual / qualifier.VISITS, 256*1024)
    common = stable(audit / AUDIT_NAME, 256*1024)
    safe_raw = stable(audit / AUDIT_REPORT, 64*1024)
    need(0 < decode(report)["peak_allocated_bytes"] <= 6*GIB, "Observed CUDA allocation must respect the original cap")
    stock = qualifier.QualificationArm(dict(
        **{k: terminal[k] for k in ("original_os_wait", "original_os_exit", "resource_stop", "worker_closed", "stdout_sha256")},
        terminal_sha256=terminal_sha, output_files=files), report, visits, stdout)
    safe = decode(safe_raw)
    need(exact(safe, arm_fixed(cfg, mode, request_sha, common, visits, report),
               (*STATS, "selected_mask_stream_sha256")), "Exact original common audit, receipt lineage and clocks required")
    ledger = [decode(line) for line in common.splitlines()]
    selected = ctx.common.verify_consumption(ledger, decode(visits), mode)
    need(selected == safe["selected_mask_stream_sha256"] and safe["consumed_visits"] == 32
         and safe["prefetched_unconsumed"] in (0, 1) and safe["fetched_visits"] == 32 + safe["prefetched_unconsumed"]
         and (safe["unconsumed_prefetch_sha256"] is None) == (safe["prefetched_unconsumed"] == 0)
         and safe["common_targets_total"] == sum(row["common_nonempty"] for row in ledger)
         and safe["common_empty_visits"] == sum(row["common_nonempty"] == 0 for row in ledger)
         and safe["changed_mask_visits"] == sum(row["common_changed_masks"] > 0 for row in ledger) > 0,
         "Actual selected mask stream and common count readback must agree")
    need(decode(stdout.splitlines()[-1]) == dict(status=ARM_STATUS, report_sha256=digest(safe_raw),
                                                common_ledger_sha256=digest(common)),
         "Original worker stdout must bind the closed common audit")
    return stock, (common, selected, safe), terminal["worker"], digest(safe_raw)


def verify_common_pair(a, b, changes):
    need(a[0] == b[0] and a[1] != b[1]
         and a[2]["changed_mask_visits"] == b[2]["changed_mask_visits"] > 0,
         "Ordered shared common ledger and distinct consumed mask streams required")
    need(all(changes[key] == 0 for key in ("differing_prefilter_count_visits", "differing_prefilter_labels_visits",
              "differing_prefilter_box_visits", "differing_postfilter_count_visits", "differing_postfilter_box_visits"))
         and changes["differing_postfilter_mask_visits"] == a[2]["changed_mask_visits"],
         "Exact common nonmask fields and stock mask-change count required")


def verify_closed(ctx, cfg, root, request_sha, wait_path, wait_sha, destination):
    check_runtime(cfg, "-1")
    owner = decode(stable(root / "OWNER_BIRTH_PRIVATE.json", 4096))
    need(ctx.stock.alive(read_birth(owner)) is False, "Original DGX owner must close before scientific readback")
    summary_raw = stable(root / SUMMARY, 16*1024)
    summary = decode(summary_raw)
    fixed = dict(status=PENDING, authority=AUTHORITY, owner=owner, request_sha256=request_sha,
                 source_manifest_sha256=manifest(cfg), failure=None, **SCOPE)
    need(exact(summary, fixed, ("terminal_sha256s",)) and set(summary["terminal_sha256s"]) == set(MODES)
         and all(is_hash(v) for v in summary["terminal_sha256s"].values()), "Exact two-arm original closure summary")
    external_wait = stable(wait_path, 64*1024)
    need(is_hash(wait_sha) and digest(external_wait) == wait_sha, "Exact external original owner wait digest required")
    verify_external_wait(decode(external_wait), owner, request_sha, digest(summary_raw), ctx.stock.alive)
    lock = stock_binding(ctx, cfg)
    qualifier_path = ctx.stock.CODE / "ssdd_phase_qualification.py"
    qualifier_raw = stable(qualifier_path, 64*1024)
    need(digest(qualifier_raw) == QUALIFIER_SHA, "Exact stock paired short independent validator")
    qualifier = module(qualifier_path, qualifier_raw, "_ssdd_float_exact_stock_pair_qualifier")
    sys.path.insert(0, str(ctx.stock.CODE))
    fit_ids = ctx.stock.fit_ids(lock)
    float_inputs(cfg, Path(lock["layout"]), fit_ids)
    a, b = (closed_arm(ctx, qualifier, cfg, root, mode, request_sha, summary["terminal_sha256s"][mode]) for mode in MODES)
    need(owner["boot_id"] == a[2]["boot_id"] == b[2]["boot_id"]
         and len({owner["pid"], a[2]["pid"], b[2]["pid"]}) == 3
         and owner["created"] <= a[2]["created"] < b[2]["created"], "Exact ordered distinct owner and worker births")
    pair = qualifier.verify_pair(a[0], b[0], protocol=cfg["training_protocol"],
                                source_manifest_sha256=manifest(cfg), expected_fit_ids=fit_ids)
    verify_common_pair(a[1], b[1], pair["changes"])
    result = dict(status=ACCEPTED, authority=AUTHORITY, policy=POLICY,
                  request_sha256=request_sha, original_owner_wait_sha256=wait_sha, original_owner_summary_sha256=digest(summary_raw),
                  source_manifest_sha256=manifest(cfg), training_protocol=cfg["training_protocol"],
                  implementation_commit=cfg["implementation_commit"], training_seed=5201, initialization_seed=5201,
                  label_binding=LABEL_BINDING,
                  common_ledger_sha256=digest(a[1][0]), common_targets_total=a[1][2]["common_targets_total"],
                  native_selected_mask_stream_sha256=a[1][1], center_selected_mask_stream_sha256=b[1][1],
                  native_common_report_sha256=a[3], center_common_report_sha256=b[3],
                  stock_training_qualification=pair, **SCOPE)
    raw = encode(result)
    write(destination, raw)
    output(dict(status=ACCEPTED, result_sha256=digest(raw)))


def main():
    cli = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    cli.add_argument("--request", required=True, type=Path)
    cli.add_argument("--request-sha256", required=True)
    action = cli.add_mutually_exclusive_group(required=True)
    action.add_argument("--check-source", action="store_true")
    action.add_argument("--run", action="store_true")
    action.add_argument("--worker", choices=MODES)
    action.add_argument("--verify-closed", action="store_true")
    cli.add_argument("--original-wait", type=Path)
    cli.add_argument("--original-wait-sha256")
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    need(all((v is not None) == args.verify_closed for v in (args.original_wait, args.original_wait_sha256, args.output)),
         "Original external wait is only for independent closed verification")
    cfg, root, leases, code = get_request(args.request, args.request_sha256)
    ctx = dependencies(cfg, code)
    if args.check_source:
        output(dict(status="SSDD_DGX_FLOAT_COMMON_SOURCE_PREFLIGHT_PASSED", authority=AUTHORITY,
                    source_manifest_sha256=manifest(cfg), label_binding=LABEL_BINDING))
    elif args.run:
        return run_owner(ctx, cfg, root, leases, args.request, args.request_sha256)
    elif args.worker:
        run_arm(ctx, cfg, root, code, args.worker, args.request_sha256)
    else:
        destination = args.output
        need(destination.is_absolute() and destination.parent.resolve(strict=True) == destination.parent
             and not destination.exists() and not destination.is_symlink()
             and destination != root and root not in destination.parents, "Fresh separate verification output required")
        verify_closed(ctx, cfg, root, args.request_sha256, args.original_wait, args.original_wait_sha256, destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
