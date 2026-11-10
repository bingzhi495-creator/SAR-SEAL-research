"""Admit one Windows SSDD FIT-only CPU replay or one GPU short arm."""
import argparse
from hashlib import sha256
from importlib import import_module
from importlib.metadata import version
import json
from math import isfinite
import os
from pathlib import Path
import platform
import re
import stat
import sys


SELF = "ssdd_windows_phase_entry.py"
LOCK_STATUS = "SSDD_WINDOWS_Q100_STAGE_LOCK_V1"
REPLAY_STATUS = "SSDD_WINDOWS_Q100_CPU_REPLAY_COMPLETE_V1"
TERMINAL_STATUS = "SSDD_WINDOWS_Q100_CPU_REPLAY_ORIGINAL_CLOSED_V1"
DGX_PROTOCOL = "5d69d04342e241e32d56c219d99213aed0242e62"
WORKER_MANIFEST_SHA = "046ddaa300e428140952a09931269f6914c789a63fabdc94f98484c3f2efaa91"
GIB, MIB = 1024**3, 1024**2
RESOURCE_PROFILE = {
    "windows_guard_source_sha256": "b5f03331136321715870e1f64b95d670e821ae5a16c90e5480ff1935b9af03c6",
    "windows_short_owner_source_sha256": "e373975394c2602f13afb3f67c974833673312b7068aceec586e6c3cfeec208f",
    "host_available_floor_bytes": 16 * GIB, "worker_peak_rss_cap_bytes": 10 * GIB, "wall_seconds_cap": 300,
    "cuda_immediate_free_floor_bytes": 12 * GIB, "torch_allocator_cap_bytes": 6 * GIB,
}
Q100_SHA = "40524fc63797d2fcab7b50b6ee1f041f9429df73e505cb9e647a5a8cdf6f2e2f"
COCO_SHA = "a44613a4ecd6b5ba61a62002c600b0b6cb7a9da2936a45317ec4b62c635fb99b"
RECIPE_SHA = "ba9508d9acab5ff68b10db8a1aff5d9001868d6dda00e39f299d6f5cc1b8011b"
FIT_ARCHIVE_SHA = "1c8f9b75e79fbc02c983ccd283a70082fcd5ab0d6538c777e8b286dbfdb78100"
FIT_INDEX_SHA = "62966857c78cea1e3b060700c62a7b896942686c2e674d3eb09f516a92250bc1"
FIT_ORDER_SHA = "42b93cbf2424e1c1ffa45c9dc4b32c3537fca6fe3a224e81762fcc339276e4cd"
FIT_ROSTER_SHA = "315f7cfad9a3782865989ced212a6eeea1052e31251a01f24dbe2758bbc4f0e9"
FIT_MANIFEST_SHA = "5bf021523d1ba3507a5ef7ba6a731da8699c37eb32d6b8ea962c52dcfbaafb47"
SCOPE = dict(FIT=True, DEV=False, TEST=False, AP=False)
LAYOUT_READBACK = dict(
    status="SSDD_FIT741_ONLY_LAYOUT_INDEPENDENT_BYTES_VERIFIED_V1", images=741, annotations=1661,
    unique_encoded_images=730, fit_archive_sha256=FIT_ARCHIVE_SHA, fit_index_sha256=FIT_INDEX_SHA,
    native_sorted_id_order_sha256=FIT_ORDER_SHA, original_annotation_order_sha256=FIT_ORDER_SHA,
    content_roster_sha256=FIT_ROSTER_SHA, portable_manifest_sha256=FIT_MANIFEST_SHA,
    model_executed=False, training=False, **SCOPE,
)
CANDIDATES = {
    "ssdd_phase_init.py": "444e91009bce7a56e65a9bd2110c3479a86609d29b5ccdc925b06ec013b7894a",
    "ssdd_fit_only.py": "e421295d4956e09ae42c776e7f7d68f821ab136f1c6ddcca004e97cf0435b646",
    "ssdd_phase_train.py": "6785b1f9ffe23b6b2c1da5939d10d8b5877b93127316f535af5890ca05f4653c",
}
SOURCES = {
    **CANDIDATES,
    "ssdd_phase_dependencies.py": "811bfe310a86223589c876e51a46953a318918a5d31b9e72ee7ac14604948cd7",
    "ssdd_phase_fit_replay.py": "14b097a19c8b951e7d8e246e75e855b2eb574bb1149aa3945b9cf15b91a2cacf",
    "ssdd_fit_stage_readback.py": "4ca757bcfdd630ce8a58fab54d27e14631420f54711db185e5d408f46adc5dc4",
    "ssdd_fit_stage.py": "5623d554d918ae2dc678d52dce96d86aaa407d6fb0b03fb4cf478d07a4676a88",
}
WINDOWS_OVERLAY_SOURCE = (
    "HRSID_NATIVE_FIT_E1_20261109/train_hrsid_native_e1.py",
    "23bf7b432763f86f57322b3c1e09cf7966fa57c80e0ce74e9185b8977a414362",
)
PACKAGES = {"torch": "2.10.0+cu130", "torchvision": "0.25.0+cu130", "rfdetr": "1.12.0",
            "pytorch-lightning": "2.6.6", "numpy": "1.26.4", "Pillow": "12.3.0", "pycocotools": "2.0.11"}
WINDOWS_CODECS = {
    "PIL._imaging": "16a7c881fd3624c8428078ce24460861e91fc5cc43de1b3614dc2531fe327e2e",
    "pycocotools._mask": "bcc52682fb8cd20dcc58b09f009566a07d0cf8bd27354aa7f386b5e18d5e5b20",
}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def exact(value, expected, extras=(), *, reason):
    require(type(value) is dict and set(value) == expected.keys() | set(extras)
            and all(type(value[key]) is type(item) and value[key] == item for key, item in expected.items()), reason)


def hex_id(value, length=64):
    return type(value) is str and re.fullmatch(rf"[a-f0-9]{{{length}}}", value) is not None


def unique(items):
    value = dict(items)
    require(len(value) == len(items), "Duplicate locked JSON field")
    return value


def invalid_constant(token):
    raise ValueError("Nonfinite locked JSON value " + token)


def document(raw):
    value = json.loads(raw, object_pairs_hook=unique, parse_constant=invalid_constant)
    require(type(value) is dict, "Expected a locked JSON object")
    return value


def ref_path(reference, limit, *, allow_empty=False, basename=None):
    exact(reference, {}, ("path", "bytes", "sha256"), reason="Expected closed path, size and digest")
    require(type(reference["path"]) is str and type(reference["bytes"]) is int
            and (0 if allow_empty else 1) <= reference["bytes"] <= limit and hex_id(reference["sha256"]),
            "Expected a bounded closed artifact reference")
    path = Path(reference["path"])
    require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file()
            and reference["path"] in (str(path), path.as_posix())
            and (basename is None or path.name == basename), "Expected canonical closed artifact path")
    return path


def read_ref(reference, limit, *, allow_empty=False, basename=None):
    path = ref_path(reference, limit, allow_empty=allow_empty, basename=basename)
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and before.st_size == reference["bytes"], "Closed file size changed")
        raw = stream.read(reference["bytes"] + 1)
        after = os.fstat(stream.fileno())
    def token(meta):
        return meta.st_dev, meta.st_ino, meta.st_size, meta.st_mtime_ns, meta.st_ctime_ns
    require(len(raw) == reference["bytes"] and sha256(raw).hexdigest() == reference["sha256"]
            and token(before) == token(after) == token(path.stat()), "Exact original artifact bytes required")
    return raw


def replay_identity(lock):
    return dict(
        protocol=lock["protocol"], implementation_commit=lock["implementation_commit"],
        source_manifest_sha256=lock["source_manifest_sha256"], worker_source_sha256=lock["worker"]["sha256"],
        interpreter_sha256=lock["python"]["sha256"],
        fit_readback_sha256=lock["fit_readback"]["sha256"], fit_archive_sha256=FIT_ARCHIVE_SHA,
        fit_index_sha256=FIT_INDEX_SHA, fit_order_sha256=FIT_ORDER_SHA, recipe_sha256=RECIPE_SHA,
        initial_online_sha256=Q100_SHA, runtime="Windows_AMD64_CPython_3.10.13_CPU", packages=PACKAGES,
    )


def verify_stock_replay(stock):
    require(type(stock) is dict, "Expected actual stock replay safe summary")
    fixed = dict(status="SSDD_FIT_NATIVE_CENTER_REPLAY_COMPLETE", images=741, views=741, seed_base=5311,
                 exact_shared_geometry_rng_images=True, exact_common_nonmask_fields=True,
                 forward_calls=0, optimizer_updates=0, **SCOPE)
    require(all(type(stock.get(key)) is type(value) and stock[key] == value for key, value in fixed.items()),
            "Exact 741-view stock paired replay and FIT-only scope required")
    capacity = stock["capacity"]
    require(type(capacity) is dict and set(capacity) == {"limit", "native_max_present", "center_max_present"}
            and type(capacity["limit"]) is int and capacity["limit"] == 100
            and all(type(capacity[key]) is int and 0 < capacity[key] <= 100
                    for key in ("native_max_present", "center_max_present")),
            "The complete stock replay must respect Q100 capacity")
    by_path = stock["by_path"]
    require(type(by_path) is dict and set(by_path) == {"direct", "crop"}
            and all(type(row["views"]) is int and row["views"] > 0 for row in by_path.values())
            and sum(row["views"] for row in by_path.values()) == 741
            and type(stock["membership"]["common_nonempty"]) is int and stock["membership"]["common_nonempty"] > 0,
            "Expected actual direct, crop and nonempty paired replay support")
    changes = [row[size]["masks_changed"] for row in by_path.values() for size in ("small", "other")]
    require(all(type(value) is int and value >= 0 for value in changes) and sum(changes) > 0,
            "Observed real paired mask changes are required")


def verify_replay_receipt(lock):
    """Read the independently closed replay evidence before importing Torch."""
    refs = lock["cpu_replay"]
    exact(refs, {}, ("report", "terminal", "stdout", "stderr"), reason="Exact independently closed CPU replay references")
    report_raw = read_ref(refs["report"], MIB, basename="FIT_REPLAY_SAFE.json")
    report = document(report_raw)
    exact(report, dict(replay_identity(lock), status=REPLAY_STATUS, cuda_initialized=False, torch_threads=1, **SCOPE),
          ("stock_replay",), reason="Replay and short arm must share exact Windows source, layout and protocol")
    verify_stock_replay(report["stock_replay"])
    terminal = document(read_ref(refs["terminal"], MIB, basename="TERMINAL_PRIVATE.json"))
    exact(terminal, dict(status=TERMINAL_STATUS, original_os_wait=True, original_os_exit=0, worker_closed=True,
                         resource_stop=False, stop_reason=None, protocol=lock["protocol"],
                         worker_source_sha256=lock["worker"]["sha256"], result_sha256=refs["report"]["sha256"],
                         stdout_sha256=refs["stdout"]["sha256"], stderr_sha256=refs["stderr"]["sha256"]),
          ("pid", "birth_utc", "elapsed_seconds", "peak_rss_bytes"), reason="Original independently closed CPU replay required")
    require(type(terminal["pid"]) is int and terminal["pid"] > 0
            and type(terminal["birth_utc"]) is str and terminal["birth_utc"].endswith("Z")
            and type(terminal["elapsed_seconds"]) in (int, float) and isfinite(terminal["elapsed_seconds"])
            and 0 < terminal["elapsed_seconds"] <= RESOURCE_PROFILE["wall_seconds_cap"]
            and type(terminal["peak_rss_bytes"]) is int
            and 0 < terminal["peak_rss_bytes"] <= RESOURCE_PROFILE["worker_peak_rss_cap_bytes"],
            "Expected exact replay process identity and resource evidence")
    stdout = read_ref(refs["stdout"], MIB, basename="WORKER.stdout")
    read_ref(refs["stderr"], MIB, allow_empty=True, basename="WORKER.stderr")
    require(stdout.endswith(b"\n") and document(stdout.splitlines()[-1]) == dict(
        status=REPLAY_STATUS, protocol=lock["protocol"], result_sha256=refs["report"]["sha256"]),
        "Original replay stdout must bind the closed report")
    return report


def compact(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")


def canonical_dir(value):
    require(type(value) is str, "Expected canonical Windows directory")
    path = Path(value)
    require(path.is_absolute() and path.resolve(strict=True) == path and path.is_dir(),
            "Expected canonical Windows directory")
    return path


def read_lock(path, expected_sha, operation):
    """Bind exact isolated stage and evidence before package import."""
    lock = document(read_ref(dict(path=str(path), bytes=path.stat().st_size, sha256=expected_sha), MIB))
    exact(lock, dict(status=LOCK_STATUS, operation=operation, resources=RESOURCE_PROFILE,
                     source_manifest_sha256=WORKER_MANIFEST_SHA),
          ("mode", "protocol", "implementation_commit", "code", "python", "overlay", "worker", "fit_archive",
           "fit_layout", "fit_readback", "output", "cpu_replay", "coco"), reason="Exact Windows SSDD lock required")
    require(hex_id(lock["protocol"], 40) and lock["protocol"] != DGX_PROTOCOL
            and hex_id(lock["implementation_commit"], 40), "Distinct prospective Windows protocol required")
    if operation == "replay":
        require(lock["mode"] is lock["cpu_replay"] is lock["coco"] is None, "CPU replay admits no arm or weights")
    else:
        require(operation == "short" and lock["mode"] in ("native", "center"), "Windows GPU short only")
        verify_replay_receipt(lock)
        coco = ref_path(lock["coco"], 134545398)
        require(lock["coco"]["bytes"] == coco.stat().st_size == 134545398 and lock["coco"]["sha256"] == COCO_SHA,
                "Exact official COCO Seg Nano source required")
    require(ref_path(lock["worker"], MIB, basename=SELF) == Path(__file__).resolve(strict=True)
            and ref_path(lock["python"], 4 * MIB) == Path(sys.executable).resolve(strict=True),
            "Lock must name the exact running worker and isolated interpreter")
    for field, limit in (("worker", MIB), ("python", 4 * MIB)):
        read_ref(lock[field], limit)
    layout = canonical_dir(lock["fit_layout"])
    readback = ref_path(lock["fit_readback"], MIB, basename="READBACK_SAFE.json")
    require(layout == readback.parent / "LAYOUT", "FIT layout must match its independent staging readback")
    exact(document(read_ref(lock["fit_readback"], MIB)), LAYOUT_READBACK,
          reason="Exact independent SSDD FIT741 readback required")
    archive = ref_path(lock["fit_archive"], 33079338, basename="SSDD_FIT741_ORIGINAL_INPUTS_20261107_01.zip")
    require(lock["fit_archive"]["bytes"] == archive.stat().st_size == 33079338
            and lock["fit_archive"]["sha256"] == FIT_ARCHIVE_SHA, "Exact original FIT741 archive required")
    canonical_dir(lock["code"])
    canonical_dir(lock["overlay"])
    require(type(lock["output"]) is str, "Expected fresh Windows output path")
    output = Path(lock["output"])
    require(output.is_absolute() and canonical_dir(str(output.parent)) == output.parent
            and not output.exists() and not output.is_symlink()
            and not output.is_relative_to(layout) and not layout.is_relative_to(output),
            "Expected fresh Windows output isolated from FIT")
    return lock


def pinned_source(path, expected, limit=64 * 1024):
    return read_ref(dict(path=str(path), bytes=path.stat().st_size, sha256=expected), limit)


def offline(event, args):
    if event in ("socket.connect", "socket.getaddrinfo"):
        raise RuntimeError("Windows SSDD FIT stage is strictly offline")


def windows_available_bytes():
    import ctypes
    from ctypes import wintypes

    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD),
                    ("total_physical", ctypes.c_uint64), ("available_physical", ctypes.c_uint64),
                    ("total_page_file", ctypes.c_uint64), ("available_page_file", ctypes.c_uint64),
                    ("total_virtual", ctypes.c_uint64), ("available_virtual", ctypes.c_uint64),
                    ("available_extended_virtual", ctypes.c_uint64)]

    query = ctypes.WinDLL("kernel32", use_last_error=True).GlobalMemoryStatusEx
    query.argtypes = [ctypes.POINTER(MemoryStatus)]
    query.restype = wintypes.BOOL
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not query(ctypes.byref(status)):
        raise ctypes.WinError(ctypes.get_last_error())
    return status.available_physical


def admit_host():
    available = windows_available_bytes()
    require(available >= RESOURCE_PROFILE["host_available_floor_bytes"], "Windows host available RAM below 16 GiB")
    return available


def admit_cuda_short(cuda):
    require(cuda.is_available() and cuda.device_count() == 1 and cuda.is_bf16_supported(including_emulation=False),
            "One native BF16 CUDA device required for the short arm")
    free, total = cuda.mem_get_info(0)
    require(total >= free >= RESOURCE_PROFILE["cuda_immediate_free_floor_bytes"],
            "Windows SSDD short immediate CUDA free memory below 12 GiB")
    fraction = min(.35, RESOURCE_PROFILE["torch_allocator_cap_bytes"] / total)
    cuda.set_per_process_memory_fraction(fraction, 0)
    return free, total, fraction


def admit_windows_runtime(lock):
    replay = lock["operation"] == "replay"
    require(platform.system() == "Windows" and platform.machine() == "AMD64"
            and sys.version_info[:3] == (3, 10, 13) and sys.flags.isolated == sys.flags.utf8_mode == 1
            and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES") == ("-1" if replay else "0")
            and os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1",
            "Expected isolated, offline Windows AMD64 Python 3.10.13 and the locked device")
    sys.addaudithook(offline)
    admit_host()
    require({name: version(name) for name in PACKAGES} == PACKAGES, "Exact Windows numeric package versions required")
    code = Path(lock["code"])
    source_dir = code / "SSDD_MASK_PHASE_TRANSFER_20261110"
    for name, digest in SOURCES.items():
        pinned_source(source_dir / name, digest)
    overlay_name, overlay_sha = WINDOWS_OVERLAY_SOURCE
    pinned_source(code / overlay_name, overlay_sha)
    sys.path.insert(0, str(source_dir))
    init = import_module("ssdd_phase_init")
    require(Path(init.__file__).resolve() == source_dir / "ssdd_phase_init.py" and init.CODE == code,
            "Exact ordinary SSDD initialization module required")
    overlay = init.load_api(code / overlay_name, overlay_sha)
    overlay.OVERLAY = Path(lock["overlay"])
    overlay.admit_dependency_overlay()
    for name, digest in WINDOWS_CODECS.items():
        codec = Path(import_module(name).__file__).resolve(strict=True)
        pinned_source(codec, digest, 16 * MIB)
    torch = import_module("torch")
    torch.set_num_threads(1 if replay else 2)
    torch.set_num_interop_threads(1)
    worker, dependencies, manifest = import_module("ssdd_phase_dependencies").load_worker(CANDIDATES)
    require(manifest == lock["source_manifest_sha256"], "Exact stock SSDD training dependencies required")
    return torch, init, worker, dependencies


def read_fit_population(lock):
    layout = Path(lock["fit_layout"])
    archive = Path(lock["fit_archive"]["path"])
    result = import_module("ssdd_fit_stage_readback").readback_fit_layout(archive, layout)
    exact(result, LAYOUT_READBACK, reason="Stock replay requires the exact FIT741 archive and staged JPEG bytes")
    index = document(pinned_source(layout / "train/_annotations.coco.json", FIT_INDEX_SHA, MIB))
    ids = sorted(row["id"] for row in index["images"])
    require(len(ids) == len(set(ids)) == 741 and sha256(compact(ids)).hexdigest() == FIT_ORDER_SHA,
            "Exact sorted stock FIT741 image IDs required")
    return layout, ids


def emit(value):
    print(compact(value).decode("ascii"), flush=True)


def run_locked(lock):
    torch, init, worker, dependencies = admit_windows_runtime(lock)
    layout, ids = read_fit_population(lock)
    output = Path(lock["output"])
    if lock["operation"] == "short":
        admit_host()
        free, total, fraction = admit_cuda_short(torch.cuda)
        emit(dict(event="SSDD_WINDOWS_Q100_GPU_ADMITTED_V1", protocol=lock["protocol"], mode=lock["mode"],
                  cuda_free_bytes=free, cuda_total_bytes=total, cuda_allocator_fraction=fraction))
        worker(stage="qualification", mode=lock["mode"], output=output, fit_layout=layout,
               coco_path=Path(lock["coco"]["path"]), expected_fit_ids=ids, expected_initial_sha256=Q100_SHA,
               protocol=lock["protocol"], recipe_sha256=RECIPE_SHA, source_manifest_sha256=WORKER_MANIFEST_SHA,
               dependencies=dependencies, event_sink=emit)
        return

    require(not torch.cuda.is_initialized(), "CPU replay must leave CUDA uninitialized")
    config = init.load_api(init.CONFIG, init.CONFIG_SHA)
    init.pinned_bytes(init.RECIPE, RECIPE_SHA)
    model, train = config._load_configs(init.RECIPE, layout)
    dependencies.e1.native_recipe(model, train, import_module("rfdetr.config"))
    data = dependencies.fit_only_data(model, train, layout)
    data.setup("fit")
    stock = import_module("ssdd_phase_fit_replay").replay_fit_population(data._dataset_train, ids)
    require(not torch.cuda.is_initialized() and torch.get_num_threads() == 1, "Replay must remain one-thread CPU")
    require(read_fit_population(lock) == (layout, ids), "FIT population changed during replay")
    verify_stock_replay(stock)
    report = dict(replay_identity(lock), status=REPLAY_STATUS, cuda_initialized=False, torch_threads=1,
                  stock_replay=stock, **SCOPE)
    raw = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf8")
    output.mkdir()
    with (output / "FIT_REPLAY_SAFE.json").open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    emit(dict(status=REPLAY_STATUS, protocol=lock["protocol"], result_sha256=sha256(raw).hexdigest()))


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--operation", required=True, choices=("replay", "short"))
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--lock-sha256", required=True)
    args = parser.parse_args()
    run_locked(read_lock(args.lock, args.lock_sha256, args.operation))


if __name__ == "__main__":
    main()
