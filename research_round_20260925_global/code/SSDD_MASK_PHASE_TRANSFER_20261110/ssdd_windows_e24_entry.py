"""Admit one Windows SSDD E24 FIT arm from an independently closed short pair."""
import argparse
from hashlib import sha256
from importlib import import_module
from pathlib import Path
import re
import sys
from types import ModuleType


SELF = "ssdd_windows_e24_entry.py"
WINDOWS_ENTRY = "ssdd_windows_phase_entry.py"
ENTRY_SHA = "e29edeb0a3aff7c871123deee7c0c198ef5913fd33e7c016434a671d186796d3"
QUALIFIER_SHA = "8b76b42d59fff857dfb0c0fc9b67664817c72a494966f5f0ec808156776cdaec"
ACCEPTANCE_SHA = "5a22d2dc7a9dd67427fcedd3c6b45278c8abe8c9023ced33357206bc2d275604"
STATUS = "SSDD_WINDOWS_Q100_E24_STAGE_LOCK_V1"
MODES = ("native", "center")
LONG_WALL_SECONDS = 36000


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def load_exact(path, expected):
    require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file(),
            "Exact canonical original Windows SSDD admission entry required")
    with path.open("rb") as stream:
        raw = stream.read(65537)
    require(0 < len(raw) <= 65536 and sha256(raw).hexdigest() == expected,
            "Exact immutable Windows entry bytes required")
    result = ModuleType("_ssdd_e24_pinned_" + path.stem)
    result.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), result.__dict__)
    return result


def make_formal_lock(mode, operation_commit, windows_scope_lock, qualification_pair, worker):
    return dict(status=STATUS, stage="e24", mode=mode, operation_commit=operation_commit,
                windows_scope_lock=windows_scope_lock, qualification_pair=qualification_pair,
                worker=worker, worker_wall_seconds_cap=LONG_WALL_SECONDS)


def validate_stage_binding(lock, scope):
    require(scope["operation"] == "short" and scope["mode"] == lock["mode"] and scope["mode"] in MODES,
            "E24 must use the same qualified Windows mode and original scope")
    commit = lock["operation_commit"]
    require(type(commit) is str and re.fullmatch(r"[a-f0-9]{40}", commit) is not None and commit != scope["protocol"],
            "A distinct E24 operational commit must preserve the short scientific protocol")


def admit_lock(entry, path, expected):
    lock = entry.document(entry.pinned_source(path, expected))
    entry.exact(lock, dict(status=STATUS, stage="e24", worker_wall_seconds_cap=LONG_WALL_SECONDS),
                ("mode", "operation_commit", "windows_scope_lock", "qualification_pair", "worker"),
                reason="Exact separate Windows E24 admission lock required")
    worker = entry.ref_path(lock["worker"], entry.MIB, basename=SELF)
    require(worker == Path(__file__).resolve(strict=True), "Exact running E24 worker must be independently locked")
    entry.read_ref(lock["worker"], entry.MIB)
    original = entry.ref_path(lock["windows_scope_lock"], entry.MIB, basename="WINDOWS_SCOPE_LOCK_PRIVATE.json")
    entry.read_ref(lock["windows_scope_lock"], entry.MIB)
    scope = entry.read_lock(original, lock["windows_scope_lock"]["sha256"], "short")
    validate_stage_binding(lock, scope)
    require(Path(scope["output"]).parent == path.parent == original.parent,
            "Separate E24 lock, verified scope lock and fresh arm output must share the same owner directory")
    source = Path(scope["code"]) / "SSDD_MASK_PHASE_TRANSFER_20261110"
    for name, digest in entry.SOURCES.items():
        entry.pinned_source(source / name, digest)
    qualifier = source / "ssdd_phase_qualification.py"
    entry.pinned_source(qualifier, QUALIFIER_SHA)
    sys.path.insert(0, str(source))
    acceptance = load_exact(source / "ssdd_phase_e24_acceptance.py", ACCEPTANCE_SHA)
    require(Path(import_module("ssdd_phase_qualification").__file__).resolve(strict=True) == qualifier,
            "Exact pure SSDD short qualifier must be bound")
    layout, ids = entry.read_fit_population(scope)
    receipt = lock["qualification_pair"]
    short_bytes = entry.read_ref(receipt, 64 * 1024, basename="SHORT_PAIR_ACCEPTED_SAFE.json")
    acceptance.verify_scope(scope["protocol"], scope["source_manifest_sha256"], ids, short_bytes, receipt["sha256"])
    return lock, scope, layout, ids


def run(entry, lock, scope, layout, ids):
    torch, _, worker, dependencies = entry.admit_windows_runtime(scope)
    entry.admit_host()
    free, total, fraction = entry.admit_cuda_short(torch.cuda)
    entry.emit(dict(event="SSDD_WINDOWS_Q100_E24_GPU_ADMITTED_V1", protocol=scope["protocol"], mode=scope["mode"],
                    operational_commit=lock["operation_commit"], short_pair_sha256=lock["qualification_pair"]["sha256"],
                    cuda_free_bytes=free, cuda_total_bytes=total, cuda_allocator_fraction=fraction))
    return worker(stage="e24", mode=scope["mode"], output=Path(scope["output"]), fit_layout=layout,
                  coco_path=Path(scope["coco"]["path"]), expected_fit_ids=ids,
                  expected_initial_sha256=entry.Q100_SHA, protocol=scope["protocol"], recipe_sha256=entry.RECIPE_SHA,
                  source_manifest_sha256=scope["source_manifest_sha256"], dependencies=dependencies,
                  event_sink=entry.emit)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--windows-entry", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--lock-sha256", required=True)
    args = parser.parse_args(argv)
    require(args.windows_entry.name == WINDOWS_ENTRY, "Original Windows SSDD entry basename required")
    entry = load_exact(args.windows_entry, ENTRY_SHA)
    run(entry, *admit_lock(entry, args.lock, args.lock_sha256))


if __name__ == "__main__":
    main()
