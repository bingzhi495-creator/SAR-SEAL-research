"""Verify the exact released P297 HRSID initialization seam on DGX CPU."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


LOCK_STATUS = "P304_DGX_BC_CPU_INIT_LOCK_V1"
PASS = "P304_DGX_BC_CPU_INIT_EXACT_V1"
PATH_FIELDS = {"python", "code_root", "output"}
REF_FIELDS = {"worker", "coco", "position", "query_tail", "query_tail_receipt", "parity_receipt"}
WORKER_SHA = "89d2ac054b25764ff67c0aaf2cbd2b335e1e06bede06191d9b32253eaaba7b0c"
WORKER_BYTES = 26799
SCOPE = dict(FIT=False, DEV=False, TEST=False, AP=False, training=False, cuda_initialized=False,
             model_forward_calls=0, optimizer_updates=0)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def fixed_bytes(path, expected, maximum=65536):
    require(re.fullmatch(r"[0-9a-f]{64}", expected) is not None and path.is_absolute()
            and path.resolve(strict=True) == path and path.is_file(), "Canonical pinned input required")
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        raw = stream.read(maximum + 1)
        after = os.fstat(stream.fileno())
    token = lambda st: (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
    require(0 < len(raw) == before.st_size <= maximum and token(before) == token(after) == token(path.stat())
            and sha256(raw).hexdigest() == expected, "Exact stable pinned input bytes required")
    return raw


def read_lock(path, expected):
    def unique(pairs):
        value = dict(pairs)
        require(len(value) == len(pairs), "Duplicate CPU init lock key")
        return value

    value = json.loads(fixed_bytes(path, expected), object_pairs_hook=unique)
    require(type(value) is dict and set(value) == PATH_FIELDS | REF_FIELDS | {"status"}
            and value["status"] == LOCK_STATUS
            and all(type(value[name]) is str and Path(value[name]).is_absolute() for name in PATH_FIELDS),
            "Exact CPU init lock required")
    return value


def load_worker(ref):
    require(type(ref) is dict and set(ref) == {"path", "bytes", "sha256"}
            and type(ref["path"]) is str and ref["bytes"] == WORKER_BYTES and ref["sha256"] == WORKER_SHA,
            "Exact public P297 worker reference required")
    path = Path(ref["path"])
    raw = fixed_bytes(path, WORKER_SHA, WORKER_BYTES)
    require(path.as_posix() == ref["path"] and len(raw) == WORKER_BYTES, "Exact public P297 worker bytes required")
    worker = ModuleType("_p304_pinned_p297")
    worker.__file__ = path.as_posix()
    exec(compile(raw, str(path), "exec"), worker.__dict__)
    return worker


def cpu_scope(worker, sentinel, allowed):
    common = worker.scope_guard(sentinel / "train", allowed)

    def guard(event, args):
        if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0]))
            if path.name == "_annotations.coco.json" or path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tiff", ".tif"}:
                raise PermissionError("CPU initialization admits zero image or annotation opens")
        common(event, args)
    return guard


def verify_runtime_paths(python):
    prefix = Path(sys.prefix)
    require(python == Path(sys.executable) == prefix / "bin" / "python"
            and prefix.resolve(strict=True) == prefix and sys.prefix != sys.base_prefix,
            "Exact lock-supplied isolated DGX virtualenv Python required")
    driver = Path("/proc/driver/nvidia/version").read_text(encoding="ascii").splitlines()[0]
    require("NVIDIA UNIX Open Kernel Module for aarch64  580.173.02" in driver,
            "Exact DGX driver string required")


def forbidden(*args, **kwargs):
    raise RuntimeError("CPU initialization cannot execute FIT, a model forward or an optimizer step")


def run(lock_path, lock_sha):
    cfg = read_lock(lock_path, lock_sha)
    worker = load_worker(cfg["worker"])
    code = Path(cfg["code_root"])
    qualifier, engine, plan = worker.source_plan(code)
    require(plan["worker"] == cfg["worker"], "Exact P297 source-plan worker identity required")
    parity = plan["platform_sources"]
    p293 = worker.load_source(Path(parity["p293_source"]["path"]), parity["p293_source"]["sha256"], "_p304_p293")
    p291 = p293.pinned_p291(Path(parity["p291_source"]["path"]))
    p290 = p291.pinned_p290(Path(parity["p290_source"]["path"]))
    verify_runtime_paths(Path(cfg["python"]))
    paths = {role: worker.ref_path(cfg[role], worker.FIXED_FILES[role]) for role in REF_FIELDS - {"worker"}}
    require(len(set(paths.values())) == len(paths) and paths["coco"].name == "rf-detr-seg-nano.pt"
            and paths["position"].name == "Q100_POSITION_REFERENCE_PRIVATE.pt"
            and paths["query_tail"].name == p293.TAIL_NAME, "Exact distinct canonical initialization inputs required")
    reference = worker.read_json_ref(cfg["parity_receipt"])
    tail_receipt = worker.read_json_ref(cfg["query_tail_receipt"])
    require(reference["status"] == p293.PARITY and reference["expected_canonical_q164_sha256"] == p290.Q164
            and reference["windows_query_tail_receipt_sha256"] == cfg["query_tail_receipt"]["sha256"],
            "Exact closed canonical Q164 receipt required")
    output = Path(cfg["output"])
    require(output.parent.resolve(strict=True) == output.parent and output.parent.is_dir()
            and not output.exists() and not output.is_symlink() and not output.is_relative_to(code.parent),
            "Fresh canonical private CPU output required")
    sentinel = output / "UNUSED_FIT_LAYOUT"
    sys.addaudithook(cpu_scope(worker, sentinel, {paths[name] for name in ("coco", "position", "query_tail")}))
    # P290's CPU admission checks all common package/codec pins and sets the
    # two thread counts once. The native train entry is deliberately excluded.
    torch = p290.admit_runtime()
    stock = p290.package_root("rfdetr")
    for relative, pin in worker.STOCK_RUNTIME.items():
        p290.pinned(stock / relative, pin, 1024**2)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True
    bundle = p291.source_bundle(p290, code)
    refs = plan["scientific_sources"]
    native = engine.load_source(refs[engine.NATIVE], "_p297_native")
    base = p290.load(bundle, "base", "_p297_base")
    p290.load(bundle, "selector", "select_native_fit")
    probe = p290.load(bundle, "probe", "_p297_probe")
    state = p290.load(bundle, "state", "_p297_state")
    p290.load(bundle, "p242_init", "ssdd_phase_init")
    position_loader = p290.load(bundle, "p242_position", "_p297_position_loader")
    native.COCO, native.RECIPE, native.CONFIG = paths["coco"], bundle["recipe"][0], bundle["config"][0]
    native.RNG, native.FILTER = (Path(plan["auxiliary_sources"][role]["path"]) for role in ("rng", "filter"))
    native.run = forbidden
    base.LAYOUT, base.fit_ids = sentinel, forbidden
    require(probe.INIT_SEED == native.SEED == 5201 and probe.EXPANSION_SEED == 5201164
            and probe.STATE_SHA == p290.SOURCE_PINS["state"][1], "Exact P297 original initialization source lineage required")
    base.closed_probe = lambda: SimpleNamespace(STOCK=bundle["state"][0].parent,
                                                STATE_SHA=probe.STATE_SHA, expand_groups=probe.expand_groups)
    engine.load_source(refs["trace_resize.py"], "trace_resize")
    engine.load_source(refs["phase_patch.py"], "phase_patch")
    phase = engine.load_source(refs["phase_train.py"], "_p297_phase")
    from pytorch_lightning import seed_everything
    from rfdetr.training.module_model import RFDETRModelModule

    output.mkdir(mode=0o700)
    source_sha = engine.manifest_digest({"sources": refs})
    provenance = dict(qualification="P304_CPU_INIT_ONLY", lock_sha256=lock_sha, worker_sha256=WORKER_SHA,
                      scientific_source_manifest_sha256=source_sha, FIT=False, training=False, cuda_initialized=False)
    worker.install_initialization(native, base, state, p290, p291, p293, position_loader, paths, reference,
                                  tail_receipt, output, phase, torch, provenance=provenance)

    def deny_forward(module, _):
        if isinstance(module, RFDETRModelModule) or type(module).__module__ == "rfdetr.models.lwdetr":
            forbidden()

    handle = torch.nn.modules.module.register_module_forward_pre_hook(deny_forward)
    try:
        with patch.object(torch.optim.AdamW, "step", forbidden):
            seed_everything(native.SEED, workers=True)
            model, train, module, state_api, init = native.initialize(base, output, RFDETRModelModule)
    finally:
        handle.remove()
    tensors = module.model.state_dict()
    require(len(tensors) == 545 and all(tensor.device.type == "cpu" for tensor in tensors.values())
            and init["width100_loaded_sha256"] == p290.Q100 and init["expanded_online_sha256"] == p290.Q164
            and state_api.tensor_dict_sha256(tensors) == p290.Q164 and model.num_queries == 164
            and model.num_select == 100 and model.resolution == 576 and model.group_detr == 13
            and Path(train.dataset_dir) == sentinel and not sentinel.exists()
            and not torch.cuda.is_available() and not torch.cuda.is_initialized(),
            "Exact final CPU-only strict P297 Q164 state and unused FIT sentinel required")
    witness_path = output / "DGX_INITIALIZATION_WITNESS_SAFE.json"
    witness_ref = worker.closed_ref(witness_path)
    witness = worker.read_json_ref(witness_ref)
    require(witness["status"] == "P297_HRSID_DGX_BC_SHORT_CANONICAL_INITIALIZATION_WITNESS_V1"
            and witness["position_audit"] == reference["position_audit"]
            and witness["query_tail_audit"] == reference["query_tail_audit"]
            and all(type(witness.get(key)) is type(value) and witness[key] == value for key, value in SCOPE.items()),
            "Exact P297 initialization witness and zero scientific execution required")
    result = dict(status=PASS, lock_sha256=lock_sha, worker_sha256=WORKER_SHA,
                  scientific_source_manifest_sha256=source_sha, worker_init_witness_sha256=witness_ref["sha256"],
                  parity_receipt_sha256=cfg["parity_receipt"]["sha256"],
                  query_tail_receipt_sha256=cfg["query_tail_receipt"]["sha256"],
                  q100_sha256=p290.Q100, q164_sha256=p290.Q164, q164_cpu_tensors=545,
                  position_audit_exact=True, query_tail_audit_exact=True, **SCOPE)
    phase.exclusive_json(output / "RESULT_SAFE.json", result)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--lock-sha256", required=True)
    args = parser.parse_args(argv)
    return run(args.lock, args.lock_sha256)


if __name__ == "__main__":
    raise SystemExit(main())
