"""Windows native CUDA qualification for paired float PSeg-SSDD FIT masks."""
import argparse
from contextlib import ExitStack, contextmanager
from hashlib import sha1, sha256
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

CONFIG_STATUS = "SSDD_WINDOWS_FLOAT_COMMON_SHORT_REQUEST_V1"
AUTHORITY = "SSDD_WINDOWS_Q100_FLOAT_COMMON_SHORT_5201"
WORKER_STATUS = "SSDD_WINDOWS_FLOAT_COMMON_SHORT_ARM_CLOSED_PENDING_ACCEPTANCE"
PAIR_STATUS = "SSDD_WINDOWS_FLOAT_COMMON_SHORT_PAIR_INDEPENDENTLY_ACCEPTED_V1"
PAIR_NAME = "FLOAT_COMMON_SHORT_PAIR_ACCEPTED_SAFE.json"
WAIT_STATUS = "SSDD_WINDOWS_FLOAT_COMMON_SHORT_ORIGINAL_OWNER_WAIT_V1"
P333_OWNER_SHA = "ddf35079155045e14ebd743144d0ab538521de3a22b014c46f79edd64e690929"
P333_WORKER_SHA = "8aa1785f6727656d564177c6850120ce65d89f1ed67132e5a72ce331b97e564b"
P333_PAIR_SHA = "f79a7adcfb53d384e5b7dc6784201100e46a4b0ccd90b4fc0a294a11e3242b45"
P333_PROTOCOL = "16de15ea1bfe1a4b22f7434f16caae78054d9926"
P395_SHA = "cdf47287425e6a2cb7c2e06351f97fff3bcb3a88c02d9208774ae4d9dd47256c"
P397_SHA = "96f80f9d3a0662f93189665d2161ada7078cd33353d6e768ef0d34925f76af15"
P398_SHA = "030a24914dccc88772075745aa3cdbdaafcc4d02c9d1ea0946e661ed3602e590"
FLOAT_SHA = "97162d23a33b9387fe0b47a117f7e5b1f9dde7471033f244ab336cee1ca9039b"
INTEGER_SHA = "62966857c78cea1e3b060700c62a7b896942686c2e674d3eb09f516a92250bc1"
ROSTER_SHA = "315f7cfad9a3782865989ced212a6eeea1052e31251a01f24dbe2758bbc4f0e9"
PROTOCOL_SHA1 = "194e16f289a644430015f310dd63a3c1c1618cfd"
PROTOCOL_SHA256 = "b614bac72e0dc095c6617fa04b6cd76e50f588ec819612b2e3cbb0eab66c2b33"
LAYOUT_VALIDATOR = ("PSEG_SSDD_FLOAT_WINDOWS_COMMON_ADMISSION_20261111/float_fit_admission.py",
                    "b123f9dc2c283129b0be209b019ef6038f88bfc65365358d9ceafeb9db157eec")
FIT_BINDING = dict(integer_fit_sha256=INTEGER_SHA, float_fit_sha256=FLOAT_SHA, jpeg_roster_sha256=ROSTER_SHA,
                   p395_sha256=P395_SHA, p397_sha256=P397_SHA, p398_sha256=P398_SHA)
MODES = ("native", "center")
MIB = 1024**2


def need(ok, why):
    if not ok:
        raise ValueError(why)


def digest(raw):
    return sha256(raw).hexdigest()


def reference(path):
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest(path.read_bytes()))


def pinned(path, expected, limit):
    need(path.is_absolute() and path.resolve(strict=True) == path and path.is_file(),
         "Canonical bounded input required")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    need(0 < len(raw) <= limit and digest(raw) == expected, "Exact pinned input bytes required")
    return raw


def load_module(path, raw, name):
    mod = ModuleType(name)
    mod.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), mod.__dict__)
    return mod


def pinned_ref(entry, cfg, key, expected, limit=64 * 1024):
    ref = cfg[key]
    need(type(ref) is dict and ref.get("sha256") == expected, "Exact receipt or source digest required: " + key)
    return entry.read_ref(ref, limit)


def verify_receipts(entry, cfg, worker):
    p330 = entry.document(pinned_ref(entry, cfg, "p330_report", worker.P330_REPORT_SHA))
    terminal = entry.document(pinned_ref(entry, cfg, "p330_terminal", worker.P330_TERM_SHA))
    pair = entry.document(pinned_ref(entry, cfg, "p333_pair", P333_PAIR_SHA))
    p395 = entry.document(pinned_ref(entry, cfg, "p395_report", P395_SHA))
    p397 = entry.document(pinned_ref(entry, cfg, "p397_report", P397_SHA))
    p398 = entry.document(pinned_ref(entry, cfg, "p398_report", P398_SHA))
    need(p330["status"] == "P330_SSDD_FIXED_COMMON_CPU_PREFLIGHT_COMPLETE"
         and p330["FIT"] is True and p330["optimizer_updates"] == 0
         and terminal["status"] == "P330_SSDD_COMMON_DATASET_ORIGINAL_CPU_WAIT_V2"
         and terminal["original_os_wait"] is terminal["worker_closed"] is True
         and type(terminal["original_os_exit"]) is int and terminal["original_os_exit"] == 0,
         "Exact original P330 CPU closure required")
    need(pair["status"] == "P333_SSDD_WINDOWS_COMMON_SHORT_PAIR_INDEPENDENTLY_ACCEPTED_V1"
         and pair["training_protocol"] == P333_PROTOCOL and pair["visits_per_arm"] == 32
         and pair["optimizer_updates_per_arm"] == pair["ema_updates_per_arm"] == 8
         and pair["FIT"] is True and all(pair[k] is False for k in ("DEV", "TEST", "AP", "checkpoint_saved")),
         "Exact historical integer P333 paired qualification required")
    need(p395["status"] == "FIT741_FLOAT_POLYGON_ONLY_NATIVE_RASTER_ACTUAL"
         and p395["fit_images"] == 741 and p395["fit_rows"] == 1661
         and p395["frozen_integer_fit_sha256"] == INTEGER_SHA and p395["float_polygon_only_fit_sha256"] == FLOAT_SHA
         and p395["population"]["float_source_has_fraction"] == 1606 and p395["separate_dev_or_test_opened"] is False,
         "Exact P395 polygon-only source receipt required")
    need(p397["status"] == "FIT741_FLOAT_WINDOWS_TRAIN_ONLY_LAYOUT_EXACT_READBACK"
         and p397["images"] == 741 and p397["annotations"] == 1661 and p397["unique_jpeg"] == 730
         and p397["float_index_sha256"] == FLOAT_SHA and p397["content_roster_sha256"] == ROSTER_SHA
         and p397["fit"] is True and all(p397[k] is False for k in ("dev", "test", "ap", "training")),
         "Exact P397 float layout closure required")
    need(p398["status"] == "SSDD_FLOAT_WINDOWS_FIXED_COMMON_CPU_POPULATION_ACTUAL"
         and p398["visits_per_arm"] == 741 and p398["ordered_common_ledger_exact"] is True
         and p398["float_index_sha256"] == FLOAT_SHA and p398["content_roster_sha256"] == ROSTER_SHA
         and p398["counters"]["common_nonempty"] == 1457
         and p398["optimizer_updates"] == p398["model_forward_calls"] == 0
         and p398["fit"] is True and all(p398[k] is False for k in ("dev", "test", "ap")),
         "Exact P398 full FIT common-support CPU evidence required")


def context(path, expected):
    raw = pinned(path, expected, 64 * 1024)
    preliminary = json.loads(raw)
    ref = preliminary["p333_owner"]
    owner_path = Path(ref["path"])
    owner_raw = pinned(owner_path, P333_OWNER_SHA, 64 * 1024)
    p333 = load_module(owner_path, owner_raw, "_windows_float_original_p333_owner")
    old = p333.owner_module()
    entry = old.module(old.HERE / "ssdd_windows_phase_entry.py", old.ENTRY_SHA, "_windows_float_stock_entry")
    cfg = entry.document(raw)
    entry.exact(cfg, {"status": CONFIG_STATUS, "training_protocol": PROTOCOL_SHA1}, (
        "implementation_commit", "operation_commit", "output_root", "worker", "p333_owner", "p333_worker",
        "p330_report", "p330_terminal", "p333_pair", "p395_report", "p397_report", "p398_report",
        "float_layout", "float_overlay", "protocol_document", "coco"), reason="Exact float Windows request required")
    ids = (cfg["training_protocol"], cfg["implementation_commit"], cfg["operation_commit"])
    need(all(entry.hex_id(i, 40) for i in ids)
         and len({*ids, old.PROTOCOL, entry.DGX_PROTOCOL, P333_PROTOCOL}) == 6,
         "Distinct frozen prospective protocol, implementation and operation identities required")
    need(pinned_ref(entry, cfg, "p333_owner", P333_OWNER_SHA) == owner_raw
         and sha1(pinned_ref(entry, cfg, "protocol_document", PROTOCOL_SHA256)).hexdigest() == PROTOCOL_SHA1,
         "Exact original P333 owner and prospective protocol are required")
    worker_path = entry.ref_path(cfg["worker"], 64 * 1024)
    need(worker_path == Path(__file__).resolve(strict=True), "Exact new dual-role source required")
    entry.read_ref(cfg["worker"], 64 * 1024)
    source = entry.ref_path(cfg["p333_worker"], 64 * 1024)
    worker = load_module(source, pinned_ref(entry, cfg, "p333_worker", P333_WORKER_SHA),
                         "_windows_float_original_p333_worker")
    verify_receipts(entry, cfg, worker)
    coco = entry.ref_path(cfg["coco"], 134545398)
    need(coco.stat().st_size == cfg["coco"]["bytes"] == 134545398 and cfg["coco"]["sha256"] == entry.COCO_SHA,
         "Exact stock COCO initialization reference required")
    overlay = pinned_ref(entry, cfg, "float_overlay", FLOAT_SHA, MIB)
    float_layout = entry.canonical_dir(cfg["float_layout"])
    output = Path(cfg["output_root"])
    need(output.is_absolute() and output.parent.resolve(strict=True) == output.parent == old.ROOT
         and output.name.startswith("SSDD_WINDOWS_Q100_FLOAT_COMMON_") and not output.is_symlink()
         and output != float_layout and output not in float_layout.parents and float_layout not in output.parents,
         "Fresh isolated float CUDA output required")
    validator = old.module(old.CODE / LAYOUT_VALIDATOR[0], LAYOUT_VALIDATOR[1], "_windows_float_layout_validator")
    original_manifest = worker.manifest

    def manifest(active_entry, active_cfg):
        return digest(worker.encoded(dict(p333_common_manifest=original_manifest(active_entry, active_cfg),
            p333_owner_sha256=P333_OWNER_SHA, p333_worker_sha256=P333_WORKER_SHA, p333_pair_sha256=P333_PAIR_SHA,
            layout_validator_sha256=LAYOUT_VALIDATOR[1], float_binding=FIT_BINDING,
            protocol_sha256=PROTOCOL_SHA256)))

    worker.AUTHORITY, worker.WORKER_STATUS, worker.PAIR_STATUS = AUTHORITY, WORKER_STATUS, PAIR_STATUS
    worker.manifest = manifest
    return SimpleNamespace(p333=p333, old=old, entry=entry, cfg=cfg, worker=worker, worker_path=worker_path,
                           validator=validator, overlay=overlay, output=output, float_layout=float_layout)


def select_layout(ctx, scope, integer, ids):
    floating = ctx.float_layout
    need(floating != integer and integer not in floating.parents and floating not in integer.parents,
         "Exact separate float FIT layout required")
    integer_raw = (integer / "train/_annotations.coco.json").read_bytes()
    float_raw = (floating / "train/_annotations.coco.json").read_bytes()
    need(digest(integer_raw) == INTEGER_SHA and float_raw == ctx.overlay and len(ids) == 741,
         "Exact integer baseline and P395 float annotation bytes required")
    checks = importlib.import_module("ssdd_fit_stage_readback")
    ctx.validator.verify_float_layout(Path(scope["fit_archive"]["path"]), integer, floating, ids, checks)
    return floating


@contextmanager
def float_population(entry, validator):
    """Use the prospective float data only after original stock data admission."""
    original = entry.read_fit_population

    def read(scope):
        integer, ids = original(scope)
        floating = validator(scope, integer, ids)
        need(floating != integer and integer not in floating.parents and floating not in integer.parents,
             "Exact separate float FIT layout required")
        return floating, ids

    with patch.object(entry, "read_fit_population", read):
        yield


def child_main(argv):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for key in ("code-root", "config", "scope"):
        parser.add_argument("--" + key, type=Path, required=True)
    for key in ("config-sha256", "scope-sha256"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    args = parser.parse_args(argv)
    ctx = context(args.config, args.config_sha256)
    need(args.code_root == ctx.old.CODE, "Exact original Windows stock code root required")
    original_module = ctx.worker.module

    def config(_entry, path, expected):
        need((path, expected) == (args.config, args.config_sha256), "Exact worker request required")
        return ctx.cfg, ctx.output

    with ExitStack() as stack:
        def module(path, expected, name):
            imported = original_module(path, expected, name)
            if name == "windows_entry":
                stack.enter_context(float_population(imported, lambda scope, integer, ids:
                    select_layout(ctx, scope, integer, ids)))
            return imported

        stack.enter_context(patch.object(ctx.worker, "module", module))
        stack.enter_context(patch.object(ctx.worker, "read_config", config))
        ctx.worker.run(args)
    return 0


def original_owner_events():
    return [dict(status="SSDD_WINDOWS_Q100_ORIGINAL_ARM_START", stage="short", mode=mode) for mode in MODES] + [
        dict(status="SSDD_WINDOWS_Q100_ORIGINAL_OWNER_RETURN", stage="short", resource_closed=True, failure=None)]


def verify_original_events(events):
    expected = original_owner_events()
    need(len(events) == len(expected) and all(
        type(a) is dict and set(a) == set(b) and all(type(a[k]) is type(v) and a[k] == v for k, v in b.items())
        for a, b in zip(events, expected, strict=True)), "Exact ordered original Windows owner stdout required")


def require_wait(entry, args, legacy, record, config_sha):
    wait = entry.document(pinned(args.original_wait, args.original_wait_sha256, 64 * 1024))
    expected = dict(status=WAIT_STATUS, original_os_wait=True, original_os_exit=0, owner_birth=record["owner_birth"],
                    owner_config_sha256=config_sha, owner_source_sha256=digest(Path(__file__).read_bytes()))
    entry.exact(wait, expected, ("combined_stdout_sha256",), reason="Exact original float owner OS wait required")
    need(entry.hex_id(wait["combined_stdout_sha256"], 64) and not legacy.live(record["owner_birth"]),
         "Original float Windows owner must be closed")
    original_stdout = pinned(args.original_stdout, wait["combined_stdout_sha256"], 64 * 1024)
    verify_original_events([entry.document(line) for line in original_stdout.splitlines() if line])
    return wait


def owner_main(argv):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    action = parser.add_mutually_exclusive_group(required=True)
    for name in ("preflight-cpu", "run", "accept"):
        action.add_argument("--" + name, action="store_true")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--original-wait", type=Path)
    parser.add_argument("--original-wait-sha256")
    parser.add_argument("--original-stdout", type=Path)
    args = parser.parse_args(argv)
    need(all((value is not None) == args.accept for value in (
        args.original_wait, args.original_wait_sha256, args.original_stdout)),
        "External original owner wait and stdout are only for independent acceptance")
    ctx = context(args.config, args.config_sha256)
    short, legacy, runtime = ctx.old.load_guards("short")
    lock = dict(ctx.cfg, stage="short")
    cfg_ref = legacy.ref(args.config)
    need(cfg_ref["sha256"] == args.config_sha256, "Exact frozen prospective request required")
    base, replay, ids = ctx.old.inputs(ctx.entry, legacy, runtime, lock)
    select_layout(ctx, base, Path(base["fit_layout"]), ids)
    if args.preflight_cpu:
        need(not ctx.output.exists() and "torch" not in sys.modules, "Fresh no-CUDA input preflight required")
        ctx.entry.emit(dict(status="SSDD_WINDOWS_FLOAT_COMMON_SHORT_CPU_PREFLIGHT_PASSED",
                            config_sha256=args.config_sha256, source_manifest_sha256=ctx.worker.manifest(ctx.entry, lock),
                            fit_binding=FIT_BINDING, fit_images=len(ids), fit_rows=1661,
                            model_forward_calls=0, optimizer_updates=0, torch_imported=False, **ctx.worker.SCOPE))
        return 0

    raw = legacy.P189["stable"](legacy.STATE, short.STATE_LIMIT)
    state = json.loads(raw)
    ctx.old.check_idle(state)
    ctx.old.verify_main_handoff(state, ctx.entry, legacy)
    ctx.p333.attach_common_stage(ctx.old, ctx.worker, ctx.worker_path, args.config, args.config_sha256)
    if args.accept:
        def accept_cas(original, new, _name):
            return cas(original, new, "SSDD_WINDOWS_FLOAT_COMMON_SHORT_INDEPENDENTLY_ACCEPTED")

        cas = legacy.cas
        with patch.object(ctx.p333, "require_wait", require_wait), patch.object(ctx.p333, "PAIR_NAME", PAIR_NAME), \
                patch.object(legacy, "cas", accept_cas):
            return ctx.p333.accept(args, ctx.old, ctx.worker, ctx.entry, short, legacy, lock, ctx.output,
                                   cfg_ref, raw, state, base, replay, ids)
    need(AUTHORITY not in state and not ctx.output.exists(), "Fresh float Windows authority and output required")
    return ctx.old.execute(ctx.entry, short, legacy, runtime, lock, ctx.output,
                           cfg_ref, raw, state, base, replay, None)


if __name__ == "__main__":
    arguments = sys.argv[1:]
    sys.exit(child_main(arguments) if arguments and arguments[0] == "--code-root" else owner_main(arguments))
