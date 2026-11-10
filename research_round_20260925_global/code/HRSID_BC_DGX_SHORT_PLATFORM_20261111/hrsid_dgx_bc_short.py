"""Run one DGX-only HRSID fixed-common B/C FIT qualification arm."""
import argparse
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import re
import stat
import sys
from types import ModuleType, SimpleNamespace


STATUS = "P297_HRSID_DGX_BC_SHORT_PLATFORM_LOCK_V1"
SELF = Path(__file__).resolve()
REPLAY = "HRSID_FIT_RESIZE_PHASE_REPLAY_20261109"
PARITY = "HRSID_DGX_Q164_INIT_20261111"
QUALIFIER = "one_pass_common_qualification.py"
QUALIFIER_SHA = "4b2bb3bbe00aaf28c748f66a540182993589039304ea1d966d1dde7e7c1980d8"
MODES = ("stepwise_center", "one_pass")
Q164 = "63ff93afdd977ba79b5b5a46ba8b4839ef6c40eb0722b29e4667d4932404fbc2"
INDEX = "e95cf26cc60a2c6fb651e907541380a256f2df7b1e153244a6c0f803b0b19f8f"
INDEX_BYTES = 30076663
MANIFEST_SHA = "a7e4d2a7ff9e8ff238adbf94a1d0397553d8746e2558c316db4e417f86f8fec9"
GiB = 1024**3
SOURCE_FILES = {
    "fit_query_export.py": "HRSID_FIT_NATIVE_QUERY_RETENTION_20261109",
    "phase_patch.py": "HRSID_FIT_RESIZE_PHASE_TRAIN_20261109",
    "phase_train.py": "HRSID_FIT_RESIZE_PHASE_TRAIN_20261109",
    "phase_repeat_train.py": "HRSID_MASK_PHASE_REPEAT_TRAIN_20261109",
    "train_hrsid_coco_native_e24_filtered.py": "HRSID_NATIVE_EMPTY_MASK_FILTER_20261109",
}
PLATFORM_SOURCES = {
    "p290_source": (PARITY + "/hrsid_dgx_q164_parity.py",
                    "22e12203a8ad09a5f24911c316983b29ae5ce3d75bfde6065ecf3859c5970f5d"),
    "p291_source": (PARITY + "/hrsid_dgx_q164_position_parity.py",
                    "ca1cfc94ec954fadfc48aa164d527d0914bba80224be589fe8a9d1d33e62a04b"),
    "p293_source": (PARITY + "/hrsid_dgx_q164_query_tail_parity.py",
                    "b1c1b623294f7282401e58dcc463a3d8160c0f109af7ca83ed8b3b2ce69b4d5d"),
}
FIXED_FILES = {
    "fit_manifest": (1386385, MANIFEST_SHA),
    "fit_decode_receipt": (1762, "6f073cbf6dcf3ad08d68c405589323e321a04f441665e479d2cdd6505f48e6ae"),
    "coco": (134545398, "a44613a4ecd6b5ba61a62002c600b0b6cb7a9da2936a45317ec4b62c635fb99b"),
    "position": (3542121, "f8c2c8d83b4efa917a1e731a312fc0adaaafba89dda61ef845b61ced0916f9ee"),
    "query_tail": (851968, "bd0cc43395fec526d93d96e83a54a383dddaa4c96cd98c6f81aba47b596a5f43"),
    "query_tail_receipt": (1733, "be947ee8f1007d7b12f3e4b226574437e7826da67f5c638cb7efcffcef90c603"),
    "parity_receipt": (3154, "ecc4dab87ba55cda369021657b79f2a38ca733bd9d14f0b014371b518f7eae06"),
}
RECIPE_AUX = {
    "rng": ("RFDETR_NATIVE_E1_CHECKPOINT_API_RECOVERY_20261107/e1_resume.py",
            "014e087379ffe6e8d32329c337d6b625b98dfc2d235cb321becebfdf3832b013"),
    "filter": ("HRSID_NATIVE_EMPTY_MASK_FILTER_20261109/filter_empty_masks.py",
               "3b164cbf8c104539eabee8775c8e1f297d9a1607e37473c4d7a4a9b561f93471"),
}
STOCK_RUNTIME = {
    "models/transformer.py": "279c0af7e999ff4ce58845c94c87bd9211356d62cb1eeaeb626fe2680fbb8f4d",
    "models/criterion.py": "fe3a1b05acc7edc4476c03954d75e2b9c91d5612da333782aecd23d0d1cea5c3",
    "models/matcher.py": "2521d58fd5c3270f8f352c14ae299364347bd27792e18696de8bb20c637765f6",
    "utilities/tensors.py": "122f44a151db786ef4845c5d89e5cb2209624651d3638f007e3c0867f7312827",
    "training/module_data.py": "624bc77f7028bc53e0ca57dcbf5be1a7d334685147dbfab65e9acaa063d715b5",
    "datasets/coco.py": "be64b20a178d9cde41db69e01e17e1296e36eee791e837294c3f930ae56f03b9",
    "datasets/_torchvision.py": "03c12e881f733d4f6b469f784187e461753b8775b7e35e57c1d8a6c35c46ef56",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value, width=64):
    return type(value) is str and re.fullmatch("[0-9a-f]{" + str(width) + "}", value) is not None


def canonical(path, directory=False):
    require(path.is_absolute() and path.resolve(strict=True) == path
            and (path.is_dir() if directory else path.is_file()), "Canonical file or directory required: " + path.name)
    return path


def checked_file(path, limit=65536, expected=None):
    canonical(path)
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= limit, "Bounded regular input required")
        raw = stream.read(before.st_size + 1)
        after = os.fstat(stream.fileno())
    signature = lambda entry: (entry.st_dev, entry.st_ino, entry.st_size, entry.st_mtime_ns, entry.st_ctime_ns)
    require(signature(before) == signature(after) == signature(path.stat()) and len(raw) == before.st_size,
            "Stable closed bytes required: " + path.name)
    ref = dict(path=path.as_posix(), bytes=len(raw), sha256=sha256(raw).hexdigest())
    if expected is not None:
        require(ref["sha256"] == expected, "Exact SHA-256 required: " + path.name)
    return ref, raw


def closed_ref(path, limit=65536, expected=None):
    return checked_file(path, limit, expected)[0]


def object_from(raw):
    def unique(items):
        value = dict(items)
        require(len(value) == len(items), "Duplicate JSON key")
        return value
    def reject_constant(value):
        raise ValueError("Nonfinite JSON constant " + value)
    value = json.loads(raw, object_pairs_hook=unique, parse_constant=reject_constant)
    require(type(value) is dict, "JSON object required")
    return value


def ref_path(ref, expected=None):
    require(type(ref) is dict and set(ref) == {"path", "bytes", "sha256"}
            and type(ref["path"]) is str and type(ref["bytes"]) is int and ref["bytes"] > 0
            and digest(ref["sha256"]), "Exact closed reference schema required")
    if expected is not None:
        require((ref["bytes"], ref["sha256"]) == expected, "Unexpected precommitted input digest or byte count")
    path = canonical(Path(ref["path"]))
    require(path.as_posix() == ref["path"] and path.stat().st_size == ref["bytes"], "Exact canonical reference required")
    return path


def read_json_ref(ref):
    path = ref_path(ref)
    actual, raw = checked_file(path, expected=ref["sha256"])
    require(actual == ref, "Closed JSON reference changed")
    return object_from(raw)


def load_source(path, source_sha, name):
    _, raw = checked_file(path, expected=source_sha)
    module = ModuleType(name)
    module.__file__ = path.as_posix()
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def source_plan(code):
    canonical(code, directory=True)
    require(code.name == "code", "Exact ordinary code root required")
    qualifier = load_source(code / REPLAY / QUALIFIER, QUALIFIER_SHA, "_p297_qualified_b_c")
    engine = qualifier.configured_engine(code / REPLAY / qualifier.ENGINE)
    require(qualifier.MODES == MODES and engine.SELF == QUALIFIER
            and engine.INITIAL_EXPANDED_SHA == Q164 and engine.FIT_INDEX_SHA == INDEX,
            "Exact original B/C scientific qualifier required")
    scientific = {
        name: closed_ref(code / SOURCE_FILES.get(name, REPLAY) / name, expected=pin)
        for name, pin in {**engine.SOURCE_PINS, QUALIFIER: QUALIFIER_SHA}.items()
    }
    operations = {
        role: closed_ref(code / relative, expected=pin) for role, (relative, pin) in PLATFORM_SOURCES.items()
    }
    p293 = load_source(Path(operations["p293_source"]["path"]), operations["p293_source"]["sha256"], "_p297_p293")
    p291 = p293.pinned_p291(Path(operations["p291_source"]["path"]))
    p290 = p291.pinned_p290(Path(operations["p290_source"]["path"]))
    bundle = p291.source_bundle(p290, code)
    auxiliary = {role: dict(path=path.as_posix(), bytes=len(raw), sha256=sha256(raw).hexdigest())
                 for role, (path, raw) in bundle.items()}
    auxiliary.update({role: closed_ref(code / relative, expected=pin)
                      for role, (relative, pin) in RECIPE_AUX.items()})
    return qualifier, engine, dict(scientific_sources=scientific, platform_sources=operations, auxiliary_sources=auxiliary,
                                  worker=closed_ref(SELF))


def platform_lock(path, expected):
    require(digest(expected), "Exact prospective platform lock SHA-256 required")
    _, raw = checked_file(path, expected=expected)
    lock = object_from(raw)
    keys = {"status", "python", "code_root", "work_root", "training_protocol", "implementation_commit",
            "worker_sha256", "fit_layout", "owner_birth_ref"} | set(FIXED_FILES) | set(PLATFORM_SOURCES)
    require(set(lock) == keys and lock["status"] == STATUS and digest(lock["training_protocol"], 40)
            and digest(lock["implementation_commit"], 40) and digest(lock["worker_sha256"]),
            "Exact prospective platform lock schema required")
    require(all(type(lock[key]) is str for key in ("python", "code_root", "work_root", "fit_layout")),
            "Exact string platform paths required")
    require(lock["worker_sha256"] == closed_ref(SELF)["sha256"], "Exact executing worker SHA-256 required")
    return lock


def arm_args(cfg, lock):
    work = canonical(Path(lock["work_root"]), directory=True)
    require(cfg.mode in MODES, "Exact fixed-common B/C mode required")
    root = canonical(work / cfg.mode, directory=True)
    output, audit = root / "ACTUAL", root / "AUDIT"
    require(not any(path.exists() or path.is_symlink() for path in (output, audit)), "Fresh qualification outputs required")
    return SimpleNamespace(mode=cfg.mode, stage="qualification", training_seed=5202, output=output, audit_output=audit,
                           protocol=lock["training_protocol"], expected_initial_sha256=Q164,
                           training_lock=cfg.training_lock, training_lock_sha256=cfg.training_lock_sha256)


def runtime_guard(lock, p290):
    python = Path(lock["python"])
    prefix = Path(sys.prefix)
    require(python.is_absolute() and python == Path(sys.executable) == prefix / "bin" / "python"
            and prefix.resolve(strict=True) == prefix and sys.prefix != sys.base_prefix
            and sys.version_info[:3] == (3, 12, 14)
            and platform.system() == "Linux" and platform.machine() == "aarch64"
            and sys.flags.isolated == sys.flags.utf8_mode == 1 and sys.dont_write_bytecode
            and os.sched_getaffinity(0) == {0, 1} and os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
            and all(os.environ.get(key) == "1" for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")),
            "Exact isolated offline DGX ARM GPU runtime and CPU0/1 affinity required")
    require({key: version(key) for key in p290.PACKAGES} == p290.PACKAGES, "Exact DGX runtime package versions required")
    import psutil
    birth = read_json_ref(lock["owner_birth_ref"])
    require(set(birth) == {"pid", "created", "boot_id"} and type(birth["pid"]) is int
            and type(birth["created"]) is float and birth["pid"] == os.getppid()
            and birth["boot_id"] == Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
            and birth["created"] == psutil.Process(birth["pid"]).create_time()
            and psutil.virtual_memory().available >= 40 * GiB,
            "Exact live original owner birth and at least 40 GiB host available required")
    stock = p290.package_root("rfdetr")
    for relative, pin in {**p290.RFDETR_PINS, **STOCK_RUNTIME}.items():
        p290.pinned(stock / relative, pin, 1024**2)
    for name, (pattern, pin) in p290.EXTENSIONS.items():
        files = tuple(p290.package_root(name).glob(pattern))
        require(len(files) == 1, "Exact DGX codec extension required")
        p290.pinned(files[0], pin, 16 * 1024**2)
    driver = Path("/proc/driver/nvidia/version").read_text(encoding="ascii").splitlines()[0]
    require("NVIDIA UNIX Open Kernel Module for aarch64  580.173.02" in driver, "Exact DGX CUDA driver required")


def gpu_guard(torch):
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1
            and torch.cuda.get_device_capability(0) == (12, 1)
            and "GB10" in torch.cuda.get_device_name(0)
            and torch.cuda.is_bf16_supported(including_emulation=False)
            and torch.cuda.mem_get_info(0)[0] >= 12 * GiB,
            "Exact single DGX GB10, native BF16 and at least 12 GiB available CUDA memory required")


def verify_decode_receipt(value):
    require(value.get("status") == "HRSID_FIT2937_DGX_TRANSPORT_AND_INDEPENDENT_STOCK_DECODE_ACCEPTED"
            and all(type(value.get(name)) is int and value[name] == 0
                    for name in ("transport_original_exit", "independent_decode_original_exit")),
            "Exact independently closed DGX FIT transport and stock decode required")
    expected = dict(status="HRSID_FIT2937_DGX_INDEPENDENT_STOCK_DECODE_ACCEPTED_V1",
                    images=2937, annotations=9260, encoded_sha256_match=True, decoded_rgb_sha256_match=True,
                    png_bytes=3698804355, image_size=[800, 800], native_index_sha256=INDEX,
                    manifest_sha256=MANIFEST_SHA,
                    FIT=True, DEV=False, TEST=False, AP=False, model_forward_calls=0)
    observed = value["independent_readback"]
    require(all(type(observed.get(key)) is type(item) and observed[key] == item for key, item in expected.items()),
            "Exact independently verified stock FIT dimensions, counts and scope required")


def verify_fit_manifest(layout, ref):
    sidecar = layout.with_name(layout.name + ".manifest.private.json")
    require(ref_path(ref, FIXED_FILES["fit_manifest"]) == sidecar, "Exact adjacent P289 FIT manifest required")
    actual, raw = checked_file(sidecar, ref["bytes"], ref["sha256"])
    value = object_from(raw)
    require(actual == ref and value["native_index_sha256"] == INDEX and value["native_index_bytes"] == INDEX_BYTES
            and value["selected_encoded_png_bytes"] == 3698804355 and len(value["images"]) == 2937,
            "Exact closed original P289 image manifest required")
    return {row["native_basename"]: (row["image_id"], row["encoded_bytes"]) for row in value["images"]}


def fit_ids(layout, *, index_ref=None, counts=(2937, 9260), roster=None):
    canonical(layout, directory=True)
    train = canonical(layout / "train", directory=True)
    require({p.name for p in layout.iterdir()} == {"train"}, "Exact FIT-only root")
    index = train / "_annotations.coco.json"
    pin = INDEX if index_ref is None else index_ref["sha256"]
    ref, raw = checked_file(index, 32 * 1024**2, pin)
    if roster is not None:
        require(ref["bytes"] == INDEX_BYTES and len(roster) == counts[0], "Exact P289 index and manifest byte census")
    if index_ref is not None:
        require(ref == index_ref, "Exact FIT index bytes required")
    value = object_from(raw)
    images, annotations = value["images"], value["annotations"]
    require(type(images) is list and type(annotations) is list
            and (len(images), len(annotations)) == counts, "Exact FIT index census required")
    ids, names = set(), set()
    for row in images:
        image_id, name = row["id"], row["file_name"]
        require(type(image_id) is int and image_id not in ids
                and type(name) is str and name == Path(name).name and name.endswith(".png")
                and name not in names and row["width"] == row["height"] == 800, "Exact unique FIT row and PNG basename")
        if roster is not None:
            require(name in roster and roster[name][0] == image_id
                    and (train / name).stat().st_size == roster[name][1], "Exact P289 PNG ID and byte-size identity required")
        ids.add(image_id)
        names.add(name)
    require(all(row["image_id"] in ids for row in annotations), "Every FIT annotation must refer to a FIT image")
    present = tuple(train.iterdir())
    require({path.name for path in present} == names | {index.name}
            and all(path.is_file() and not path.is_symlink() for path in present), "Exact FIT PNG roster required")
    return ids


def scope_guard(train, references):
    index = train / "_annotations.coco.json"
    blocked = {".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".zip", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    def guard(event, args):
        if event.startswith("socket.") and event != "socket.__new__":
            raise PermissionError("Offline FIT qualification")
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0]))
        if path.suffix.lower() not in blocked and path.name != index.name:
            return
        access = args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
        allowed = path in references or path == index or (path.parent == train and path.suffix == ".png")
        if access or not allowed or path.resolve(strict=True) != path:
            raise PermissionError("Only exact immutable FIT and initialization scientific inputs are admitted")
    return guard


def install_initialization(native, base, state, p290, p291, p293, position_loader, paths, reference,
                           tail_receipt, audit, phase, torch, *, provenance):
    original = native.initialize
    tensor_digest = state.tensor_dict_sha256
    expected_prefix = tail_receipt["query_prefix_sha256"]
    require(type(expected_prefix) is dict and set(expected_prefix) == set(p290.QUERIES)
            and expected_prefix[p293.QUERY_KEY] == p293.QUERY_PREFIX, "Exact canonical query prefixes required")
    typed_tail = p293.verify_tail_receipt(p290, p291, tail_receipt,
                                        {"query_tail_file_sha256": FIXED_FILES["query_tail"][1]}, expected_prefix)

    def initialize(active_base, output, factory):
        require(active_base is base, "Exact DGX FIT base namespace required")
        position_audit = None

        def build(model_config, train_config, **kwargs):
            nonlocal position_audit
            module = factory(model_config, train_config, **kwargs)
            if model_config.num_queries == 100:
                require(position_audit is None, "Exactly one CPU Q100 initialization required")
                load_position = lambda: position_loader._load_position(
                    paths["position"], FIXED_FILES["position"][1], torch, tensor_digest)
                position_audit, _ = p291.canonicalize(module.model, load_position, torch, tensor_digest, p290)
                require(position_audit == reference["position_audit"], "Exact observed DGX Q100 parity required")
            return module

        model, train, module, state_api, info = original(base, output, build)
        require(position_audit is not None and info["width100_loaded_sha256"] == p290.Q100
                and info["expanded_online_sha256"] == p293.POSITION_ONLY_Q164, "Exact observed pre-tail Q100/Q164 required")
        observed = p290.observe(module.model, 164, torch, tensor_digest)
        require(tensor_digest(module.model.state_dict()) == p293.POSITION_ONLY_Q164
                and observed["queries"][p293.QUERY_KEY]["sha256"] == p293.POSITION_ONLY_QUERY
                and observed["queries"][p293.QUERY_KEY]["group_tail_sha256"] == p293.POSITION_ONLY_TAIL
                and all(observed["queries"][key]["group_prefix_sha256"] == expected_prefix[key] for key in p290.QUERIES),
                "Exact intermediate expanded grouped tensor identities required")
        tail = p293.load_tail(p290, paths["query_tail"], FIXED_FILES["query_tail"][1], typed_tail, tensor_digest, torch)
        tail_audit = p293.copy_query_tail(module.model, tail, torch, tensor_digest, p293.POSITION_ONLY_Q164,
                                         Q164, p293.QUERY_PREFIX)
        require(tail_audit == reference["query_tail_audit"]
                and state_api.tensor_dict_sha256(module.model.state_dict()) == Q164, "Exact observed final DGX Q164 required")
        phase.exclusive_json(audit / "DGX_INITIALIZATION_WITNESS_SAFE.json", dict(
            status="P297_HRSID_DGX_BC_SHORT_CANONICAL_INITIALIZATION_WITNESS_V1",
            q100_sha256=p290.Q100, q164_sha256=Q164, initialization_seed=5201, expansion_seed=5201164,
            parity_receipt_sha256=FIXED_FILES["parity_receipt"][1], position_audit=position_audit,
            query_tail_audit=tail_audit, optimizer_updates=0, model_forward_calls=0, DEV=False, TEST=False, AP=False, **provenance))
        info["expanded_online_sha256"] = Q164
        return model, train, module, state_api, info

    native.initialize = initialize


def run(cfg):
    lock = platform_lock(cfg.platform_lock, cfg.platform_lock_sha256)
    code = Path(lock["code_root"])
    qualifier, engine, plan = source_plan(code)
    require(all(lock[role] == ref for role, ref in plan["platform_sources"].items()), "Exact platform source references required")
    p293 = load_source(Path(lock["p293_source"]["path"]), lock["p293_source"]["sha256"], "_p297_p293")
    p291 = p293.pinned_p291(Path(lock["p291_source"]["path"]))
    p290 = p291.pinned_p290(Path(lock["p290_source"]["path"]))
    runtime_guard(lock, p290)
    args = arm_args(cfg, lock)
    engine.PRIVATE = Path(lock["work_root"])
    engine.RELEASE = code.parent
    engine.FIT_LAYOUT = Path(lock["fit_layout"])
    training = engine.read_training_lock(args)
    require(training["implementation_commit"] == lock["implementation_commit"]
            and training["sources"] == plan["scientific_sources"], "Exact original B/C source roster and DGX implementation required")
    engine.isolated_outputs(args.output, args.audit_output)
    bundle = p291.source_bundle(p290, code)
    paths = {key: ref_path(lock[key], pin) for key, pin in FIXED_FILES.items()}
    require(paths["coco"].name == "rf-detr-seg-nano.pt" and paths["query_tail"].name == p293.TAIL_NAME
            and paths["position"].name == "Q100_POSITION_REFERENCE_PRIVATE.pt"
            and len(set(paths.values())) == len(paths)
            and all(not path.is_relative_to(engine.FIT_LAYOUT) for path in paths.values()),
            "Exact separated COCO and canonical initialization references required")
    aux = {}
    for role, (relative, pin) in RECIPE_AUX.items():
        aux[role] = code / relative
        p290.pinned(aux[role], pin)
    sys.addaudithook(scope_guard(engine.FIT_LAYOUT / "train", {paths[k] for k in ("coco", "position", "query_tail")}))
    import torch
    gpu_guard(torch)
    verify_decode_receipt(read_json_ref(lock["fit_decode_receipt"]))
    roster = verify_fit_manifest(engine.FIT_LAYOUT, lock["fit_manifest"])
    ids = fit_ids(engine.FIT_LAYOUT, roster=roster)
    reference = read_json_ref(lock["parity_receipt"])
    tail_receipt = read_json_ref(lock["query_tail_receipt"])
    require(reference["status"] == p293.PARITY and reference["expected_canonical_q164_sha256"] == Q164
            and reference["windows_query_tail_receipt_sha256"] == FIXED_FILES["query_tail_receipt"][1],
            "Exact prior independently verified DGX canonical initialization required")
    refs = training["sources"]
    native = engine.load_source(refs[engine.NATIVE], "_p297_native")
    base = p290.load(bundle, "base", "_p297_base")
    p290.load(bundle, "selector", "select_native_fit")
    probe = p290.load(bundle, "probe", "_p297_probe")
    state = p290.load(bundle, "state", "_p297_state")
    p290.load(bundle, "p242_init", "ssdd_phase_init")
    position_loader = p290.load(bundle, "p242_position", "_p297_position_loader")
    native.COCO, native.RECIPE, native.CONFIG = paths["coco"], bundle["recipe"][0], bundle["config"][0]
    native.RNG, native.FILTER = aux["rng"], aux["filter"]
    base.LAYOUT, base.fit_ids = engine.FIT_LAYOUT, lambda: ids
    require(probe.INIT_SEED == native.SEED == 5201 and probe.EXPANSION_SEED == 5201164
            and probe.STATE_SHA == p290.SOURCE_PINS["state"][1], "Exact stock grouped query expansion required")
    base.closed_probe = lambda: SimpleNamespace(STOCK=bundle["state"][0].parent,
                                                STATE_SHA=probe.STATE_SHA, expand_groups=probe.expand_groups)
    engine.load_source(refs["trace_resize.py"], "trace_resize")
    engine.load_source(refs["phase_patch.py"], "phase_patch")
    phase = engine.load_source(refs["phase_train.py"], "_p297_phase")
    repeat = engine.load_source(refs["phase_repeat_train.py"], "_p297_repeat")
    modules = {Path(name).stem: engine.load_source(refs[name], Path(name).stem) for name in qualifier.RUNTIME_FILES}
    require(modules["one_pass_common_pair_transform"].MODES == MODES, "Exact unmodified B/C transform modes required")
    from pytorch_lightning import seed_everything
    from rfdetr.datasets import _torchvision, coco
    from rfdetr.training import module_data, trainer as trainer_api

    install_initialization(native, base, state, p290, p291, p293, position_loader, paths, reference,
                           tail_receipt, args.audit_output, phase, torch,
                           provenance=dict(mode=args.mode, training_protocol=args.protocol,
                                           implementation_commit=training["implementation_commit"],
                                           platform_lock_sha256=cfg.platform_lock_sha256,
                                           worker_sha256=lock["worker_sha256"], fit_index_sha256=INDEX,
                                           fit_manifest_sha256=MANIFEST_SHA,
                                           scientific_source_manifest_sha256=engine.manifest_digest(training)))
    engine.run_engine(args, training, None, phase=phase, repeat=repeat, native=native, base=base,
                      binding=modules["one_pass_common_binding"], module_data=module_data, trainer_api=trainer_api,
                      coco=coco, native_transforms=_torchvision, seed_everything=seed_everything, torch=torch)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--check-source", action="store_true")
    parser.add_argument("--code-root", type=Path)
    for name in ("platform-lock", "training-lock"):
        parser.add_argument("--" + name, type=Path)
        parser.add_argument("--" + name + "-sha256")
    parser.add_argument("--mode", choices=MODES)
    cfg = parser.parse_args(argv)
    inputs = (cfg.platform_lock, cfg.platform_lock_sha256, cfg.training_lock, cfg.training_lock_sha256, cfg.mode)
    if cfg.check_source:
        if cfg.code_root is None or any(value is not None for value in inputs):
            parser.error("Source check requires only --code-root")
        print(json.dumps(source_plan(cfg.code_root)[2], sort_keys=True, allow_nan=False))
        return 0
    if cfg.code_root is not None or any(value is None for value in inputs):
        parser.error("An actual qualification requires exact platform and training locks, SHA-256 values and mode")
    return run(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
