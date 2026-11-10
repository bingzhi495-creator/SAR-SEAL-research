"""Check unchanged official COCO Q100 and native HRSID Q164 initialization on DGX CPU."""
from argparse import ArgumentParser
from hashlib import sha256
from importlib.metadata import version
from importlib.util import find_spec
import json
import os
from pathlib import Path
import platform
import re
import sys
from types import ModuleType, SimpleNamespace


COCO_SHA = "a44613a4ecd6b5ba61a62002c600b0b6cb7a9da2936a45317ec4b62c635fb99b"
Q100 = "40524fc63797d2fcab7b50b6ee1f041f9429df73e505cb9e647a5a8cdf6f2e2f"
Q164 = "63ff93afdd977ba79b5b5a46ba8b4839ef6c40eb0722b29e4667d4932404fbc2"
P242_RAW_Q100 = "7b270a9889ac56b815442309e42ef9a7f7937c9ab28df5042cda7f73e384b9cd"
P242_RAW_POSITION = "f5bef45a496862ad91550217654f25e83a76a62a7f42a8dc200010ca9b7966ed"
WINDOWS_POSITION = "5ccc0fd712422fb88f720506835142abc77b98ae97252473cc17e48711ca7aae"
POSITION = "backbone.0.encoder.encoder.embeddings.position_embeddings"
QUERIES = ("refpoint_embed.weight", "query_feat.weight")
PARITY = "HRSID_DGX_Q100_Q164_COCO_INIT_PARITY_V1"
MISMATCH = "HRSID_DGX_Q100_Q164_COCO_INIT_MISMATCH_V1"
SOURCE_PINS = {
    "native": ("code/HRSID_NATIVE_EMPTY_MASK_FILTER_20261109/train_hrsid_coco_native_e24_filtered.py",
               "11d17cab878d62576b8d19e6231caab90c9990c03909cb01e1d04468add452a6"),
    "base": ("code/HRSID_NATIVE_FIT_E1_20261109/train_hrsid_native_e1.py",
             "23bf7b432763f86f57322b3c1e09cf7966fa57c80e0ce74e9185b8977a414362"),
    "probe": ("code/HRSID_REAL_FIT_NATIVE_MATCH_20261109_NUMERIC/probe_native_match.py",
              "fbc793f8df253c9f340ffa671cb502c0a9a770facce126f15ad7d15de5b416f0"),
    "selector": ("code/HRSID_REAL_FIT_NATIVE_MATCH_20261109_NUMERIC/select_native_fit.py",
                 "59e6129f8f10ccfb0e83fe816f6b41c3b5cd9bdeccfae6ebb7cb719c9c4df34f"),
    "state": ("code/RFDETR_E24_OPTIMIZER_PRESERVATION_20261108/state_preservation.py",
              "8dda41131a2930aacf00c51dbe8cfa4817c0a3cd4b02c6bb606948db77341678"),
    "config": ("RFDETR_REAL_SSDD_DATA_QUALIFICATION_DRAFT_20261107_01/entry.py",
               "f0642e8f33e75a2440eac0968dd76f270d445498ba88e812543902e7d0ad8453"),
    "recipe": ("configs/RFDETR_SEG_NANO_BASELINE_576_E100_20261107.yaml",
               "ba9508d9acab5ff68b10db8a1aff5d9001868d6dda00e39f299d6f5cc1b8011b"),
}
PACKAGES = {
    "torch": "2.10.0+cu130", "torchvision": "0.25.0+cu130", "rfdetr": "1.12.0",
    "pytorch-lightning": "2.6.6", "numpy": "1.26.4", "Pillow": "12.3.0",
    "pycocotools": "2.0.11", "faster-coco-eval": "1.8.0", "psutil": "6.1.1",
    "pydantic": "2.14.3", "supervision": "0.31.5",
}
RFDETR_PINS = {
    "models/lwdetr.py": "248faf15d673bc18f51f60bb55245f64eee40eb9c8adacea398aebdadad2243b",
    "models/weights.py": "162a354b54d8e73ccdac77960bea1d9db46ade68847f1eaed289f541b34e0c14",
    "training/callbacks/ema.py": "41ab4c516a5a11b8e239f2173b384064d8a02774bd042e0a99840cca228afbda",
    "training/module_model.py": "3ce535eee30dbfed2493e22581e5a9c71fdc77a915dbf8e633d966258fa6ea8f",
    "training/trainer.py": "491b925e695668382a8e645afbdc63e00d2fdca3fc912857c61cebbff25f5f04",
}
EXTENSIONS = {
    "PIL": ("_imaging.*.so", "2bdbf62e38af4c59c6563b9391f11d411e42f5ced5dc0991aaf31bdda72928a8"),
    "pycocotools": ("_mask.*.so", "53d99dd5a70f01aed23bcbfa59aa4f82f0dc68c48cbc4a676a9d5df9e31f1597"),
}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def pinned(path, expected, maximum=256 * 1024):
    require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file(), "Exact canonical pinned source")
    with path.open("rb") as stream:
        first = os.fstat(stream.fileno())
        raw = stream.read(maximum + 1)
        last = os.fstat(stream.fileno())
    token = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
    require(0 < len(raw) == first.st_size <= maximum and token(first) == token(last) == token(path.stat())
            and sha256(raw).hexdigest() == expected, "Exact stable source SHA-256 required: " + path.name)
    return raw


def source_bundle(code):
    require(code.is_absolute() and code.resolve(strict=True) == code and code.name == "code", "Canonical research code root required")
    return {role: (code.parent / relative, pinned(code.parent / relative, digest))
            for role, (relative, digest) in SOURCE_PINS.items()}


def load(bundle, role, alias):
    path, raw = bundle[role]
    require(alias not in sys.modules, "Fresh isolated scientific source namespace required")
    result = ModuleType(alias)
    result.__file__ = str(path)
    sys.modules[alias] = result
    exec(compile(raw, str(path), "exec"), result.__dict__)
    return result


def package_root(name):
    spec = find_spec(name)
    paths = tuple(spec.submodule_search_locations)
    require(len(paths) == 1, "Exact installed package root required: " + name)
    return Path(paths[0]).resolve(strict=True)


def admit_runtime():
    require(platform.system() == "Linux" and platform.machine() == "aarch64" and sys.version_info[:3] == (3, 12, 14)
            and sys.flags.isolated == sys.flags.utf8_mode == 1 and sys.dont_write_bytecode
            and os.sched_getaffinity(0) == {0, 1} and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
            and all(os.environ.get(key) == "1" for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")),
            "Exact isolated offline DGX ARM CPU runtime and affinity required")
    require({key: version(key) for key in PACKAGES} == PACKAGES, "Exact existing DGX runtime package set required")
    rf = package_root("rfdetr")
    for relative, digest in RFDETR_PINS.items():
        pinned(rf / relative, digest, 1024**2)
    for name, (glob, digest) in EXTENSIONS.items():
        files = tuple(package_root(name).glob(glob))
        require(len(files) == 1, "Exact existing DGX compiled codec required")
        pinned(files[0], digest, 16 * 1024**2)
    import psutil
    import torch

    require(psutil.virtual_memory().available >= 40 * 1024**3 and torch.__version__ == PACKAGES["torch"]
            and not torch.cuda.is_available() and not torch.cuda.is_initialized(),
            "Exact DGX CPU-only Torch and at least 40 GiB available host memory required")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    return torch


def scope_guard(coco):
    blocked_suffixes = {".pt", ".pth", ".ckpt", ".safetensors", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".zip"}

    def guard(event, args):
        require(event not in ("socket.connect", "socket.getaddrinfo", "subprocess.Popen", "os.system"),
                "Offline and in-process initialization only")
        if event != "open" or not isinstance(args[0], (str, bytes)):
            return
        path = Path(os.path.abspath(os.fsdecode(args[0])))
        if path.suffix.lower() not in blocked_suffixes and path.name != "_annotations.coco.json":
            return
        mode, flags = args[1:3]
        writing = (isinstance(mode, str) and any(ch in mode for ch in "wax+")
                   or isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
        require(path == coco and path.resolve(strict=True) == path and not writing,
                "Only the admitted read-only official COCO asset may be opened")

    return guard


def observe(model, width, torch, digest):
    state = model.state_dict()
    require(len(state) == 545 and all(value.device.type == "cpu" for value in state.values()),
            "Expected 545 stock CPU model tensors")
    position = state[POSITION]
    require(tuple(position.shape) == (1, 2305, 384) and position.dtype == torch.float32
            and bool(torch.isfinite(position).all()), "Expected exact finite RF-DETR position tensor")
    queries = {}
    for key in QUERIES:
        value = state[key]
        require(value.ndim == 2 and value.shape[0] == 13 * width and value.dtype == torch.float32
                and bool(torch.isfinite(value).all()), "Expected finite grouped CPU query table")
        grouped = value.detach().reshape(13, width, -1)
        queries[key] = dict(shape=list(value.shape), sha256=digest({key: value}),
                            group_prefix_sha256=digest({key: grouped[:, :100].contiguous()}),
                            group_tail_sha256=(digest({key: grouped[:, 100:].contiguous()}) if width == 164 else None))
    return dict(tensors=len(state), position_sha256=digest({POSITION: position}), queries=queries)


def result_from(info, observations):
    fixed = dict(native_original_width=100, query_width=164, query_groups=13, class_head_slots=2,
                 coco_asset_sha256=COCO_SHA, coco_asset_bytes=134545398, initialized_from_SSDD=False)
    require(all(type(info.get(key)) is type(value) and info[key] == value for key, value in fixed.items())
            and set(observations) == {"q100", "q164"} and all(row["tensors"] == 545 for row in observations.values()),
            "Exact original Q100/Q164 COCO-only initialization observation required")
    raw, expanded = (info[key] for key in ("width100_loaded_sha256", "expanded_online_sha256"))
    require(all(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) for value in (raw, expanded)),
            "Expected exact native full-state digests")
    q100 = observations["q100"]
    both = raw == Q100 and expanded == Q164
    require(not both or q100["position_sha256"] == observations["q164"]["position_sha256"] == WINDOWS_POSITION,
            "Full parity must retain the canonical position tensor")
    return dict(status=PARITY if both else MISMATCH, expected_q100_sha256=Q100, expected_q164_sha256=Q164,
                observed_q100_sha256=raw, observed_q164_sha256=expanded,
                expected_windows_position_sha256=WINDOWS_POSITION, expected_p242_raw_position_sha256=P242_RAW_POSITION,
                matches_prior_p242_uncorrected_q100_fingerprint=(raw == P242_RAW_Q100
                                                                  and q100["position_sha256"] == P242_RAW_POSITION),
                initialization_seed=5201, expansion_seed=5201164, input_resolution=576, num_select=100,
                diagnostic=observations, cuda_initialized=False, FIT=False, DEV=False, TEST=False, AP=False,
                model_forward_calls=0, optimizer_updates=0, reference_tensor_substitution=False, **fixed)


def original_initialize(bundle, coco, sentinel, torch):
    native = load(bundle, "native", "_p290_exact_native")
    base = load(bundle, "base", "_p290_exact_base")
    load(bundle, "selector", "select_native_fit")
    probe = load(bundle, "probe", "_p290_exact_probe")
    state = load(bundle, "state", "_p290_exact_state_digest")
    from pytorch_lightning import seed_everything
    from rfdetr.training.module_model import RFDETRModelModule

    native.COCO, native.RECIPE, native.CONFIG = coco, bundle["recipe"][0], bundle["config"][0]
    base.LAYOUT = sentinel
    require(native.SEED == probe.INIT_SEED == 5201 and probe.EXPANSION_SEED == 5201164
            and native.WIDTH == probe.WIDTH_LARGE == 164 and probe.GROUPS == 13
            and probe.STATE_SHA == SOURCE_PINS["state"][1] and native.CONFIG_SHA == SOURCE_PINS["config"][1]
            and native.RECIPE_SHA == SOURCE_PINS["recipe"][1], "Exact original initialization source lineage required")
    observations = {}

    def expand_once(model, width):
        require(not observations and width == 164, "Exactly one original grouped CPU expansion required")
        observations["q100"] = observe(model, 100, torch, state.tensor_dict_sha256)
        probe.expand_groups(model, width)
        observations["q164"] = observe(model, width, torch, state.tensor_dict_sha256)
        for key in QUERIES:
            require(observations["q100"]["queries"][key]["group_prefix_sha256"]
                    == observations["q164"]["queries"][key]["group_prefix_sha256"],
                    "Original 100 per-group query prefix changed")

    base.closed_probe = lambda: SimpleNamespace(STOCK=bundle["state"][0].parent,
                                                STATE_SHA=probe.STATE_SHA, expand_groups=expand_once)
    seed_everything(native.SEED, workers=True)
    model, _, module, _, info = native.initialize(base, sentinel, RFDETRModelModule)
    require(model.resolution == 576 and model.num_queries == 164 and model.num_select == 100
            and state.tensor_dict_sha256(module.model.state_dict()) == info["expanded_online_sha256"]
            and observe(module.model, 164, torch, state.tensor_dict_sha256) == observations["q164"]
            and not torch.cuda.is_initialized() and not sentinel.exists(),
            "Exact strict expanded CPU endpoint and unused FIT sentinel required")
    return result_from(info, observations)


def main(argv=None):
    parser = ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--check-source", action="store_true")
    parser.add_argument("--coco", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    bundle = source_bundle(args.code_root)
    if args.check_source:
        if args.coco is not None or args.output is not None:
            parser.error("--check-source takes no input asset")
        for path, raw in bundle.values():
            if path.suffix == ".py":
                compile(raw, str(path), "exec")
        print("P290_HRSID_DGX_Q164_SOURCE_CHECK_PASSED", flush=True)
        return 0
    coco, output = args.coco, args.output
    if (coco is None or output is None or not coco.is_absolute() or not output.is_absolute()
            or coco.name != "rf-detr-seg-nano.pt" or output.suffix != ".json"):
        parser.error("Exact canonical local official COCO file and fresh safe JSON output required")
    require(coco.resolve(strict=True) == coco and coco.is_file() and output.parent.resolve(strict=True) == output.parent
            and not output.exists() and not output.is_symlink(), "Canonical isolated input/output paths required")
    sentinel = output.parent / "UNUSED_HRSID_Q164_FIT_LAYOUT"
    require(not sentinel.exists() and not sentinel.is_symlink(), "FIT sentinel must not exist")
    sys.addaudithook(scope_guard(coco))
    torch = admit_runtime()
    value = original_initialize(bundle, coco, sentinel, torch)
    value.update(source_sha256={role: digest for role, (_, digest) in SOURCE_PINS.items()}, runtime=PACKAGES,
                 runtime_rf_detr_source_sha256=RFDETR_PINS, cpu_affinity=[0, 1], cpu_threads=2)
    raw = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    with output.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    print(raw.decode("utf-8").rstrip("\n"), flush=True)
    return 0 if value["status"] == PARITY else 4


if __name__ == "__main__":
    raise SystemExit(main())

