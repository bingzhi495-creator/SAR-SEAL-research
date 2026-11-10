"""Run one original image-only export or one paired CPU score for a closed SSDD DEV187 request."""

from argparse import ArgumentParser
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import sys
from types import ModuleType


MIB = 1024**2
REQUEST_STATUS = "P284_SSDD_DEV187_STAGE_REQUEST_V1"
REPORT_EVENT = "P284_SSDD_DEV187_STAGE_REPORT_WRITTEN_V1"
EXPORT_REPORT = "EXPORT_REPORT_PRIVATE.json"
SCORE_REPORT = "SCORE_SAFE.json"
ARTIFACTS = "ARTIFACTS"
ARMS = ("native", "center")
EXPORTS = {"EXPORT_NATIVE": "native", "EXPORT_CENTER": "center"}
STAGES = (*EXPORTS, "SCORE")
IMAGE_MANIFEST_SHA = "18df007e15e3e838e607f178f14e7a26f2b37b992ee39bf091e0b74cf084fb13"
INTEGER_GT_SHA = "5657600d6ea198f79bd5d557d2a1aadce5bec031d2ecf72a98716248d1183a55"
INTEGER_GT_BYTES = 118319
SOURCE_DIR = "SSDD_MASK_PHASE_TRANSFER_20261110"
PINS = {
    "ssdd_phase_init": "444e91009bce7a56e65a9bd2110c3479a86609d29b5ccdc925b06ec013b7894a",
    "ssdd_phase_dev_admission": "6b5ffa17ecb06ace03038193415261405d2c90b85ebaed362223c88bdf86585b",
    "ssdd_phase_dev_score": "483e6251184b258475b97fd664aa76c784d599ce805cdd9d8db2f5e06a7e1696",
}
COMMON = {"status", "stage", "training_protocol", "evaluation_protocol", "training_pair_ref",
          "native_arm_ref", "center_arm_ref", "image_manifest_ref", "output_directory"}
CLOSURE = dict(original_os_wait=True, original_os_exit=0, resource_stop=False, worker_closed=True)


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def hex_digest(value, size=64):
    return type(value) is str and re.fullmatch(f"[0-9a-f]{{{size}}}", value) is not None


def keys(document, expected, reason):
    require(type(document) is dict and document.keys() == expected, reason)


def unique(pairs):
    result = dict(pairs)
    require(len(result) == len(pairs), "Duplicate request fields are forbidden")
    return result


def reject_constant(_):
    raise ValueError("Nonfinite request fields are forbidden")


def path_value(raw):
    require(type(raw) is str and raw and all(ord(c) >= 32 for c in raw), "Canonical absolute path required")
    path = Path(raw)
    require(path.is_absolute() and ".." not in path.parts and raw in (str(path), path.as_posix()),
            "Canonical absolute path required")
    require(not any(re.fullmatch(r"(?:official[-_])?(?:test|testing)(?:[._-].*)?", part, re.IGNORECASE)
                    for part in path.parts), "Reserved TEST paths are outside the SSDD DEV187 stage")
    return path


def ref_path(ref, maximum, *, name=None):
    keys(ref, {"path", "bytes", "sha256"}, "Exact file reference required")
    path = path_value(ref["path"])
    require(type(ref["bytes"]) is int and 0 < ref["bytes"] <= maximum and hex_digest(ref["sha256"])
            and (name is None or path.name == name), "Exact bounded file reference required")
    return path


def validate_arm_request(arm):
    keys(arm, {"closure", "report_ref", "predictions_ref", "export_manifest_ref", "checkpoint_ref"},
         "Exact independently closed SSDD export roles required")
    closure = arm["closure"]
    keys(closure, CLOSURE.keys() | {"terminal_sha256", "report_sha256"}, "Exact SSDD export closure fields required")
    require(all(type(closure[key]) is type(value) and closure[key] == value for key, value in CLOSURE.items())
            and hex_digest(closure["terminal_sha256"]) and hex_digest(closure["report_sha256"]),
            "Original SSDD export OS wait and worker closure required")
    report = ref_path(arm["report_ref"], MIB, name=EXPORT_REPORT)
    predictions = ref_path(arm["predictions_ref"], 64 * MIB, name="PREDICTIONS.json")
    manifest = ref_path(arm["export_manifest_ref"], 4 * MIB, name="IMAGE_MANIFEST.json")
    checkpoint = ref_path(arm["checkpoint_ref"], 1024 * MIB, name="checkpoint_23.ckpt")
    require(closure["report_sha256"] == arm["report_ref"]["sha256"]
            and predictions.parent == manifest.parent == report.parent / ARTIFACTS,
            "Original export closure and exact report/artifact layout required")
    return report, predictions, manifest, checkpoint


def read_request(path, expected, stage):
    require(stage in STAGES and hex_digest(expected), "Exact SSDD stage and owner request SHA-256 required")
    path_value(path.as_posix())
    require(path.resolve(strict=True) == path and path.is_file(), "Canonical SSDD stage request required")
    with path.open("rb") as stream:
        raw = stream.read(MIB + 1)
    require(0 < len(raw) <= MIB and sha256(raw).hexdigest() == expected, "Exact SSDD stage request SHA-256 mismatch")
    request = json.loads(raw, object_pairs_hook=unique, parse_constant=reject_constant)
    extras = {"archive_ref", "dev_root"} if stage in EXPORTS else {"integer_gt_ref", *ARMS}
    keys(request, COMMON | extras, "Owner request must have the exact stage scope")
    require(request["status"] == REQUEST_STATUS and request["stage"] == stage
            and all(hex_digest(request[field], 40) for field in ("training_protocol", "evaluation_protocol")),
            "Owner request must bind the exact stage and protocols")
    refs = tuple(ref_path(request[key], MIB) for key in
                 ("training_pair_ref", "native_arm_ref", "center_arm_ref", "image_manifest_ref"))
    require(len(set(refs)) == len(refs) and request["image_manifest_ref"]["sha256"] == IMAGE_MANIFEST_SHA,
            "Exact separate accepted receipts and original DEV187 manifest required")
    out = path_value(request["output_directory"])
    require(out.resolve() == out and out.parent.is_dir() and not out.exists() and not out.is_symlink(),
            "Fresh canonical stage output directory required")
    if stage in EXPORTS:
        archive = ref_path(request["archive_ref"], 1024 * MIB, name="checkpoint_23.ckpt")
        dev_root = path_value(request["dev_root"])
        require(not out.is_relative_to(dev_root) and not dev_root.is_relative_to(out)
                and not archive.is_relative_to(out), "Separate DEV images, E24 archive and export destination required")
    else:
        gt = request["integer_gt_ref"]
        gt_path = ref_path(gt, MIB)
        require(gt["sha256"] == INTEGER_GT_SHA and gt["bytes"] == INTEGER_GT_BYTES,
                "Exact original DEV187 integer GT reference required")
        inputs = tuple(path for mode in ARMS for path in validate_arm_request(request[mode]))
        require(len(set((*refs, gt_path, *inputs))) == len(refs) + 1 + len(inputs)
                and request["native"]["closure"]["terminal_sha256"] != request["center"]["closure"]["terminal_sha256"],
                "Distinct closed export processes, arms and score inputs required")
    return request


def load_published(code_root, *, scoring):
    require(code_root.is_absolute() and code_root.resolve(strict=True) == code_root and code_root.is_dir(),
            "Canonical published code root required")
    previous, loaded = {}, {}
    selected = PINS if scoring else {key: PINS[key] for key in ("ssdd_phase_init", "ssdd_phase_dev_admission")}
    try:
        for name, digest in selected.items():
            path = code_root / SOURCE_DIR / (name + ".py")
            require(path.resolve(strict=True) == path and path.is_file(), "Canonical published stage source required")
            with path.open("rb") as stream:
                raw = stream.read(128 * 1024 + 1)
            require(0 < len(raw) <= 128 * 1024 and sha256(raw).hexdigest() == digest,
                    "Exact published SSDD source required")
            previous[name] = sys.modules.get(name)
            module = ModuleType(name)
            module.__file__ = str(path)
            sys.modules[name] = loaded[name] = module
            exec(compile(raw, str(path), "exec"), module.__dict__)
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    return loaded


def admit_runtime(stage):
    device = "0" if stage in EXPORTS else "-1"
    require(sys.version_info[:3] == (3, 10, 13) and sys.flags.isolated == sys.flags.utf8_mode == 1
            and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES") == device
            and os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1",
            "Exact isolated offline SSDD worker Python and stage device required")
    require(version("numpy") == "1.26.4" and version("pycocotools") == "2.0.11",
            "Exact original DEV187 numeric and COCO runtime required")


def reject_network(event, _):
    if event in ("socket.connect", "socket.getaddrinfo"):
        raise PermissionError("Offline SSDD DEV187 stage required")


def scope_guard(*, allowed, images=(), data_root=None):
    """Guard known data/weights from other opens and keep exact JPEG read coverage."""
    allowed, images = set(allowed), set(images)
    protected_suffixes = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".ckpt", ".pt", ".pth", ".safetensors"}
    protected_names = {"_annotations.coco.json"}
    seen = set()

    def hook(event, args):
        if event != "open" or not isinstance(args[0], (str, bytes)):
            return
        path = Path(os.path.abspath(os.fsdecode(args[0])))
        protected = (path in allowed or path.suffix.lower() in protected_suffixes
                     or path.name.lower() in protected_names
                     or (data_root is not None and path.is_relative_to(data_root)))
        if not protected:
            return
        mode, flags = args[1:3]
        writing = (isinstance(mode, str) and any(char in mode for char in "wax+")
                   or isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT
                                                          | os.O_TRUNC | os.O_APPEND))
        if writing or path not in allowed or path.resolve(strict=True) != path:
            raise PermissionError("SSD DEV187 attempted to open data outside its exact read-only stage roles")
        if path in images:
            seen.add(path)

    return hook, seen


def load_recipe(init, dev_root):
    init.pinned_bytes(init.RECIPE, init.RECIPE_SHA)
    entry = init.load_api(init.CONFIG, init.CONFIG_SHA)
    model, train = entry._load_configs(init.RECIPE, dev_root)
    from rfdetr import config

    init.load_api(init.E1, init.E1_SHA).native_recipe(model, train, config)
    return model, train


def configure_export():
    import torch

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
            "Exactly one preadmitted CUDA export device required")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False


def read_artifact(ref, maximum):
    path = ref_path(ref, maximum)
    require(path.resolve(strict=True) == path and path.is_file(), "Canonical admitted artifact required")
    with path.open("rb") as stream:
        raw = stream.read(ref["bytes"] + 1)
    require(len(raw) == ref["bytes"] and sha256(raw).hexdigest() == ref["sha256"],
            "Exact admitted artifact byte length and SHA-256 required")
    return raw


def train_args(request):
    return dict(native_arm_ref=request["native_arm_ref"], center_arm_ref=request["center_arm_ref"],
                training_protocol=request["training_protocol"], evaluation_protocol=request["evaluation_protocol"])


def run_export(request, api, init, *, event_sink):
    mode = EXPORTS[request["stage"]]
    pair_ref = request["training_pair_ref"]
    accepted = train_args(request)
    _, arms = api.read_pair(pair_ref, **accepted)
    arm = arms[mode]
    archive_ref = request["archive_ref"]
    archive = api.reference(archive_ref, max_bytes=1024 * MIB, suffix=".ckpt")
    require(archive_ref["sha256"] == arm["checkpoint_sha256"] and archive_ref["bytes"] == arm["checkpoint_bytes"],
            "Only the accepted checkpoint for the selected export role is admitted")
    dev_root = path_value(request["dev_root"])
    image_manifest_ref = request["image_manifest_ref"]
    items, _ = api.load_image_manifest(image_manifest_ref, dev_root)
    image_paths = {Path(item["image_ref"]["path"]) for item in items}
    input_refs = (pair_ref, accepted["native_arm_ref"], accepted["center_arm_ref"], image_manifest_ref)
    allowed = {*image_paths, archive, *(Path(ref["path"]) for ref in input_refs)}
    guard, seen = scope_guard(allowed=allowed, images=image_paths, data_root=dev_root)
    sys.addaudithook(guard)
    model, train = load_recipe(init, dev_root)
    configure_export()
    output = path_value(request["output_directory"])
    output.mkdir()
    artifact_dir = output / ARTIFACTS
    result = api.export_arm(
        pair_ref, **accepted, mode=mode, archive_ref=archive_ref, dev_manifest_ref=image_manifest_ref,
        dev_root=dev_root, model_config=model, train_config=train, output_directory=artifact_dir,
        event_sink=event_sink,
    )
    info = result["result"]
    require(result["status"] == api.EXPORT_STATUS and result["mode"] == mode
            and info["GT_read"] is info["TEST"] is info["AP"] is info["training"] is False
            and info["float32_matmul_precision"] == "highest" and info["cudnn_tf32"] is False
            and len(image_paths) == 187 and seen == image_paths,
            "Exact original image-only FP32 DEV187 export required")
    for role, name, maximum in (("predictions_ref", "PREDICTIONS.json", 64 * MIB),
                                ("image_manifest_ref", "IMAGE_MANIFEST.json", 4 * MIB)):
        require(ref_path(info[role], maximum, name=name) == artifact_dir / name,
                "Exact original DEV187 export output path and bound required")
        read_artifact(info[role], maximum)
    require(set(artifact_dir.iterdir()) == {artifact_dir / "PREDICTIONS.json", artifact_dir / "IMAGE_MANIFEST.json"},
            "Exact two original DEV187 inference artifacts required")
    return result


def run_score(request, api, scorer):
    pair_ref = request["training_pair_ref"]
    accepted = train_args(request)
    _, trained = api.read_pair(pair_ref, **accepted)
    for mode in ARMS:
        ref = request[mode]["checkpoint_ref"]
        arm = trained[mode]
        require(ref["sha256"] == arm["checkpoint_sha256"] and ref["bytes"] == arm["checkpoint_bytes"],
                "Closed score inputs must retain both independently accepted E24 checkpoints")
    sys.addaudithook(scope_guard(allowed=())[0])

    def artifact(ref, maximum):
        return scorer.Artifact(ref, read_artifact(ref, maximum))

    arms = {}
    for mode in ARMS:
        arm = request[mode]
        arms[mode] = scorer.ClosedExport(
            arm["closure"], artifact(arm["report_ref"], MIB), artifact(arm["predictions_ref"], 64 * MIB),
            artifact(arm["export_manifest_ref"], 4 * MIB), arm["checkpoint_ref"],
        )
    return scorer.score_closed_pair(
        **arms, training_pair=artifact(pair_ref, MIB),
        image_manifest=artifact(request["image_manifest_ref"], MIB),
        integer_gt=artifact(request["integer_gt_ref"], MIB),
        training_protocol=accepted["training_protocol"], evaluation_protocol=accepted["evaluation_protocol"],
    )


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def print_event(value):
    print(json_bytes(value).decode("utf-8"), end="", flush=True)


def main(argv=None):
    parser = ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    args = parser.parse_args(argv)
    request = read_request(args.request, args.request_sha256, args.stage)
    admit_runtime(args.stage)
    sys.addaudithook(reject_network)
    modules = load_published(args.code_root, scoring=args.stage == "SCORE")
    api = modules["ssdd_phase_dev_admission"]
    output = path_value(request["output_directory"])
    if args.stage in EXPORTS:
        result = run_export(request, api, modules["ssdd_phase_init"], event_sink=print_event)
        report = output / EXPORT_REPORT
    else:
        result = run_score(request, api, modules["ssdd_phase_dev_score"])
        output.mkdir()
        report = output / SCORE_REPORT
    payload = json_bytes(result)
    with report.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    print_event(dict(status=REPORT_EVENT, stage=args.stage, request_sha256=args.request_sha256,
                     report_sha256=sha256(payload).hexdigest()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
