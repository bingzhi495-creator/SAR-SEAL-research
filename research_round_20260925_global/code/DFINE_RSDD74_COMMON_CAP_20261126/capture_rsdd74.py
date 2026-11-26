"""Frozen HRSID D-FINE E24 author capture on the exact 74-chip Instance-RSDD reserve."""
from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

CAPTURE_SOURCE = "11df18cb1eb0dbcff66b576d0c082b45896e5e254adb66b98fdfaff366d7391a"
LEGACY_SOURCE = "ee1db45898c42262ba263e15a14ee24c72ef1ee00d5015b5bd243daaeeee01d3"
INTAKE = "da9d92ea16e74f988d46d9855d26b530111b4c5e650fef513895c5d2e4441065"
INTAKE_TERMINAL = "b97ba047bf9422346f210b2a513be606cdc648cccdc0e8545affdd510f9ffaeb"
MANIFEST = "154d0d748b96df0d56ac249adcb8b6ae63275a09dfd7a0d7480839005e044338"
ROSTER = "2614e567e9ef1b2c791c1b357b03adfe7302ece53e911c1963c40e623ad405fc"
INTAKE_WORKER = "65b0a52d2c81535e61249c7079f65ede96f41e66ca18777ab1931d1c897c9c88"
CAPTURED = "DFINE_E24_RSDD74_NATIVE_CAPTURE_COMPLETE"
VERIFIED = "DFINE_E24_RSDD74_NATIVE_CAPTURE_ACCEPTED"
TARGET_PATH = "PSEG_TEST232_DFINE_FROZEN_CAPTURE_20261121/target_capture.py"
REFERENCE_PATH = "DFINE_HRSID_TO_SSDD_DEV187_CAPTURE_20261120"
REPLAY_PATH = "DFINE_HRSID_TO_SSDD_POINT_TRANSFER_20261120/ssdd_replay.py"
MIB = 1024**2


def need(condition, error):
    if not condition:
        raise ValueError(error)


def hex_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def pin(path, digest, limit=MIB):
    with Path(path).open("rb") as file:
        value = file.read(limit+1)
    need(len(value) <= limit and sha256(value).hexdigest() == digest, f"Exact closed input {Path(path).name}")
    return value


def closure(result, result_sha, terminal, terminal_sha, phase, worker_key, worker):
    summary = json.loads(pin(result, result_sha))
    observed = json.loads(pin(terminal, terminal_sha))
    need(observed["phase"] == phase and type(observed["actual_exit"]) is int and observed["actual_exit"] == 0
         and observed["output_footer_verified"] is True and observed["result_sha256"] == result_sha
         and observed["worker_sha256"] == summary[worker_key] == worker,
         "Observed natural exit and exact result footer are required")
    return summary


def ordered_roster(manifest):
    need(set(manifest) == {"expected_ids", "items"}, "Original image-only intake layout")
    before = manifest["items"]
    ids = [row["image_id"] for row in before]
    need(len(ids) == len(set(ids)) == 74 and ids == manifest["expected_ids"] and all(type(i) is int for i in ids),
         "All 74 original IDs in exact intake order")
    need(Counter(row["sensor"] for row in before) == {"GF-3": 42, "TerraSAR-X": 32}
         and len({row["acquisition_group"] for row in before}) == 13, "Exact original sensor and parent census")
    rows = []
    inventory = sha256()
    for row in before:
        fname = row["source_image_name"]
        ref = row["image_ref"]
        need(Path(fname).name == fname and fname.lower().endswith(".jpg") and fname.isascii()
             and row["original_hw"] == [512, 512] and set(ref) == {"path", "bytes", "sha256"}
             and ref["path"] == f"images/{fname}" and type(ref["bytes"]) is int and 0 < ref["bytes"] < 16*MIB
             and hex_digest(ref["sha256"]) and hex_digest(row["rgb_sha256"]), "Exact official native JPEG record")
        rows.append(dict(image_id=row["image_id"], original_hw=row["original_hw"],
                         image_ref=dict(ref, path=fname), rgb_sha256=row["rgb_sha256"]))
        inventory.update(f"{fname}\t{ref['bytes']}\t{ref['sha256']}\n".encode("ascii"))
    need(len({r["image_ref"]["path"] for r in rows}) == len({r["image_ref"]["sha256"] for r in rows})
         == len({r["rgb_sha256"] for r in rows}) == 74, "Unique original files and native RGB arrays")
    return rows, inventory.hexdigest()


def read_intake(a):
    result = closure(a.intake / "intake/RESULT_SAFE.json", INTAKE, a.intake / "observed_intake/TERMINAL.json",
                     INTAKE_TERMINAL, "capture", "worker_sha256", INTAKE_WORKER)
    need(result["status"] == "INSTANCE_RSDD_RESERVE74_IMAGE_INTAKE_COMPLETE_V1"
         and result["image_manifest_sha256"] == MANIFEST and result["roster_sha256"] == ROSTER
         and result["images"] == result["exact_qualified_rf_rgb"] == result["member_crc32_verified"] == 74
         and result["model_forwards"] == result["optimizer_updates"] == result["target_polygon_rows_opened"] == 0,
         "Original verified reserve image-only intake")
    return ordered_roster(json.loads(pin(a.intake / "intake/IMAGE_MANIFEST_PRIVATE.json", MANIFEST)))


def read_rgb(root, row, base):
    import numpy as np
    from PIL import Image, ImageOps

    ref = row["image_ref"]
    raw = base.pinned(root / ref["path"], ref["sha256"], 16*MIB, ref["bytes"])
    with Image.open(BytesIO(raw)) as image:
        need(image.format == "JPEG", "Original author JPEG")
        rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"), dtype=np.uint8).copy()
    need(rgb.shape == (512, 512, 3) and sha256(rgb.tobytes(order="C")).hexdigest() == row["rgb_sha256"],
         "Exact closed intake 512-square RGB bytes")
    return rgb


def source(path, expected, name, overrides=()):
    text = pin(path, expected).decode("utf-8")
    for old, new in overrides:
        need(text.count(old) == 1, "An exact original author capture source fragment is required")
        text = text.replace(old, new, 1)
    module = ModuleType(name)
    module.__file__ = str(path)
    exec(compile(text, str(path), "exec"), module.__dict__)
    return module


def export_module(code):
    projected = (
        ('"Exactly the 232 JPEGs in the image-only root"', '"Exactly the 74 original author JPEGs"'),
        ('stats["probability_files"] == 232', 'stats["probability_files"] == 74'),
        ("images=232, model_forwards=232", "images=74, model_forwards=74"),
        ("raw_queries=232*300, internal_top_rows=232*300", "raw_queries=74*300, internal_top_rows=74*300"),
        ("native_dimensions_verified=232,", "native_dimensions_verified=74,"),
        ("historical_target_exposure=True, FIT=False, DEV=False, TEST=True, GT_opened=False, AP=False, **stats",
         "historical_target_exposure=True, FIT=False, DEV=False, TEST=True, GT_opened=False, AP=False,\n"
         "        author_TEST=False, author_partition='train', prelock_commit=args.prelock_commit, "
         "intake_result_sha256=args.intake_result_sha, intake_terminal_sha256=args.intake_terminal_sha, "
         "label_files_opened=0, **stats"),
        ('event="SSDD_TEST232_NATIVE_CAPTURE_PROGRESS"', 'event="RSDD74_NATIVE_CAPTURE_PROGRESS"'),
    )
    module = source(code / TARGET_PATH, CAPTURE_SOURCE, "_frozen_dfine_rsdd74", projected)
    module.__file__ = __file__
    module.STATUS = CAPTURED
    module.read_rgb = read_rgb
    return module


def install_numpy(args):
    sys.path.insert(0, str(args.numpy_overlay))
    import numpy as np

    need(np.__version__ == "1.26.4" and Path(np.__file__).is_relative_to(args.numpy_overlay),
         "Original isolated NumPy 1.26.4 ABI overlay")


def setup(args):
    need(sys.flags.isolated and sys.dont_write_bytecode, "Use isolated frozen runtime")
    if args.phase == "source-check":
        export_module(args.code)
        source(args.code / REPLAY_PATH, LEGACY_SOURCE, "_dfine_reference_readback")
        print(json.dumps(dict(status="SOURCE_CHECK_OK", real_payload_opened=0)))
        return
    need(args.output is not None and not args.output.exists() and hex_digest(args.protocol_sha)
         and len(args.prelock_commit) == 40 and all(c in "0123456789abcdef" for c in args.prelock_commit),
         "Fresh output under verified published prelock")
    pin(args.protocol, args.protocol_sha, 64*1024)
    rows, inventory = read_intake(args)
    install_numpy(args)

    def offline(event, _):
        if event.startswith("socket."):
            raise PermissionError("Offline frozen target handling")
    sys.addaudithook(offline)
    if args.phase == "capture":
        module = export_module(args.code)
        args.manifest = args.intake / "intake/IMAGE_MANIFEST_PRIVATE.json"
        args.manifest_sha = MANIFEST
        args.test_root = (args.intake / "intake/images").resolve(strict=True)
        args.reference_dir = args.code / REFERENCE_PATH
        args.intake_result_sha = INTAKE
        args.intake_terminal_sha = INTAKE_TERMINAL
        module.INVENTORY_SHA = inventory
        module.manifest_rows = lambda _path, _digest, _base: rows
        module.run(args)
    else:
        verify_capture(args, rows, inventory)


def verify_capture(a, rows, inventory):
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "CPU-only readback")
    worker = sha256(Path(__file__).read_bytes()).hexdigest()
    report = closure(a.capture / "RESULT_SAFE.json", a.capture_sha, a.capture_terminal, a.capture_terminal_sha,
                     "capture", "export_source_sha256", worker)
    module = export_module(a.code)
    need(report["status"] == CAPTURED and report["protocol_sha256"] == a.protocol_sha
         and report["prelock_commit"] == a.prelock_commit and report["image_inventory_sha256"] == inventory
         and report["intake_result_sha256"] == INTAKE and report["intake_terminal_sha256"] == INTAKE_TERMINAL
         and report["manifest_sha256"] == MANIFEST and report["ema_sha256"] == module.EMA_SHA
         and report["e24_acceptance_sha256"] == module.ACCEPT_SHA and report["model_forwards"] == report["images"]
         == report["native_dimensions_verified"] == report["probability_files"] == 74
         and report["GT_opened"] is report["AP"] is report["author_TEST"] is False
         and report["label_files_opened"] == report["optimizer_updates"] == 0
         and report["original_author_postprocess_parity"] is True
         and report["exact_author_binary_replay"] == report["post_nms_rows"]
         and report["exact_top100_binary_replay"] == report["rows_at_100"] == report["probability_rows"],
         "Exact closed native D-FINE capture receipt")
    sys.path.insert(0, str(a.capture / "CODEC_PRIVATE"))
    legacy = source(a.code / REPLAY_PATH, LEGACY_SOURCE, "_dfine_reference_readback")
    author = module.load_source(a.code / REFERENCE_PATH, "author_export_reference.py", module.BASE_SHA)
    author.cpu_runtime()
    expected = {f"{ordinal:04d}.npy" for ordinal in range(1, 75)}
    need({path.name for path in (a.capture / "PROBABILITY_PRIVATE").iterdir()} == expected,
         "Exact native sidecar inventory")
    count = slots = empty = 0
    for _, native, audit, planes in legacy.blocks(SimpleNamespace(capture=a.capture), author, rows, report, True):
        need(len(native) == len(audit) == len(planes), "Exact candidate audit and immutable probability plane alignment")
        count += 1
        slots += len(native)
        empty += sum(candidate["mask_area"] == 0 for candidate in audit)
    need(count == 74 and slots == report["rows_at_100"] and empty == report["empty_at_100"],
         "Complete independent native readback counts")
    flags = ("manifest_sha256", "image_inventory_sha256", "intake_result_sha256", "intake_terminal_sha256",
             "ema_sha256", "e24_acceptance_sha256", "predictions_sha256", "candidate_audit_sha256",
             "probability_index_sha256", "probability_rows", "rows_at_100", "post_nms_rows", "probability_files",
             "empty_at_100", "original_author_postprocess_parity", "author_TEST", "author_partition", "GT_opened", "AP")
    result = {key: report[key] for key in flags}
    result.update(status=VERIFIED, protocol_sha256=a.protocol_sha, prelock_commit=a.prelock_commit,
                  worker_sha256=worker, capture_result_sha256=a.capture_sha,
                  capture_terminal_sha256=a.capture_terminal_sha, images=count, model_forwards=0,
                  independent_probability_and_candidate_census=True, exact_top100_slot_population=(slots == 7400),
                  label_files_opened=0, optimizer_updates=0)
    a.output.mkdir()
    digest = author.write_new(a.output / "RESULT_SAFE.json", result)
    print(json.dumps(dict(status=VERIFIED, result_sha256=digest)), flush=True)


def main():
    p = ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("phase", choices=("source-check", "capture", "verify-capture"))
    p.add_argument("--code", type=Path, required=True)
    for key in ("intake", "protocol", "output", "acceptance", "author-source", "author-tree", "ema", "codec-wheel",
                "numpy-overlay", "capture", "capture-terminal"):
        p.add_argument("--"+key, type=Path)
    for key in ("protocol-sha", "prelock-commit", "acceptance-sha", "capture-sha", "capture-terminal-sha"):
        p.add_argument("--"+key)
    args = p.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve(strict=key != "output"))
    setup(args)


if __name__ == "__main__":
    main()
