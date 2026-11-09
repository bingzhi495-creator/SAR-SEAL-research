"""Admit independently closed SSDD Q100 E24 pairs to fixed DEV187 image-only export."""

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from types import ModuleType, SimpleNamespace


CODE = Path(__file__).resolve().parent.parent
SOURCES = {
    "worker": ("RFDETR_NATIVE_E24_DEV_EXPORT_WORKER_CANDIDATE_20261107/e24_dev_export_worker.py",
               "7bf88b0880a360f52de9263b215bfda30d34ab14d91183dc33a13a13b8aa26f5"),
    "long": ("RFDETR_NATIVE_UNINTERRUPTED_E100_WORKER_CANDIDATE_20261107/e100_worker.py",
             "ecc8c67bf67e4e58d0d697bf4ac48b9ff4a339ff1d162019ce95b73a0fed0144"),
    "boundary": ("RFDETR_NATIVE_E1_CHECKPOINT_API_RECOVERY_20261107/e1_resume.py",
                 "014e087379ffe6e8d32329c337d6b625b98dfc2d235cb321becebfdf3832b013"),
    "pilot": ("RFDETR_NATIVE_E1_EMA_PILOT_WORKER_CANDIDATE_20261107/ema_pilot_worker.py",
              "e9fd227e098742e5406833877d13a91dced5bf64a12b0a338c98ce24d91c594f"),
    "raw_export": ("RFDETR_NATIVE_E1_EXPORT_CANDIDATE_20261107/e1_export.py",
                   "4e1fe8e825f31152380ee4191adb92987a828669eb14189058622ac2d22c4039"),
    "canonical": ("RFDETR_CANONICAL_EXPORT_DRAFT_20261107_01/canonical_export.py",
                  "b2e99e581acb0236f6cee5542321bb3e01b2c620f2c727062b28b71eedfd1820"),
}
RECIPE_SHA = "ba9508d9acab5ff68b10db8a1aff5d9001868d6dda00e39f299d6f5cc1b8011b"
Q100_SHA = "40524fc63797d2fcab7b50b6ee1f041f9429df73e505cb9e647a5a8cdf6f2e2f"
COCO_SHA = "a44613a4ecd6b5ba61a62002c600b0b6cb7a9da2936a45317ec4b62c635fb99b"
IMAGE_MANIFEST_SHA = "18df007e15e3e838e607f178f14e7a26f2b37b992ee39bf091e0b74cf084fb13"
PAIR_STATUS = "SSDD_MASK_PHASE_E24_PAIR_INDEPENDENTLY_ACCEPTED_V1"
ARM_STATUS = "SSDD_MASK_PHASE_E24_ARM_INDEPENDENTLY_ACCEPTED_V1"
EXPORT_STATUS = "SSDD_MASK_PHASE_DEV187_IMAGE_ONLY_EXPORT_PENDING_INDEPENDENT_ACCEPTANCE_V1"
PHASE_MARKER = "sar_seal_ssdd_mask_resize_phase_v1"
ARCHIVE_NAME = "checkpoint_23.ckpt"
MIB = 1024**2
ARMS = ("native", "center")
SEED, EPOCHS, VISITS, UPDATES, DEV_IMAGES, Q100 = 5201, 24, 17856, 4464, 187, 100
FIT_ONLY = dict(FIT=True, DEV=False, TEST=False, AP=False)
ARM_DIGESTS = ("result_sha256", "epoch_audit_sha256", "visit_ledger_sha256", "stdout_sha256", "terminal_sha256")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def is_hex(value, width=64):
    return type(value) is str and re.fullmatch(f"[0-9a-f]{{{width}}}", value) is not None


def matches(value, expected):
    return type(value) is dict and all(type(value.get(key)) is type(item) and value[key] == item
                                       for key, item in expected.items())


def exact(value, expected, reason, *, extra=()):
    require(matches(value, expected) and value.keys() == expected.keys() | set(extra), reason)


def reference(ref, *, max_bytes, suffix):
    """Use the original pinned-file contract for canonical paths on either platform."""
    exact(ref, {}, "Exact original file reference required", extra=("path", "bytes", "sha256"))
    raw = ref["path"]
    require(type(raw) is str and raw and not any(ord(char) < 32 for char in raw)
            and type(ref["bytes"]) is int and 0 < ref["bytes"] <= max_bytes and is_hex(ref["sha256"]),
            "Exact bounded original file reference required")
    path = Path(raw)
    require(path.is_absolute() and raw in (str(path), path.as_posix()) and ".." not in path.parts
            and path.suffix == suffix and path.resolve(strict=True) == path and path.is_file(),
            "Canonical original file required")
    return path


def _unique(pairs):
    value = dict(pairs)
    require(len(value) == len(pairs), "Duplicate JSON fields are forbidden")
    return value


def _reject_constant(_):
    raise ValueError("Nonfinite JSON fields are forbidden")


def read_receipt(ref, *, limit=MIB):
    path = reference(ref, max_bytes=limit, suffix=".json")
    with path.open("rb") as stream:
        raw = stream.read(ref["bytes"] + 1)
    require(len(raw) == ref["bytes"] and hashlib.sha256(raw).hexdigest() == ref["sha256"],
            "Exact closed JSON size and SHA-256 required")
    return json.loads(raw, object_pairs_hook=_unique, parse_constant=_reject_constant)


def phase_marker(mode, training_protocol, fit_ids_sha256, source_manifest_sha256):
    return dict(training_protocol=training_protocol, stage="e24", mask_resize_mode=mode,
                seed=SEED, initialization_seed=SEED, num_queries=Q100, num_select=Q100,
                query_groups=13, class_head_slots=2, coco_asset_sha256=COCO_SHA, initial_online_sha256=Q100_SHA,
                recipe_sha256=RECIPE_SHA, fit_ids_sha256=fit_ids_sha256, source_manifest_sha256=source_manifest_sha256)


def read_pair(pair_ref, *, native_arm_ref, center_arm_ref, training_protocol, evaluation_protocol):
    """Read the independently pinned pair and both original safe arm receipts."""
    require(is_hex(training_protocol, 40) and is_hex(evaluation_protocol, 40), "Exact owner protocol pins required")
    pair = read_receipt(pair_ref)
    arm_pins = tuple(f"{mode}_{item}_sha256" for mode in ARMS for item in ("arm_receipt", "checkpoint"))
    pair_pins = ("source_manifest_sha256", "qualification_pair_sha256", "fit_ids_sha256", "ordered_fit_visits_sha256")
    exact(pair, dict(status=PAIR_STATUS, training_protocol=training_protocol, seed=SEED,
                     initial_online_sha256=Q100_SHA, coco_asset_sha256=COCO_SHA, recipe_sha256=RECIPE_SHA,
                     completed_epochs=EPOCHS, visits_per_arm=VISITS, optimizer_updates_per_arm=UPDATES,
                     ema_updates_per_arm=UPDATES, ordered_fit_visits_match=True, **FIT_ONLY),
          "Exact independently accepted SSDD Q100 E24 pair required", extra=(*pair_pins, *arm_pins))
    require(all(is_hex(pair[key]) for key in (*pair_pins, *arm_pins))
            and len({pair[key] for key in arm_pins}) == len(arm_pins), "Distinct exact SSDD training arm pins required")
    arms = {}
    for mode, ref in zip(ARMS, (native_arm_ref, center_arm_ref), strict=True):
        arm = read_receipt(ref)
        require(ref["sha256"] == pair[f"{mode}_arm_receipt_sha256"], "Accepted SSDD arm receipt digest mismatch")
        exact(arm, dict(status=ARM_STATUS, qualification_pair_sha256=pair["qualification_pair_sha256"],
                        original_os_wait=True, original_os_exit=0, resource_stop=False, worker_closed=True,
                        completed_epochs=EPOCHS, visits=VISITS, optimizer_updates=UPDATES, ema_updates=UPDATES,
                        checkpoint_sha256=pair[f"{mode}_checkpoint_sha256"], **FIT_ONLY),
              "Exact independently closed SSDD Q100 E24 arm required",
              extra=("phase", "checkpoint_bytes", "filtered_empty_rows_total", *ARM_DIGESTS))
        exact(arm["phase"], phase_marker(mode, training_protocol, pair["fit_ids_sha256"], pair["source_manifest_sha256"]),
              "Exact SSDD phase marker identity required")
        require(all(is_hex(arm[key]) for key in ARM_DIGESTS) and type(arm["checkpoint_bytes"]) is int
                and 0 < arm["checkpoint_bytes"] <= 1024 * MIB and type(arm["filtered_empty_rows_total"]) is int
                and arm["filtered_empty_rows_total"] >= 0, "Exact bounded SSDD archive and arm evidence required")
        arms[mode] = arm
    require(all(arms["native"][key] != arms["center"][key] for key in ("terminal_sha256", "result_sha256")),
            "Distinct original terminal and worker evidence required")
    return pair, arms


def _file_digest(stream, algorithm):
    digest = hashlib.new(algorithm)
    for block in iter(lambda: stream.read(MIB), b""):
        digest.update(block)
    return digest


def load_sources():
    """Load byte-pinned historical code, with portability changes confined to its namespaces."""
    modules = {}
    for role, (relative, digest) in SOURCES.items():
        path = CODE / relative
        require(path.resolve(strict=True) == path, "Canonical historical export source required")
        with path.open("rb") as stream:
            raw = stream.read(64 * 1024 + 1)
        require(0 < len(raw) <= 64 * 1024 and hashlib.sha256(raw).hexdigest() == digest,
                "Historical export source digest mismatch")
        module = ModuleType("_ssdd_phase_dev_" + role)
        module.__file__ = str(path)
        exec(compile(raw, str(path), "exec"), module.__dict__)
        modules[role] = module
    modules["pilot"].original_ref = reference
    if not hasattr(hashlib, "file_digest"):
        modules["worker"].hashlib = SimpleNamespace(sha256=hashlib.sha256, file_digest=_file_digest)
    return modules


def bind_archive_validator(admitted_arm, *, long_worker):
    """Add mode/seed validation before the historical full E24 archive check."""
    phase = dict(admitted_arm["phase"])

    def validate(checkpoint, epoch, *, boundary_api, recipe_sha256):
        require(type(epoch) is int and epoch == EPOCHS and recipe_sha256 == RECIPE_SHA,
                "Exact fixed SSDD E24 archive recipe required")
        require(type(checkpoint.get("epoch")) is int and checkpoint["epoch"] == EPOCHS - 1
                and type(checkpoint.get("global_step")) is int and checkpoint["global_step"] == UPDATES,
                "Exact original E24 optimizer cursor required")
        exact(checkpoint.get(PHASE_MARKER), phase, "Exact SSDD checkpoint phase marker required")
        require(matches(checkpoint.get("args"), {"seed": SEED}), "Exact saved RF-DETR seed 5201 required")
        require(matches(checkpoint.get("model_config"), dict(num_queries=Q100, num_select=Q100, group_detr=13,
                                                             resolution=576, num_classes=1, segmentation_head=True)),
                "Exact saved SSDD Nano Q100 model configuration required")
        exact(checkpoint.get(long_worker.BOUNDARY_KEY), long_worker.endpoint_marker(EPOCHS, RECIPE_SHA),
              "Exact original native E24 boundary marker required")
        long_worker.validate_archive_progress(checkpoint, epoch, boundary_api=boundary_api, recipe_sha256=recipe_sha256)

    return validate


def load_image_manifest(ref, dev_root):
    """Rebase only original JPEG basenames, preserving the frozen manifest byte identities."""
    require(type(ref) is dict and ref.get("sha256") == IMAGE_MANIFEST_SHA,
            "Exact historical DEV187 image-only manifest digest required")
    manifest = read_receipt(ref)
    exact(manifest, dict(schema="Root-RFDETR-DEV187-image-only-manifest-v1", population="internal_DEV187",
                         model_execution=False, AP=False, TEST=False),
          "Fixed SSDD image-only DEV187 manifest required", extra=("expected_ids", "items"))
    ids, rows = manifest["expected_ids"], manifest["items"]
    require(type(ids) is list and type(rows) is list and len(ids) == len(rows) == DEV_IMAGES
            and all(type(value) is int and value >= 0 for value in ids) and len(set(ids)) == DEV_IMAGES,
            "Exact ordered logical DEV187 image identities required")
    root = Path(dev_root)
    valid = root / "valid"
    require(root.is_absolute() and root.resolve(strict=True) == root and valid.resolve(strict=True) == valid
            and valid.is_dir(), "Canonical owner-admitted DEV-only valid directory required")
    items, names = [], set()
    for image_id, row in zip(ids, rows, strict=True):
        exact(row, dict(image_id=image_id), "Exact original DEV image order required", extra=("image_ref", "original_hw"))
        hw, original = row["original_hw"], row["image_ref"]
        exact(original, {}, "Exact original DEV image reference required", extra=("path", "bytes", "sha256"))
        require(type(original["path"]) is str and type(hw) is list and len(hw) == 2
                and all(type(value) is int and value > 0 for value in hw), "Exact original variable JPEG dimensions required")
        old_path = PurePosixPath(original["path"])
        name = old_path.name
        require(old_path.is_absolute() and old_path.parent.name == "valid" and ".." not in old_path.parts
                and re.fullmatch(r"[A-Za-z0-9_.-]+\.jpg", name) and name not in names,
                "Distinct original DEV JPEG basenames required")
        names.add(name)
        image_ref = dict(original, path=str(valid / name))
        reference(image_ref, max_bytes=16 * MIB, suffix=".jpg")
        items.append(dict(row, image_ref=image_ref))
    return items, ids


def export_arm(pair_ref, *, native_arm_ref, center_arm_ref, training_protocol, evaluation_protocol, mode,
               archive_ref, dev_manifest_ref, dev_root, model_config, train_config, output_directory, event_sink=None):
    """Run the unchanged native exporter for one closed paired E24 checkpoint."""
    require(mode in ARMS, "Expected a prelocked SSDD phase arm")
    _, arms = read_pair(pair_ref, native_arm_ref=native_arm_ref, center_arm_ref=center_arm_ref,
                        training_protocol=training_protocol, evaluation_protocol=evaluation_protocol)
    arm = arms[mode]
    archive = reference(archive_ref, max_bytes=1024 * MIB, suffix=".ckpt")
    require(archive.name == ARCHIVE_NAME and archive_ref["sha256"] == arm["checkpoint_sha256"]
            and archive_ref["bytes"] == arm["checkpoint_bytes"], "Exact accepted SSDD checkpoint_23.ckpt required")
    items, ids = load_image_manifest(dev_manifest_ref, dev_root)
    sources = load_sources()
    result = sources["worker"].run_e24_dev_export(
        archive_ref, items, ids, model_config, train_config, recipe_sha256=RECIPE_SHA,
        validate_archive_progress=bind_archive_validator(arm, long_worker=sources["long"]),
        boundary_api=sources["boundary"], pilot_api=sources["pilot"],
        export_e1_image=sources["raw_export"].export_e1_image,
        canonical_export=sources["canonical"].export_coco_mask_predictions,
        output_directory=output_directory, event_sink=event_sink,
    )
    require(matches(result, dict(status="NATIVE_E24_EMA_DEV187_EXPORT_COMPLETE_PENDING_ROOT", fixed_epoch=EPOCHS,
                                 optimizer_updates=UPDATES, checkpoint_sha256=arm["checkpoint_sha256"],
                                 recipe_sha256=RECIPE_SHA, image_count=DEV_IMAGES, native_count_total=DEV_IMAGES * Q100,
                                 full_DEV=True, qualified=False, terminal_self_certified=False,
                                 GT_read=False, TEST=False, AP=False, training=False)),
            "Complete original image-only SSDD Q100 DEV187 export required")
    return dict(status=EXPORT_STATUS, training_protocol=training_protocol, evaluation_protocol=evaluation_protocol,
                mode=mode, training_seed=SEED, training_pair_sha256=pair_ref["sha256"],
                training_arm_sha256=(native_arm_ref if mode == "native" else center_arm_ref)["sha256"],
                image_manifest_sha256=IMAGE_MANIFEST_SHA, result=result)
