"""Score independently closed SSDD phase exports on the exact original DEV187 cohort."""

from contextlib import redirect_stdout
from copy import deepcopy
from hashlib import sha256
from importlib.metadata import version
from io import StringIO
import json
import math
from pathlib import PurePosixPath, PureWindowsPath
from typing import NamedTuple

from ssdd_phase_dev_admission import (
    ARMS, CODE, COCO_SHA, EXPORT_STATUS, IMAGE_MANIFEST_SHA, PAIR_STATUS, Q100_SHA, RECIPE_SHA,
    exact, is_hex, matches, require,
)
from ssdd_phase_init import load_api


MIB = 1024**2
INTEGER_GT_SHA = "5657600d6ea198f79bd5d557d2a1aadce5bec031d2ecf72a98716248d1183a55"
INTEGER_GT_BYTES = 118319
DEV_IMAGES, DEV_ANNOTATIONS, NATIVE_ROWS = 187, 380, 18700
METRIC_NAMES = {"AP": 0, "AP50": 1, "AP75": 2, "APs": 3, "AR100": 8}
EXPORT_TOTALS = ("image_count", "native_count_total", "unmapped_slot1_total", "prediction_count",
                 "zero_prediction_images", "codec_roundtrip_masks", "nonempty_masks", "all_zero_masks")
SOURCES = {
    "bridge": ("RFDETR_NATIVE_DEV187_OUTPUT_BRIDGE_CANDIDATE_20261107/bridge.py",
               "b84a362f4ba6524a27e8b5d9c77345f3685ef25d1ad8a88b9d82fe2ea400d961"),
    "adapter": ("RFDETR_NATIVE_DEV187_CPU_EVAL_ADAPTER_CANDIDATE_20261107/adapter.py",
                "2596195c30f87a0e626719e68a8a28aaf3d905e2f1aba02ca64941f30c2a3b59"),
    "official": ("DIFFSAR_NATIVE_CPU_COCO_ADAPTER_DRAFT_20261107_01/worker.py",
                 "f9fc79403d12d8840c3269b04ba22d7584b8eb5e3c592ead2b6359ad7df65748"),
    "sensitivity": ("SAMRefiner_FULL187_GT_SENSITIVITY_DRAFT_20261025_01/worker.py",
                    "043a0d3e8b157765413b91cd298d9d108370115cea79721dc0e27cc09d93e349"),
}


class Artifact(NamedTuple):
    ref: dict
    raw: bytes


class ClosedExport(NamedTuple):
    closure: dict
    report: Artifact
    predictions: Artifact
    export_manifest: Artifact
    checkpoint_ref: dict


def artifact_path(ref, limit, *, name=None):
    exact(ref, {}, "Exact owner-admitted artifact reference required", extra=("path", "bytes", "sha256"))
    path_text = ref["path"]
    require(type(path_text) is str and path_text and all(ord(char) >= 32 for char in path_text)
            and type(ref["bytes"]) is int and 0 < ref["bytes"] <= limit and is_hex(ref["sha256"]),
            "Exact bounded owner-admitted artifact reference required")
    windows = PureWindowsPath(path_text)
    path = windows if windows.is_absolute() else PurePosixPath(path_text)
    require(path.is_absolute() and ".." not in path.parts and path_text in (str(path), path.as_posix())
            and (name is None or path.name == name), "Canonical owner-admitted artifact path required")
    return path


def unique(pairs):
    result = dict(pairs)
    require(len(pairs) == len(result), "Duplicate closed SSDD JSON fields are forbidden")
    return result


def reject_constant(_):
    raise ValueError("Nonfinite closed SSDD JSON number")


def read_artifact(artifact, limit, shape, *, name=None):
    require(type(artifact) is Artifact, "Exact caller-supplied closed artifact required")
    path = artifact_path(artifact.ref, limit, name=name)
    raw = artifact.raw
    require(type(raw) is bytes and len(raw) == artifact.ref["bytes"]
            and sha256(raw).hexdigest() == artifact.ref["sha256"], "Exact closed artifact size and SHA-256 required")
    value = json.loads(raw, object_pairs_hook=unique, parse_constant=reject_constant)
    require(type(value) is shape, "Unexpected closed SSDD artifact JSON shape")
    return path, value


def fixed_pair(artifact, protocol):
    path, value = read_artifact(artifact, MIB, dict)
    arm_pins = tuple(f"{mode}_{key}_sha256" for mode in ARMS for key in ("arm_receipt", "checkpoint"))
    other_pins = ("source_manifest_sha256", "qualification_pair_sha256", "fit_ids_sha256", "ordered_fit_visits_sha256")
    exact(value, dict(status=PAIR_STATUS, training_protocol=protocol, seed=5201, initial_online_sha256=Q100_SHA,
                      coco_asset_sha256=COCO_SHA, recipe_sha256=RECIPE_SHA, completed_epochs=24,
                      visits_per_arm=17856, optimizer_updates_per_arm=4464, ema_updates_per_arm=4464,
                      ordered_fit_visits_match=True, FIT=True, DEV=False, TEST=False, AP=False),
          "Exact independently accepted SSDD E24 Q100 training pair required", extra=(*arm_pins, *other_pins))
    require(all(is_hex(value[key]) for key in (*arm_pins, *other_pins))
            and len({value[key] for key in arm_pins}) == len(arm_pins), "Distinct exact SSDD E24 arm pins required")
    return path, value


def fixed_images(artifact):
    path, value = read_artifact(artifact, MIB, dict)
    require(artifact.ref["sha256"] == IMAGE_MANIFEST_SHA, "Exact fixed source DEV187 image manifest required")
    exact(value, dict(schema="Root-RFDETR-DEV187-image-only-manifest-v1", population="internal_DEV187",
                      model_execution=False, AP=False, TEST=False),
          "Exact SSDD DEV187 source manifest required", extra=("expected_ids", "items"))
    ids, items = value["expected_ids"], value["items"]
    require(type(ids) is type(items) is list and len(ids) == len(items) == DEV_IMAGES
            and all(type(i) is int and i >= 0 for i in ids) and len(set(ids)) == DEV_IMAGES,
            "Exact original ordered DEV187 logical image identities required")
    for image_id, item in zip(ids, items, strict=True):
        exact(item, dict(image_id=image_id), "Original DEV187 image identity and order differ",
              extra=("original_hw", "image_ref"))
        hw = item["original_hw"]
        require(type(hw) is list and len(hw) == 2 and all(type(n) is int and n > 0 for n in hw),
                "Exact original DEV187 image dimensions required")
        artifact_path(item["image_ref"], 16 * MIB)
    return path, ids, items


def validate_gt(gt, ids, items):
    images, annotations, categories = (gt.get(key) for key in ("images", "annotations", "categories"))
    require(type(images) is type(annotations) is type(categories) is list and len(images) == DEV_IMAGES
            and len(annotations) == DEV_ANNOTATIONS and len(categories) == 1 and matches(categories[0], {"id": 1}),
            "Exact original DEV187/380 ship GT population required")
    for image, item in zip(images, items, strict=True):
        height, width = item["original_hw"]
        require(matches(image, dict(id=item["image_id"], height=height, width=width)),
                "GT image order, identity or original dimensions differ")
    admitted = set(ids)
    require(all(type(row) is dict and type(row.get("id")) is int
                and type(row.get("image_id")) is int and row["image_id"] in admitted
                and matches(row, {"category_id": 1}) for row in annotations)
            and len({row["id"] for row in annotations}) == DEV_ANNOTATIONS,
            "Exact original unique DEV187 GT annotation IDs required")


def check_arm(arm, mode, pair, training_pair_sha, training_protocol, evaluation_protocol, image_manifest_sha):
    require(type(arm) is ClosedExport, "Exact independently closed SSDD export inputs required")
    report_path, report = read_artifact(arm.report, MIB, dict)
    exact(arm.closure, dict(original_os_wait=True, original_os_exit=0, resource_stop=False, worker_closed=True,
                          report_sha256=arm.report.ref["sha256"]), "Exact independent SSDD export process closure required",
          extra=("terminal_sha256",))
    require(is_hex(arm.closure["terminal_sha256"]), "Exact independent SSDD export terminal digest required")
    expected_checkpoint = pair[f"{mode}_checkpoint_sha256"]
    expected_arm = pair[f"{mode}_arm_receipt_sha256"]
    checkpoint_path = artifact_path(arm.checkpoint_ref, 1024 * MIB, name="checkpoint_23.ckpt")
    require(arm.checkpoint_ref["sha256"] == expected_checkpoint, "Exact accepted SSDD arm checkpoint identity required")
    exact(report, dict(status=EXPORT_STATUS, training_protocol=training_protocol, evaluation_protocol=evaluation_protocol,
                       mode=mode, training_seed=5201, training_pair_sha256=training_pair_sha,
                       training_arm_sha256=expected_arm, image_manifest_sha256=image_manifest_sha),
          "SSDD export must bind its original pair, arm, mode and image manifest", extra=("result",))
    result = report["result"]
    require(matches(result, dict(status="NATIVE_E24_EMA_DEV187_EXPORT_COMPLETE_PENDING_ROOT", fixed_epoch=24,
                                 optimizer_updates=4464, checkpoint_sha256=expected_checkpoint,
                                 recipe_sha256=RECIPE_SHA, image_count=DEV_IMAGES, native_count_total=NATIVE_ROWS,
                                 full_DEV=True, qualified=False, terminal_self_certified=False, GT_read=False,
                                 TEST=False, AP=False, training=False, tensor_precision="float32", autocast=False,
                                 checkpoint_weights_only_cpu=True)), "Exact complete image-only SSDD E24 export required")
    predictions_path, predictions = read_artifact(arm.predictions, 64 * MIB, list, name="PREDICTIONS.json")
    manifest_path, index = read_artifact(arm.export_manifest, 4 * MIB, dict, name="IMAGE_MANIFEST.json")
    require(result.get("predictions_ref") == arm.predictions.ref and result.get("image_manifest_ref") == arm.export_manifest.ref
            and predictions_path.parent == manifest_path.parent,
            "Original export report, prediction path and index reference differ")
    totals = {name: result.get(name) for name in EXPORT_TOTALS}
    require(all(type(value) is int and 0 <= value <= NATIVE_ROWS for value in totals.values()),
            "Exact original SSDD export counters required")
    exact(index, dict(schema="RFDETR-E24-canonical-DEV-image-index-v1", checkpoint_sha256=expected_checkpoint,
                      recipe_sha256=RECIPE_SHA, prediction_offset_base=0, native_rank_base=1,
                      predictions_ref=arm.predictions.ref, **totals),
          "SSDD export index must bind the evaluated checkpoint and exact prediction artifact",
          extra=("images",))
    safe = dict(training_arm_sha256=expected_arm, checkpoint_sha256=expected_checkpoint,
                export_report_sha256=arm.report.ref["sha256"], prediction_sha256=arm.predictions.ref["sha256"],
                export_manifest_sha256=arm.export_manifest.ref["sha256"],
                export_terminal_sha256=arm.closure["terminal_sha256"], **totals)
    return (report_path, predictions_path, manifest_path, checkpoint_path), index, predictions, safe


def score_one(official, gt, predictions, ids):
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    captured = []

    def evaluator(*args):
        actual = COCOeval(*args)
        captured.append(actual)
        return actual

    coco = COCO()
    coco.dataset = deepcopy(gt)
    with redirect_stdout(StringIO()):
        coco.createIndex()
        require(list(coco.imgs) == ids and len(coco.anns) == DEV_ANNOTATIONS and set(coco.cats) == {1},
                "Exact original COCO index population required")
        stats = official.evaluate_population(dict(COCO=COCO, COCOeval=evaluator), coco, predictions, ids)
    require(len(captured) == 1 and len(stats) == 12
            and all(type(x) is float and math.isfinite(x) and -1. <= x <= 1. for x in stats),
            "Original COCOeval must return one complete finite COCO12 result")
    evaluation, = captured
    index, = np.flatnonzero(np.isclose(evaluation.params.iouThrs, .90))
    precision = evaluation.eval["precision"][index, :, :, 0, -1]
    valid = precision[precision > -1]
    return dict(**{key: stats[i] for key, i in METRIC_NAMES.items()}, all12=stats,
                AP90=float(valid.mean()) if valid.size else -1.)


def paired_scores(official, gt, predictions, ids):
    result = {mode: score_one(official, gt, predictions[mode], ids) for mode in ARMS}
    a, b = (result[mode] for mode in ARMS)
    return dict(**result, center_minus_native_points={
        key: 100. * (b[key] - a[key]) if min(a[key], b[key]) >= 0 else None for key in (*METRIC_NAMES, "AP90")
    })


def score_closed_pair(*, native, center, training_pair, image_manifest, integer_gt, training_protocol,
                      evaluation_protocol, floating_gt=None, expected_floating_gt_sha256=None):
    """Compute fixed SSDD paired scores only from caller-admitted closed bytes."""
    require(is_hex(training_protocol, 40) and is_hex(evaluation_protocol, 40), "Exact owner protocol pins required")
    pair_path, pair = fixed_pair(training_pair, training_protocol)
    source_path, ids, items = fixed_images(image_manifest)
    gt_path, gt = read_artifact(integer_gt, MIB, dict)
    require(integer_gt.ref["sha256"] == INTEGER_GT_SHA and integer_gt.ref["bytes"] == INTEGER_GT_BYTES,
            "Exact prelocked original SSDD integer GT required")
    validate_gt(gt, ids, items)
    paths = [pair_path, source_path, gt_path]
    float_gt = None
    if floating_gt is None:
        require(expected_floating_gt_sha256 is None, "A prelocked floating-GT sensitivity requires its exact artifact")
    else:
        require(is_hex(expected_floating_gt_sha256) and expected_floating_gt_sha256 != INTEGER_GT_SHA,
                "The distinct floating-GT sensitivity digest must be prelocked")
        float_path, float_gt = read_artifact(floating_gt, MIB, dict)
        require(floating_gt.ref["sha256"] == expected_floating_gt_sha256,
                "Exact prelocked floating-GT sensitivity artifact required")
        validate_gt(float_gt, ids, items)
        paths.append(float_path)

    inputs, arms = {}, {}
    for mode, arm in zip(ARMS, (native, center), strict=True):
        arm_paths, index, predictions, safe = check_arm(
            arm, mode, pair, training_pair.ref["sha256"], training_protocol,
            evaluation_protocol, image_manifest.ref["sha256"])
        paths.extend(arm_paths)
        inputs[mode] = index, predictions
        arms[mode] = safe
    require(len(set(paths)) == len(paths)
            and arms["native"]["export_terminal_sha256"] != arms["center"]["export_terminal_sha256"],
            "Both SSDD export arms require distinct original paths and independent terminal identities")

    require(version("pycocotools") == "2.0.11", "Exact original pycocotools 2.0.11 CPU backend required")
    roles = ("bridge", "adapter", "official") if float_gt is None else tuple(SOURCES)
    apis = {role: load_api(CODE / SOURCES[role][0], SOURCES[role][1]) for role in roles}
    if float_gt is not None:
        apis["sensitivity"].validate_GT_pair(gt, float_gt)

    normalized = {}
    for mode, (index, predictions) in inputs.items():
        document = apis["bridge"].group_canonical_rows(index, predictions, items)
        flat, _ = apis["adapter"].normalize_export(document, gt["images"])
        require(flat == predictions, "Original SSDD mapped prediction rows and rank order must be preserved")
        normalized[mode] = flat
    primary = paired_scores(apis["official"], gt, normalized, ids)
    sensitivity = None if float_gt is None else paired_scores(apis["official"], float_gt, normalized, ids)
    return dict(status="SSDD_MASK_PHASE_DEV187_CPU_SCORE_PENDING_INDEPENDENT_ACCEPTANCE_V1",
                training_protocol=training_protocol, evaluation_protocol=evaluation_protocol,
                training_pair_sha256=training_pair.ref["sha256"], image_manifest_sha256=image_manifest.ref["sha256"],
                integer_gt_sha256=INTEGER_GT_SHA, floating_gt_sha256=expected_floating_gt_sha256,
                images=DEV_IMAGES, annotations=DEV_ANNOTATIONS, primary_GT="INTEGER_PRIMARY",
                sensitivity_GT=None if float_gt is None else "LOCAL_CANDIDATE_FLOAT",
                arms=arms, primary=primary, sensitivity=sensitivity, FIT=False, DEV=True, TEST=False,
                AP=True, training=False)

