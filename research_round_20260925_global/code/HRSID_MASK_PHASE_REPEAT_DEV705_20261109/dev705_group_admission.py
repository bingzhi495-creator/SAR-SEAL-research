"""Read independently closed 5201/5202 DEV705 inputs for P-prefix AP diagnostics."""

from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re
from types import ModuleType


DRIVER_SOURCE = (
    Path(__file__).resolve().parent.parent / "HRSID_MASK_PHASE_PAIRED_GEOMETRY_20261109" / "score_closed_pair.py"
)
DRIVER_SHA256 = "a046cab04371b2e0932ad817c8ad41381ccb7134a30842c37d9797e54b2f6e14"
GROUP_MAP_SHA256 = "9f45d673649d2aa65bb38fc2a29b27981e0b51f2a879c3f9d309b9da28e511be"
GROUP_COUNT = 28
PROTOCOLS = {
    5201: ("d66baf3c57fafdba4242ad1b55e08cf76bfd77f7", "4e5ea5db93423d4f4c0a814ca18e8386313cb58a"),
    5202: ("a49361b92977a605267025b1e7798420936cbe53", "0410a3268ba9192735588003124f3e364a6b494d"),
}
CLOSED = dict(original_os_wait=True, original_os_exit=0, resource_stop=False)


def load_driver():
    path = DRIVER_SOURCE
    if path.resolve(strict=True) != path:
        raise ValueError("Canonical immutable P204 driver path required")
    limit = 64 * 1024
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if not 0 < len(raw) <= limit or sha256(raw).hexdigest() != DRIVER_SHA256:
        raise ValueError("Exact immutable P204 driver required")
    driver = ModuleType("_dev705_group_admission_p204")
    driver.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), driver.__dict__)
    return driver


def load_population(driver, manifest_ref, gt_ref):
    """Return the exact raw GT, original image order and internal P-prefix membership."""
    manifest_path, gt_path = driver.pin(manifest_ref), driver.pin(gt_ref)
    driver.require(manifest_ref["sha256"] == driver.IMAGE_MANIFEST_SHA and gt_ref["sha256"] == driver.GT_SHA,
                   "Fixed original DEV705 GT and image-only manifest digests required")
    manifest = driver.read_json(manifest_path, driver.IMAGE_MANIFEST_SHA, 4 * driver.MIB)
    gt = driver.read_json(gt_path, driver.GT_SHA, 4 * driver.MIB)
    images = gt["images"]
    driver.require(len(manifest) == len(images) == driver.DEV_IMAGES
                   and len(gt["annotations"]) == driver.GT_COUNT and [row["id"] for row in gt["categories"]] == [1],
                   "Fixed raw DEV705 scope and ship category required")
    groups = {}
    for member, image in zip(manifest, images, strict=True):
        image_id, name = member["image_id"], member["file_name"]
        driver.require(type(image_id) is int and image_id == image["id"] and image_id not in groups
                       and type(name) is str and name == image["file_name"]
                       and (member["width"], member["height"]) == (image["width"], image["height"]) == driver.CANVAS,
                       "Original manifest and raw GT image order and identities must agree")
        prefix = PurePosixPath(name).stem.split("_", 1)[0]
        driver.require(re.fullmatch(r"P[0-9]+", prefix) is not None, "Canonical P-prefix proxy group required")
        groups[image_id] = prefix
    canonical = json.dumps(sorted(groups.items()), ensure_ascii=True, separators=(",", ":")) + "\n"
    driver.require(len(set(groups.values())) == GROUP_COUNT
                   and sha256(canonical.encode("ascii")).hexdigest() == GROUP_MAP_SHA256,
                   "Exact fixed DEV705 P-prefix membership required")
    return gt, images, groups


def read_closed_pair(driver, pair_ref, *, training_seed, training_protocol, evaluation_protocol):
    """Return the verified export and training receipts under an external digest pin."""
    driver.require(type(training_seed) is int
                   and PROTOCOLS.get(training_seed) == (training_protocol, evaluation_protocol),
                   "Exact prelocked 5201 or 5202 protocol pair required")
    pair = driver.read_json(driver.pin(pair_ref), pair_ref["sha256"])
    seed_fields = {} if training_seed == 5201 else {"seed": training_seed}
    driver.exact(pair, dict(status=driver.PAIR_STATUS, protocol=evaluation_protocol, **seed_fields),
                 "Exact seed-specific closed export pair required", extra=("training_pair", "gt", "arms"))
    driver.pin(pair["gt"])
    driver.require(pair["gt"]["sha256"] == driver.GT_SHA, "Fixed original DEV705 GT digest required")
    training_ref = pair["training_pair"]
    training = driver.read_json(driver.pin(training_ref), training_ref["sha256"])
    arm_digests = tuple(f"{name}_{kind}_sha256" for name in driver.ARMS for kind in ("arm_receipt", "checkpoint"))
    driver.exact(training, dict(status=driver.TRAINING_PAIR_STATUS, training_protocol=training_protocol,
                                ordered_fit_visits_match=True, completed_epochs=24, visits_per_arm=70560,
                                **seed_fields), "Exact seed-specific accepted E24 training pair required",
                 extra=arm_digests)
    driver.require(all(driver.is_digest(training[key]) for key in arm_digests)
                   and len({training[key] for key in arm_digests}) == len(arm_digests),
                   "Distinct accepted E24 arm receipt and checkpoint digests required")
    driver.exact(pair["arms"], {}, "Exactly native and center export arms required", extra=driver.ARMS)
    for arm in pair["arms"].values():
        driver.exact(arm, CLOSED, "Both original export processes must have closed cleanly",
                     extra=("report", "predictions"))
    return pair, training


def _closed_score(driver, score_ref, predictions_sha, checkpoint_sha, training_protocol, evaluation_protocol):
    report = driver.read_json(driver.pin(score_ref), score_ref["sha256"])
    driver.exact(
        report,
        dict(status="HRSID_DEV705_E24_NATIVE_COCO_COMPLETE", protocol=evaluation_protocol,
             training_protocol=training_protocol, checkpoint_sha256=checkpoint_sha, predictions_sha256=predictions_sha,
             initialization="official_COCO_only", original_dev_gt_sha256=driver.GT_SHA, images=driver.DEV_IMAGES,
             excluded_empty_gt_sensitivity=driver.EMPTY_COUNT, max_dets=[1, 10, 100], units="fraction",
             official_TEST_opened=False, official_TEST_AP=False),
        "Exact closed COCO report bound to original predictions, checkpoint and DEV GT required",
        extra=("primary", "sensitivity"),
    )
    driver.require(all(type(value) is int for value in report["max_dets"]), "Exact COCO max detections required")
    for population, count in (("primary", driver.GT_COUNT), ("sensitivity", driver.GT_COUNT - driver.EMPTY_COUNT)):
        score = report[population]
        driver.exact(score, dict(annotations=count), "Exact original COCO score population required",
                     extra=("AP", "AP75", "AP90", "APs", "AR100", "all12"))
        vector = score["all12"]
        driver.require(type(vector) is list and len(vector) == 12
                       and all(type(value) in (int, float) and (value == -1 or 0 <= value <= 1) for value in vector)
                       and all(type(score[key]) in (int, float) and 0 <= score[key] <= 1 and score[key] == vector[index]
                               for key, index in (("AP", 0), ("AP75", 2))),
                       "Exact original COCO fractional AP and AP75 required")
    return {key: report["primary"][key] for key in ("AP", "AP75")}


def read_arm(driver, closure_export, score_ref, *, training_protocol, evaluation_protocol, checkpoint_sha, images):
    """Return original-order COCO rows, accepted full-population AP/AP75 and path-free pins."""
    driver.require((training_protocol, evaluation_protocol) in PROTOCOLS.values() and driver.is_digest(checkpoint_sha),
                   "Exact prelocked DEV705 arm protocols and checkpoint digest required")
    driver.exact(closure_export, CLOSED, "Original export must have closed cleanly", extra=("report", "predictions"))
    predictions_ref = closure_export["predictions"]
    driver.pin(predictions_ref)
    expected = _closed_score(driver, score_ref, predictions_ref["sha256"], checkpoint_sha,
                             training_protocol, evaluation_protocol)
    previous_protocol = driver.TRAINING_PROTOCOL
    driver.TRAINING_PROTOCOL = training_protocol
    try:
        _, blocks = driver.read_arm(closure_export, checkpoint_sha, evaluation_protocol, images)
    finally:
        driver.TRAINING_PROTOCOL = previous_protocol
    flat = []
    for block in blocks:
        ranks, rows = block["native_ranks"], block["coco"]
        driver.require(len(ranks) == len(rows)
                       and all(type(rank) is int and 1 <= rank <= driver.NATIVE_ROWS for rank in ranks)
                       and all(a < b for a, b in zip(ranks, ranks[1:]))
                       and all(type(row["score"]) in (int, float) and 0 <= row["score"] <= 1 for row in rows)
                       and all(a["score"] >= b["score"] for a, b in zip(rows, rows[1:])),
                       "Exact original mapped native ranks and nonincreasing scores required")
        flat.extend(rows)
    pins = dict(checkpoint_sha256=checkpoint_sha, export_report_sha256=closure_export["report"]["sha256"],
                predictions_sha256=predictions_ref["sha256"], score_report_sha256=score_ref["sha256"])
    return flat, expected, pins
