"""Run frozen paired geometry on independently closed common-FIT DEV705 exports."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from types import ModuleType


HERE = Path(__file__).resolve().parent
DRIVER_SOURCE = HERE.parent / "HRSID_MASK_PHASE_PAIRED_GEOMETRY_20261109" / "score_closed_pair.py"
DRIVER_SHA = "a046cab04371b2e0932ad817c8ad41381ccb7134a30842c37d9797e54b2f6e14"
ADMISSION_SOURCE = HERE / "common_fit_eval_admission.py"
ADMISSION_SHA = "7a38fa028c07bf16d2da42935c5ee117f9e9ee5a88ede4337a43b1199c5ced13"
PHASE_SOURCE = HERE.parent / "HRSID_MASK_PHASE_E24_DEV705_20261109" / "hrsid_mask_phase_dev705.py"
PAIR_STATUS = "HRSID_MASK_COMMON_FIT_E24_CLOSED_EXPORT_SCORE_PAIR_V1"
OUTPUT_STATUS = "HRSID_MASK_COMMON_FIT_E24_DEV705_PAIRED_GEOMETRY_COMPLETE"
SCORE_STATUS = "HRSID_DEV705_E24_NATIVE_COCO_COMPLETE"
CLOSED_PROCESS = dict(original_os_wait=True, original_os_exit=0, resource_stop=False)


def load_source(path, expected_sha, name):
    if path.resolve(strict=True) != path:
        raise ValueError("Canonical immutable source path required")
    limit = 64 * 1024
    with path.open("rb") as stream:
        source = stream.read(limit + 1)
    if not 0 < len(source) <= limit or sha256(source).hexdigest() != expected_sha:
        raise ValueError("Exact immutable source required: " + path.name)
    module = ModuleType(name)
    module.__file__ = str(path)
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def load_driver():
    return load_source(DRIVER_SOURCE, DRIVER_SHA, "_common_fit_geometry_driver")


def validate_score_population(driver, population, annotations):
    indices = {"AP": 0, "AP75": 2, "APs": 3, "AR100": 8}
    driver.exact(population, {"annotations": annotations}, "Exact native COCO score population required",
                 extra=(*indices, "AP90", "all12"))
    values = population["all12"]
    driver.require(type(values) is list and len(values) == 12, "Exact native COCO score vector required")
    fractions = [*values, *(population[key] for key in (*indices, "AP90"))]
    driver.require(
        all(type(value) in (int, float) and (value == -1 or 0 <= value <= 1) for value in fractions)
        and all(population[key] == values[index] for key, index in indices.items()),
        "Exact native COCO fractional scores required",
    )


def score_report(driver, closure, *, evaluation_protocol, training_protocol, checkpoint_sha, predictions_sha):
    driver.exact(closure, dict(CLOSED_PROCESS, owner_closed=True, worker_closed=True),
                 "Original COCO score owner and worker must have closed cleanly", extra=("report",))
    report_pin = closure["report"]
    report_path = driver.pin(report_pin)
    report = driver.read_json(report_path, report_pin["sha256"])
    driver.exact(
        report,
        dict(status=SCORE_STATUS, protocol=evaluation_protocol, training_protocol=training_protocol,
             checkpoint_sha256=checkpoint_sha, predictions_sha256=predictions_sha,
             initialization="official_COCO_only", original_dev_gt_sha256=driver.GT_SHA,
             images=driver.DEV_IMAGES, excluded_empty_gt_sensitivity=driver.EMPTY_COUNT,
             max_dets=[1, 10, 100], units="fraction", official_TEST_opened=False, official_TEST_AP=False),
        "Exact closed COCO report bound to the accepted checkpoint, prediction stream and DEV GT required",
        extra=("primary", "sensitivity"),
    )
    driver.require(all(type(value) is int for value in report["max_dets"]), "Exact COCO max detections required")
    validate_score_population(driver, report["primary"], driver.GT_COUNT)
    validate_score_population(driver, report["sensitivity"], driver.GT_COUNT - driver.EMPTY_COUNT)
    return report_pin["sha256"]


def score_pair(pair_path, pair_sha256, *, native_arm_receipt, center_arm_receipt, training_protocol):
    """Consume the externally pinned owner receipt and reuse the original geometry driver."""
    driver = load_driver()
    admission = load_source(ADMISSION_SOURCE, ADMISSION_SHA, "_common_fit_geometry_admission")
    pair_path = Path(pair_path)
    closed = driver.read_json(pair_path, pair_sha256)
    driver.exact(closed, dict(status=PAIR_STATUS, training_seed=admission.TRAINING_SEED),
                 "Exact closed common FIT export and score pair required",
                 extra=("protocol", "training_pair", "gt", "arms"))
    evaluation_protocol = closed["protocol"]
    driver.require(driver.is_digest(evaluation_protocol, 40), "Exact evaluation protocol commit required")
    driver.pin(closed["gt"])
    driver.require(closed["gt"]["sha256"] == driver.GT_SHA, "Fixed original DEV GT digest required")
    training_pin = closed["training_pair"]
    training_path = driver.pin(training_pin)
    phase = admission.load_phase(PHASE_SOURCE)
    admission.read_pair(phase, training_path, training_pin["sha256"], native_path=native_arm_receipt,
                        center_path=center_arm_receipt, mode="native", training_protocol=training_protocol)
    trained = driver.read_json(training_path, training_pin["sha256"])

    arms = closed["arms"]
    driver.exact(arms, {}, "Exactly native and center common FIT arms required", extra=driver.ARMS)
    export_arms, score_digests = {}, {}
    for mode in driver.ARMS:
        arm = arms[mode]
        driver.exact(arm, CLOSED_PROCESS, "Original export must have closed cleanly",
                     extra=("report", "predictions", "score_closure"))
        predictions = arm["predictions"]
        driver.pin(predictions)
        score_digests[mode] = score_report(
            driver, arm["score_closure"], evaluation_protocol=evaluation_protocol,
            training_protocol=training_protocol, checkpoint_sha=trained[f"{mode}_checkpoint_sha256"],
            predictions_sha=predictions["sha256"],
        )
        export_arms[mode] = {key: value for key, value in arm.items() if key != "score_closure"}

    original_read = driver.read_json
    projected = dict(status=driver.PAIR_STATUS, protocol=evaluation_protocol,
                     training_pair=training_pin, gt=closed["gt"], arms=export_arms)

    def admitted_pair_read(path, digest, limit=driver.MIB):
        if path == pair_path and digest == pair_sha256:
            return projected
        return original_read(path, digest, limit)

    def admitted_training_pair(pin):
        driver.require(pin == training_pin, "The independently accepted common FIT training pair is required")
        return trained

    # The driver has a fresh namespace. Only the two admitted receipt reads
    # and the training protocol differ from the original scoring path.
    driver.read_json = admitted_pair_read
    driver.training_pair = admitted_training_pair
    driver.TRAINING_PROTOCOL = training_protocol
    result = driver.score_pair(pair_path, pair_sha256)
    return dict(result, status=OUTPUT_STATUS, training_seed=admission.TRAINING_SEED,
                support_policy=admission.POLICY, score_report_sha256s=score_digests)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--pair", type=Path, required=True)
    parser.add_argument("--pair-sha256", required=True)
    parser.add_argument("--native-arm-receipt", type=Path, required=True)
    parser.add_argument("--center-arm-receipt", type=Path, required=True)
    parser.add_argument("--training-protocol", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    if (not output.is_absolute() or output.parent.resolve(strict=True) != output.parent
            or output.exists() or output.is_symlink()):
        raise ValueError("Fresh canonical output file required")
    result = score_pair(args.pair, args.pair_sha256, native_arm_receipt=args.native_arm_receipt,
                        center_arm_receipt=args.center_arm_receipt, training_protocol=args.training_protocol)
    data = (json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode()
    with output.open("xb") as stream:
        stream.write(data)


if __name__ == "__main__":
    main()
