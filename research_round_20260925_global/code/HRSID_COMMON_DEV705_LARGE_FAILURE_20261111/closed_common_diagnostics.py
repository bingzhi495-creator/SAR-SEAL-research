"""Analyze a closed common-support HRSID DEV705 pair using its admitted predictions."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import re
from types import ModuleType


REPLAY = Path("HRSID_FIT_RESIZE_PHASE_REPLAY_20261109")
GATE_PATH = REPLAY / "common_fit_score_pair.py"
GATE_SHA = "5e0814da015ae788450ce723af2d10061b3b673e0f1c41ef1edb949032c07697"
PHASE_PATH = Path("HRSID_MASK_PHASE_E24_DEV705_20261109/hrsid_mask_phase_dev705.py")
PHASE_SHA = "488f9ed73943cc842162c0d45822b6bac42aecabf37c8ecf0d4dde8e96c0959b"
SOURCE_LIMIT = 128 * 1024
ARMS = ("native", "center")


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def pinned_source(path, expected):
    require(path.resolve(strict=True) == path and path.is_file(), "Exact canonical source required")
    with path.open("rb") as stream:
        raw = stream.read(SOURCE_LIMIT + 1)
    require(0 < len(raw) <= SOURCE_LIMIT and sha256(raw).hexdigest() == expected,
            "Exact source digest required: " + path.name)
    return raw


def load_source(path, expected, name):
    raw = pinned_source(path, expected)
    module = ModuleType(name)
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def load_gate(code_root):
    pinned_source(code_root / PHASE_PATH, PHASE_SHA)
    return load_source(code_root / GATE_PATH, GATE_SHA, "_p282_common_fit_score_pair")


def verified_inputs(code_root, pair, pair_sha256, *, native_arm_receipt, center_arm_receipt, training_protocol):
    """Return the exact admitted GT, ordered native/center COCO rows, and digest-only provenance."""
    require(type(pair_sha256) is str and re.fullmatch(r"[a-f0-9]{64}", pair_sha256) is not None,
            "Exact closed export/score pair SHA-256 required")
    require(type(training_protocol) is str and re.fullmatch(r"[a-f0-9]{40}", training_protocol) is not None,
            "Exact common FIT training protocol required")
    gate = load_gate(Path(code_root).absolute())
    driver = gate.load_driver()
    read_json, read_arm = driver.read_json, driver.read_arm
    population, captures = [], []

    def capture_gt(path, digest, limit=driver.MIB):
        value = read_json(path, digest, limit)
        if digest == driver.GT_SHA:
            population.append(value)
        return value

    def capture_arm(arm, checkpoint, protocol, images):
        directory, blocks = read_arm(arm, checkpoint, protocol, images)
        captures.append((checkpoint, blocks))
        return directory, blocks

    gate.load_driver = lambda: driver
    driver.read_json = capture_gt
    driver.read_arm = capture_arm
    result = gate.score_pair(
        Path(pair), pair_sha256, native_arm_receipt=Path(native_arm_receipt),
        center_arm_receipt=Path(center_arm_receipt), training_protocol=training_protocol,
    )
    require(result["status"] == gate.OUTPUT_STATUS and result["official_TEST_opened"] is False
            and result["closed_export_pair_sha256"] == pair_sha256
            and result["training_protocol"] == training_protocol
            and result["original_dev_gt_sha256"] == driver.GT_SHA
            and len(population) == 1 and len(captures) == 2 and tuple(driver.ARMS) == ARMS,
            "One exact common DEV705 population and two checked prediction streams required")

    prediction_rows, arm_provenance = {}, {}
    for mode, (checkpoint, blocks) in zip(ARMS, captures, strict=True):
        arm = result["arms"][mode]
        require(arm["checkpoint_sha256"] == checkpoint, "Checked prediction checkpoint or arm order changed")
        prediction_rows[mode] = [row for block in blocks for row in block["coco"]]
        arm_provenance[mode] = {key: arm[key] for key in
                                ("checkpoint_sha256", "export_report_sha256", "predictions_sha256")}
        arm_provenance[mode]["score_report_sha256"] = result["score_report_sha256s"][mode]
    provenance = {key: result[key] for key in ("closed_export_pair_sha256", "training_pair_receipt_sha256",
                                              "original_dev_gt_sha256", "training_protocol", "training_seed",
                                              "support_policy")}
    provenance.update(evaluation_protocol=result["protocol"], arms=arm_provenance)
    return population[0], prediction_rows["native"], prediction_rows["center"], provenance


DIAG_DIR = Path("HRSID_MASK_PHASE_REPEAT_DEV705_20261109")
DIAG_PINS = {
    "dev705_group_sensitivity": "f7a004ed3d128cffd2a7114618b892d964f1c00d946377044d0be09d46a871b9",
    "dev705_paired_coverage": "2ac4abb1a0cb9e684f6bde6b2990419eb0ec209d469411b240dc7dd520e69d3d",
    "dev705_score_template": "65388e3f32388f8aea10b022e1a9c229848f10b19401fcecc292e410a48e5ee4",
    "dev705_paired_mask_geometry": "37b6fb7eb7485e3c3d204da2689e0089285b1009c6e8e5d828eedd2a798f1002",
    "dev705_paired_mask_failure_scope": "2d226adb92a16fed3a2888d42316dcd8963d289c872e8cc15eb0a729770cf369",
}
OUTPUT_STATUS = "HRSID_COMMON_FIT_DEV705_CLOSED_DIAGNOSTICS_COMPLETE_V1"


def load_diagnostics(code_root):
    import sys

    previous = {name: sys.modules.get(name) for name in DIAG_PINS}
    modules = {}
    try:
        for name, digest in DIAG_PINS.items():
            module = load_source(code_root / DIAG_DIR / (name + ".py"), digest, name)
            sys.modules[name] = modules[name] = module
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    return modules


def score_with_large(score, gt, native_rows, center_rows):
    evaluate_cell, evaluator_type = score.evaluate_cell, score.COCOeval
    large_ap = []

    def measure_cell(*args, **kwargs):
        captured = []

        def capture(*ev_args):
            evaluator = evaluator_type(*ev_args)
            captured.append(evaluator)
            return evaluator

        score.COCOeval = capture
        try:
            result = evaluate_cell(*args, **kwargs)
        finally:
            score.COCOeval = evaluator_type
        evaluator, = captured
        params = evaluator.params
        area = list(params.areaRngLbl).index("large")
        max_det = list(params.maxDets).index(100)
        require(list(params.catIds) == [1] and list(params.areaRng[area]) == [9216, 1e10],
                "Exact official COCO large ship area and maxDets required")
        precision = evaluator.eval["precision"][:, :, 0, area, max_det]
        valid = precision[precision > -1]
        large_ap.append(float(valid.mean()) if valid.size else -1.)
        return result

    score.evaluate_cell = measure_cell
    try:
        result = score.evaluate_pair(gt, native_rows, center_rows)
    finally:
        score.evaluate_cell = evaluate_cell
    require(len(large_ap) == 4, "Exactly four original COCO large-area cells required")
    for (geometry, donor), value in zip(((g, d) for g in ARMS for d in ARMS), large_ap, strict=True):
        result["cells"][geometry][donor]["APl"] = value
    aa, ab, ba, bb = large_ap
    for name, left, right in (
        ("score_on_native_geometry", aa, ab), ("score_on_center_geometry", ba, bb),
        ("geometry_with_native_scores", aa, ba), ("geometry_with_center_scores", ab, bb),
    ):
        result["contrasts_points"][name]["APl"] = 100. * (right - left) if min(left, right) >= 0 else None
    return result


def failure_with_support(failure, gt, native_rows, center_rows):
    from contextlib import redirect_stdout
    from io import StringIO

    from pycocotools import mask
    from pycocotools.coco import COCO

    match_official = failure.match_official
    match_ids = []

    def capture(*args):
        matches = match_official(*args)
        match_ids.append(set(matches))
        return matches

    failure.match_official = capture
    try:
        result = failure.evaluate_pair(gt, native_rows, center_rows)
    finally:
        failure.match_official = match_official
    native, center = match_ids
    with redirect_stdout(StringIO()):
        coco = COCO()
        coco.dataset = gt
        coco.createIndex()

    support = {key: dict(eligible_gt=0, both=0, native_only=0, center_only=0, neither=0)
               for key in failure.STRATA}
    cases = ("neither", "center_only", "native_only", "both")
    for ann in gt["annotations"]:
        target = coco.annToRLE(ann)
        if mask.area(target) == 0 or ann.get("iscrowd", 0) or ann.get("ignore", 0):
            continue
        row = support[failure.stratum(target, ann["area"])]
        row["eligible_gt"] += 1
        index = 2 * (ann["id"] in native) + (ann["id"] in center)
        row[cases[index]] += 1
    require(
        sum(row["eligible_gt"] for row in support.values()) + sum(result["exclusions"].values())
        == len(gt["annotations"])
        and all(row["eligible_gt"] == sum(row[key] for key in cases)
                and row["both"] == result["strata"][stratum]["paired_gt"] for stratum, row in support.items()),
        "Complete original eligible GT and official paired matches must reconcile by stratum",
    )
    return result, support


def analyze_diagnostics(code_root, gt, native_rows, center_rows):
    """Analyze already admitted inputs without opening labels or predictions again."""
    modules = load_diagnostics(Path(code_root).absolute())
    score = modules["dev705_score_template"]
    coverage = modules["dev705_paired_coverage"]
    failure = modules["dev705_paired_mask_failure_scope"]
    args = (gt, native_rows, center_rows)
    scored = score_with_large(score, *args)
    covered = coverage.evaluate_pair(*args)
    failed, support = failure_with_support(failure, *args)
    rows = scored["retained_prefix_rows"]
    require(rows == covered["equal_nonempty_prefix"]["rows_per_arm"] == failed["retained_prefix_rows"],
            "The same equal nonempty candidate quota is required for all three diagnostics")
    return dict(
        status=OUTPUT_STATUS, official_TEST_opened=False, equal_prefix_score_template=scored,
        paired_candidate_coverage=covered, paired_mask_failure_scope=failed,
        eligible_gt_match_support_by_stratum=support,
    )


def main(argv=None):
    parser = ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--pair", type=Path, required=True)
    parser.add_argument("--pair-sha256", required=True)
    parser.add_argument("--native-arm-receipt", type=Path, required=True)
    parser.add_argument("--center-arm-receipt", type=Path, required=True)
    parser.add_argument("--training-protocol", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.output.exists() and not args.output.is_symlink() and args.output.parent.is_dir(),
            "Fresh output in an existing destination directory required")
    gt, native, center, provenance = verified_inputs(
        args.code_root, args.pair, args.pair_sha256, native_arm_receipt=args.native_arm_receipt,
        center_arm_receipt=args.center_arm_receipt, training_protocol=args.training_protocol,
    )
    result = dict(analyze_diagnostics(args.code_root, gt, native, center), provenance=provenance)
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
