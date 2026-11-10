"""Verify original B/C DEV705 score closures and reuse paired geometry on CPU."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace


PLAN_STATUS = "HRSID_ONE_PASS_COMMON_FIT_E24_CLOSED_EXPORT_SCORE_PAIR_V1"
OUTPUT_STATUS = "HRSID_ONE_PASS_COMMON_FIT_E24_DEV705_PAIRED_GEOMETRY_COMPLETE"
STAGE_STATUS = {
    "export": "HRSID_ONE_PASS_COMMON_FIT_DEV705_ORIGINAL_EXPORT_CLOSED_V1",
    "score": "HRSID_ONE_PASS_COMMON_FIT_DEV705_ORIGINAL_SCORE_CLOSED_V1",
}
BRIDGE_SHA = "5e0814da015ae788450ce723af2d10061b3b673e0f1c41ef1edb949032c07697"
ADMISSION_SHA = "86d4ce7c6efc13387a62c3391ae2cbb13fa3ffb8310f0bfbd0bd535ddbd17ec5"
POSITIONAL = {"native": "stepwise_center", "center": "one_pass"}
MODES = tuple(POSITIONAL.values())


def load_support(code):
    """Load the immutable B/C admission and original CPU geometry in fresh namespaces."""
    if not code.is_absolute() or code.resolve(strict=True) != code:
        raise ValueError("Canonical ordinary code root required")
    common = code / "HRSID_FIT_RESIZE_PHASE_REPLAY_20261109"
    bridge_path = common / "common_fit_score_pair.py"
    if bridge_path.resolve(strict=True) != bridge_path:
        raise ValueError("Canonical original score bridge source required")
    with bridge_path.open("rb") as stream:
        raw = stream.read(64 * 1024 + 1)
    if not 0 < len(raw) <= 64 * 1024 or sha256(raw).hexdigest() != BRIDGE_SHA:
        raise ValueError("Exact immutable original score bridge required")
    bridge = ModuleType("_bc_closed_original_score_bridge")
    bridge.__file__ = str(bridge_path)
    exec(compile(raw, str(bridge_path), "exec"), bridge.__dict__)
    admission = bridge.load_source(
        common / "one_pass_common_eval_admission.py", ADMISSION_SHA, "_bc_closed_eval_admission",
    )
    original = admission.load_original(common / "common_fit_eval_admission.py")
    phase = original.load_phase(bridge.PHASE_SOURCE)
    driver = bridge.load_driver()
    driver.require(admission.MODES == MODES and driver.ARMS == tuple(POSITIONAL),
                   "Expected true B/C modes and original geometry argument order")
    return SimpleNamespace(bridge=bridge, admission=admission, original=original, phase=phase, driver=driver)


def _pin(driver, reference, seen, limit, *, basename=None):
    driver.exact(reference, {}, "Exact closed path, bytes and digest required", extra=("path", "bytes", "sha256"))
    legacy_pin = {key: reference[key] for key in ("path", "sha256")}
    path = driver.pin(legacy_pin)
    driver.require(type(reference["bytes"]) is int and 0 < reference["bytes"] <= limit
                   and path.stat().st_size == reference["bytes"] and (basename is None or path.name == basename)
                   and path not in seen, "Distinct original bounded artifact path and byte length required")
    seen.add(path)
    return path, legacy_pin


def _require_close(driver, value, operation, identity, terminals, *, extras, extra_identity=None):
    fixed = dict(identity, status=STAGE_STATUS[operation], owner_closed=True, worker_closed=True,
                 original_os_wait=True, original_os_exit=0, resource_stop=False)
    fixed.update(extra_identity or {})
    driver.exact(value, fixed, "Exact original closed B/C " + operation + " operation and lineage required",
                 extra=("terminal_sha256", "stdout_sha256", *extras))
    terminal = value["terminal_sha256"]
    driver.require(driver.is_digest(terminal) and driver.is_digest(value["stdout_sha256"])
                   and terminal not in terminals, "Replayed or invalid original B/C terminal evidence")
    terminals.add(terminal)


def _closed_arms(support, plan, training_arms, seen, training_sha):
    driver, bridge = support.driver, support.bridge
    driver.exact(plan["arms"], {}, "Exactly the two true B/C evaluation arms required", extra=MODES)
    terminals = {arm["terminal_sha256"] for arm in training_arms.values()}
    export_arms, score_digests, stage_terminals = {}, {}, {}
    for mode in MODES:
        arm = plan["arms"][mode]
        driver.exact(arm, {}, "Each B/C arm must carry original export and score closures", extra=("export", "score"))
        export, score = arm["export"], arm["score"]
        checkpoint = training_arms[mode]["checkpoint_sha256"]
        identity = dict(mode=mode, evaluation_protocol=plan["protocol"], training_protocol=plan["training_protocol"],
                        training_pair_sha256=training_sha, checkpoint_sha256=checkpoint)
        _require_close(driver, export, "export", identity, terminals, extras=("report", "predictions"))
        report_path, report_pin = _pin(driver, export["report"], seen, driver.MIB,
                                       basename="DEV_NATIVE_EXPORT_SAFE.json")
        predictions_path, predictions_pin = _pin(driver, export["predictions"], seen, 512 * driver.MIB,
                                                 basename="DEV_NATIVE_PREDICTIONS_PRIVATE.jsonl")
        driver.require(report_path.parent == predictions_path.parent, "Original report and predictions must share an arm")
        _require_close(driver, score, "score", identity, terminals, extras=("report",), extra_identity=dict(
            export_terminal_sha256=export["terminal_sha256"], export_report_sha256=report_pin["sha256"],
            predictions_sha256=predictions_pin["sha256"], original_dev_gt_sha256=driver.GT_SHA,
        ))
        _, score_pin = _pin(driver, score["report"], seen, driver.MIB)
        report = driver.read_json(report_path, report_pin["sha256"])
        report_identity = dict(
            status=driver.EXPORT_STATUS, protocol=plan["protocol"], training_protocol=plan["training_protocol"],
            checkpoint_sha256=checkpoint, predictions_sha256=predictions_pin["sha256"],
            image_manifest_sha256=driver.IMAGE_MANIFEST_SHA, GT_opened=False, official_TEST_opened=False, AP=False,
        )
        driver.require(type(report) is dict and all(type(report.get(key)) is type(value) and report[key] == value
                                                   for key, value in report_identity.items()),
                       "Exact original export report bound to its admitted B/C checkpoint required")
        score_digests[mode] = bridge.score_report(
            driver, dict(bridge.CLOSED_PROCESS, owner_closed=True, worker_closed=True, report=score_pin),
            evaluation_protocol=plan["protocol"], training_protocol=plan["training_protocol"],
            checkpoint_sha=checkpoint, predictions_sha=predictions_pin["sha256"],
        )
        stage_terminals[mode] = {"export": export["terminal_sha256"], "score": score["terminal_sha256"]}
        export_arms[mode] = dict(bridge.CLOSED_PROCESS, report=report_pin, predictions=predictions_pin)
    return export_arms, score_digests, stage_terminals


def _geometry_labels(metrics):
    point_five = dict(metrics["point_five"])
    common = dict(metrics["common_pair_geometry"])
    for alias, mode in POSITIONAL.items():
        point_five[f"{mode}_only"] = point_five.pop(f"{alias}_only")
        common[mode] = common.pop(alias)
    common["one_pass_minus_stepwise_center"] = common.pop("center_minus_native")
    return dict(metrics, point_five=point_five, common_pair_geometry=common,
                candidates={mode: metrics["candidates"][alias] for alias, mode in POSITIONAL.items()},
                strict_transition_axes=dict(row="stepwise_center", column="one_pass"))


def score_pair(code, plan_path, plan_sha256, *, evaluation_protocol, training_pair_sha256):
    support = load_support(code)
    driver, admission = support.driver, support.admission
    driver.require(driver.is_digest(evaluation_protocol, 40) and driver.is_digest(training_pair_sha256),
                   "Externally pinned evaluation protocol and accepted training pair digest required")
    plan = driver.read_json(plan_path, plan_sha256)
    driver.exact(plan, dict(status=PLAN_STATUS, stage="e24", training_seed=support.original.TRAINING_SEED,
                           protocol=evaluation_protocol), "Exact closed prospective B/C evaluation plan required",
                 extra=("training_protocol", "implementation_commit", "qualification_pair_sha256",
                        "training_pair", "stepwise_center_arm_receipt", "one_pass_arm_receipt", "gt", "arms"))
    driver.require(evaluation_protocol != plan["training_protocol"], "Evaluation must use its separately locked protocol")
    seen = {plan_path}
    pair_path, training_pin = _pin(driver, plan["training_pair"], seen, 8192)
    driver.require(training_pin["sha256"] == training_pair_sha256, "Exact independently accepted training pair pin")
    arm_paths = {mode: _pin(driver, plan[f"{mode}_arm_receipt"], seen, 8192)[0] for mode in MODES}
    pair, training_arms = admission.read_pair(
        support.original, support.phase, pair_path, training_pair_sha256,
        stepwise_center_path=arm_paths["stepwise_center"], one_pass_path=arm_paths["one_pass"],
        training_protocol=plan["training_protocol"], implementation_commit=plan["implementation_commit"],
        qualification_pair_sha256=plan["qualification_pair_sha256"],
    )
    driver.require(all(plan[f"{mode}_arm_receipt"]["sha256"] == pair[f"{mode}_arm_receipt_sha256"] for mode in MODES),
                   "Exact independently accepted per-arm receipt pins required")
    _, gt_pin = _pin(driver, plan["gt"], seen, 4 * driver.MIB)
    driver.require(gt_pin["sha256"] == driver.GT_SHA, "Fixed original DEV GT digest required")
    exports, scores, terminals = _closed_arms(support, plan, training_arms, seen, training_pair_sha256)

    # The old driver is a fresh private namespace. These are positional
    # inputs to its unchanged geometry and never become new receipt bytes.
    projected = dict(status=driver.PAIR_STATUS, protocol=evaluation_protocol, training_pair=training_pin, gt=gt_pin,
                     arms={alias: exports[mode] for alias, mode in POSITIONAL.items()})
    checkpoint_pins = {f"{alias}_checkpoint_sha256": training_arms[mode]["checkpoint_sha256"]
                       for alias, mode in POSITIONAL.items()}
    original_read = driver.read_json

    def projected_read(path, digest, limit=driver.MIB):
        if path == plan_path and digest == plan_sha256:
            return projected
        return original_read(path, digest, limit)

    def verified_training_pair(pin):
        driver.require(pin == training_pin, "Exact original accepted B/C training pair required")
        return checkpoint_pins

    driver.read_json = projected_read
    driver.training_pair = verified_training_pair
    driver.TRAINING_PROTOCOL = plan["training_protocol"]
    result = driver.score_pair(plan_path, plan_sha256)
    return dict(
        result, status=OUTPUT_STATUS, training_seed=support.original.TRAINING_SEED, support_policy=admission.POLICY,
        arms={mode: result["arms"][alias] for alias, mode in POSITIONAL.items()}, metrics=_geometry_labels(result["metrics"]),
        score_report_sha256s=scores, original_stage_terminal_sha256s=terminals, original_COCO_score_reports_verified=True,
        terminal_proof_scope="external independently verified original OS closure",
        training_implementation_commit=plan["implementation_commit"], qualification_pair_sha256=plan["qualification_pair_sha256"],
        geometry_driver_positional_inputs=POSITIONAL.copy(),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("code", "plan", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("plan-sha256", "evaluation-protocol", "training-pair-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    output = args.output
    if not output.is_absolute() or output.parent.resolve(strict=True) != output.parent or output.exists() or output.is_symlink():
        raise ValueError("Fresh canonical safe B/C geometry result required")
    result = score_pair(args.code, args.plan, args.plan_sha256, evaluation_protocol=args.evaluation_protocol,
                        training_pair_sha256=args.training_pair_sha256)
    raw = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with output.open("xb") as stream:
        if stream.write(raw) != len(raw):
            raise OSError("Incomplete safe B/C geometry output")
        stream.flush()
        os.fsync(stream.fileno())


if __name__ == "__main__":
    main()
