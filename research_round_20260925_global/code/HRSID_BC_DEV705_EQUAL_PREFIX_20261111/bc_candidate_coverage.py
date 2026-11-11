"""Compute secondary equal-count B/C coverage from verified closed DEV705 inputs."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
from types import ModuleType


GEOMETRY = ("HRSID_FIT_RESIZE_PHASE_REPLAY_20261109/one_pass_common_score_pair.py",
            "46bcb96d6b4b4fd34184fe5976588a46427c51d149a71ca05ac9d794fce66707")
COVERAGE = ("HRSID_MASK_PHASE_REPEAT_DEV705_20261109/dev705_paired_coverage.py",
            "2ac4abb1a0cb9e684f6bde6b2990419eb0ec209d469411b240dc7dd520e69d3d")
P309_SOURCE_SHA256 = "48a64174cadf433dba55f1cff403d6b5a61492e16f4cc94a459c069a354ca44c"
POSITIONAL = {"native": "stepwise_center", "center": "one_pass"}
SCOPES = ("full", "equal_mapped_prefix", "equal_nonempty_prefix")


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def read_pinned(path, expected, limit=64 * 1024):
    require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file(),
            "Exact canonical closed file required")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    require(0 < len(raw) <= limit and sha256(raw).hexdigest() == expected,
            "Pinned closed file size or digest differs")
    return raw


def load_source(code, pin, name):
    relative, expected = pin
    path = code / relative
    raw = read_pinned(path, expected)
    module = ModuleType(name)
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def label_coverage(scope):
    transitions = {}
    marginals = {}
    for threshold, areas in scope["any_candidate_transitions"].items():
        transitions[threshold] = {}
        marginals[threshold] = {}
        for area, row in areas.items():
            transitions[threshold][area] = dict(neither=row["neither"], both=row["both"],
                                               stepwise_center_only=row["native_only"],
                                               one_pass_only=row["center_only"])
            marginals[threshold][area] = dict(
                stepwise_center=row["both"] + row["native_only"],
                one_pass=row["both"] + row["center_only"],
                one_pass_minus_stepwise_center=row["center_only"] - row["native_only"],
            )
    return dict(scope, arms={mode: scope["arms"][alias] for alias, mode in POSITIONAL.items()},
                any_candidate_transitions=transitions, any_candidate_by_size=marginals)


def coverage_from_closed(args):
    geometry = load_source(args.code, GEOMETRY, "_bc_coverage_geometry")
    coverage = load_source(args.code, COVERAGE, "_bc_equal_candidate_coverage")
    support = geometry.load_support(args.code)
    driver = support.driver

    p309 = driver.decode(read_pinned(args.p309_wait, args.p309_wait_sha256))
    require(p309["status"] == "ROOT_ORIGINAL_TOOL_WAIT_OBSERVED_V1"
            and p309["source_sha256"] == P309_SOURCE_SHA256
            and p309["original_os_wait"] is True and p309["process_group_closed"] is True
            and type(p309["original_os_exit"]) is int and p309["original_os_exit"] == 0,
            "Successful original external P309 wait required")
    driver.exact(driver.decode(p309["combined_stdout"]),
                 dict(failure=None, postclose_plan_sha256=args.plan_sha256, resource_closed=True,
                      status="P309_WINDOWS_HRSID_BC_DEV705_CLOSED_PENDING_INDEPENDENT_ACCEPTANCE"),
                 "Exact original P309 wait must bind the closed B/C plan")

    p260 = driver.decode(read_pinned(args.p260_wait, args.p260_wait_sha256))
    require(p260["status"] == "BOUNDED_CPU_FIT_ORIGINAL_WAIT"
            and p260["worker_sha256"] == GEOMETRY[1]
            and p260["original_os_wait"] is True and p260["process_group_closed"] is True
            and type(p260["original_os_exit"]) is int and p260["original_os_exit"] == 0
            and p260["stop_reason"] is None,
            "Successful original P260 CPU wait required")
    raw = read_pinned(args.p260_result, args.p260_result_sha256)
    driver.exact(p260["result"], dict(bytes=len(raw), sha256=args.p260_result_sha256),
                 "P260 wait must bind the admitted result bytes")
    prior = driver.decode(raw)
    require(prior["status"] == geometry.OUTPUT_STATUS
            and prior["protocol"] == args.evaluation_protocol
            and prior["closed_export_pair_sha256"] == args.plan_sha256
            and prior["training_pair_receipt_sha256"] == args.training_pair_sha256
            and prior["original_COCO_score_reports_verified"] is True
            and prior["official_TEST_opened"] is False,
            "Exact previously verified P260 geometry scope required")

    ground_truth, blocks = [], []
    original_read = driver.read_json
    original_read_arm = driver.read_arm

    def capture_gt(path, digest, limit=driver.MIB):
        result = original_read(path, digest, limit)
        if digest == driver.GT_SHA:
            ground_truth.append(result)
        return result

    def capture_arm(arm, checkpoint, protocol, images):
        result = original_read_arm(arm, checkpoint, protocol, images)
        blocks.append((checkpoint, result[1]))
        return result

    driver.read_json = capture_gt
    driver.read_arm = capture_arm
    geometry.load_support = lambda code: support
    replay = geometry.score_pair(args.code, args.plan, args.plan_sha256,
                                 evaluation_protocol=args.evaluation_protocol,
                                 training_pair_sha256=args.training_pair_sha256)
    require(replay == prior and len(ground_truth) == 1 and len(blocks) == 2,
            "Pinned original P260 result must replay exactly")
    require([checkpoint for checkpoint, _ in blocks] ==
            [prior["arms"][mode]["checkpoint_sha256"] for mode in POSITIONAL.values()],
            "Coverage rows must retain original B/C arm order")
    rows = [[row for block in arm_blocks for row in block["coco"]] for _, arm_blocks in blocks]
    result = coverage.evaluate_pair(ground_truth[0], *rows)
    return dict(result, status="HRSID_BC_DEV705_PAIRED_CANDIDATE_COVERAGE_COMPLETE",
                **{scope: label_coverage(result[scope]) for scope in SCOPES},
                evaluation_protocol=args.evaluation_protocol, training_pair_sha256=args.training_pair_sha256,
                original_dev_gt_sha256=driver.GT_SHA, p309_wait_sha256=args.p309_wait_sha256,
                postclose_plan_sha256=args.plan_sha256, p260_wait_sha256=args.p260_wait_sha256,
                p260_result_sha256=args.p260_result_sha256, coverage_source_sha256=COVERAGE[1])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("code", "plan", "p309-wait", "p260-wait", "p260-result", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("plan-sha256", "p309-wait-sha256", "p260-wait-sha256", "p260-result-sha256",
                 "evaluation-protocol", "training-pair-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    output = args.output
    require(output.is_absolute() and output.parent.resolve(strict=True) == output.parent
            and not output.exists() and not output.is_symlink(), "Fresh canonical aggregate output required")
    result = coverage_from_closed(args)
    raw = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with output.open("xb") as stream:
        if stream.write(raw) != len(raw):
            raise OSError("Incomplete aggregate result")
        stream.flush()
        os.fsync(stream.fileno())


if __name__ == "__main__":
    main()
