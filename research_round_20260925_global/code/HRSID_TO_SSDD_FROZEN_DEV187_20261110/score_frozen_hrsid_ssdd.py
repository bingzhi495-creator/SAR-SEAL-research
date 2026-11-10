"""Score closed frozen HRSID E24 exports on SSDD's fixed integer DEV187."""

import argparse
from builtins import __import__ as standard_import
from hashlib import sha256
from importlib.metadata import version
import json
from math import fsum, isfinite
import os
from pathlib import Path
import re
from types import ModuleType


CODE = Path(__file__).resolve().parent.parent
SCORE_SOURCE = {
    "ssdd_phase_init": "444e91009bce7a56e65a9bd2110c3479a86609d29b5ccdc925b06ec013b7894a",
    "ssdd_phase_dev_admission": "6b5ffa17ecb06ace03038193415261405d2c90b85ebaed362223c88bdf86585b",
    "ssdd_phase_dev_score": "483e6251184b258475b97fd664aa76c784d599ce805cdd9d8db2f5e06a7e1696",
}
IMAGE_MANIFEST_SHA = "18df007e15e3e838e607f178f14e7a26f2b37b992ee39bf091e0b74cf084fb13"
INTEGER_GT_SHA = "5657600d6ea198f79bd5d557d2a1aadce5bec031d2ecf72a98716248d1183a55"
INTEGER_GT_BYTES = 118319
TRAIN_PROTOCOLS = {
    5201: "d66baf3c57fafdba4242ad1b55e08cf76bfd77f7",
    5202: "a49361b92977a605267025b1e7798420936cbe53",
    5203: "a49361b92977a605267025b1e7798420936cbe53",
}
MODES = ("native", "center")
METRICS = ("AP", "AP50", "AP75", "AP90", "APs", "AR100")
PLAN_STATUS = "FROZEN_HRSID_TO_SSDD_DEV187_CLOSED_PLAN_V1"
EXPORT_STATUS = "FROZEN_HRSID_TO_SSDD_DEV187_EXPORT_COMPLETE_PENDING_INDEPENDENT_ACCEPTANCE_V1"
SCORE_STATUS = "FROZEN_HRSID_TO_SSDD_DEV187_CPU_SCORE_PENDING_INDEPENDENT_ACCEPTANCE_V1"
REPORT_NAME = "DEV_NATIVE_EXPORT_SAFE.json"
PREDICTION_NAME = "DEV_NATIVE_PREDICTIONS_PRIVATE.jsonl"
MIB = 1024**2


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def hex_digest(value, length=64):
    return type(value) is str and re.fullmatch(f"[0-9a-f]{{{length}}}", value) is not None


def exact(value, fixed, extra=()):
    return (type(value) is dict and set(value) == set(fixed).union(extra)
            and all(type(value[key]) is type(expected) and value[key] == expected
                    for key, expected in fixed.items()))


def nonnegative_number(value):
    return (type(value) is int and 0 <= value < 2**63
            or type(value) is float and isfinite(value) and value >= 0)


def bounded_int(value, upper):
    return type(value) is int and 0 <= value <= upper


def token(stat):
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def stable_file(path, digest, limit, expected_bytes=None):
    require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file()
            and hex_digest(digest), "Canonical absolute pinned closed file required")
    before = token(path.stat())
    require(0 < before[2] <= limit and (expected_bytes is None or before[2] == expected_bytes),
            "Closed file size outside exact admission")
    with path.open("rb") as stream:
        require(token(os.fstat(stream.fileno())) == before, "Closed file changed before read")
        raw = stream.read(before[2] + 1)
        require(token(os.fstat(stream.fileno())) == before, "Closed file changed during read")
    require(len(raw) == before[2] and token(path.stat()) == before and sha256(raw).hexdigest() == digest,
            "Exact stable closed file SHA-256 required")
    return raw


def ref_path(ref, limit, name=None):
    require(exact(ref, {}, ("path", "bytes", "sha256")) and type(ref["path"]) is str
            and bounded_int(ref["bytes"], limit) and ref["bytes"] > 0 and hex_digest(ref["sha256"]),
            "Exact bounded closure reference required")
    path = Path(ref["path"])
    require(path.is_absolute() and ".." not in path.parts and ref["path"] in (str(path), path.as_posix())
            and (name is None or path.name == name), "Canonical closure reference path required")
    return path


def read_ref(ref, limit, name=None):
    return stable_file(ref_path(ref, limit, name), ref["sha256"], limit, ref["bytes"])


def unique_fields(pairs):
    result = dict(pairs)
    require(len(result) == len(pairs), "Duplicate JSON fields are forbidden")
    return result


def reject_constant(_):
    raise ValueError("Nonfinite JSON number is forbidden")


def document(raw):
    return json.loads(raw.decode("utf-8"), object_pairs_hook=unique_fields, parse_constant=reject_constant)


def load_scoring():
    dependencies = {}

    def pinned_import(name, globals=None, locals=None, fromlist=(), level=0):
        if not level and name in dependencies:
            return dependencies[name]
        return standard_import(name, globals, locals, fromlist, level)

    for name, digest in SCORE_SOURCE.items():
        path = CODE / "SSDD_MASK_PHASE_TRANSFER_20261110" / (name + ".py")
        source = stable_file(path, digest, MIB)
        module = ModuleType(name)
        module.__file__ = str(path)
        module.__dict__["__builtins__"] = dict(vars(standard_import("builtins")), __import__=pinned_import)
        exec(compile(source, str(path), "exec"), module.__dict__)
        dependencies[name] = module
    scoring = dependencies["ssdd_phase_dev_score"]
    path, digest = scoring.SOURCES["official"]
    return scoring, scoring.load_api(CODE / path, digest)


def admit_entry(entry, protocol, image_manifest):
    identity = ("seed", "mode", "source_commit", "checkpoint_sha256", "source_pair_receipt_sha256",
                "source_arm_receipt_sha256", "terminal", "report", "predictions")
    require(exact(entry, {}, identity) and type(entry["seed"]) is int and entry["seed"] in TRAIN_PROTOCOLS
            and entry["mode"] in MODES and hex_digest(entry["source_commit"], 40)
            and all(hex_digest(entry[key]) for key in identity[3:6]), "Exact preadmitted HRSID arm identity required")
    seed, mode = entry["seed"], entry["mode"]
    terminal_ref, report_ref, predictions_ref = (entry[key] for key in ("terminal", "report", "predictions"))
    terminal_path = ref_path(terminal_ref, MIB, "TERMINAL_PRIVATE.json")
    report_path = ref_path(report_ref, MIB, REPORT_NAME)
    predictions_path = ref_path(predictions_ref, 64 * MIB, PREDICTION_NAME)
    require(report_path.parent == predictions_path.parent, "Closed report and predictions must share one arm output")
    terminal = document(read_ref(terminal_ref, MIB))
    terminal_fixed = dict(original_os_wait=True, original_os_exit=0, resource_stop=False, worker_closed=True,
                          seed=seed, mode=mode)
    terminal_digests = ("export_lock_sha256", "stdout_sha256", "stderr_sha256")
    require(exact(terminal, terminal_fixed, (*terminal_digests, "elapsed_seconds", "peak_rss_bytes"))
            and all(hex_digest(terminal[key]) for key in terminal_digests)
            and nonnegative_number(terminal["elapsed_seconds"])
            and bounded_int(terminal["peak_rss_bytes"], 2**63-1), "Exact independently closed original export required")
    report = document(read_ref(report_ref, MIB))
    fixed = dict(status=EXPORT_STATUS, evaluation_protocol=protocol, source_training_protocol=TRAIN_PROTOCOLS[seed],
                 source_commit=entry["source_commit"], source_seed=seed, source_mode=mode,
                 checkpoint_sha256=entry["checkpoint_sha256"], source_pair_receipt_sha256=entry["source_pair_receipt_sha256"],
                 source_arm_receipt_sha256=entry["source_arm_receipt_sha256"], target_image_manifest_sha256=image_manifest,
                 predictions_sha256=predictions_ref["sha256"], images=187, raw_queries=30668, native_top_rows=18700,
                 model_forward_calls=187, precision="FP32_TF32_disabled", query_width=164, num_select=100,
                 checkpoint_ema_updates=17640, training=False, target_adaptation=False, GT_opened=False,
                 official_TEST_opened=False, AP=False)
    extras = ("target_image_zip_sha256", "mapped_rows", "unmapped_slot1_rows", "empty_image_predictions", "device",
              "elapsed_seconds", "peak_allocated_bytes", "peak_reserved_bytes")
    require(exact(report, fixed, extras) and hex_digest(report["target_image_zip_sha256"])
            and report["device"] in ("cpu", "cuda") and nonnegative_number(report["elapsed_seconds"])
            and bounded_int(report["peak_allocated_bytes"], 2 * MIB * 1024)
            and bounded_int(report["peak_reserved_bytes"], 3 * MIB * 1024)
            and report["peak_allocated_bytes"] <= report["peak_reserved_bytes"]
            and (report["device"] == "cuda" or report["peak_reserved_bytes"] == 0)
            and bounded_int(report["mapped_rows"], 18700) and bounded_int(report["unmapped_slot1_rows"], 18700)
            and report["mapped_rows"] + report["unmapped_slot1_rows"] == 18700
            and bounded_int(report["empty_image_predictions"], 187), "Exact scope, lineage and accounting of frozen export required")
    return terminal_path, report_path, predictions_path, terminal, report


def score_rows(raw, items, report):
    require(raw.endswith(b"\n"), "Complete frozen prediction JSONL required")
    lines = raw[:-1].split(b"\n")
    require(len(lines) == len(items) == 187, "Exactly 187 ordered prediction blocks required")
    flat, unmapped, empty_images = [], 0, 0
    for line, item in zip(lines, items, strict=True):
        block = document(line)
        image_id, hw = item["image_id"], item["original_hw"]
        require(exact(block, dict(image_id=image_id, raw_query_count=164, native_count=100),
                      ("unmapped_slot1_count", "native_ranks", "coco")), "Unchanged native width, original image ID and top100 required")
        rows, ranks, skipped = (block[key] for key in ("coco", "native_ranks", "unmapped_slot1_count"))
        require(type(rows) is type(ranks) is list and bounded_int(skipped, 100) and len(rows) == len(ranks) == 100-skipped
                and all(bounded_int(rank, 100) and rank > 0 for rank in ranks)
                and all(a < b for a, b in zip(ranks, ranks[1:])), "Unchanged retained native ranks and slot accounting required")
        previous = 1.
        for row in rows:
            require(exact(row, dict(image_id=image_id, category_id=1), ("score", "segmentation"))
                    and nonnegative_number(row["score"]) and row["score"] <= previous,
                    "Original category, native score and descending order required")
            rle = row["segmentation"]
            require(exact(rle, dict(size=hw), ("counts",)) and all(type(n) is int for n in rle["size"])
                    and type(rle["counts"]) is str and rle["counts"] and rle["counts"].isascii(),
                    "Original integer image-grid compressed RLE required")
            previous = row["score"]
        flat.extend(rows)
        unmapped += skipped
        empty_images += not rows
    require((len(flat), unmapped, empty_images) == (report["mapped_rows"], report["unmapped_slot1_rows"],
                                                  report["empty_image_predictions"]), "Frozen prediction accounting changed")
    return flat


def compare(arms):
    rows = []
    for seed in TRAIN_PROTOCOLS:
        a, b = (arms[seed, mode]["metrics"] for mode in MODES)
        rows.append(dict(seed=seed, center_minus_native_points={
            key: 100. * (b[key] - a[key]) if min(a[key], b[key]) >= 0 else None for key in METRICS}))
    mean = {}
    for key in METRICS:
        values = [row["center_minus_native_points"][key] for row in rows]
        mean[key] = fsum(values) / len(values) if all(value is not None for value in values) else None
    return dict(by_seed=rows, mean_center_minus_native_points=mean)


def evaluate_plan(plan_path, plan_sha256, protocol, *, seed=None, mode=None):
    """Require owner-authorized closed exports, then read fixed GT and call original SSDD COCO scoring."""
    require(hex_digest(protocol, 40) and ((seed is None and mode is None)
            or (type(seed) is int and seed in TRAIN_PROTOCOLS and mode in MODES)), "Exact protocol and complete arm selector required")
    plan = document(stable_file(plan_path, plan_sha256, MIB))
    require(exact(plan, dict(status=PLAN_STATUS, evaluation_protocol=protocol), ("image_manifest", "integer_gt", "arms"))
            and type(plan["arms"]) is list, "Exact closed evaluation plan required")
    expected = {(n, m) for n in TRAIN_PROTOCOLS for m in MODES} if seed is None else {(seed, mode)}
    require(len(plan["arms"]) == len(expected), "Plan must supply exactly the selected arm or all six arms")
    source_ref, gt_ref = (plan[name] for name in ("image_manifest", "integer_gt"))
    paths = [plan_path, ref_path(source_ref, MIB), ref_path(gt_ref, MIB)]
    require(source_ref["sha256"] == IMAGE_MANIFEST_SHA and gt_ref["sha256"] == INTEGER_GT_SHA
            and gt_ref["bytes"] == INTEGER_GT_BYTES, "Exact historical DEV187 image manifest and primary integer GT required")

    admitted = {}
    for entry in plan["arms"]:
        terminal_path, report_path, pred_path, terminal, report = admit_entry(entry, protocol, source_ref["sha256"])
        key = entry["seed"], entry["mode"]
        require(key in expected and key not in admitted, "Duplicate or unexpected closed source arm")
        paths.extend((terminal_path, report_path, pred_path))
        admitted[key] = entry, terminal, report
    require(len(set(paths)) == len(paths), "Every admitted input requires a distinct original path")
    require(len({report["device"] for _, _, report in admitted.values()}) == 1
            and len({report["target_image_zip_sha256"] for _, _, report in admitted.values()}) == 1,
            "All exports must use one device type and one exact target image ZIP")
    for key in ("checkpoint_sha256", "source_arm_receipt_sha256"):
        require(len({entry[key] for entry, _, _ in admitted.values()}) == len(expected), "Distinct original source arms required")
    require(len({entry["terminal"]["sha256"] for entry, _, _ in admitted.values()}) == len(expected)
            and len({terminal["export_lock_sha256"] for _, terminal, _ in admitted.values()}) == len(expected),
            "Every export needs its own independently closed terminal and lock")
    if seed is None:
        pairs = set()
        for source_seed in TRAIN_PROTOCOLS:
            a, b = (admitted[source_seed, name][0] for name in MODES)
            require(a["source_pair_receipt_sha256"] == b["source_pair_receipt_sha256"]
                    and a["source_commit"] == b["source_commit"], "Native and center must use the same accepted source pair and code")
            pairs.add(a["source_pair_receipt_sha256"])
        require(len(pairs) == len(TRAIN_PROTOCOLS), "Distinct accepted source pairs required across seeds")

    require(version("pycocotools") == "2.0.11", "Exact original pycocotools 2.0.11 CPU backend required")
    scoring, official = load_scoring()
    _, ids, items = scoring.fixed_images(scoring.Artifact(source_ref, read_ref(source_ref, MIB)))
    distinct = len({item["image_ref"]["sha256"] for item in items})
    require(distinct == 183, "Exact fixed 183 distinct target JPEG identities required")
    gt = document(read_ref(gt_ref, MIB))
    scoring.validate_gt(gt, ids, items)

    results = {}
    for key in ((n, m) for n in TRAIN_PROTOCOLS for m in MODES if (n, m) in expected):
        entry, terminal, report = admitted[key]
        rows = score_rows(read_ref(entry["predictions"], 64 * MIB), items, report)
        metrics = scoring.score_one(official, gt, rows, ids)
        results[key] = dict(seed=key[0], mode=key[1], source_training_protocol=TRAIN_PROTOCOLS[key[0]],
                            source_commit=entry["source_commit"], checkpoint_sha256=entry["checkpoint_sha256"],
                            source_pair_receipt_sha256=entry["source_pair_receipt_sha256"],
                            source_arm_receipt_sha256=entry["source_arm_receipt_sha256"],
                            export_terminal_sha256=entry["terminal"]["sha256"], export_lock_sha256=terminal["export_lock_sha256"],
                            export_report_sha256=entry["report"]["sha256"], predictions_sha256=entry["predictions"]["sha256"],
                            mapped_rows=report["mapped_rows"], empty_image_predictions=report["empty_image_predictions"], metrics=metrics)
    _, _, first = next(iter(admitted.values()))
    return dict(status=SCORE_STATUS, evaluation_protocol=protocol, closed_plan_sha256=plan_sha256,
                image_manifest_sha256=source_ref["sha256"], integer_gt_sha256=gt_ref["sha256"],
                image_zip_sha256=first["target_image_zip_sha256"], images=187, annotations=380, distinct_jpeg_sha256=distinct,
                device=first["device"], primary_GT="INTEGER_PRIMARY", max_dets=[1, 10, 100], units="fraction",
                arms=list(results.values()), paired_summary=compare(results) if seed is None else None,
                target_adaptation=False, training=False, DEV=True, official_TEST_opened=False, AP=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=tuple(TRAIN_PROTOCOLS))
    parser.add_argument("--mode", choices=MODES)
    args = parser.parse_args()
    output = args.output
    try:
        require(output.is_absolute() and output.parent.resolve(strict=True) == output.parent and not output.exists(),
                "Fresh absolute output path in an existing canonical directory required")
        result = evaluate_plan(args.plan, args.plan_sha256, args.protocol, seed=args.seed, mode=args.mode)
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    except (OSError, ValueError) as exc:
        parser.exit(3, f"Frozen SSDD score refused: {exc}\n")


if __name__ == "__main__":
    main()
