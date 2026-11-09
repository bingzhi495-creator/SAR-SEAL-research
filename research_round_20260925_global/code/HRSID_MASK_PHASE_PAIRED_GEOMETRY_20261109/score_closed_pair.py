"""Score two pinned, OS-closed HRSID E24 native DEV705 exports on CPU."""

import argparse
from hashlib import sha256
from importlib.metadata import version
import json
import math
from pathlib import Path
import re
from types import ModuleType


PAIR_STATUS = "HRSID_MASK_PHASE_E24_CLOSED_EXPORT_PAIR_V1"
TRAINING_PAIR_STATUS = "HRSID_MASK_PHASE_E24_PAIR_READY_V1"
EXPORT_STATUS = "HRSID_DEV705_E24_NATIVE_EXPORT_COMPLETE"
OUTPUT_STATUS = "HRSID_MASK_PHASE_E24_DEV705_PAIRED_GEOMETRY_COMPLETE"
TRAINING_PROTOCOL = "d66baf3c57fafdba4242ad1b55e08cf76bfd77f7"
GT_SHA = "393b3da233991dc5fafc98b4ddeaa341b042d9571434ef8ce53c3f2fdada5628"
IMAGE_MANIFEST_SHA = "39a40d089d4a54448fe6efca8ea5cc98ad1eb7c0587815f3dd9a3b097173e78f"
CORE_SHA = "d8e3703ae4439e384cfa0c8cdb5dfc388e73b08b8ddf0903ede98151947e4f91"
DEV_IMAGES, GT_COUNT, EMPTY_COUNT = 705, 1782, 4
CANVAS = (800, 800)  # width, height
WIDTH, NATIVE_ROWS, UPDATES = 164, 100, 17640
ARMS = ("native", "center")
MIB = 1024**2


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def is_digest(value, size=64):
    return isinstance(value, str) and len(value) == size and re.fullmatch(r"[0-9a-f]+", value) is not None


def exact(actual, expected, reason, *, extra=()):
    require(
        isinstance(actual, dict)
        and actual.keys() == expected.keys() | set(extra)
        and all(type(actual[key]) is type(value) and actual[key] == value for key, value in expected.items()),
        reason,
    )


def unique_object(items):
    value = dict(items)
    require(len(items) == len(value), "Duplicate JSON key")
    return value


def json_constant(value):
    raise ValueError("Non-finite JSON constant: " + value)


def decode(data):
    return json.loads(data, object_pairs_hook=unique_object, parse_constant=json_constant)


def canonical(path):
    require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file(), "Canonical input file required")
    return path


def pin(value):
    exact(value, {}, "Exact path and digest required", extra=("path", "sha256"))
    require(isinstance(value["path"], str) and is_digest(value["sha256"]), "Path and lowercase SHA-256 required")
    return canonical(Path(value["path"]))


def read_bytes(path, digest, limit):
    require(is_digest(digest), "Lowercase SHA-256 required")
    with canonical(path).open("rb") as stream:
        data = stream.read(limit + 1)
    require(0 < len(data) <= limit and sha256(data).hexdigest() == digest, "Pinned input size or SHA-256 mismatch")
    return data


def read_json(path, digest, limit=MIB):
    return decode(read_bytes(path, digest, limit))


def training_pair(value):
    pair = read_json(pin(value), value["sha256"])
    digests = tuple(f"{arm}_{item}_sha256" for arm in ARMS for item in ("arm_receipt", "checkpoint"))
    exact(
        pair, dict(status=TRAINING_PAIR_STATUS, training_protocol=TRAINING_PROTOCOL,
                   ordered_fit_visits_match=True, completed_epochs=24, visits_per_arm=70560),
        "Exact accepted E24 training pair required", extra=digests,
    )
    require(all(is_digest(pair[key]) for key in digests) and len({pair[key] for key in digests}) == len(digests),
            "Distinct accepted E24 arm receipt and checkpoint digests required")
    return pair


def read_blocks(path, digest, images, report):
    blocks, total_bytes = [], 0
    mapped = unmapped = 0
    calculated = sha256()
    size = [CANVAS[1], CANVAS[0]]
    with path.open("rb") as stream:
        for image in images:
            line = stream.readline(512 * MIB - total_bytes + 1)
            total_bytes += len(line)
            require(total_bytes <= 512 * MIB and line.endswith(b"\n"), "Complete bounded native prediction JSONL")
            calculated.update(line)
            block = decode(line)
            exact(
                block, dict(image_id=image["id"], raw_query_count=WIDTH, native_count=NATIVE_ROWS),
                "Exact original image order and native query width required",
                extra=("unmapped_slot1_count", "native_ranks", "coco"),
            )
            rows = block["coco"]
            slot1 = block["unmapped_slot1_count"]
            require(
                isinstance(rows, list) and isinstance(block["native_ranks"], list)
                and type(slot1) is int and 0 <= slot1 <= NATIVE_ROWS and len(rows) + slot1 == NATIVE_ROWS,
                "Exact native candidate accounting required",
            )
            for row in rows:
                exact(row, dict(image_id=image["id"], category_id=1), "Exact mapped ship mask required",
                      extra=("score", "segmentation"))
                rle = row["segmentation"]
                exact(rle, dict(size=size), "Exact original-canvas RLE required", extra=("counts",))
                require(isinstance(rle["counts"], str) and re.fullmatch(r"[0-o]+", rle["counts"]) is not None,
                        "Expected ASCII compressed RLE")
            mapped += len(rows)
            unmapped += slot1
            blocks.append(block)
        require(stream.read(1) == b"", "Unexpected additional prediction lines")
    require(calculated.hexdigest() == digest, "Pinned prediction SHA-256 mismatch")
    require(mapped == report["mapped_rows"] and unmapped == report["unmapped_slot1_rows"],
            "Export report and prediction stream candidate counts differ")
    return blocks


def read_arm(arm, expected_checkpoint, protocol, images):
    exact(arm, dict(original_os_wait=True, original_os_exit=0, resource_stop=False),
          "Both original export processes must have closed cleanly", extra=("report", "predictions"))
    report_pin, predictions_pin = arm["report"], arm["predictions"]
    report_path, predictions_path = pin(report_pin), pin(predictions_pin)
    require(report_path.parent == predictions_path.parent and report_path != predictions_path,
            "One closed export directory per arm required")
    report = read_json(report_path, report_pin["sha256"])
    exact(
        report,
        dict(
            status=EXPORT_STATUS, protocol=protocol, checkpoint_sha256=expected_checkpoint,
            checkpoint_ema_updates=UPDATES, training_protocol=TRAINING_PROTOCOL,
            initialization="official_COCO_only", image_manifest_sha256=IMAGE_MANIFEST_SHA,
            predictions_sha256=predictions_pin["sha256"], images=DEV_IMAGES,
            raw_queries=DEV_IMAGES * WIDTH, native_top_rows=DEV_IMAGES * NATIVE_ROWS,
            num_select=NATIVE_ROWS, query_width=WIDTH, precision="FP32_TF32_disabled",
            GT_opened=False, official_TEST_opened=False, AP=False,
        ),
        "Exact closed native export report required",
        extra=("mapped_rows", "unmapped_slot1_rows", "peak_allocated", "elapsed_seconds"),
    )
    require(
        all(type(report[key]) is int and report[key] >= 0
            for key in ("mapped_rows", "unmapped_slot1_rows", "peak_allocated"))
        and report["mapped_rows"] + report["unmapped_slot1_rows"] == DEV_IMAGES * NATIVE_ROWS
        and type(report["elapsed_seconds"]) is float
        and math.isfinite(report["elapsed_seconds"]) and report["elapsed_seconds"] >= 0,
        "Exact closed export totals and telemetry required",
    )
    blocks = read_blocks(predictions_path, predictions_pin["sha256"], images, report)
    return report_path.parent, blocks


def score_pair(pair_path, pair_sha256):
    require(version("numpy") == "1.26.4" and version("pycocotools") == "2.0.11",
            "Pinned canonical CPU mask runtime required")
    pair = read_json(pair_path, pair_sha256)
    exact(pair, dict(status=PAIR_STATUS), "Exact closed export pair receipt required",
          extra=("protocol", "training_pair", "gt", "arms"))
    protocol = pair["protocol"]
    require(is_digest(protocol, 40), "Exact evaluation protocol commit required")
    trained = training_pair(pair["training_pair"])
    gt_path = pin(pair["gt"])
    require(pair["gt"]["sha256"] == GT_SHA, "Fixed original DEV GT digest required")
    gt = read_json(gt_path, GT_SHA, 4 * MIB)
    images = gt["images"]
    require(
        len(images) == DEV_IMAGES and len(gt["annotations"]) == GT_COUNT
        and all((image["width"], image["height"]) == CANVAS for image in images),
        "Fixed original DEV scope required",
    )
    arms = pair["arms"]
    exact(arms, {}, "Exactly native and center arms required", extra=ARMS)
    native, center = (
        read_arm(arms[name], trained[f"{name}_checkpoint_sha256"], protocol, images) for name in ARMS
    )
    require(native[0] != center[0], "Distinct closed arm export directories required")
    core_path = Path(__file__).resolve().with_name("paired_geometry.py")
    core = ModuleType("_pinned_paired_geometry")
    exec(compile(read_bytes(core_path, CORE_SHA, 64 * 1024), str(core_path), "exec"), core.__dict__)
    metrics = core.evaluate_pair(gt, native[1], center[1])
    require(metrics["exclusions"]["empty"] == EMPTY_COUNT, "Fixed original DEV empty raster count required")
    return dict(
        status=OUTPUT_STATUS, protocol=protocol, training_protocol=TRAINING_PROTOCOL,
        closed_export_pair_sha256=pair_sha256, training_pair_receipt_sha256=pair["training_pair"]["sha256"],
        original_dev_gt_sha256=GT_SHA, geometry_core_sha256=CORE_SHA,
        arms={
            name: dict(checkpoint_sha256=trained[f"{name}_checkpoint_sha256"],
                       export_report_sha256=arms[name]["report"]["sha256"],
                       predictions_sha256=arms[name]["predictions"]["sha256"])
            for name in ARMS
        },
        official_TEST_opened=False, COCO_AP=False, metrics=metrics,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--pair", type=Path, required=True)
    parser.add_argument("--pair-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    require(
        output.is_absolute() and output.parent.resolve(strict=True) == output.parent
        and not output.exists() and not output.is_symlink(),
        "Fresh canonical output file required",
    )
    result = score_pair(args.pair, args.pair_sha256)
    data = (json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode()
    with output.open("xb") as stream:
        stream.write(data)


if __name__ == "__main__":
    main()
