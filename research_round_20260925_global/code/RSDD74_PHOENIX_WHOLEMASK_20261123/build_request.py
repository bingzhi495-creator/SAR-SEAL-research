"""Build the exact, label-free ordered whole-mask request from closed runs."""
import argparse
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

from PIL import Image
from pycocotools import mask as codec


STOCK_SHA = "41163bd418d13a05f9c52b28e7821458bfb5eafca0fcf761682436fbff75f71c"
STOCK_RESULT_SHA = "f43fb04035045b1e30b7ae06ced778e66fcc4cd3e34077f83ebff6096dacc71d"
STOCK_TERM_SHA = "af977d49d1a9c06399a145aaff5e19c7e2d36d0bc8311d88014f66079bbbdaf5"
ORIGINAL_REQUEST_SHA = "a40c0b8a3b6473a4c4f1f6c58c04f9cf6070f34c3ef7331783134a1dd77447ac"
ORIGINAL_RESULT_SHA = "08ba5d901c7c6d0b17a8b17cd8a3bb5f646139e66e2aae2459554cacb0fabd2a"
ORIGINAL_TERM_SHA = "f0e8fbaa487e0819c5f6e7ac2ecaadcee86edbf7d871accb27600021089008de"
ROSTER_SHA = "91f95a50e19c9809b9a84f9e5825d2b6a31476d970dae727f144e852b60d4d20"
STATUS = "PHOENIX_RSDD74_WHOLE_REQUEST"
HW = [512, 512]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(blob):
    return sha256(blob).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def pin(path, expected):
    data = path.read_bytes()
    require(digest(data) == expected, "Input digest mismatch: " + path.name)
    return data


def closed_result(directory, phase, result_sha, terminal_sha):
    result = json.loads(pin(directory / phase / "RESULT_SAFE.json", result_sha))
    terminal = json.loads(pin(directory / ("observed_" + phase) / "TERMINAL.json", terminal_sha))
    require(terminal["actual_exit"] == 0 and terminal["output_footer_verified"] is True
            and terminal["result_sha256"] == result_sha
            and terminal["worker_sha256"] == result["worker_sha256"], "Closed successful observation is required")
    return result


def request_rows(original_images, stock_rows):
    rows = []
    for number, (original, stock) in enumerate(zip(original_images, stock_rows, strict=True), 1):
        name = f"image_{number:03}.png"
        require(original["image_id"] == stock["image_id"] and original["image_file"] == name
                and len(stock["coco"]) + stock["unmapped_slot1_count"] == stock["native_count"],
                "Original image order or mapped candidate count differs")
        candidates = []
        for position, prediction in enumerate(stock["coco"], 1):
            rle = prediction["segmentation"]
            require(prediction["image_id"] == stock["image_id"] and set(rle) == {"size", "counts"}
                    and rle["size"] == HW and isinstance(rle["counts"], str), "Original candidate identity differs")
            candidates.append({"position": position, "segmentation": rle})
        for item in original["candidates"]:
            actual = candidates[item["position"] - 1]
            require(actual["position"] == item["position"] and actual["segmentation"] == item["segmentation"],
                    "Archived matched request used different mask pixels")
        rows.append({"image_file": name, "encoded_sha256": original["encoded_sha256"],
                     "decoded_rgb_sha256": original["decoded_rgb_sha256"], "candidates": candidates})
    return rows


def build(args):
    pin(args.protocol, args.protocol_sha256)
    closure = closed_result(args.baseline, "request", ORIGINAL_RESULT_SHA, ORIGINAL_TERM_SHA)
    require(closure["request_sha256"] == ORIGINAL_REQUEST_SHA
            and closure["input_lineage"]["predictions_sha256"] == STOCK_SHA
            and closure["target_labels_opened"] is False, "Closed label-free source run required")
    closed_result(args.stock_root, "capture", STOCK_RESULT_SHA, STOCK_TERM_SHA)
    original = json.loads(pin(args.baseline / "request" / "REQUEST_PRIVATE.json", ORIGINAL_REQUEST_SHA))
    source = pin(args.stock_root / "capture" / "PREDICTIONS_PRIVATE.jsonl", STOCK_SHA)
    rows = request_rows(original["images"], [json.loads(line) for line in source.splitlines()])
    identity = b"".join(encoded({"ordinal": i, "decoded_rgb_sha256": row["decoded_rgb_sha256"],
                                "candidates": row["candidates"]}) for i, row in enumerate(rows, 1))
    require(digest(identity) == ROSTER_SHA, "Ordered native population identity differs")
    counts = Counter(images=len(rows))
    images = []
    for row in rows:
        raw = pin(args.baseline / "request" / row["image_file"], row["encoded_sha256"])
        with Image.open(BytesIO(raw)) as im:
            require(im.format == "PNG" and im.mode == "RGB" and list(im.size) == HW
                    and digest(im.tobytes()) == row["decoded_rgb_sha256"], "Exact native RGB differs")
        images.append((row["image_file"], raw))
        counts["native_candidates"] += len(row["candidates"])
        for candidate in row["candidates"]:
            area = int(codec.area(candidate["segmentation"]))
            counts["native_empty"] += area == 0
            counts["lvis_candidates"] += area > 0
            counts["coco_candidates"] += area * 100 > HW[0] * HW[1]
    require(counts == dict(images=74, native_candidates=7346, native_empty=1267,
                           lvis_candidates=6079, coco_candidates=172), "Whole-mask area-gate counts differ")
    request = encoded({"status": STATUS, "protocol_sha256": args.protocol_sha256, "images": rows})
    args.output.mkdir()
    for name, raw in images:
        (args.output / name).write_bytes(raw)
    (args.output / "REQUEST_PRIVATE.json").write_bytes(request)
    return dict(status=STATUS + "_COMPLETE", phase="request", request_sha256=digest(request),
                worker_sha256=digest(Path(__file__).read_bytes()), protocol_sha256=args.protocol_sha256,
                ordered_native_population_sha256=ROSTER_SHA, stock_sha256=STOCK_SHA,
                target_labels_opened=False, counts=dict(counts))


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for field in ("baseline", "stock-root", "protocol", "output"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    args = parser.parse_args()
    result = build(args)
    body = encoded(result)
    (args.output / "RESULT_SAFE.json").write_bytes(body)
    print(json.dumps({"status": result["status"], "result_sha256": digest(body)}), flush=True)


if __name__ == "__main__":
    main()
