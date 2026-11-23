"""Build a label-free Phoenix XL1 request from the closed native Umbra carrier."""

import argparse
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path
import sys

from PIL import Image
from pycocotools import mask as codec


PROTOCOL_NAME = "UMBRA12_PHOENIX_WHOLE_PRELOCK_20261123.md"
PROTOCOL_SHA = "49fb049a6986d1f3d34adb7469b3239f5742701723e84f4dce8e60a4425210cb"
CARRIER_CLOSURE = "29f6f58a12cf72cb0da444c24be9939d0dacbab56d6a664f289a1a5f5b88a665"
ROSTER = "e626c38ea53727a2d235a80e675735898bbce5defce698f04fb602d674a87a56"
STATUS = "PHOENIX_UMBRA12_WHOLE_REQUEST"
HW = [1024, 1024]
PHASES = {
    "capture": "UMBRA12_FROZEN_CAPTURE_COMPLETE",
    "verify-capture": "UMBRA12_FROZEN_CAPTURE_READBACK_COMPLETE",
    "prepare-edits": "UMBRA12_FROZEN_EDITS_PREPARED",
    "verify-edits": "UMBRA12_FROZEN_EDITS_READBACK_COMPLETE",
}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(data):
    return sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def pin(path, expected):
    require(path.resolve(strict=True) == path, "Expected regular pinned input path")
    data = path.read_bytes()
    require(digest(data) == expected, "Input digest differs: " + path.name)
    return data


def observed_carrier(args):
    closure = json.loads(pin(args.carrier_closure, CARRIER_CLOSURE))
    require(closure["status"] == "FROZEN_BLIND_CANDIDATES_AND_EDIT_LEDGER_VERIFIED"
            and closure["anonymous_roster_sha256"] == ROSTER
            and closure["images"] == closure["detector_forwards"] == 12
            and closure["optimizer_updates"] == 0 and closure["scoring_executed"] is False
            and closure["target_labels_opened"] is False and closure["exact_native_parity"] is True
            and closure["exact_support_and_xor"] is True and closure["exact_independent_replay"] is True
            and set(closure["observed"]) == set(PHASES), "Exact label-free carrier closure required")
    stages = {}
    for phase, status in PHASES.items():
        binding = closure["observed"][phase]
        term = json.loads(pin(args.capture_root / ("observed_" + phase) / "TERMINAL.json", binding["terminal_sha256"]))
        report = json.loads(pin(args.capture_root / phase / "RESULT_SAFE.json", binding["result_sha256"]))
        require(term["phase"] == report["phase"] == phase and report["status"] == status
                and type(term["actual_exit"]) is int and term["actual_exit"] == binding["actual_exit"] == 0
                and term["output_footer_verified"] is binding["output_footer_verified"] is True
                and term["result_sha256"] == binding["result_sha256"]
                and term["worker_sha256"] == report["worker_sha256"] == binding["worker_sha256"]
                and report["protocol_sha256"] == closure["protocol_sha256"]
                and report["anonymous_roster_sha256"] == ROSTER and report["intake_sha256"] == closure["intake_sha256"]
                and report["checkpoint_sha256"] == closure["checkpoint_sha256"]
                and report["images"] == 12 and report["target_labels_opened"] is False
                and report["optimizer_updates"] == 0, "Missing observed label-free closure: " + phase)
        stages[phase] = report
    cap, checked = stages["capture"], stages["verify-capture"]
    prepared, replayed = stages["prepare-edits"], stages["verify-edits"]
    require(cap["model_forward_calls"] == 12 and checked["model_forward_calls"] == 0
            and checked["capture_sha256"] == closure["observed"]["capture"]["result_sha256"]
            and checked["exact_native_parity"] is True and cap["counts"] == checked["counts"]
            and cap["predictions_sha256"] == checked["predictions_sha256"] == closure["prediction_ledger_sha256"]
            and cap["audit_sha256"] == checked["audit_sha256"] == closure["candidate_audit_sha256"]
            and prepared["edits_sha256"] == replayed["edits_sha256"] == closure["edit_ledger_sha256"]
            and prepared["detector_forwards"] == replayed["detector_forwards"] == 0
            and prepared["point_head_sha256"] == replayed["point_head_sha256"] == closure["point_head_sha256"]
            and replayed["prepared_result_sha256"] == closure["observed"]["prepare-edits"]["result_sha256"]
            and replayed["capture_readback_result_sha256"] == closure["observed"]["verify-capture"]["result_sha256"]
            and replayed["exact_independent_replay"] is True, "Closed capture/readback and replay lineage differ")
    return closure, cap["counts"]["mapped"]


def native_rgb(path, expected_png, expected_rgb):
    source = pin(path, expected_png)
    with Image.open(BytesIO(source), formats=["PNG"]) as image:
        require(image.format == "PNG" and image.mode == "L" and image.size == tuple(HW) and not image.info,
                "Original anonymous native grayscale input required")
        rgb = image.convert("RGB")
    require(digest(rgb.tobytes()) == expected_rgb, "Native grayscale reproduction differs from candidate audit")
    stream = BytesIO()
    rgb.save(stream, format="PNG")
    exported = stream.getvalue()
    with Image.open(BytesIO(exported), formats=["PNG"]) as check:
        require(check.mode == "RGB" and check.size == tuple(HW) and digest(check.tobytes()) == expected_rgb,
                "Transport PNG changed a native pixel")
    return exported


def admit_stock(ordinal, pred, audit):
    candidates = pred["coco"]
    mapped = audit["candidates"]
    ranks = pred["native_ranks"]
    require(pred["image_id"] == audit["image_id"] == ordinal and audit["original_hw"] == HW
            and pred["raw_query_count"] == 164 and pred["native_count"] == 100
            and len(candidates) + pred["unmapped_slot1_count"] == 100
            and len(candidates) == len(mapped) == len(ranks) == len(audit["query_ids"])
            and ranks == audit["native_ranks"] and ranks == sorted(set(ranks))
            and all(type(rank) is int and 1 <= rank <= 100 for rank in ranks),
            "Original compact candidate population or joint rank is inconsistent")
    admitted = []
    previous_score = math.inf
    for position, (row, entry, rank, query) in enumerate(zip(candidates, mapped, ranks, audit["query_ids"], strict=True), 1):
        rle = row["segmentation"]
        score = row["score"]
        require(row["image_id"] == ordinal and row["category_id"] == 1 and entry["position"] == position
                and entry["native_rank"] == rank and entry["query_id"] == query and entry["score"] == score
                and math.isfinite(score) and 0 <= score <= 1 and score <= previous_score
                and set(rle) == {"size", "counts"} and rle["size"] == HW
                and isinstance(rle["counts"], str) and bool(rle["counts"]),
                "Stock candidate position, score, or native RLE differs")
        previous_score = score
        area = int(codec.area(rle))
        require(0 <= area <= HW[0] * HW[1] and entry["mask_area"] == area, "Stock foreground area disagrees")
        if area > 0:
            admitted.append(dict(position=position, segmentation=rle))
    return admitted


def request_data(args):
    require(args.protocol.name == PROTOCOL_NAME and args.protocol_sha256 == PROTOCOL_SHA, "New Phoenix prelock required")
    pin(args.protocol, PROTOCOL_SHA)
    closure, original_count = observed_carrier(args)
    roster_doc = json.loads(pin(args.roster, ROSTER))
    roster = roster_doc["images"]
    require(roster_doc["schema"] == "sar-hull-roster-v1" and len(roster) == 12, "Frozen anonymous image roster required")
    cap = args.capture_root / "capture"
    native = [json.loads(line) for line in pin(cap / "PREDICTIONS_PRIVATE.jsonl", closure["prediction_ledger_sha256"]).splitlines()]
    audits = [json.loads(line) for line in pin(cap / "CANDIDATE_AUDIT_PRIVATE.jsonl", closure["candidate_audit_sha256"]).splitlines()]
    images = []
    pngs = []
    counts = Counter(images=12, native_candidates=0, native_empty=0, nonempty_candidates=0)
    for ordinal, (item, pred, audit) in enumerate(zip(roster, native, audits, strict=True), 1):
        anonymous = f"UMB-{ordinal:03d}"
        source_path = args.intake / "images" / (anonymous + ".png")
        require(item["image_id"] == anonymous and [item["height"], item["width"]] == HW
                and audit["source_image_name"] == source_path.name and audit["acquisition_group"] == anonymous
                and audit["sensor"] == "Umbra" and audit["image_sha256"] == item["png_sha256"],
                "Ordered native image, cohort geometry or original PNG digest differs")
        rgb_sha = audit["rgb_sha256"]
        transport_bytes = native_rgb(source_path, item["png_sha256"], rgb_sha)
        admitted = admit_stock(ordinal, pred, audit)
        total_slots = len(pred["coco"])
        name = f"image_{ordinal:03d}.png"
        pngs.append((name, transport_bytes))
        images.append(dict(ordinal=ordinal, image_id=ordinal, image_file=name, hw=HW,
                           encoded_sha256=digest(transport_bytes), decoded_rgb_sha256=rgb_sha,
                           native_candidate_count=total_slots, candidates=admitted))
        counts.update(native_candidates=total_slots, native_empty=total_slots - len(admitted),
                      nonempty_candidates=len(admitted))
    require(counts["native_candidates"] == original_count
            and counts["native_empty"] + counts["nonempty_candidates"] == original_count
            and len({row["decoded_rgb_sha256"] for row in images}) == len(images) == 12,
            "Closed native candidate census or original image uniqueness differs")
    lineage = dict(carrier_closure_sha256=CARRIER_CLOSURE, carrier_protocol_sha256=closure["protocol_sha256"],
                   anonymous_roster_sha256=ROSTER, native_intake_manifest_sha256=closure["intake_sha256"],
                   native_prediction_ledger_sha256=closure["prediction_ledger_sha256"],
                   native_candidate_audit_sha256=closure["candidate_audit_sha256"],
                   native_edit_ledger_sha256=closure["edit_ledger_sha256"],
                   checkpoint_sha256=closure["checkpoint_sha256"],
                   ordered_native_population_sha256=digest(b"".join(encoded(row) for row in images)))
    return dict(status=STATUS, protocol_sha256=PROTOCOL_SHA, lineage=lineage, counts=dict(counts), images=images), pngs


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("intake", "roster", "capture-root", "carrier-closure", "protocol", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    args = parser.parse_args()
    require(sys.flags.isolated and sys.dont_write_bytecode, "Use isolated Python with bytecode writes disabled")
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve(strict=name != "output"))
    require(not args.output.exists(), "Fresh request directory required")
    request, pngs = request_data(args)
    request_bytes = encoded(request)
    args.output.mkdir()
    for name, data in pngs:
        (args.output / name).write_bytes(data)
    (args.output / "REQUEST_PRIVATE.json").write_bytes(request_bytes)
    result = dict(status=STATUS + "_COMPLETE", phase="request", protocol_sha256=PROTOCOL_SHA,
                  request_sha256=digest(request_bytes), worker_sha256=digest(Path(__file__).read_bytes()),
                  lineage=request["lineage"], counts=request["counts"], target_labels_opened=False,
                  native_detector_forwards=0, optimizer_updates=0)
    body = encoded(result)
    (args.output / "RESULT_SAFE.json").write_bytes(body)
    print(json.dumps(dict(status=result["status"], result_sha256=digest(body))), flush=True)


if __name__ == "__main__":
    main()
