"""Count only pooled editing opportunities from the verified closed Umbra ledger."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path


WINDOWS = 12
CANVAS = [1024, 1024]
PHASES = {
    "capture": "UMBRA12_FROZEN_CAPTURE_COMPLETE",
    "verify-capture": "UMBRA12_FROZEN_CAPTURE_READBACK_COMPLETE",
    "prepare-edits": "UMBRA12_FROZEN_EDITS_PREPARED",
    "verify-edits": "UMBRA12_FROZEN_EDITS_READBACK_COMPLETE",
}
PROTOCOL = "ff70161af804adc9f52c044771b504884ea80b7b2242ffc38fab51b55f2839ad"
CARRIER_SAFE = "29f6f58a12cf72cb0da444c24be9939d0dacbab56d6a664f289a1a5f5b88a665"
PRELOCK = "58bab8f3482ff1c6dedfa301f895225991ece91fcdbaaf093d792f54060bbaaf"
OBSERVE = "e3f1529066d5897cfb98664d6b691ec32b27273e9a4ceeb6d51f158837da59e4"
POLICY = "d753f673dcd43f1a65b817be176b4d77a05332a66d01fe98c7833f86248b5feb"
WRITER = "be1516489123153a018ebc721806190d83f05285617e7c108e9b3c1064f66655"
INTAKE_PINS = {
    "intake_terminal": "1aa3cb27c6f5205503c51b48f6a3a03104194365b3914639da507b5629f296b3",
    "intake_result": "52482b1bb53d3ec8ca2ee4eebd9984aa5cacbb156873177841a5a2e2c4cdb7ec",
    "intake_readback": "11a99f558de5ebc9bf38bb70f0ec4516e68a37887b91f5825e3a6f41e7db784c",
    "packet_receipt": "004c0557b4e7ff7e1299227b601c438039fa69ea4a15c291f8964a763779cd81",
}
ALLOWED_ARMS = {"fixed", "free_matched_count", "free_natural", "native_matched_count",
                "attraction_dynamic", "attraction_static_at_dynamic_count", "learned_at_attraction_count"}


def require(test, message):
    if not test:
        raise ValueError(message)


def digest(raw):
    return sha256(raw).hexdigest()


def unique_object(items):
    result = {}
    for key, value in items:
        require(key not in result, "Duplicate JSON key")
        result[key] = value
    return result


def parse(raw):
    return json.loads(raw, object_pairs_hook=unique_object)


def pin(path, expected):
    require(path.resolve(strict=True) == path, "Input must be an exact regular path")
    raw = path.read_bytes()
    require(digest(raw) == expected, "Pinned bytes differ")
    return raw


def primary_edit(edit):
    require(type(edit) is dict and set(edit) == {"swaps", "rle"},
            "Expected original signed edit record")
    swaps, rle = edit["swaps"], edit["rle"]
    require(type(swaps) is int and 1 <= swaps <= 8, "Expected a strictly positive fixed budget")
    require(type(rle) is dict and set(rle) == {"size", "counts"} and rle["size"] == CANVAS
            and isinstance(rle["counts"], str) and bool(rle["counts"])
            and rle["counts"].isascii(), "Expected canonical native writer RLE")
    return swaps, rle["counts"]


def aggregate(raw):
    """Return only four counts. Never decode or return a spatial array."""
    lines = raw.splitlines()
    require(len(lines) == WINDOWS, "Expected the twelve frozen acquisition rows")
    positive_groups = different_groups = positive_slots = different_slots = 0
    for ordinal, line in enumerate(lines, 1):
        row = parse(line)
        require(type(row) is dict and set(row) == {"image_id", "changes"}
                and type(row["image_id"]) is int and row["image_id"] == ordinal
                and type(row["changes"]) is dict, "Ordered immutable acquisitions required")
        positives = differences = 0
        for compact_slot, edits in row["changes"].items():
            require(len(compact_slot) <= 3 and compact_slot.isascii() and compact_slot.isdecimal()
                    and not compact_slot.startswith("0") and 1 <= int(compact_slot) <= 100,
                    "Expected a compact unique detector slot")
            require(type(edits) is dict and edits and set(edits) <= ALLOWED_ARMS
                    and ("fixed" in edits) == ("free_matched_count" in edits),
                    "Expected both equal-count arms together or neither")
            if "fixed" not in edits:
                continue
            fixed_count, fixed_rle = primary_edit(edits["fixed"])
            free_count, free_rle = primary_edit(edits["free_matched_count"])
            require(fixed_count == free_count, "Realized per-slot signed budgets differ")
            positives += 1
            differences += fixed_rle != free_rle
        positive_slots += positives
        different_slots += differences
        positive_groups += positives > 0
        different_groups += differences > 0
    return {
        "positive_fixed_pair_acquisitions": positive_groups, "positive_fixed_pair_slots": positive_slots,
        "different_equal_count_acquisitions": different_groups, "different_equal_count_slots": different_slots,
    }


def verify_carrier(release, carrier_run, prelock, intake_paths):
    pin(prelock, PRELOCK)
    pin(release / "UMBRA12_FROZEN_CARRIER_PRELOCK_20261123.md", PROTOCOL)
    evidence = parse(pin(release / "results" / "UMBRA12_FROZEN_CARRIER_ACTUAL_SAFE_20261123.json",
                         CARRIER_SAFE))
    require(evidence["protocol_sha256"] == PROTOCOL and evidence["images"] == WINDOWS
            and evidence["exact_native_parity"] is evidence["exact_support_and_xor"]
            is evidence["exact_independent_replay"] is True
            and evidence["target_labels_opened"] is evidence["scoring_executed"] is False,
            "Original frozen and independently verified carrier required")

    code = release / "code"
    carrier_code = code / "UMBRA12_FROZEN_CARRIER_20261123"
    pin(code / "UEIKAP_RESERVED_HULL_FROZEN_20261122" / "observe.py", OBSERVE)
    pin(code / "UEIKAP_RFDETR_CARRIER_20261122" / "edit_rf.py", POLICY)
    pin(code / "PSEG_RFDETR_FROZEN_SOURCE_HEAD_20261122" / "replay.py", WRITER)

    observed, results = evidence["observed"], {}
    for phase, status in PHASES.items():
        item = observed[phase]
        worker = "capture_umbra.py" if phase in ("capture", "verify-capture") else "edit_umbra.py"
        pin(carrier_code / worker, item["worker_sha256"])
        terminal = parse(pin(carrier_run / ("observed_" + phase) / "TERMINAL.json",
                             item["terminal_sha256"]))
        result = parse(pin(carrier_run / phase / "RESULT_SAFE.json", item["result_sha256"]))
        require(terminal["phase"] == result["phase"] == phase and result["status"] == status
                and type(terminal["actual_exit"]) is int and terminal["actual_exit"] == item["actual_exit"] == 0
                and terminal["output_footer_verified"] is item["output_footer_verified"] is True
                and terminal["result_sha256"] == item["result_sha256"]
                and terminal["worker_sha256"] == item["worker_sha256"] == result["worker_sha256"],
                "Exact natural exit and observed worker footer required")
        require(result["protocol_sha256"] == PROTOCOL and result["images"] == WINDOWS
                and result["intake_sha256"] == evidence["intake_sha256"]
                and result["anonymous_roster_sha256"] == evidence["anonymous_roster_sha256"]
                and result["checkpoint_sha256"] == evidence["checkpoint_sha256"]
                and result["predictions_sha256"] == evidence["prediction_ledger_sha256"]
                and result["audit_sha256"] == evidence["candidate_audit_sha256"]
                and result["target_labels_opened"] is False and result["optimizer_updates"] == 0,
                "Original frozen phase lineage required")
        results[phase] = result

    capture, capture_readback, prepared, checked = (results[phase] for phase in PHASES)
    intake_terminal, intake, intake_readback, packet = (
        parse(pin(intake_paths[key], expected)) for key, expected in INTAKE_PINS.items()
    )
    require(intake_terminal["phase"] == "native-window-intake" and intake_terminal["actual_exit"] == 0
            and intake_terminal["output_footer_verified"] is True
            and intake_terminal["result_sha256"] == INTAKE_PINS["intake_result"]
            and intake_readback["extraction_result_sha256"] == INTAKE_PINS["intake_result"]
            and intake_readback["extraction_actual_exit"] == 0 and intake_readback["annotations_read"] == 0
            and intake["verified_native_windows"] == intake_readback["unique_pixel_arrays"] == WINDOWS
            and intake["manifest_sha256"] == intake_readback["manifest_sha256"]
            == packet["frozen_manifest_sha256"] == evidence["intake_sha256"]
            and packet["anonymous_roster_sha256"] == evidence["anonymous_roster_sha256"]
            and packet["windows_per_packet"] == WINDOWS and packet["labels_present"] == packet["model_outputs_present"] == 0,
            "Exact original intake readback and blind packet preparation required")
    capture_links = {
        "intake_terminal_sha256": INTAKE_PINS["intake_terminal"],
        "intake_result_sha256": INTAKE_PINS["intake_result"],
        "intake_readback_sha256": INTAKE_PINS["intake_readback"],
        "reviewer_receipt_sha256": INTAKE_PINS["packet_receipt"],
    }
    require(all(capture[key] == capture_readback[key] == val for key, val in capture_links.items())
            and capture["model_forward_calls"] == WINDOWS and capture_readback["model_forward_calls"] == 0
            and capture_readback["exact_native_parity"] is True and capture["counts"] == capture_readback["counts"]
            and capture_readback["capture_sha256"] == observed["capture"]["result_sha256"],
            "Original capture to independent readback links differ")

    edit_links = {
        "capture_result_sha256": observed["capture"]["result_sha256"],
        "capture_terminal_sha256": observed["capture"]["terminal_sha256"],
        "capture_readback_result_sha256": observed["verify-capture"]["result_sha256"],
        "capture_readback_terminal_sha256": observed["verify-capture"]["terminal_sha256"],
        "capture_worker_sha256": observed["capture"]["worker_sha256"],
        "point_head_sha256": evidence["point_head_sha256"],
        "edits_sha256": evidence["edit_ledger_sha256"],
    }
    require(all(prepared[key] == checked[key] == val for key, val in edit_links.items())
            and prepared["exact_support_and_xor"] is checked["exact_support_and_xor"] is True
            and prepared["exact_independent_replay"] is False and checked["exact_independent_replay"] is True
            and prepared["detector_forwards"] == checked["detector_forwards"] == 0
            and checked["prepared_result_sha256"] == observed["prepare-edits"]["result_sha256"]
            and checked["prepared_terminal_sha256"] == observed["prepare-edits"]["terminal_sha256"],
            "Original preparation and exact independent edit replay links differ")
    return evidence["edit_ledger_sha256"], capture["counts"]["eligible"]


def run(args):
    require(not args.result.exists(), "Use a new result destination")
    require(args.result.parent.resolve(strict=True) == args.result.parent
            and args.carrier_run not in (args.result.parent, *args.result.parent.parents)
            and args.release not in (args.result.parent, *args.result.parent.parents),
            "Use a private result outside the carrier and release roots")
    ledger_hash, max_slots = verify_carrier(args.release, args.carrier_run, args.prelock,
                                            {key: getattr(args, key) for key in INTAKE_PINS})
    raw = pin(args.carrier_run / "prepare-edits" / "EDITS_PRIVATE.jsonl", ledger_hash)
    counts = aggregate(raw)
    require(counts["positive_fixed_pair_slots"] <= max_slots, "Positive opportunity count exceeds area gate")
    compatible = counts["positive_fixed_pair_slots"] >= 20 and counts["positive_fixed_pair_acquisitions"] >= 8
    result = {
        "status": "UMBRA12_COORDINATOR_BLIND_EDIT_COVERAGE_VERIFIED",
        "phase": "verify-edits",
        "worker_sha256": digest(Path(__file__).read_bytes()),
        "coverage_prelock_sha256": PRELOCK,
        "carrier_evidence_sha256": CARRIER_SAFE,
        "frozen_protocol_sha256": PROTOCOL,
        "verified_edit_ledger_sha256": ledger_hash,
        "acquisitions": WINDOWS,
        "pooled_counts": counts,
        "original_gate_not_excluded_by_opportunity_ceiling": compatible,
        "opportunity_ceiling_each_reference": True,
        "original_image_bytes_opened": False,
        "reviewer_returns_opened": False,
        "scoring_executed": False,
    }
    result_bytes = (json.dumps(result, sort_keys=True, allow_nan=False, indent=2) + "\n").encode()
    with args.result.open("xb") as file:
        file.write(result_bytes)
        file.flush()
        os.fsync(file.fileno())
    print(json.dumps({"status": result["status"], "result_sha256": digest(result_bytes)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("prelock", "release", "carrier-run", *map(lambda name: name.replace("_", "-"), INTAKE_PINS), "result"):
        parser.add_argument("--" + name, type=Path, required=True)
    try:
        run(parser.parse_args())
    except (ValueError, KeyError, TypeError, OSError) as error:
        # Keep potentially identifying private schema values and filesystem paths out of logs.
        if type(error) is ValueError:
            parser.exit(1, f"Coverage verification failed: {error}\n")
        parser.exit(1, "Coverage verification failed on input schema or access.\n")


if __name__ == "__main__":
    main()

