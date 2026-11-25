"""Freeze both independently returned supplemental references before scoring."""

import argparse
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
from types import ModuleType


PRELOCK_SHA256 = "67be071057cbdc8a63fdefea7514e92a097e6f388f91a9ebd6fa3079908ba6f5"
PACKET_PRELOCK_SHA256 = "775451046ef9ff3e084e80f7fd0df8bbe9a5883cc22243ed0583d3cfec923d4d"
VALIDATOR_SHA256 = "e49bc030433ef833db6574fa75426774d65f4abfd2d7f7c5c8eedf848e34d9db"
ROLES = ("A", "B")
WINDOWS = tuple(f"UMB-{index:03d}" for index in range(101, 125))
STATUS = "UMBRA_SUPPLEMENT_DUAL_REVIEW_ADMISSION_COMPLETE"
ROSTER_FILE = "ROSTER_PINNED.json"
LEDGER_FILE = "ANNOTATION_ADMISSION_PRIVATE.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return sha256(raw).hexdigest()


def pin(path, expected):
    raw = path.read_bytes()
    require(digest(raw) == expected, f"SHA-256 mismatch for {path.name}")
    return raw


def encode(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def write_exact(path, raw):
    with path.open("xb") as output:
        require(output.write(raw) == len(raw), "Short write")
        output.flush()
        os.fsync(output.fileno())
    require(path.read_bytes() == raw, "Admission output readback differs")


class BytesSnapshot(BytesIO):
    def __init__(self, name, raw):
        super().__init__(raw)
        self.name = name

    def read_text(self, *, encoding):
        return self.getvalue().decode(encoding)


def parse_originals(validator_path, roster_bytes, reviews):
    require(set(reviews) == set(ROLES) and digest(reviews["A"]) != digest(reviews["B"]),
            "Two distinct original role-bound reviews are required")
    validator = ModuleType("umbra_supplement_polygon_validator")
    source = pin(validator_path, VALIDATOR_SHA256)
    exec(compile(source, str(validator_path), "exec"), validator.__dict__)
    roster = validator.read_roster(BytesSnapshot(ROSTER_FILE, roster_bytes))
    expected_order = tuple(key for key in WINDOWS if key in roster)
    require(tuple(roster) == expected_order and 18 <= len(roster) <= 24
            and set(roster.values()) == {(1024, 1024)}, "Supplemental packet roster differs")
    parsed = {}
    for role in ROLES:
        embedded_role, items = validator.read_review(BytesSnapshot(f"REVIEWER_{role}_ORIGINAL.json", reviews[role]), roster)
        require(embedded_role == role, f"The role-{role} polygon file has a different reviewer role")
        parsed[role] = items
    return validator, roster, parsed


def freeze_originals(validator_path, roster_bytes, reviews, output, lineage):
    validator, roster, parsed = parse_originals(validator_path, roster_bytes, reviews)
    report = validator.agreement_report(roster, parsed["A"], parsed["B"])
    complete = [key for key in roster if all(parsed[role][key][0] in validator.COMPLETE for role in ROLES)]
    require(len(complete) == report["agreement"]["jointly_completed_windows"], "Admission denominator differs")
    identity = dict(lineage, roster_sha256=digest(roster_bytes),
                    validator_sha256=VALIDATOR_SHA256, scope_windows=len(roster),
                    review_sha256={role: digest(reviews[role]) for role in ROLES}, predictions_opened=False)
    aggregates = {key: report[key] for key in ("a_review", "b_review", "status_pairs", "agreement")}
    ledger = encode(dict(schema="umbra-supplement-dual-review-admission-v1", **identity, **aggregates,
                         jointly_completed_ids=complete, windows=report["windows"]))
    receipt = dict(status=STATUS, phase="request", worker_sha256=digest(Path(__file__).read_bytes()),
                   admission_sha256=digest(ledger), **identity, **aggregates)
    output.mkdir()
    for role in ROLES:
        write_exact(output / f"REVIEWER_{role}_ORIGINAL.json", reviews[role])
    write_exact(output / ROSTER_FILE, roster_bytes)
    write_exact(output / LEDGER_FILE, ledger)
    write_exact(output / "RESULT_SAFE.json", encode(receipt))
    return receipt


def packet_binding(run, terminal_sha256, announced_worker):
    terminal = json.loads(pin(run / "observed_packet/TERMINAL.json", terminal_sha256))
    result = json.loads(pin(run / "actual/RESULT_SAFE.json", terminal["result_sha256"]))
    require(terminal["phase"] == result["phase"] == "packet-build"
            and type(terminal["actual_exit"]) is int and terminal["actual_exit"] == 0
            and terminal["output_footer_verified"] is True
            and terminal["worker_sha256"] == result["worker_sha256"] == announced_worker
            and result["status"] == "TWO_BLIND_SUPPLEMENT_REVIEW_PACKETS_VERIFIED"
            and result["prelock_sha256"] == PACKET_PRELOCK_SHA256
            and result["requested_windows"] == 24
            and 18 <= result["windows_per_packet"] <= 24
            and result["windows_per_packet"] + result["unavailable_windows"] == 24
            and all(result[key] == 0 for key in ("labels_present", "model_outputs_present", "source_coordinates_present")),
            "A closed prediction-blind supplemental packet build is required")
    roster_bytes = pin(run / "actual/ROSTER.json", result["anonymous_roster_sha256"])
    require(len(json.loads(roster_bytes)["images"]) == result["windows_per_packet"], "Packet denominator differs")
    lineage = dict(protocol_sha256=PRELOCK_SHA256, packet_terminal_sha256=terminal_sha256,
                   packet_result_sha256=terminal["result_sha256"], packet_worker_sha256=announced_worker)
    return roster_bytes, lineage


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("protocol", "validator", "packet-run", "a", "b", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("packet-terminal-sha256", "packet-worker-sha256", "a-sha256", "b-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    pin(args.protocol, PRELOCK_SHA256)
    require(not args.output.exists(), "Use a fresh review admission directory")
    roster, lineage = packet_binding(args.packet_run, args.packet_terminal_sha256, args.packet_worker_sha256)
    reviews = {"A": pin(args.a, args.a_sha256), "B": pin(args.b, args.b_sha256)}
    receipt = freeze_originals(args.validator, roster, reviews, args.output, lineage)
    print(json.dumps(dict(status=STATUS, result_sha256=digest(encode(receipt)))), flush=True)


if __name__ == "__main__":
    main()
