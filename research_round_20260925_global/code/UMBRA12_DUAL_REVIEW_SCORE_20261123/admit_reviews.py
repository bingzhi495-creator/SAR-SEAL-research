"""Admit both original Umbra reviewer returns before model-output access."""

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from types import ModuleType


PROTOCOL_NAME = "UMBRA12_FROZEN_CARRIER_PRELOCK_20261123.md"
PROTOCOL_SHA256 = "ff70161af804adc9f52c044771b504884ea80b7b2242ffc38fab51b55f2839ad"
ROSTER_SHA256 = "e626c38ea53727a2d235a80e675735898bbce5defce698f04fb602d674a87a56"
VALIDATOR_SHA256 = "e49bc030433ef833db6574fa75426774d65f4abfd2d7f7c5c8eedf848e34d9db"
WINDOWS = tuple(f"UMB-{index:03d}" for index in range(1, 13))
ADMISSION = "ANNOTATION_ADMISSION_PRIVATE.json"
ORIGINALS = {"A": "REVIEWER_A_ORIGINAL.json", "B": "REVIEWER_B_ORIGINAL.json"}
STATUS = "UMBRA12_DUAL_REVIEW_ADMISSION_COMPLETE"


@dataclass(frozen=True)
class PinnedJson:
    name: str
    raw: bytes

    def read_text(self, *, encoding):
        return self.raw.decode(encoding)


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(payload):
    return sha256(payload).hexdigest()


def pinned(path, expected):
    raw = path.read_bytes()
    require(digest(raw) == expected, f"SHA-256 mismatch for {path.name}")
    return raw


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def write_exact(path, payload):
    with path.open("xb") as stream:
        require(stream.write(payload) == len(payload), f"Short write for {path.name}")
        stream.flush()
        os.fsync(stream.fileno())
    require(path.read_bytes() == payload, f"Byte readback mismatch for {path.name}")


def validate(args):
    require(args.protocol.name == PROTOCOL_NAME and not args.output.exists(),
            "Expected the published prospective protocol and fresh output")
    pinned(args.protocol, PROTOCOL_SHA256)

    # Freeze both input byte strings before inspecting roles or polygon coordinates.
    review_hashes = {"A": args.a_sha256, "B": args.b_sha256}
    require(review_hashes["A"] != review_hashes["B"], "Expected two distinct original role-specific files")
    returns = {
        "A": PinnedJson(ORIGINALS["A"], pinned(args.a, review_hashes["A"])),
        "B": PinnedJson(ORIGINALS["B"], pinned(args.b, review_hashes["B"])),
    }
    roster_snapshot = PinnedJson("ROSTER_PINNED.json", pinned(args.roster, ROSTER_SHA256))
    validator = ModuleType("umbra_review_validator")
    validator.__file__ = str(args.validator)
    source = pinned(args.validator, VALIDATOR_SHA256)
    exec(compile(source, str(args.validator), "exec"), validator.__dict__)

    roster = validator.read_roster(roster_snapshot)
    require(tuple(roster) == WINDOWS and all(size == (1024, 1024) for size in roster.values()),
            "Expected the ordered twelve original native canvases")
    for role, original in returns.items():
        header = validator.read_json(original)
        require(isinstance(header, dict) and header.get("reviewer") == role,
                f"Expected reviewer {role} in the supplied role-{role} file")

    a_role, a_items = validator.read_review(returns["A"], roster)
    b_role, b_items = validator.read_review(returns["B"], roster)
    require((a_role, b_role) == ("A", "B"), "Expected exact review roles A and B")
    report = validator.agreement_report(roster, a_items, b_items)
    jointly_completed = sorted(
        window["image_id"] for window in report["windows"]
        if window["a_status"] in validator.COMPLETE and window["b_status"] in validator.COMPLETE
    )
    require(len(jointly_completed) == report["agreement"]["jointly_completed_windows"],
            "Jointly completed windows disagree with frozen reviewer agreement")
    identity = dict(protocol_sha256=PROTOCOL_SHA256, validator_sha256=VALIDATOR_SHA256,
                    roster_sha256=ROSTER_SHA256, review_sha256=review_hashes,
                    scope_windows=len(roster), predictions_opened=False)
    return returns, roster_snapshot, report, jointly_completed, identity


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("validator", "protocol", "roster", "a", "b", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    for role in ("a", "b"):
        parser.add_argument(f"--{role}-sha256", required=True,
                            help=f"SHA-256 of the separately received role-{role.upper()} original JSON")
    args = parser.parse_args()
    worker_sha = digest(Path(__file__).read_bytes())

    returns, roster, report, jointly_completed, identity = validate(args)
    aggregate = {key: report[key] for key in ("a_review", "b_review", "status_pairs", "agreement")}
    ledger = json_bytes(dict(schema="umbra12-dual-review-admission-v1", **identity, **aggregate,
                            jointly_completed_ids=jointly_completed, windows=report["windows"]))
    result = json_bytes(dict(status=STATUS, phase="request", worker_sha256=worker_sha,
                             admission_sha256=digest(ledger), **identity, **aggregate))
    args.output.mkdir()
    for snapshot in (*returns.values(), roster):
        write_exact(args.output / snapshot.name, snapshot.raw)
    write_exact(args.output / ADMISSION, ledger)
    write_exact(args.output / "RESULT_SAFE.json", result)
    print(json.dumps(dict(status=STATUS, result_sha256=digest(result))), flush=True)


if __name__ == "__main__":
    main()
