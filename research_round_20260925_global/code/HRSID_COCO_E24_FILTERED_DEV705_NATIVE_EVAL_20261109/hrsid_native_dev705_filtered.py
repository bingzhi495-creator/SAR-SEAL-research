"""Admit the repaired HRSID E24 endpoint to the pinned DEV705 evaluator."""
from argparse import ArgumentParser
from dataclasses import dataclass
from functools import partial
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
from types import ModuleType

ORIGINAL_SHA = "1ce0eaa2106fe52452129fad534d0fc11bb8ad5fc4611f70858dbdf6f87a5ec3"
TRAINING_PROTOCOL = "cff45e6e91975e571278a0cf689f180e6b2a652e"
FILTER_SHA = "3b164cbf8c104539eabee8775c8e1f297d9a1607e37473c4d7a4a9b561f93471"
ENDPOINT_KEY = "sar_seal_hrsid_coco_fit_native_endpoint_v1"
STEPS = 24 * 735


@dataclass(frozen=True)
class AcceptedEndpoint:
    checkpoint_sha256: str
    filtered_empty_rows_total: int


def need(condition, message):
    if not condition:
        raise ValueError(message)


def checked_bytes(path, expected_sha256, size_limit):
    need(path.is_absolute() and path.resolve(strict=True) == path, "Use exact resolved input path")
    with path.open("rb") as stream:
        raw = stream.read(size_limit + 1)
    need(0 < len(raw) <= size_limit and sha256(raw).hexdigest() == expected_sha256,
         "Pinned input size or SHA-256 mismatch")
    return raw


def load_original_evaluator(path):
    raw = checked_bytes(path, ORIGINAL_SHA, 64 * 1024)
    module = ModuleType("_pinned_hrsid_native_dev705")
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def read_acceptance(path, expected_sha256):
    """Read a bridge receipt written only after independent training qualification."""
    receipt = json.loads(checked_bytes(path, expected_sha256, 8192))
    required = dict(
        status="HRSID_FILTERED_E24_DEV705_READY_V1",
        training_protocol=TRAINING_PROTOCOL,
        original_os_wait=True, original_os_exit=0, resource_stop=False,
        completed_epochs=24, optimizer_updates=STEPS, ema_updates=STEPS,
        target_filter_sha256=FILTER_SHA,
    )
    need(isinstance(receipt, dict)
         and set(receipt) == required.keys() | {"filtered_empty_rows_total", "checkpoint_sha256"}
         and all(type(receipt[key]) is type(value) and receipt[key] == value for key, value in required.items()),
         "Exact independent repaired E24 qualification required")
    count = receipt["filtered_empty_rows_total"]
    digest = receipt["checkpoint_sha256"]
    need(type(count) is int and count >= 1 and isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest),
         "Qualified filter count and E24 archive SHA-256 required")
    return AcceptedEndpoint(digest, count)


def check_filtered_endpoint(checkpoint, training_protocol, *, original_check, qualification):
    """Verify the two repaired marker fields, then reuse all original checks."""
    need(training_protocol == TRAINING_PROTOCOL, "Exact repaired HRSID training protocol")
    marker = checkpoint[ENDPOINT_KEY]
    count = marker.get("filtered_empty_rows_total")
    need(marker.get("target_filter_sha256") == FILTER_SHA
         and type(count) is int and count == qualification.filtered_empty_rows_total,
         "Exact repaired endpoint filter SHA-256 and independent row count")
    # The original checker intentionally accepts only the original marker schema.
    normalized_marker = dict(marker)
    del normalized_marker["target_filter_sha256"]
    del normalized_marker["filtered_empty_rows_total"]
    original_check(dict(checkpoint, **{ENDPOINT_KEY: normalized_marker}), training_protocol)


def main():
    entry = ArgumentParser(add_help=False, allow_abbrev=False)
    entry.add_argument("--base-source", type=Path, required=True)
    entry.add_argument("--acceptance", type=Path, required=True)
    entry.add_argument("--acceptance-sha256", required=True)
    args, downstream = entry.parse_known_args()
    base = load_original_evaluator(args.base_source)
    acceptance = read_acceptance(args.acceptance, args.acceptance_sha256)
    base.check_endpoint = partial(check_filtered_endpoint, original_check=base.check_endpoint,
                                  qualification=acceptance)

    def bound(operation):
        def run(parsed):
            need(parsed.training_protocol == TRAINING_PROTOCOL
                 and parsed.checkpoint_sha256 == acceptance.checkpoint_sha256,
                 "Exact repaired E24 protocol and independently accepted archive SHA-256")
            operation(parsed)
        return run

    base.run_export = bound(base.run_export)
    base.run_score = bound(base.run_score)
    sys.argv[1:] = downstream
    base.main()


if __name__ == "__main__":
    main()

