"""Admit both closed HRSID mask-resize E24 arms to fixed DEV705 evaluation."""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
from types import ModuleType


FILTERED_SOURCE_SHA = "9a69500d10d5b0a371ea504f1ba89b7335aca130c1a28d96c4f93a87b2a4b9c0"
TRAINING_PROTOCOL = "d66baf3c57fafdba4242ad1b55e08cf76bfd77f7"
FILTER_SHA = "3b164cbf8c104539eabee8775c8e1f297d9a1607e37473c4d7a4a9b561f93471"
INITIAL_SHA = "63ff93afdd977ba79b5b5a46ba8b4839ef6c40eb0722b29e4667d4932404fbc2"
PHASE_TRAINER_SHA = "8550b43791a6cb6f53736dc7e64b9c5c6204723cf4c07c3b112491ba1bd74043"
PHASE_PATCH_SHA = "811eccf2b42319272aea8b3f50a71bc029734d006bba43260f66117e9421948e"
NATIVE_TRAINER_SHA = "11d17cab878d62576b8d19e6231caab90c9990c03909cb01e1d04468add452a6"
PHASE_MARKER = "sar_seal_hrsid_mask_resize_phase_v1"
ARM_STATUS = "HRSID_MASK_PHASE_E24_ARM_ACCEPTED_V1"
PAIR_STATUS = "HRSID_MASK_PHASE_E24_PAIR_READY_V1"
EPOCHS = 24
UPDATES = 17640
VISITS = 70560


def require(condition, message):
    if not condition:
        raise ValueError(message)


def is_sha256(value):
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def checked_bytes(path, expected, limit):
    path = Path(path)
    require(path.is_absolute() and path.resolve(strict=True) == path, "Use exact resolved input path")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    require(0 < len(raw) <= limit and sha256(raw).hexdigest() == expected,
            "Pinned input size or SHA-256 mismatch")
    return raw


def unique_object(items):
    value = dict(items)
    require(len(value) == len(items), "Qualification JSON contains duplicate keys")
    return value


def checked_receipt(path, digest):
    return json.loads(checked_bytes(path, digest, 8192), object_pairs_hook=unique_object)


def exact_fields(actual, expected, message, *, extras=()):
    require(isinstance(actual, dict) and actual.keys() == expected.keys() | set(extras)
            and all(type(actual[key]) is type(value) and actual[key] == value
                    for key, value in expected.items()), message)


def expected_phase(mode):
    return dict(protocol=TRAINING_PROTOCOL, mask_resize_mode=mode, stage="e24", seed=5201,
                phase_trainer_sha256=PHASE_TRAINER_SHA, phase_patch_sha256=PHASE_PATCH_SHA,
                native_trainer_sha256=NATIVE_TRAINER_SHA, initial_expanded_sha256=INITIAL_SHA)


def read_arm(path, digest, mode, checkpoint_sha):
    arm = checked_receipt(path, digest)
    required = dict(status=ARM_STATUS, original_os_wait=True, original_os_exit=0, resource_stop=False,
                    completed_epochs=EPOCHS, optimizer_updates=UPDATES, ema_updates=UPDATES,
                    target_filter_sha256=FILTER_SHA, checkpoint_sha256=checkpoint_sha)
    digests = ("visit_ledger_sha256", "result_sha256", "stdout_sha256", "terminal_sha256", "phase_provenance_sha256")
    exact_fields(arm, required, "Exact independently accepted E24 arm required",
                 extras=("phase", "filtered_empty_rows_total", *digests))
    exact_fields(arm["phase"], expected_phase(mode), "Exact arm training phase and provenance required")
    require(all(is_sha256(arm[key]) for key in digests), "Arm evidence digests required")
    count = arm["filtered_empty_rows_total"]
    require(type(count) is int and 0 <= count <= 164 * VISITS, "Qualified filter row count required")
    return arm


def read_pair(path, digest, *, native_path, center_path, mode):
    """Read both independently accepted arms and return the requested arm."""
    require(mode in ("native", "center"), "Authorized evaluation arm required")
    pair = checked_receipt(path, digest)
    required = dict(status=PAIR_STATUS, training_protocol=TRAINING_PROTOCOL, ordered_fit_visits_match=True,
                    completed_epochs=EPOCHS, visits_per_arm=VISITS)
    digests = ("native_arm_receipt_sha256", "center_arm_receipt_sha256",
               "native_checkpoint_sha256", "center_checkpoint_sha256")
    exact_fields(pair, required, "Exact independently ready E24 pair required", extras=digests)
    require(all(is_sha256(pair[key]) for key in digests)
            and pair["native_arm_receipt_sha256"] != pair["center_arm_receipt_sha256"]
            and pair["native_checkpoint_sha256"] != pair["center_checkpoint_sha256"],
            "Both distinct E24 arm and checkpoint digests required")
    native = read_arm(native_path, pair["native_arm_receipt_sha256"], "native", pair["native_checkpoint_sha256"])
    center = read_arm(center_path, pair["center_arm_receipt_sha256"], "center", pair["center_checkpoint_sha256"])
    return native if mode == "native" else center


def load_evaluator(filtered_source, base_source):
    """Load the byte-pinned P153 compatibility layer and P149 evaluator."""
    raw = checked_bytes(filtered_source, FILTERED_SOURCE_SHA, 64 * 1024)
    name = "_hrsid_dev705_phase_pinned_p153"
    filtered = ModuleType(name)
    filtered.__file__ = str(filtered_source)
    previous = sys.modules.get(name)
    sys.modules[name] = filtered
    try:
        exec(compile(raw, str(filtered_source), "exec"), filtered.__dict__)
    finally:
        if previous is None:
            del sys.modules[name]
        else:
            sys.modules[name] = previous
    return filtered, filtered.load_original_evaluator(base_source)


def bind_evaluator(base, filtered, arm):
    """Authorize one closed arm while preserving the native export and score operations."""
    original_endpoint = base.check_endpoint
    phase = arm["phase"].copy()
    accepted = filtered.AcceptedEndpoint(arm["checkpoint_sha256"], arm["filtered_empty_rows_total"])
    # This module namespace is a fresh, byte-pinned copy of P153.
    filtered.TRAINING_PROTOCOL = TRAINING_PROTOCOL

    def check_endpoint(checkpoint, training_protocol):
        exact_fields(checkpoint.get(PHASE_MARKER), phase, "Exact accepted checkpoint phase marker required")
        filtered.check_filtered_endpoint(checkpoint, training_protocol, original_check=original_endpoint,
                                         qualification=accepted)

    def bound(operation):
        def run(args):
            require(args.training_protocol == TRAINING_PROTOCOL
                    and args.checkpoint_sha256 == accepted.checkpoint_sha256,
                    "Exact paired protocol and independently accepted arm archive SHA-256 required")
            return operation(args)
        return run

    base.check_endpoint = check_endpoint
    base.run_export = bound(base.run_export)
    base.run_score = bound(base.run_score)


def main():
    entry = ArgumentParser(add_help=False, allow_abbrev=False)
    entry.add_argument("--filtered-source", type=Path, required=True)
    entry.add_argument("--base-source", type=Path, required=True)
    entry.add_argument("--pair-receipt", type=Path, required=True)
    entry.add_argument("--pair-receipt-sha256", required=True)
    entry.add_argument("--native-arm-receipt", type=Path, required=True)
    entry.add_argument("--center-arm-receipt", type=Path, required=True)
    entry.add_argument("--arm", choices=("native", "center"), required=True)
    args, downstream = entry.parse_known_args()

    arm = read_pair(args.pair_receipt, args.pair_receipt_sha256, native_path=args.native_arm_receipt,
                    center_path=args.center_arm_receipt, mode=args.arm)
    filtered, base = load_evaluator(args.filtered_source, args.base_source)
    bind_evaluator(base, filtered, arm)
    original_args = sys.argv[1:]
    sys.argv[1:] = downstream
    try:
        base.main()
    finally:
        sys.argv[1:] = original_args


if __name__ == "__main__":
    main()
