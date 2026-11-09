"""Admit one independently accepted HRSID repeat to the unchanged DEV705 evaluator."""
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
import sys
from types import ModuleType


TRAINING_PROTOCOL = "a49361b92977a605267025b1e7798420936cbe53"
REPEAT_SEEDS = (5202, 5203)
REPEAT_TRAINER_SHA = "42a174573a04c529dfbae943bcc88ef4ed5d793d7eda6766511396034fda321f"
P203_SHA = "488f9ed73943cc842162c0d45822b6bac42aecabf37c8ecf0d4dde8e96c0959b"
WITNESS_FIELD = "repeat_seed_witness_sha256"


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def load_phase(path):
    """Load an isolated, byte-pinned copy of the P203 evaluation bridge."""
    path = Path(path)
    require(path.is_absolute() and path.resolve(strict=True) == path, "Exact resolved P203 source required")
    limit = 64 * 1024
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    require(0 < len(raw) <= limit and sha256(raw).hexdigest() == P203_SHA,
            "Exact immutable P203 evaluation source required")
    phase = ModuleType("_repeat_scoped_p203_dev705")
    phase.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), phase.__dict__)
    phase.TRAINING_PROTOCOL = TRAINING_PROTOCOL
    return phase


def read_pair(phase, path, digest, *, native_path, center_path, mode, seed):
    """Read the exact repeat pair and both arms before loading the scientific evaluator."""
    require(type(seed) is int and seed in REPEAT_SEEDS and mode in ("native", "center")
            and phase.TRAINING_PROTOCOL == TRAINING_PROTOCOL, "Exact authorized DEV705 repeat seed and arm required")
    path, native_path, center_path = (Path(item) for item in (path, native_path, center_path))
    require(len({path, native_path, center_path}) == 3, "Distinct pair and arm receipt paths required")
    original_read = phase.checked_receipt
    original_phase = phase.expected_phase
    witnesses = {}

    def checked_repeat_receipt(receipt_path, expected_digest):
        receipt = original_read(receipt_path, expected_digest)
        require(type(receipt) is dict, "Expected repeat receipt object")
        if receipt_path == path:
            value = receipt.pop("seed", None)
            require(type(value) is int and value == seed, "Exact authorized repeat pair seed required")
        else:
            witness = receipt.pop(WITNESS_FIELD, None)
            require(phase.is_sha256(witness), "Original repeat seed witness SHA-256 required for both arms")
            witnesses[receipt_path] = witness
        return receipt

    def expected_phase(arm_mode):
        return dict(original_phase(arm_mode), seed=seed, phase_trainer_sha256=REPEAT_TRAINER_SHA)

    # P203 enforces all original pair, closure, endpoint and digest requirements.
    phase.checked_receipt = checked_repeat_receipt
    phase.expected_phase = expected_phase
    try:
        arm = phase.read_pair(path, digest, native_path=native_path, center_path=center_path, mode=mode)
    finally:
        phase.checked_receipt = original_read
        phase.expected_phase = original_phase
    require(witnesses[native_path] != witnesses[center_path], "Distinct original repeat seed witnesses required")
    return dict(arm, **{WITNESS_FIELD: witnesses[native_path if mode == "native" else center_path]})


def bind_evaluator(phase, base, filtered, arm, *, seed):
    """Retain P203's endpoint and arm gates and add the saved runtime seed check."""
    require(type(seed) is int and seed in REPEAT_SEEDS and arm["phase"]["seed"] == seed
            and arm["phase"]["protocol"] == phase.TRAINING_PROTOCOL == TRAINING_PROTOCOL,
            "Exact repeat seed and protocol before DEV705 binding required")
    phase.bind_evaluator(base, filtered, arm)
    check_phase_endpoint = base.check_endpoint

    def check_endpoint(checkpoint, training_protocol):
        args = checkpoint.get("args")
        require(type(args) is dict and type(args.get("seed")) is int and args["seed"] == seed,
                "Exact repeat checkpoint args.seed required")
        return check_phase_endpoint(checkpoint, training_protocol)

    base.check_endpoint = check_endpoint


def main():
    entry = ArgumentParser(add_help=False, allow_abbrev=False)
    entry.add_argument("--phase-source", type=Path, required=True)
    entry.add_argument("--filtered-source", type=Path, required=True)
    entry.add_argument("--base-source", type=Path, required=True)
    entry.add_argument("--pair-receipt", type=Path, required=True)
    entry.add_argument("--pair-receipt-sha256", required=True)
    entry.add_argument("--native-arm-receipt", type=Path, required=True)
    entry.add_argument("--center-arm-receipt", type=Path, required=True)
    entry.add_argument("--arm", choices=("native", "center"), required=True)
    entry.add_argument("--training-seed", type=int, choices=REPEAT_SEEDS, required=True)
    args, downstream = entry.parse_known_args()

    phase = load_phase(args.phase_source)
    arm = read_pair(phase, args.pair_receipt, args.pair_receipt_sha256, native_path=args.native_arm_receipt,
                    center_path=args.center_arm_receipt, mode=args.arm, seed=args.training_seed)
    filtered, base = phase.load_evaluator(args.filtered_source, args.base_source)
    bind_evaluator(phase, base, filtered, arm, seed=args.training_seed)
    original_args = sys.argv[1:]
    sys.argv[1:] = downstream
    try:
        base.main()
    finally:
        sys.argv[1:] = original_args


if __name__ == "__main__":
    main()
