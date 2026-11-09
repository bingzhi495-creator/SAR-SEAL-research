"""Admit an independently accepted common-support E24 arm to the original DEV705 scorer."""

from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
import re
import sys
from types import ModuleType


P203_SHA = "488f9ed73943cc842162c0d45822b6bac42aecabf37c8ecf0d4dde8e96c0959b"
FIRST_PROTOCOL = "d66baf3c57fafdba4242ad1b55e08cf76bfd77f7"
REPEAT_PROTOCOL = "a49361b92977a605267025b1e7798420936cbe53"
TRAINING_SEED, INITIALIZATION_SEED = 5202, 5201
ARM_STATUS = "HRSID_MASK_COMMON_FIT_E24_ARM_ACCEPTED_V1"
PAIR_STATUS = "HRSID_MASK_COMMON_FIT_E24_PAIR_READY_V1"
COMMON_MARKER = "sar_seal_hrsid_mask_common_support_v1"
POLICY = "ordered_common_nonempty_prepared_rows_v1"
CHECKPOINT_NAME = "checkpoint_23.ckpt"
FIT_INDEX_SHA = "e95cf26cc60a2c6fb651e907541380a256f2df7b1e153244a6c0f803b0b19f8f"
GT_SHA = "393b3da233991dc5fafc98b4ddeaa341b042d9571434ef8ce53c3f2fdada5628"
IMAGE_MANIFEST_SHA = "39a40d089d4a54448fe6efca8ea5cc98ad1eb7c0587815f3dd9a3b097173e78f"
EMPTY_RASTERS_SHA = "f3f2d644b2e0ce475b4a46b382ec7ec51f2efa4609c8f85b4f8f552ff57d5889"
ARMS = ("native", "center")
ARM_DIGESTS = (
    "visit_ledger_sha256", "result_sha256", "stdout_sha256", "terminal_sha256", "phase_provenance_sha256",
    "repeat_seed_witness_sha256", "common_audit_sha256",
)


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def is_commit(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{40}", value) is not None


def load_phase(path):
    """Load P203 as an isolated namespace with its byte identity preserved."""
    path = Path(path)
    require(path.is_absolute() and path.resolve(strict=True) == path, "Exact resolved P203 source required")
    limit = 64 * 1024
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    require(0 < len(raw) <= limit and sha256(raw).hexdigest() == P203_SHA,
            "Exact immutable P203 evaluation source required")
    phase = ModuleType("_common_support_scoped_p203")
    phase.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), phase.__dict__)
    return phase


def _read_arm(phase, path, pair, identity, mode):
    arm = phase.checked_receipt(path, pair[f"{mode}_arm_receipt_sha256"])
    phase.exact_fields(
        arm,
        dict(
            status=ARM_STATUS, original_os_wait=True, original_os_exit=0, resource_stop=False,
            completed_epochs=24, optimizer_updates=17640, ema_updates=17640,
            target_filter_sha256=phase.FILTER_SHA, filtered_empty_rows_total=0,
            checkpoint_sha256=pair[f"{mode}_checkpoint_sha256"],
            common_consumed_ledger_sha256=pair["shared_consumed_ledger_sha256"],
        ),
        "Exact independently accepted common FIT E24 arm required",
        extras=("phase", "common_support", *ARM_DIGESTS),
    )
    require(all(phase.is_sha256(arm[key]) for key in ARM_DIGESTS), "Exact common FIT arm evidence digests required")
    phase.exact_fields(
        arm["common_support"], dict(identity, policy=POLICY, mode=mode, stage="e24"),
        "Exact accepted common-support policy, source and qualification required",
    )
    phase.exact_fields(
        arm["phase"],
        dict(
            protocol=identity["training_protocol"], mask_resize_mode=mode, stage="e24", seed=TRAINING_SEED,
            initial_expanded_sha256=phase.INITIAL_SHA, phase_patch_sha256=phase.PHASE_PATCH_SHA,
            native_trainer_sha256=phase.NATIVE_TRAINER_SHA,
        ),
        "Exact accepted common FIT phase and seed required",
        extras=("phase_trainer_sha256",),
    )
    require(phase.is_sha256(arm["phase"]["phase_trainer_sha256"]), "Exact common FIT trainer digest required")
    return arm


def read_pair(phase, path, digest, *, native_path, center_path, mode, training_protocol):
    """Verify both independently accepted common arms before scientific admission."""
    require(
        mode in ARMS and is_commit(training_protocol) and training_protocol not in (FIRST_PROTOCOL, REPEAT_PROTOCOL)
        and phase.TRAINING_PROTOCOL in (FIRST_PROTOCOL, training_protocol) and phase.is_sha256(digest),
        "Exact externally locked common FIT training protocol and pair digest required",
    )
    path, native_path, center_path = (Path(item) for item in (path, native_path, center_path))
    require(len({path, native_path, center_path}) == 3, "Distinct common FIT pair and arm receipt paths required")
    pair = phase.checked_receipt(path, digest)
    fixed_identity = dict(
        training_protocol=training_protocol, training_seed=TRAINING_SEED, initialization_seed=INITIALIZATION_SEED,
        initial_expanded_sha256=phase.INITIAL_SHA, fit_index_sha256=FIT_INDEX_SHA,
    )
    identity_digests = ("source_manifest_sha256", "qualification_pair_sha256")
    arm_pins = tuple(f"{name}_{kind}_sha256" for name in ARMS for kind in ("arm_receipt", "checkpoint"))
    phase.exact_fields(
        pair,
        dict(
            fixed_identity, status=PAIR_STATUS, ordered_common_fit_visits_match=True,
            completed_epochs=24, visits_per_arm=70560, optimizer_updates_per_arm=17640, ema_updates_per_arm=17640,
            FIT=True, DEV=False, TEST=False, AP=False,
        ),
        "Exact independently accepted common FIT E24 pair required",
        extras=("implementation_commit", "changed_mask_visits", "shared_consumed_ledger_sha256",
                *identity_digests, *arm_pins),
    )
    require(
        is_commit(pair["implementation_commit"]) and pair["implementation_commit"] not in (FIRST_PROTOCOL, REPEAT_PROTOCOL)
        and all(phase.is_sha256(pair[key]) for key in ("shared_consumed_ledger_sha256", *identity_digests, *arm_pins))
        and len({pair[key] for key in arm_pins}) == len(arm_pins)
        and type(pair["changed_mask_visits"]) is int and 0 < pair["changed_mask_visits"] <= 70560,
        "Distinct fixed common arms, source pins and actual mask changes required",
    )
    identity = dict(fixed_identity, **{key: pair[key] for key in ("implementation_commit", *identity_digests)})
    native = _read_arm(phase, native_path, pair, identity, "native")
    center = _read_arm(phase, center_path, pair, identity, "center")
    require(
        native["phase"]["phase_trainer_sha256"] == center["phase"]["phase_trainer_sha256"]
        and all(native[key] != center[key] for key in
                ("terminal_sha256", "repeat_seed_witness_sha256", "common_audit_sha256")),
        "Same common FIT trainer and distinct original arm evidence required",
    )
    phase.TRAINING_PROTOCOL = training_protocol
    return native if mode == "native" else center


def bind_evaluator(phase, base, filtered, arm):
    """Keep the original scorer, extending admission to the common E24 marker."""
    protocol = phase.TRAINING_PROTOCOL
    require(
        is_commit(protocol) and protocol not in (FIRST_PROTOCOL, REPEAT_PROTOCOL)
        and arm["status"] == ARM_STATUS and arm["phase"]["protocol"] == protocol
        and type(arm["phase"]["seed"]) is int and arm["phase"]["seed"] == TRAINING_SEED
        and arm["common_support"]["training_protocol"] == protocol,
        "An independently admitted common FIT arm must bind this evaluator",
    )
    require(
        (base.DEV_IMAGES, base.GT_COUNT, base.EMPTY_COUNT, base.WIDTH, base.NATIVE_ROWS, base.STEPS)
        == (705, 1782, 4, 164, 100, 17640)
        and base.FILES["DEV_ORIGINAL_GT_PRIVATE.json"] == GT_SHA
        and base.FILES["DEV_IMAGE_ONLY_MANIFEST_PRIVATE.json"] == IMAGE_MANIFEST_SHA
        and base.FILES["DEV_EMPTY_RASTERS_PRIVATE.json"] == EMPTY_RASTERS_SHA,
        "Exact original DEV705 population, ground truth and native mask scorer required",
    )
    require(not getattr(base, "_common_fit_admitted", False), "A fresh P149 evaluator namespace is required")
    common_marker = arm["common_support"].copy()
    phase.bind_evaluator(base, filtered, arm)
    original_endpoint, original_export = base.check_endpoint, base.run_export

    def check_endpoint(checkpoint, training_protocol):
        args = checkpoint.get("args")
        require(training_protocol == protocol and type(args) is dict
                and type(args.get("seed")) is int and args["seed"] == TRAINING_SEED,
                "Exact common FIT checkpoint args.seed and protocol required")
        phase.exact_fields(checkpoint.get(COMMON_MARKER), common_marker, "Exact admitted common-support checkpoint marker")
        return original_endpoint(checkpoint, training_protocol)

    def run_export(args):
        require(isinstance(args.checkpoint, Path) and args.checkpoint.name == CHECKPOINT_NAME,
                "Exact admitted E24 checkpoint_23.ckpt filename required")
        return original_export(args)

    base.check_endpoint = check_endpoint
    base.run_export = run_export
    base._common_fit_admitted = True


def main():
    entry = ArgumentParser(add_help=False, allow_abbrev=False)
    for name in ("phase-source", "filtered-source", "base-source", "pair-receipt", "native-arm-receipt", "center-arm-receipt"):
        entry.add_argument("--" + name, type=Path, required=True)
    entry.add_argument("--pair-receipt-sha256", required=True)
    entry.add_argument("--training-protocol", required=True)
    entry.add_argument("--arm", choices=ARMS, required=True)
    args, downstream = entry.parse_known_args()

    phase = load_phase(args.phase_source)
    arm = read_pair(
        phase, args.pair_receipt, args.pair_receipt_sha256, native_path=args.native_arm_receipt,
        center_path=args.center_arm_receipt, mode=args.arm, training_protocol=args.training_protocol,
    )
    filtered, base = phase.load_evaluator(args.filtered_source, args.base_source)
    bind_evaluator(phase, base, filtered, arm)
    original_args = sys.argv[1:]
    sys.argv[1:] = [*downstream, "--training-protocol", args.training_protocol]
    try:
        base.main()
    finally:
        sys.argv[1:] = original_args


if __name__ == "__main__":
    main()
