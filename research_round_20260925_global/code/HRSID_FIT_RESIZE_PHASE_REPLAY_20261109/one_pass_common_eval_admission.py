"""Admit independently closed fixed-support B/C E24 endpoints to DEV705."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import sys
from types import ModuleType


ORIGINAL_SHA256 = "7a38fa028c07bf16d2da42935c5ee117f9e9ee5a88ede4337a43b1199c5ced13"
TRAINER_SHA256 = "d379a3726a7833697a9405b2c22206dc4e3f9ab3fa44e4d8d6e3c7c3d68c9797"
SOURCE_MANIFEST_SHA256 = "8f00f78afc2f49fb6298de0ff44cfc959fe61b0e44ad0e07897352909f081b92"
ARM_STATUS = "HRSID_ONE_PASS_COMMON_FIT_E24_ARM_ACCEPTED_V1"
PAIR_STATUS = "HRSID_ONE_PASS_COMMON_FIT_E24_PAIR_READY_V1"
POLICY = "one_pass_on_stepwise_common_nonempty_rows_v1"
MODES = ("stepwise_center", "one_pass")
OLD_PROTOCOLS = (
    "d66baf3c57fafdba4242ad1b55e08cf76bfd77f7", "a49361b92977a605267025b1e7798420936cbe53",
    "04308c52bd2b4730f1474335c6316f6b8a11762e", "3ee0dc2f5f6508c238b760b9b234edc016e89f54",
    "91728d38a359e4a207e3f741490f09060c46663f",
)


def load_original(path):
    """Load the immutable A/B evaluator binder into an isolated namespace."""
    path = Path(path)
    if not path.is_absolute() or path.resolve(strict=True) != path:
        raise ValueError("Exact resolved common FIT evaluator source required")
    with path.open("rb") as stream:
        raw = stream.read(64 * 1024 + 1)
    if not 0 < len(raw) <= 64 * 1024 or sha256(raw).hexdigest() != ORIGINAL_SHA256:
        raise ValueError("Exact immutable common FIT evaluator binder required")
    original = ModuleType("_bc_scoped_common_fit_eval_admission")
    original.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), original.__dict__)
    return original


def canonical_receipt(phase, path, digest):
    raw = phase.checked_bytes(path, digest, 8192)
    value = json.loads(raw, object_pairs_hook=phase.unique_object)
    expected = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    phase.require(raw == expected, "Exact canonical independent B/C receipt required")
    return value


def read_pair(original, phase, path, digest, *, stepwise_center_path, one_pass_path, training_protocol,
              implementation_commit, qualification_pair_sha256):
    """Verify both original accepted B/C arms without opening evaluation inputs."""
    phase.require(all(original.is_commit(value) and value not in OLD_PROTOCOLS
                      for value in (training_protocol, implementation_commit))
                  and phase.TRAINING_PROTOCOL in (original.FIRST_PROTOCOL, training_protocol)
                  and all(phase.is_sha256(value) for value in (digest, qualification_pair_sha256)),
                  "Exact externally locked B/C formal protocol, implementation and short qualification required")
    paths = tuple(map(Path, (path, stepwise_center_path, one_pass_path)))
    phase.require(len(set(paths)) == 3, "Distinct original B/C pair and both arm receipt paths required")
    pair = canonical_receipt(phase, paths[0], digest)
    identity = dict(training_protocol=training_protocol, implementation_commit=implementation_commit,
                    training_seed=original.TRAINING_SEED, initialization_seed=original.INITIALIZATION_SEED,
                    initial_expanded_sha256=phase.INITIAL_SHA, fit_index_sha256=original.FIT_INDEX_SHA,
                    source_manifest_sha256=SOURCE_MANIFEST_SHA256, qualification_pair_sha256=qualification_pair_sha256)
    arm_pins = tuple(f"{mode}_{kind}_sha256" for mode in MODES
                     for kind in ("arm_receipt", "checkpoint", "selected_mask_stream"))
    pins = ("shared_consumed_ledger_sha256", *arm_pins)
    branches = ("consumed_direct_visits", "consumed_crop_visits", "changed_crop_visits")
    phase.exact_fields(
        pair, dict(identity, status=PAIR_STATUS, policy=POLICY, stage="e24", ordered_common_fit_visits_match=True,
                   completed_epochs=phase.EPOCHS, visits_per_arm=phase.VISITS,
                   optimizer_updates_per_arm=phase.UPDATES, ema_updates_per_arm=phase.UPDATES,
                   verified_selected_mask_visits_per_arm=phase.VISITS, FIT=True, DEV=False, TEST=False, AP=False),
        "Exact independently accepted B/C E24 pair required", extras=(*pins, *branches),
    )
    direct, crop, changed = (pair[key] for key in branches)
    phase.require(all(phase.is_sha256(pair[key]) for key in pins)
                  and len({pair[key] for key in pins}) == len(pins)
                  and all(type(value) is int for value in (direct, crop, changed))
                  and direct > 0 and direct + crop == phase.VISITS and 0 < changed <= crop,
                  "Distinct B/C arm, checkpoint and stream digests with direct and changed crop evidence required")

    arms = {}
    for mode, arm_path in zip(MODES, paths[1:], strict=True):
        arm = canonical_receipt(phase, arm_path, pair[f"{mode}_arm_receipt_sha256"])
        phase.exact_fields(
            arm, dict(status=ARM_STATUS, original_os_wait=True, original_os_exit=0, resource_stop=False,
                      completed_epochs=phase.EPOCHS, optimizer_updates=phase.UPDATES, ema_updates=phase.UPDATES,
                      target_filter_sha256=phase.FILTER_SHA, filtered_empty_rows_total=0,
                      checkpoint_sha256=pair[f"{mode}_checkpoint_sha256"],
                      common_consumed_ledger_sha256=pair["shared_consumed_ledger_sha256"],
                      verified_selected_mask_visits=phase.VISITS,
                      verified_selected_mask_stream_sha256=pair[f"{mode}_selected_mask_stream_sha256"]),
            "Exact accepted B/C E24 arm, endpoint and selected stream required",
            extras=("phase", "common_support", *original.ARM_DIGESTS),
        )
        phase.exact_fields(arm["common_support"], dict(identity, policy=POLICY, stage="e24", mode=mode),
                           "Exact fixed B/C support and own mode required")
        phase.exact_fields(
            arm["phase"], dict(phase.expected_phase(mode), protocol=training_protocol, seed=original.TRAINING_SEED,
                               phase_trainer_sha256=TRAINER_SHA256), "Exact original B/C E24 phase required",
        )
        phase.require(all(phase.is_sha256(arm[key]) for key in original.ARM_DIGESTS),
                      "Exact closed B/C evidence digests required")
        arms[mode] = arm
    stepwise, one_pass = (arms[mode] for mode in MODES)
    phase.require(stepwise["visit_ledger_sha256"] == one_pass["visit_ledger_sha256"]
                  and all(stepwise[key] != one_pass[key] for key in
                          ("terminal_sha256", "stdout_sha256", "repeat_seed_witness_sha256", "common_audit_sha256")),
                  "Matched native visit stream and distinct original B/C arm closures required")
    phase.TRAINING_PROTOCOL = training_protocol
    return pair, arms


def bind_evaluator(original, phase, base, filtered, arm):
    """Scope only the receipt status for the original endpoint/export/score binder."""
    phase.require(arm["status"] == ARM_STATUS and arm["phase"]["mask_resize_mode"] in MODES
                  and arm["phase"]["phase_trainer_sha256"] == TRAINER_SHA256
                  and arm["common_support"]["policy"] == POLICY,
                  "A verified B/C endpoint is required for evaluator binding")
    previous = original.ARM_STATUS
    original.ARM_STATUS = ARM_STATUS
    try:
        original.bind_evaluator(phase, base, filtered, arm)
    finally:
        original.ARM_STATUS = previous


def main(argv=None):
    entry = ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("common-admission-source", "phase-source", "filtered-source", "base-source",
                 "pair-receipt", "stepwise-center-arm-receipt", "one-pass-arm-receipt"):
        entry.add_argument("--" + name, required=True, type=Path)
    for name in ("pair-receipt-sha256", "training-protocol", "implementation-commit", "qualification-pair-sha256"):
        entry.add_argument("--" + name, required=True)
    entry.add_argument("--arm", choices=MODES, required=True)
    args, downstream = entry.parse_known_args(argv)

    original = load_original(args.common_admission_source)
    phase = original.load_phase(args.phase_source)
    _, arms = read_pair(
        original, phase, args.pair_receipt, args.pair_receipt_sha256,
        stepwise_center_path=args.stepwise_center_arm_receipt, one_pass_path=args.one_pass_arm_receipt,
        training_protocol=args.training_protocol, implementation_commit=args.implementation_commit,
        qualification_pair_sha256=args.qualification_pair_sha256,
    )
    filtered, base = phase.load_evaluator(args.filtered_source, args.base_source)
    bind_evaluator(original, phase, base, filtered, arms[args.arm])
    original_args = sys.argv[1:]
    sys.argv[1:] = [*downstream, "--training-protocol", args.training_protocol]
    try:
        base.main()
    finally:
        sys.argv[1:] = original_args


if __name__ == "__main__":
    main()
