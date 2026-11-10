"""Independent fixed-support B/C HRSID E24 training evidence checks.

Callers authorize exact releases and supply bytes/streams from independently
OS-closed original processes. This module discovers no data or checkpoints.
"""

from hashlib import sha256
from io import BytesIO
from typing import NamedTuple

import common_fit_e24_ledger as shared
from common_fit_e24_acceptance import EPOCHS, UPDATES, VISITS, ClosedE24Arm, accept_arm, canonical, scoped_repeat
from common_fit_pair_receipt import SCOPE, compact, digest, encode_receipt, exact, is_hex, require
import one_pass_common_short_receipt as short


TRAINER_SOURCE = "one_pass_common_e24.py"
TRAINER_SOURCE_SHA256 = "d379a3726a7833697a9405b2c22206dc4e3f9ab3fa44e4d8d6e3c7c3d68c9797"
ARM_STATUS = "HRSID_ONE_PASS_COMMON_FIT_E24_ARM_ACCEPTED_V1"
PAIR_STATUS = "HRSID_ONE_PASS_COMMON_FIT_E24_PAIR_READY_V1"


class AcceptedBCE24Pair(NamedTuple):
    stepwise_center_receipt: bytes
    one_pass_receipt: bytes
    pair_receipt: bytes
    ledger_stats: dict


def admit_qualification(*, training_protocol, implementation_commit, source_sha256s, trainer_source,
                        qualification_protocol, qualification_implementation_commit,
                        qualification_pair_bytes, qualification_pair_sha256):
    """Bind the already accepted short pair to the authorized formal worker."""
    require(all(is_hex(value, 40) and value not in short.OLD_PROTOCOLS for value in
                (training_protocol, implementation_commit, qualification_protocol, qualification_implementation_commit)),
            "Exact prospective B/C formal and short release identities required")
    require(trainer_source == TRAINER_SOURCE,
            "A distinct formal B/C trainer source name is required")
    require(type(source_sha256s) is dict
            and source_sha256s == {**short.SOURCE_PINS, TRAINER_SOURCE: TRAINER_SOURCE_SHA256},
            "Exact qualified B/C source pins and a fresh formal worker are required")
    require(training_protocol != qualification_protocol and implementation_commit != qualification_implementation_commit,
            "Formal training and short qualification release identities must be distinct")
    receipt = canonical(qualification_pair_bytes, "B/C short pair")
    require(is_hex(qualification_pair_sha256) and digest(qualification_pair_bytes) == qualification_pair_sha256,
            "Exact independently accepted B/C short pair digest required")
    fixed = dict(training_seed=short.TRAINING_SEED, initialization_seed=short.INITIALIZATION_SEED,
                 initial_expanded_sha256=short.INITIAL_EXPANDED_SHA, fit_index_sha256=short.FIT_INDEX_SHA)
    digests = ("arm_receipt_sha256", "terminal_sha256", "selected_mask_stream_sha256")
    arm_digests = tuple(f"{mode}_{field}" for mode in short.MODES for field in digests)
    exact(receipt, dict(status=short.PAIR_STATUS, stage="qualification", policy=short.POLICY,
                        training_protocol=qualification_protocol, implementation_commit=qualification_implementation_commit,
                        source_manifest_sha256=digest(compact(short.SOURCE_PINS)), **fixed,
                        ordered_common_fit_visits_match=True, visits_per_arm=32, optimizer_updates_per_arm=8,
                        ema_updates_per_arm=8, verified_selected_mask_visits_per_arm=32, **SCOPE),
          "Exact independently accepted B/C short pair is required",
          extra=(*arm_digests, "shared_consumed_ledger_sha256", "consumed_direct_visits", "consumed_crop_visits",
                 "changed_crop_visits", "changed_mask_visits"))
    direct, crop, changed, changed_total = (
        receipt[key] for key in ("consumed_direct_visits", "consumed_crop_visits", "changed_crop_visits",
                                 "changed_mask_visits")
    )
    require(all(is_hex(receipt[key]) for key in (*arm_digests, "shared_consumed_ledger_sha256"))
            and all(receipt[f"stepwise_center_{field}"] != receipt[f"one_pass_{field}"] for field in digests)
            and all(type(value) is int for value in (direct, crop, changed, changed_total))
            and direct > 0 and direct + crop == 32 and 0 < changed <= crop and changed == changed_total,
            "Exact distinct short arm closures, direct equality and crop mask differences required")
    return dict(training_protocol=training_protocol, implementation_commit=implementation_commit, **fixed,
                source_manifest_sha256=digest(compact(source_sha256s)),
                qualification_pair_sha256=qualification_pair_sha256)


class SelectedMaskReadback:
    """Tee the shared consumed ledger into the two expected selected streams."""

    def __init__(self, stream):
        self.stream = stream
        self.selected = {mode: sha256() for mode in short.MODES}
        self.direct = self.crop = self.changed_crop = self.nonempty_direct = 0

    def readline(self, limit):
        raw = self.stream.readline(limit)
        if type(raw) is not bytes or not raw.endswith(b"\n") or limit != shared.COMMON_LINE_LIMIT + 1:
            return raw
        require(len(raw) < limit, "Oversized B/C consumed row")
        row = shared.decode(raw, "B/C consumed")
        required = ("crop", "native_present", "center_present", "native_nonempty", "center_nonempty",
                    "common_nonempty", "common_changed_masks", "ordinal", "image_id", "epoch", "batch",
                    *short.SELECTED_MASK.values())
        require(all(key in row for key in required), "Missing B/C consumed row field")
        require(row["native_present"] == row["center_present"], "Both arms must retain the same B crop roster")
        if row["crop"]:
            self.crop += 1
            self.changed_crop += int(row["common_changed_masks"] != 0)
        else:
            require(row["common_changed_masks"] == 0
                    and row["native_nonempty"] == row["center_nonempty"] == row["common_nonempty"],
                    "Direct visits must have identical masks and support")
            self.direct += 1
            self.nonempty_direct += int(row["common_nonempty"] != 0)
        item = {key: row[key] for key in ("ordinal", "image_id", "epoch", "batch")}
        for mode, field in short.SELECTED_MASK.items():
            self.selected[mode].update(compact(dict(item, mask_sha256=row[field])) + b"\n")
        return raw


def verify_bc_e24_ledgers(*, stepwise_center_common, one_pass_common, stepwise_center_visits, one_pass_visits,
                          stepwise_center_safe, one_pass_safe):
    """Verify equal ordered nonmask evidence and reconcile actual selected-stream witnesses."""
    receipts = (stepwise_center_safe, one_pass_safe)
    normalized = []
    for mode, alias, safe in zip(short.MODES, ("native", "center"), receipts, strict=True):
        require(type(safe) is dict and safe.get("status") == short.AUDIT_STATUS
                and all(type(safe.get(key)) is dict and safe[key].get(field) == mode
                        for key, field in (("common_support", "mode"), ("phase", "mask_resize_mode")))
                and type(safe.get("ledger")) is dict and short.SELECTED_STATS <= set(safe["ledger"]),
                "Exact B/C E24 consumed-stream witness required")
        # Supply only the shared ledger semantics to the original reader.
        normalized.append(dict(safe, status=shared.AUDIT_STATUS, common_support=dict(safe["common_support"], mode=alias),
                               phase=dict(safe["phase"], mask_resize_mode=alias),
                               ledger={key: value for key, value in safe["ledger"].items()
                                       if key not in short.SELECTED_STATS}))
    tee = SelectedMaskReadback(stepwise_center_common)
    result = shared.verify_e24_ledgers(native_common=tee, center_common=one_pass_common,
                                       native_visits=stepwise_center_visits, center_visits=one_pass_visits,
                                       native_common_safe=normalized[0], center_common_safe=normalized[1])
    require(tee.nonempty_direct > 0 and tee.changed_crop == result["ledger"]["changed_mask_visits"] > 0
            and tee.direct + tee.crop == result["visits_per_arm"],
            "B/C E24 requires actual nonempty direct equality and changed crop masks")
    branches = dict(consumed_direct_visits=tee.direct, consumed_crop_visits=tee.crop, changed_crop_visits=tee.changed_crop)
    selected = {mode: value.hexdigest() for mode, value in tee.selected.items()}
    require(len(set(selected.values())) == 2, "Distinct B/C E24 selected-mask streams required")
    for mode, safe in zip(short.MODES, receipts, strict=True):
        exact(safe["ledger"], dict(result["ledger"], **branches, verified_selected_mask_visits=result["visits_per_arm"],
                                   verified_selected_mask_stream_sha256=selected[mode]),
              "Closed actual selected-mask stream does not match independent B/C ledger readback")
    result["stepwise_center_visit_ledger_sha256"] = result.pop("native_visit_ledger_sha256")
    result["one_pass_visit_ledger_sha256"] = result.pop("center_visit_ledger_sha256")
    result["stepwise_center_only_nonempty_total"] = result.pop("native_only_nonempty_total")
    result["one_pass_only_nonempty_total"] = result.pop("center_only_nonempty_total")
    return dict(result, **branches, **{f"{mode}_selected_mask_stream_sha256": value for mode, value in selected.items()})


def scoped_bc_repeat(protocol, trainer_sha256):
    """Preserve native checks and scope only the legacy mode allowlists."""
    verifier = scoped_repeat(protocol, trainer_sha256)
    original_marker = verifier.expected_phase("center")
    phase = verifier._phase

    def expected_phase(mode):
        require(mode in short.MODES, "Exact fixed B/C E24 mode required")
        return dict(original_marker, mask_resize_mode=mode)

    def split_native_stdout(raw, mode):
        require(mode in short.MODES and type(raw) is bytes and raw.endswith(b"\n"),
                "Complete original B/C stdout required")
        lines = raw.splitlines(keepends=True)
        require(len(lines) >= 2 and phase.same_fields(phase.object_from(lines[-1], "B/C phase stdout trailer"),
                                                     dict(event=phase.TRAILER, mask_resize_mode=mode, protocol=protocol)),
                "Exact final original B/C phase completion event required")
        for line in lines[:-1]:
            if phase.TRAILER.encode() in line and line.lstrip().startswith(b"{"):
                require(phase.object_from(line, "possible repeated B/C phase event").get("event") != phase.TRAILER,
                        "Duplicate B/C phase completion event")
        return raw[:-len(lines[-1])]

    verifier.expected_phase = expected_phase
    phase.split_native_stdout = split_native_stdout
    return verifier


def checked_audit(arm, mode, identity, verifier):
    require(type(arm) is ClosedE24Arm and callable(arm.checkpoint_reader)
            and all(type(raw) is bytes and raw for raw in arm[1:-1]), "Original closed B/C E24 arm bytes required")
    verifier._native.validate_closure(arm.closure, arm.stdout)
    safe = canonical(arm.common_audit, "B/C E24 worker receipt")
    exact(safe, dict(status=short.AUDIT_STATUS, native_result_sha256=digest(arm.result),
                     repeat_seed_witness_sha256=digest(arm.seed_witness), optimizer_updates=UPDATES,
                     ema_updates=UPDATES, completed_epochs=EPOCHS, **SCOPE),
          "Exact B/C E24 worker audit and native evidence required",
          extra=("common_support", "phase", "consumed_ledger_sha256", "consumed_ledger_bytes", "ledger"))
    exact(safe["phase"], verifier.expected_phase(mode), "Exact new B/C phase and its own arm required")
    exact(safe["common_support"], dict(identity, policy=short.POLICY, stage="e24", mode=mode),
          "Exact new B/C common-support, short qualification and E24 source identities required")
    return safe


def accept_e24_pair(*, training_protocol, implementation_commit, source_sha256s, qualification_protocol,
                    qualification_implementation_commit, qualification_pair_bytes, qualification_pair_sha256,
                    stepwise_center, one_pass, stepwise_center_common, one_pass_common, expected_fit_ids):
    """Accept B and C only after both original owners and workers have closed."""
    identity = admit_qualification(
        training_protocol=training_protocol, implementation_commit=implementation_commit, source_sha256s=source_sha256s,
        trainer_source=TRAINER_SOURCE, qualification_protocol=qualification_protocol,
        qualification_implementation_commit=qualification_implementation_commit,
        qualification_pair_bytes=qualification_pair_bytes, qualification_pair_sha256=qualification_pair_sha256,
    )
    verifier = scoped_bc_repeat(training_protocol, source_sha256s[TRAINER_SOURCE])
    arms = (stepwise_center, one_pass)
    safes = tuple(checked_audit(arm, mode, identity, verifier) for arm, mode in zip(arms, short.MODES, strict=True))
    qualification = canonical(qualification_pair_bytes, "B/C short pair")
    require(len({arm.closure["terminal_sha256"] for arm in arms} |
                {qualification[f"{mode}_terminal_sha256"] for mode in short.MODES}) == 4
            and stepwise_center.closure["stdout_sha256"] != one_pass.closure["stdout_sha256"],
            "Independent fresh original E24 closures required for both B/C arms")
    ledger = verify_bc_e24_ledgers(
        stepwise_center_common=stepwise_center_common, one_pass_common=one_pass_common,
        stepwise_center_visits=BytesIO(stepwise_center.visits), one_pass_visits=BytesIO(one_pass.visits),
        stepwise_center_safe=safes[0], one_pass_safe=safes[1],
    )
    require((ledger["completed_epochs"], ledger["visits_per_arm"], ledger["optimizer_updates_per_arm"],
             ledger["ema_updates_per_arm"]) == (EPOCHS, VISITS, UPDATES, UPDATES), "Exact B/C E24 budget required")
    expected_fit_ids = tuple(expected_fit_ids)
    accepted = []
    for arm, safe, mode in zip(arms, safes, short.MODES, strict=True):
        receipt = accept_arm(arm, mode, safe, verifier, expected_fit_ids, ledger[f"{mode}_visit_ledger_sha256"])
        accepted.append(dict(receipt, status=ARM_STATUS,
                             verified_selected_mask_visits=VISITS,
                             verified_selected_mask_stream_sha256=ledger[f"{mode}_selected_mask_stream_sha256"]))
    archive_a, archive_b = (canonical(arm.result, "native E24 result")["archive_refs"] for arm in arms)
    clocks = ("epoch", "adam_parameter_clocks", "native_base_lrs", "native_next_lrs")
    require(len({ref["archive_sha256"] for ref in archive_a + archive_b}) == 6
            and all(all(left[key] == right[key] for key in clocks) for left, right in zip(archive_a, archive_b, strict=True))
            and accepted[0]["repeat_seed_witness_sha256"] != accepted[1]["repeat_seed_witness_sha256"],
            "Distinct original B/C archives and repeat witnesses with the same optimizer schedule required")
    raw_a, raw_b = map(encode_receipt, accepted)
    pair = dict(status=PAIR_STATUS, policy=short.POLICY, stage="e24", **identity,
                ordered_common_fit_visits_match=True, completed_epochs=EPOCHS,
                visits_per_arm=VISITS, optimizer_updates_per_arm=UPDATES, ema_updates_per_arm=UPDATES,
                verified_selected_mask_visits_per_arm=VISITS,
                shared_consumed_ledger_sha256=ledger["shared_consumed_ledger_sha256"],
                **{key: ledger[key] for key in ("consumed_direct_visits", "consumed_crop_visits", "changed_crop_visits")},
                stepwise_center_arm_receipt_sha256=digest(raw_a), one_pass_arm_receipt_sha256=digest(raw_b),
                stepwise_center_checkpoint_sha256=accepted[0]["checkpoint_sha256"],
                one_pass_checkpoint_sha256=accepted[1]["checkpoint_sha256"],
                stepwise_center_selected_mask_stream_sha256=ledger["stepwise_center_selected_mask_stream_sha256"],
                one_pass_selected_mask_stream_sha256=ledger["one_pass_selected_mask_stream_sha256"], **SCOPE)
    return AcceptedBCE24Pair(raw_a, raw_b, encode_receipt(pair), ledger)

