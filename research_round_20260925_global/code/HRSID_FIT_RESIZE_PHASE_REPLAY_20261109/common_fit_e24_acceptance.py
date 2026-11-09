"""Admit an original OS-closed common-support E24 pair for fixed DEV705 scoring.

The caller owns process identity, exact closed-file readback, and immutable
source admission. This layer verifies the training evidence and all endpoints.
"""

from hashlib import sha256
from io import BytesIO
from pathlib import Path
from types import ModuleType
from typing import NamedTuple

from common_fit_e24_ledger import verify_e24_ledgers
from common_fit_pair_receipt import SCOPE, compact, encode_receipt, exact, is_hex, require
from common_fit_train import (
    AUDIT_STATUS, COMMON_MARKER, FIT_INDEX_SHA, INITIAL_EXPANDED_SHA, INITIALIZATION_SEED, ORIGINAL_PROTOCOLS,
    POLICY, QUALIFIED_STATUS, SELF, SOURCE_PINS, SOURCES, TRAINING_SEED, object_from,
)


ARM_STATUS = "HRSID_MASK_COMMON_FIT_E24_ARM_ACCEPTED_V1"
PAIR_STATUS = "HRSID_MASK_COMMON_FIT_E24_PAIR_READY_V1"
REPEAT_SOURCE = (Path(__file__).resolve().parent.parent /
                 "HRSID_MASK_PHASE_REPEAT_ACCEPTANCE_20261109/repeat_acceptance.py")
REPEAT_SHA = "608cd8ed2418be03c5618bc758275a9246cab8c4d7afb9a737e84a467029ee47"
EPOCHS, VISITS, UPDATES = 24, 70560, 17640


class ClosedE24Arm(NamedTuple):
    closure: dict
    common_audit: bytes
    result: bytes
    epochs: bytes
    visits: bytes
    stdout: bytes
    phase_provenance: bytes
    seed_witness: bytes
    checkpoint_reader: object


class AcceptedE24Pair(NamedTuple):
    native_receipt: bytes
    center_receipt: bytes
    pair_receipt: bytes
    ledger_stats: dict


def digest(raw):
    return sha256(raw).hexdigest()


def canonical(raw, label):
    require(type(raw) is bytes and 0 < len(raw) <= 64 * 1024, f"Bounded closed {label} required")
    value = object_from(raw)
    require(encode_receipt(value) == raw, f"Canonical original {label} required")
    return value


def scoped_repeat(protocol, trainer_sha256):
    """Reuse the immutable seed, optimizer, EMA and checkpoint validator."""
    require(is_hex(protocol, 40) and protocol not in ORIGINAL_PROTOCOLS and is_hex(trainer_sha256),
            "Exact separate common-support protocol and worker digest required")
    with REPEAT_SOURCE.open("rb") as stream:
        raw = stream.read(64 * 1024 + 1)
    require(0 < len(raw) <= 64 * 1024 and digest(raw) == REPEAT_SHA, "Exact historical repeat verifier required")
    repeat = ModuleType("_common_support_scoped_repeat_acceptance")
    repeat.__file__ = str(REPEAT_SOURCE)
    exec(compile(raw, str(REPEAT_SOURCE), "exec"), repeat.__dict__)
    repeat.REPEAT_PROTOCOL = protocol
    return repeat.RepeatAcceptance(protocol, TRAINING_SEED, trainer_sha256)


def short_pair_identity(training_protocol, implementation_commit, source_sha256s, raw, expected_sha):
    require(all(is_hex(value, 40) and value not in ORIGINAL_PROTOCOLS
                for value in (training_protocol, implementation_commit))
            and type(source_sha256s) is dict and set(source_sha256s) == SOURCES
            and all(is_hex(value) for value in source_sha256s.values())
            and all(source_sha256s[key] == value for key, value in SOURCE_PINS.items()),
            "Exact prospective common-support implementation and source pins required")
    short = canonical(raw, "short pair receipt")
    require(is_hex(expected_sha) and digest(raw) == expected_sha, "Independently accepted short pair digest required")
    identity = dict(training_protocol=training_protocol, implementation_commit=implementation_commit,
                    training_seed=TRAINING_SEED, initialization_seed=INITIALIZATION_SEED,
                    initial_expanded_sha256=INITIAL_EXPANDED_SHA, fit_index_sha256=FIT_INDEX_SHA,
                    source_manifest_sha256=digest(compact(source_sha256s)))
    expected = dict(status=QUALIFIED_STATUS, **identity, ordered_common_fit_visits_match=True, visits_per_arm=32,
                    optimizer_updates_per_arm=8, ema_updates_per_arm=8, **SCOPE)
    digests = ("native_arm_receipt_sha256", "center_arm_receipt_sha256", "shared_consumed_ledger_sha256")
    exact(short, expected, "Exact same-implementation short pair required", extra=(*digests, "changed_mask_visits"))
    require(all(is_hex(short[key]) for key in digests)
            and short["native_arm_receipt_sha256"] != short["center_arm_receipt_sha256"]
            and type(short["changed_mask_visits"]) is int and 0 < short["changed_mask_visits"] <= 32,
            "Accepted short pair must have distinct arms and observed mask changes")
    return dict(identity, qualification_pair_sha256=expected_sha)


def checked_arm_audit(arm, mode, identity, legacy):
    require(type(arm) is ClosedE24Arm and callable(arm.checkpoint_reader), "Original closed E24 arm input required")
    legacy._native.validate_closure(arm.closure, arm.stdout)
    safe = canonical(arm.common_audit, "common E24 worker receipt")
    exact(safe, dict(status=AUDIT_STATUS, native_result_sha256=digest(arm.result),
                     repeat_seed_witness_sha256=digest(arm.seed_witness), optimizer_updates=UPDATES,
                     ema_updates=UPDATES, completed_epochs=EPOCHS, **SCOPE),
          "Exact common E24 audit and native evidence required",
          extra=("common_support", "phase", "consumed_ledger_sha256", "consumed_ledger_bytes", "ledger"))
    exact(safe["phase"], legacy.expected_phase(mode), "Common audit phase must match the native arm")
    exact(safe["common_support"], dict(**identity, policy=POLICY, stage="e24", mode=mode),
          "Exact common-support marker required")
    return safe


def accept_arm(arm, mode, safe, legacy, expected_fit_ids, visit_digest):
    def reader(ref):
        saved, size, archive_digest = arm.checkpoint_reader(ref)
        require(type(saved) is dict, "Exact original common-support archive payload required")
        exact(saved.get(COMMON_MARKER), safe["common_support"], "Every archive must carry its own common-support marker")
        return saved, size, archive_digest

    raw, _ = legacy.validate_arm(
        mode, closure=arm.closure, result_bytes=arm.result, epoch_bytes=arm.epochs, visit_bytes=arm.visits,
        stdout_bytes=arm.stdout, sidecar_bytes=arm.phase_provenance, seed_witness_bytes=arm.seed_witness,
        checkpoint_reader=reader, expected_fit_ids=expected_fit_ids,
    )
    native = object_from(raw)
    require(native["filtered_empty_rows_total"] == 0 and native["visit_ledger_sha256"] == visit_digest,
            "Common support requires zero native row removals and the independently checked visit stream")
    return dict(native, status=ARM_STATUS, common_support=safe["common_support"],
                common_audit_sha256=digest(arm.common_audit), common_consumed_ledger_sha256=safe["consumed_ledger_sha256"])


def accept_e24_pair(*, training_protocol, implementation_commit, source_sha256s, qualification_pair_bytes,
                    qualification_pair_sha256, native, center, native_common, center_common, expected_fit_ids):
    """Return canonical accepted arm/pair bytes after the common and native checks."""
    identity = short_pair_identity(training_protocol, implementation_commit, source_sha256s,
                                   qualification_pair_bytes, qualification_pair_sha256)
    legacy = scoped_repeat(training_protocol, source_sha256s[SELF])
    safe_a = checked_arm_audit(native, "native", identity, legacy)
    safe_b = checked_arm_audit(center, "center", identity, legacy)
    ledger = verify_e24_ledgers(native_common=native_common, center_common=center_common,
                                native_visits=BytesIO(native.visits), center_visits=BytesIO(center.visits),
                                native_common_safe=safe_a, center_common_safe=safe_b)
    expected_fit_ids = tuple(expected_fit_ids)
    a = accept_arm(native, "native", safe_a, legacy, expected_fit_ids, ledger["native_visit_ledger_sha256"])
    b = accept_arm(center, "center", safe_b, legacy, expected_fit_ids, ledger["center_visit_ledger_sha256"])
    archive_a, archive_b = (object_from(arm.result)["archive_refs"] for arm in (native, center))
    paired_clocks = ("epoch", "adam_parameter_clocks", "native_base_lrs", "native_next_lrs")
    require(all(a[key] != b[key] for key in ("checkpoint_sha256", "stdout_sha256", "terminal_sha256",
                                           "repeat_seed_witness_sha256"))
            and len({ref["archive_sha256"] for ref in archive_a + archive_b}) == 6
            and all(all(left[key] == right[key] for key in paired_clocks)
                    for left, right in zip(archive_a, archive_b, strict=True)),
            "Distinct original arm closures and archives with the same optimizer schedule required")
    a_raw, b_raw = encode_receipt(a), encode_receipt(b)
    pair = dict(status=PAIR_STATUS, **identity, ordered_common_fit_visits_match=True, completed_epochs=EPOCHS,
                visits_per_arm=VISITS, optimizer_updates_per_arm=UPDATES, ema_updates_per_arm=UPDATES,
                native_arm_receipt_sha256=digest(a_raw), center_arm_receipt_sha256=digest(b_raw),
                native_checkpoint_sha256=a["checkpoint_sha256"], center_checkpoint_sha256=b["checkpoint_sha256"],
                shared_consumed_ledger_sha256=ledger["shared_consumed_ledger_sha256"],
                changed_mask_visits=ledger["ledger"]["changed_mask_visits"], **SCOPE)
    return AcceptedE24Pair(a_raw, b_raw, encode_receipt(pair), ledger)

