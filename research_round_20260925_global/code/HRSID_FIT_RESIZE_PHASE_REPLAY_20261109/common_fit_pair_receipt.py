"""Independently verify two original OS-closed common-support short FIT streams.

The owner must first prove process closure and match the supplied stdout and
terminal digests to its original closure records. This verifier uses bytes
already acquired from those independently closed workers.
"""

from hashlib import sha256
import json
from pathlib import Path
import re
from types import ModuleType
from typing import NamedTuple

from common_fit_train import (
    AUDIT_STATUS, FIT_INDEX_SHA, INITIAL_EXPANDED_SHA, INITIALIZATION_SEED, NATIVE, ORIGINAL_PROTOCOLS,
    POLICY, QUALIFIED_STATUS, SELF, SOURCE_PINS, SOURCES, TRAINING_SEED,
)


VISITS, UPDATES = 32, 8
QUALIFIER_SHA = "ae9a915b56aa27ec3376877c38a399cd5f722ee4e5d47f61aa74c32e39862e5f"
QUALIFIER_PATH = (Path(__file__).parent.parent /
                  "HRSID_MASK_PHASE_REPEAT_QUALIFICATION_20261109/repeat_qualification.py")
COUNTS = (
    "raw_rows", "native_present", "center_present", "native_nonempty", "center_nonempty",
    "common_nonempty", "native_only_nonempty", "center_only_nonempty", "common_changed_masks",
)
DIGESTS = (
    "image_sha256", "trace_sha256", "rng_before_sha256", "rng_after_sha256", "common_row_ids_sha256",
    "common_nonmask_sha256", "native_common_masks_sha256", "center_common_masks_sha256",
)
SCOPE = dict(FIT=True, DEV=False, TEST=False, AP=False)


class ClosedArm(NamedTuple):
    closure: dict
    common_receipt: bytes
    consumed_ledger: bytes
    native_report: bytes
    native_visits: bytes
    seed_witness: bytes
    stdout: bytes


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return sha256(raw).hexdigest()


def is_hex(value, length=64):
    return type(value) is str and re.fullmatch(rf"[a-f0-9]{{{length}}}", value) is not None


def encode_receipt(receipt):
    """Use the same byte encoding as the exclusive training receipt writer."""
    return (json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def compact(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def exact(document, fields, message, *, extra=()):
    require(type(document) is dict and set(document) == set(fields) | set(extra)
            and all(type(document[key]) is type(value) and document[key] == value for key, value in fields.items()),
            message)


def canonical(raw, verifier, shape, limit):
    value = verifier.parse(raw, shape, limit)
    require(encode_receipt(value) == raw, "Expected original canonical closed JSON bytes")
    return value


def consumed_visits(raw, verifier):
    require(type(raw) is bytes and 0 < len(raw) <= 256 * 1024, "Expected bounded closed consumed ledger")
    lines = raw.splitlines(keepends=True)
    require(len(lines) == VISITS, "Expected exactly 32 consumed FIT visits")
    rows = []
    for ordinal, line in enumerate(lines, 1):
        row = verifier.parse(line, dict, 4096)
        exact(row, dict(ordinal=ordinal, epoch=1, batch=ordinal), "Exact consumed FIT cursor required",
              extra=("image_id", "crop", "flipped", *COUNTS, *DIGESTS))
        require(compact(row) + b"\n" == line, "Expected canonical LF-terminated consumed FIT rows")
        require(all(type(row[key]) is int and row[key] >= 0 for key in ("image_id", *COUNTS))
                and all(type(row[key]) is bool for key in ("crop", "flipped"))
                and all(is_hex(row[key]) for key in DIGESTS),
                "Exact consumed image, trajectory, random state and row audit required")
        common = row["common_nonempty"]
        for mode in ("native", "center"):
            require(common <= row[f"{mode}_nonempty"] <= row[f"{mode}_present"] <= row["raw_rows"]
                    and row[f"{mode}_only_nonempty"] == row[f"{mode}_nonempty"] - common,
                    "Invalid common support in consumed FIT row")
        require(row["native_nonempty"] + row["center_nonempty"] - common <= row["raw_rows"]
                and row["common_changed_masks"] <= common
                and (row["common_changed_masks"] > 0) ==
                (row["native_common_masks_sha256"] != row["center_common_masks_sha256"]),
                "Invalid common support or mask changes in consumed FIT row")
        rows.append(row)
    return rows


def verify_summary(summary, rows):
    expected = dict(consumed_visits=VISITS, common_targets_total=sum(row["common_nonempty"] for row in rows),
                    common_empty_visits=sum(row["common_nonempty"] == 0 for row in rows),
                    changed_mask_visits=sum(row["common_changed_masks"] > 0 for row in rows))
    exact(summary, expected, "Closed common FIT statistics disagree with the consumed stream",
          extra=("fetched_visits", "prefetched_unconsumed", "unconsumed_prefetch_sha256"))
    prefetched = summary["prefetched_unconsumed"]
    tail = summary["unconsumed_prefetch_sha256"]
    require(type(prefetched) is int and prefetched in (0, 1)
            and type(summary["fetched_visits"]) is int and summary["fetched_visits"] == VISITS + prefetched
            and (tail is None if prefetched == 0 else is_hex(tail)),
            "Invalid unconsumed FIT lookahead accounting")
    return expected


def verify_arm(arm, mode, identity, source_sha256s, verifier):
    require(type(arm) is ClosedArm, "Independently acquired closed arm bytes required")
    exact(arm.closure, dict(original_os_wait=True, original_os_exit=0, resource_stop=False, worker_closed=True),
          "Exact independent original worker closure required", extra=("stdout_sha256", "terminal_sha256"))
    require(is_hex(arm.closure["terminal_sha256"]) and type(arm.stdout) is bytes
            and 0 < len(arm.stdout) <= 2 * 1024**2
            and arm.closure["stdout_sha256"] == digest(arm.stdout), "Exact original terminal and stdout digests required")

    common = canonical(arm.common_receipt, verifier, dict, 64 * 1024)
    rows = consumed_visits(arm.consumed_ledger, verifier)
    expected = dict(status=AUDIT_STATUS, optimizer_updates=UPDATES, ema_updates=UPDATES,
                    completed_epochs=0, consumed_ledger_sha256=digest(arm.consumed_ledger),
                    consumed_ledger_bytes=len(arm.consumed_ledger), **SCOPE)
    exact(common, expected, "Closed common FIT arm receipt does not match the consumed stream",
          extra=("common_support", "phase", "native_result_sha256", "repeat_seed_witness_sha256", "ledger"))
    stats = verify_summary(common["ledger"], rows)
    exact(common["common_support"], dict(**identity, policy=POLICY, stage="qualification", mode=mode,
                                         qualification_pair_sha256=None), "Exact short common-support arm marker required")
    phase = dict(protocol=identity["training_protocol"], mask_resize_mode=mode, stage="qualification",
                 seed=TRAINING_SEED, phase_trainer_sha256=source_sha256s[SELF],
                 phase_patch_sha256=source_sha256s["phase_patch.py"], native_trainer_sha256=source_sha256s[NATIVE],
                 initial_expanded_sha256=INITIAL_EXPANDED_SHA)
    exact(common["phase"], phase, "Exact original training phase marker required")

    native = canonical(arm.native_report, verifier, dict, 64 * 1024)
    witness = canonical(arm.seed_witness, verifier, dict, 16 * 1024)
    canonical(arm.native_visits, verifier, list, 1024 * 1024)
    require(common["native_result_sha256"] == digest(arm.native_report)
            and common["repeat_seed_witness_sha256"] == digest(arm.seed_witness),
            "Closed common FIT arm must bind native receipt and actual seed witness")
    exact(witness, dict(status="HRSID_MASK_PHASE_REPEAT_SEED_WITNESS_V1",
                        protocol=identity["training_protocol"], mask_resize_mode=mode, stage="qualification",
                        initialization_seed=INITIALIZATION_SEED, training_seed=TRAINING_SEED, global_rank=0,
                        train_config_seed=TRAINING_SEED, torch_cpu_initial_seed=TRAINING_SEED,
                        torch_cuda_initial_seed=TRAINING_SEED, lightning_global_seed=TRAINING_SEED,
                        lightning_seed_workers=True), "Actual common FIT runtime seed witness required")
    native_rows = verifier.validate_report(native, arm.native_visits, protocol=identity["training_protocol"],
                                           seed=TRAINING_SEED, mode=mode, trainer_sha=source_sha256s[SELF])
    verifier.validate_stdout(arm.stdout, native, witness, digest(arm.seed_witness), native_rows)
    require(native["removed_empty_rows"] == 0 and native["targets_after_filter"] == stats["common_targets_total"]
            and all((a["image_id"], a["common_nonempty"], a["common_nonempty"]) ==
                    (b["image_id"], b["labels"], b["postfilter_labels"])
                    for a, b in zip(rows, native_rows, strict=True)),
            "Consumed FIT visits must match actual native batch image, labels and filtering")
    return rows, native, native_rows


def load_verifier(source):
    if source is None:
        with QUALIFIER_PATH.open("rb") as stream:
            source = stream.read(64 * 1024 + 1)
    require(type(source) is bytes and 0 < len(source) <= 64 * 1024 and digest(source) == QUALIFIER_SHA,
            "Exact independently pinned original short-window verifier required")
    module = ModuleType("_common_fit_original_short_window_verifier")
    exec(compile(source, QUALIFIER_PATH.name, "exec"), module.__dict__)
    return module


def accept_short_pair(*, training_protocol, implementation_commit, source_sha256s, native, center,
                      native_qualifier_source=None):
    """Return the canonical receipt fields consumed by the E24 training lock."""
    require(all(is_hex(value, 40) and value not in ORIGINAL_PROTOCOLS
                for value in (training_protocol, implementation_commit)),
            "Prospectively locked common FIT protocol and implementation required")
    require(type(source_sha256s) is dict and set(source_sha256s) == SOURCES
            and all(is_hex(value) for value in source_sha256s.values())
            and all(source_sha256s[key] == value for key, value in SOURCE_PINS.items()),
            "Exact common FIT worker source manifest required")
    identity = dict(training_protocol=training_protocol, implementation_commit=implementation_commit,
                    training_seed=TRAINING_SEED, initialization_seed=INITIALIZATION_SEED,
                    initial_expanded_sha256=INITIAL_EXPANDED_SHA, fit_index_sha256=FIT_INDEX_SHA,
                    source_manifest_sha256=digest(compact(source_sha256s)))
    verifier = load_verifier(native_qualifier_source)
    consumed, native_report, native_rows = verify_arm(native, "native", identity, source_sha256s, verifier)
    _, center_report, center_rows = verify_arm(center, "center", identity, source_sha256s, verifier)
    require(native.closure["terminal_sha256"] != center.closure["terminal_sha256"],
            "Distinct independently closed original workers required")
    require(native.consumed_ledger == center.consumed_ledger, "Exact ordered common FIT consumed streams must match")
    require(all(native_report[key] == center_report[key] for key in verifier.LR_FIELDS),
            "Exact paired native AdamW learning rates required")
    shared = ("image_id", "image_sha256", "cpu_rng_sha256", "labels_sha256", "postfilter_boxes_sha256")
    require(all(all(a[key] == b[key] for key in shared)
                and (a["postfilter_masks_sha256"] != b["postfilter_masks_sha256"]) == (row["common_changed_masks"] > 0)
                for row, a, b in zip(consumed, native_rows, center_rows, strict=True)),
            "Actual native batches must agree in image, RNG, nonmask supervision and reported mask changes")
    require(all(a["postfilter_masks_sha256"] == row["native_common_masks_sha256"]
                and b["postfilter_masks_sha256"] == row["center_common_masks_sha256"]
                for row, a, b in zip(consumed, native_rows, center_rows, strict=True)),
            "Exact consumed mask fingerprints must match the actual native training batches")
    changed = sum(row["common_changed_masks"] > 0 for row in consumed)
    require(changed > 0, "At least one actual common FIT mask must change")
    return dict(status=QUALIFIED_STATUS, **identity, ordered_common_fit_visits_match=True, visits_per_arm=VISITS,
                optimizer_updates_per_arm=UPDATES, ema_updates_per_arm=UPDATES, changed_mask_visits=changed,
                native_arm_receipt_sha256=digest(native.common_receipt),
                center_arm_receipt_sha256=digest(center.common_receipt),
                shared_consumed_ledger_sha256=digest(native.consumed_ledger), **SCOPE)

