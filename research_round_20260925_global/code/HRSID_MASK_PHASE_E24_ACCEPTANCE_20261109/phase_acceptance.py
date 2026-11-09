"""Independently admit closed phase-training E24 arms and their matched FIT pair."""
from hashlib import sha256
import json
from pathlib import Path
import re
from types import ModuleType

PROTOCOL = "d66baf3c57fafdba4242ad1b55e08cf76bfd77f7"
NATIVE_VALIDATOR_SHA = "71a05d4ead3848a34ec073303dacb5fdb364e7eb0764a4caafaa0d35353ea576"
FILTER_SHA = "3b164cbf8c104539eabee8775c8e1f297d9a1607e37473c4d7a4a9b561f93471"
INITIAL_SHA = "63ff93afdd977ba79b5b5a46ba8b4839ef6c40eb0722b29e4667d4932404fbc2"
PHASE_MARKER = "sar_seal_hrsid_mask_resize_phase_v1"
TRAILER = "HRSID_MASK_PHASE_E24_OUTPUT_COMPLETE"
EPOCHS, VISITS, WIDTH, UPDATES = 24, 2940, 164, 17640
SOURCE_FIELDS = ("phase_trainer_sha256", "phase_patch_sha256", "native_trainer_sha256")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def is_sha256(value):
    return type(value) is str and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf8")


def object_from(raw, name):
    def pairs(items):
        value = dict(items)
        require(len(value) == len(items), "Duplicate " + name + " key")
        return value

    def nonfinite(_):
        raise ValueError("Nonfinite JSON constant")

    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)
    except (ValueError, UnicodeError) as error:
        raise ValueError("Invalid " + name) from error
    require(type(value) is dict, "Expected object for " + name)
    return value


def same_fields(value, expected):
    return (type(value) is dict and set(value) == set(expected)
            and all(type(value[k]) is type(v) and value[k] == v for k, v in expected.items()))


def load_native_validator(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve(strict=True) == path, "Canonical native validator path")
    raw = path.read_bytes()
    require(sha256(raw).hexdigest() == NATIVE_VALIDATOR_SHA, "Exact immutable native validator")
    module = ModuleType("_phase_scoped_native_e24_validator")
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    module.PROTOCOL = PROTOCOL
    return module


def check_phase(value, mode):
    fixed = dict(protocol=PROTOCOL, mask_resize_mode=mode, stage="e24", seed=5201,
                 initial_expanded_sha256=INITIAL_SHA)
    require(mode in ("native", "center") and type(value) is dict
            and set(value) == set(fixed) | set(SOURCE_FIELDS)
            and all(type(value[k]) is type(v) and value[k] == v for k, v in fixed.items())
            and all(is_sha256(value[k]) for k in SOURCE_FIELDS), "Exact locked phase marker")


def split_native_stdout(raw, mode):
    """Verify the single phase completion trailer and return the unchanged native prefix."""
    require(mode in ("native", "center") and type(raw) is bytes and raw.endswith(b"\n"),
            "Complete original stdout and phase mode required")
    lines = raw.splitlines(keepends=True)
    expected = dict(event=TRAILER, mask_resize_mode=mode, protocol=PROTOCOL)
    require(len(lines) >= 2 and same_fields(object_from(lines[-1], "phase stdout trailer"), expected),
            "Exact final original phase completion event")
    for line in lines[:-1]:
        if TRAILER.encode() in line and line.lstrip().startswith(b"{"):
            require(object_from(line, "possible repeated phase event").get("event") != TRAILER,
                    "Duplicate phase completion event")
    return raw[:-len(lines[-1])]


def validate_phase_sidecar(raw, result_raw, expected):
    check_phase(expected, expected.get("mask_resize_mode") if type(expected) is dict else None)
    sidecar = object_from(raw, "phase provenance sidecar")
    fields = dict(expected, status="NATIVE_HRSID_MASK_RESIZE_E24_CLOSED_PENDING_INDEPENDENT_ACCEPTANCE",
                  native_result_sha256=sha256(result_raw).hexdigest())
    require(same_fields(sidecar, fields), "Exact phase sidecar, own mode, source and native result digest")
    return expected.copy()


def validate_result(native, result_bytes, epoch_bytes, visit_bytes):
    actual = object_from(result_bytes, "unmodified native training result")
    count = actual.get("filtered_empty_rows_total")
    require(type(count) is int and 0 <= count <= WIDTH * VISITS * EPOCHS, "Exact real filter count including zero")
    if count:
        return native.validate_result(result_bytes, epoch_bytes, visit_bytes)
    # The historical validator imposes positivity only at result-header validation.
    # Check every other header field there, then keep the actual zero in all ledgers,
    # original log comparison, checkpoint validation and receipt.
    native.validate_result(encoded(dict(actual, filtered_empty_rows_total=1)), epoch_bytes, visit_bytes)
    return actual


def validate_phase_run(native, closure, result_bytes, epoch_bytes, visit_bytes, stdout_bytes,
                       sidecar_bytes, expected_phase, checkpoint_reader, expected_fit_ids):
    """Validate exact closed artifacts and return arm receipt bytes and SHA-256."""
    native.validate_closure(closure, stdout_bytes)
    phase = validate_phase_sidecar(sidecar_bytes, result_bytes, expected_phase)
    prefix = split_native_stdout(stdout_bytes, phase["mask_resize_mode"])
    result = validate_result(native, result_bytes, epoch_bytes, visit_bytes)
    require(encoded(object_from(prefix.rstrip().splitlines()[-1], "native result stdout")) == encoded(result),
            "Native final stdout fields and scalar types agree")
    endpoints = native.validate_epochs(epoch_bytes, visit_bytes, prefix, result, expected_fit_ids)
    for ref in result["archive_refs"]:
        cumulative, audit = endpoints[ref["epoch"]]
        require(all(ref[k] == audit[k] for k in ("adam_parameter_clocks", "native_base_lrs", "native_next_lrs")),
                "Endpoint optimizer clocks match the epoch audit")
        saved, size, digest = checkpoint_reader(ref)
        require(type(size) is int and size == ref["archive_bytes"] and digest == ref["archive_sha256"],
                "Exact independently measured archive size and SHA-256")
        require(type(saved) is dict and same_fields(saved.get(PHASE_MARKER), phase),
                "Exact own phase marker on every fixed archive")
        native.validate_checkpoint(saved, ref, cumulative)
        del saved

    receipt = dict(status="HRSID_MASK_PHASE_E24_ARM_ACCEPTED_V1", phase=phase,
                   original_os_wait=True, original_os_exit=0, resource_stop=False,
                   completed_epochs=EPOCHS, optimizer_updates=UPDATES, ema_updates=UPDATES,
                   target_filter_sha256=FILTER_SHA, filtered_empty_rows_total=result["filtered_empty_rows_total"],
                   checkpoint_sha256=result["archive_refs"][-1]["archive_sha256"],
                   visit_ledger_sha256=sha256(visit_bytes).hexdigest(), result_sha256=sha256(result_bytes).hexdigest(),
                   stdout_sha256=closure["stdout_sha256"], terminal_sha256=closure["terminal_sha256"],
                   phase_provenance_sha256=sha256(sidecar_bytes).hexdigest())
    raw = encoded(receipt)
    return raw, sha256(raw).hexdigest()


def read_visit_epochs(raw):
    lines = raw.splitlines(keepends=True)
    require(len(lines) == EPOCHS and all(line.endswith(b"\n") for line in lines), "Exact closed 24 epoch ledger")
    for epoch, line in enumerate(lines, 1):
        row = object_from(line, "FIT visit epoch")
        visits = row.get("visits")
        require(same_fields(row, {"epoch": epoch, "visits": visits}) and type(visits) is list
                and len(visits) == VISITS and all(type(v) is list and len(v) == 2
                    and type(v[0]) is int and type(v[1]) is int and 0 <= v[1] <= WIDTH for v in visits),
                "Exact closed FIT visit shape and target counts")
        yield visits


def compare_ordered_visits(left_bytes, right_bytes):
    """Compare raw FIT order with padding; return only aggregate retention differences."""
    different = left_total = right_total = left_zero = right_zero = 0
    for left, right in zip(read_visit_epochs(left_bytes), read_visit_epochs(right_bytes), strict=True):
        for (left_id, ln), (right_id, rn) in zip(left, right, strict=True):
            require(left_id == right_id, "Ordered FIT image IDs differ")
            different += ln != rn
            left_total += ln
            right_total += rn
            left_zero += ln == 0
            right_zero += rn == 0
    return dict(ordered_fit_visits_match=True, completed_epochs=EPOCHS, visits_per_arm=EPOCHS*VISITS,
                differing_target_count_visits=different, native_retained_targets=left_total,
                center_retained_targets=right_total, native_zero_target_visits=left_zero, center_zero_target_visits=right_zero)


def create_pair_receipt(native_raw, center_raw, native_visits, center_visits):
    fixed = dict(status="HRSID_MASK_PHASE_E24_ARM_ACCEPTED_V1", original_os_wait=True, original_os_exit=0,
                 resource_stop=False, completed_epochs=EPOCHS, optimizer_updates=UPDATES, ema_updates=UPDATES,
                 target_filter_sha256=FILTER_SHA)
    extra = ("checkpoint_sha256", "visit_ledger_sha256", "result_sha256", "stdout_sha256",
             "terminal_sha256", "phase_provenance_sha256")
    arms = []
    for mode, raw, visits in (("native", native_raw, native_visits), ("center", center_raw, center_visits)):
        arm = object_from(raw, "independently accepted arm")
        require(set(arm) == set(fixed) | set(extra) | {"phase", "filtered_empty_rows_total"}
                and all(type(arm[k]) is type(v) and arm[k] == v for k, v in fixed.items())
                and all(is_sha256(arm[k]) for k in extra)
                and type(arm["filtered_empty_rows_total"]) is int
                and 0 <= arm["filtered_empty_rows_total"] <= WIDTH*VISITS*EPOCHS
                and arm["visit_ledger_sha256"] == sha256(visits).hexdigest(), "Exact independent arm and FIT ledger")
        check_phase(arm["phase"], mode)
        arms.append(arm)
    native, center = arms
    require(dict(native["phase"], mask_resize_mode="center") == center["phase"]
            and native["checkpoint_sha256"] != center["checkpoint_sha256"],
            "Distinct own archives for the same fixed source and initialization")
    stats = compare_ordered_visits(native_visits, center_visits)
    receipt = dict(status="HRSID_MASK_PHASE_E24_PAIR_READY_V1", training_protocol=PROTOCOL,
                   ordered_fit_visits_match=True, completed_epochs=EPOCHS, visits_per_arm=VISITS*EPOCHS,
                   native_arm_receipt_sha256=sha256(native_raw).hexdigest(),
                   center_arm_receipt_sha256=sha256(center_raw).hexdigest(),
                   native_checkpoint_sha256=native["checkpoint_sha256"],
                   center_checkpoint_sha256=center["checkpoint_sha256"])
    raw = encoded(receipt)
    return raw, sha256(raw).hexdigest(), stats
