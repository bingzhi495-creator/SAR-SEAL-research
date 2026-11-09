"""Independently qualify closed HRSID short windows for two confirmation seeds."""
from hashlib import sha256
import json
import math
import re
from typing import NamedTuple


PROTOCOL = "a49361b92977a605267025b1e7798420936cbe53"
HISTORICAL_PROTOCOL = "d66baf3c57fafdba4242ad1b55e08cf76bfd77f7"
HISTORICAL_PAIR_SHA = "4e7bab90d8a32bc0a404ebb7fde64472474d873529367a756acb68b099c16cde"
HISTORICAL_TRAINER_SHA = "8550b43791a6cb6f53736dc7e64b9c5c6204723cf4c07c3b112491ba1bd74043"
REPEAT_TRAINER_SHA = "42a174573a04c529dfbae943bcc88ef4ed5d793d7eda6766511396034fda321f"
PATCH_SHA = "811eccf2b42319272aea8b3f50a71bc029734d006bba43260f66117e9421948e"
NATIVE_SHA = "11d17cab878d62576b8d19e6231caab90c9990c03909cb01e1d04468add452a6"
INITIAL_SHA = "63ff93afdd977ba79b5b5a46ba8b4839ef6c40eb0722b29e4667d4932404fbc2"
CLASSIFIER_SHA = "81bccf7e3ea5e3eafb988abe6331ae7682ae95201a87b7b7c981630418d7f01f"
COCO_SHA = "a44613a4ecd6b5ba61a62002c600b0b6cb7a9da2936a45317ec4b62c635fb99b"
WIDTH100_SHA = "40524fc63797d2fcab7b50b6ee1f041f9429df73e505cb9e647a5a8cdf6f2e2f"
SEEDS = (5202, 5203)
MODES = ("native", "center")
VISITS, UPDATES = 32, 8
QUALIFICATION_STATUS = "NATIVE_HRSID_MASK_RESIZE_QUALIFICATION_COMPLETE"
LR_FIELDS = ("native_base_lrs", "native_next_lrs")
INITIAL_EVENT = "HRSID_COCO_NATIVE_INITIALIZATION"
START_EVENT = "HRSID_COCO_NATIVE_E24_STARTED"
SEED_EVENT = "HRSID_MASK_PHASE_REPEAT_SEED_OBSERVED"
CLOSE_EVENT = "HRSID_MASK_PHASE_SHORT_WINDOW_CLOSED"
FILTER_EVENT = "HRSID_NATIVE_EMPTY_MASK_FILTER_APPLIED"


class Window(NamedTuple):
    report: dict
    rows: tuple
    report_sha256: str
    visit_ledger_sha256: str
    seed_witness_sha256: str | None
    stdout_sha256: str | None
    terminal_sha256: str | None


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(raw):
    return sha256(raw).hexdigest()


def sha256_value(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def unique(pairs):
    obj = dict(pairs)
    require(len(obj) == len(pairs), "Duplicate qualification JSON fields")
    return obj


def reject_constant(value):
    raise ValueError("Nonfinite qualification JSON number: " + value)


def parse(raw, shape, limit=1024**2):
    require(type(raw) is bytes and 0 < len(raw) <= limit, "Bounded exact qualification bytes required")
    value = json.loads(raw, object_pairs_hook=unique, parse_constant=reject_constant)
    require(type(value) is shape, "Unexpected qualification JSON shape")
    return value


def exact(value, expected, reason, *, extra=()):
    require(type(value) is dict and set(value) == set(expected) | set(extra)
            and all(type(value[key]) is type(item) and value[key] == item for key, item in expected.items()), reason)


def validate_witness(raw, *, seed, mode):
    require(type(seed) is int and seed in SEEDS and mode in MODES, "Authorized repeat seed and arm required")
    expected = dict(status="HRSID_MASK_PHASE_REPEAT_SEED_WITNESS_V1", protocol=PROTOCOL, mask_resize_mode=mode,
                    stage="qualification", initialization_seed=5201, training_seed=seed, global_rank=0,
                    train_config_seed=seed, torch_cpu_initial_seed=seed, torch_cuda_initial_seed=seed,
                    lightning_global_seed=seed, lightning_seed_workers=True)
    value = parse(raw, dict, 16384)
    exact(value, expected, "Exact effective runtime repeat seed required")
    return value


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def window_rows(raw):
    rows = parse(raw, list)
    require(len(rows) == VISITS, "Exactly 32 original qualification visits")
    fingerprints = ("image_sha256", "cpu_rng_sha256", "labels_sha256", "masks_sha256",
                    "boxes_sha256", "postfilter_masks_sha256", "postfilter_boxes_sha256")
    for batch, row in enumerate(rows, 1):
        exact(row, dict(batch=batch), "Ordered short-window batch cursor required",
              extra=("image_id", "labels", "postfilter_labels", *fingerprints))
        require(all(type(row[key]) is int for key in ("image_id", "labels", "postfilter_labels"))
                and row["image_id"] >= 0 and 0 <= row["postfilter_labels"] <= row["labels"] <= 164
                and all(sha256_value(row[key]) for key in fingerprints), "Exact short-window visit identity")
        if row["labels"] == row["postfilter_labels"]:
            require(row["masks_sha256"] == row["postfilter_masks_sha256"]
                    and row["boxes_sha256"] == row["postfilter_boxes_sha256"],
                    "Unfiltered supervision must retain the same tensor digests")
    return tuple(rows)


def inputs(rows):
    return [(row["image_id"], row["image_sha256"], row["cpu_rng_sha256"]) for row in rows]


def input_digest(values):
    return digest(json.dumps(values, separators=(",", ":")).encode())


def validate_report(report, visit_bytes, *, protocol, seed, mode, trainer_sha):
    rows = window_rows(visit_bytes)
    expected = dict(status=QUALIFICATION_STATUS, protocol=protocol, mask_resize_mode=mode, stage="qualification",
                    seed=seed, phase_trainer_sha256=trainer_sha, phase_patch_sha256=PATCH_SHA,
                    native_trainer_sha256=NATIVE_SHA, initial_expanded_sha256=INITIAL_SHA,
                    initial_online_sha256=INITIAL_SHA, classifier_initial_sha256=CLASSIFIER_SHA,
                    input_pair_sha256=input_digest(inputs(rows)), visit_ledger_sha256=digest(visit_bytes),
                    visits=VISITS, optimizer_updates=UPDATES, ema_updates=UPDATES, completed_epochs=0,
                    actual_trainer_epoch_cursor=1, planned_updates=73500, warmup_updates=2205,
                    removed_empty_rows=sum(row["labels"] - row["postfilter_labels"] for row in rows),
                    targets_after_filter=sum(row["postfilter_labels"] for row in rows),
                    FIT=True, DEV=False, TEST=False, AP=False, checkpoint_saved=False)
    extra = ("classifier_online_ending_sha256", "adam_parameter_clocks", *LR_FIELDS, "loss_min", "loss_max",
             "classifier_gradient_norm_max", "peak_allocated_bytes", "peak_reserved_bytes", "seconds")
    exact(report, expected, "Exact qualified short-window report and visit ledger required", extra=extra)
    exact(report["adam_parameter_clocks"], {"8": 543}, "Exact original 543 AdamW parameter clocks required")
    ending = report["classifier_online_ending_sha256"]
    require(sha256_value(ending) and ending != CLASSIFIER_SHA, "Online classifier must change before EMA swap")
    require(all(finite(report[key]) for key in ("loss_min", "loss_max", "classifier_gradient_norm_max", "seconds"))
            and 0 <= report["loss_min"] <= report["loss_max"]
            and report["classifier_gradient_norm_max"] > 0 and report["seconds"] > 0,
            "Finite loss, gradient and duration with a nonzero update required")
    require(type(report["peak_allocated_bytes"]) is type(report["peak_reserved_bytes"]) is int
            and 0 < report["peak_allocated_bytes"] <= report["peak_reserved_bytes"] <= 6 * 1024**3,
            "Original short-window CUDA memory accounting required")
    base, next_lrs = (report[key] for key in LR_FIELDS)
    require(type(base) is type(next_lrs) is list and len(base) == len(next_lrs) == 28
            and all(finite(a) and finite(b) and a >= 0
                    and math.isclose(a * (UPDATES / 2205), b, rel_tol=1e-9, abs_tol=1e-12)
                    for a, b in zip(base, next_lrs, strict=True)), "Exact native short-window learning rates")
    return rows


def validate_stdout(stdout, report, witness, witness_sha, rows):
    require(type(stdout) is bytes and 0 < len(stdout) <= 2 * 1024**2, "Bounded original worker stdout required")
    events = [parse(line, dict) for line in stdout.splitlines() if line.lstrip().startswith(b"{")]
    expected = {
        INITIAL_EVENT: dict(class_head_slots=2, coco_asset_bytes=134545398, coco_asset_sha256=COCO_SHA,
                            expanded_online_sha256=INITIAL_SHA, initialized_from_SSDD=False,
                            native_original_width=100, query_groups=13, query_width=164,
                            training=False, width100_loaded_sha256=WIDTH100_SHA),
        START_EVENT: dict(planned_epochs=100, stop_after_epoch=24, updates_per_epoch=735, visits_per_epoch=2940),
        SEED_EVENT: dict(seed_witness_sha256=witness_sha, **witness),
        CLOSE_EVENT: report,
    }
    indices = {}
    for name, fields in expected.items():
        positions = [index for index, row in enumerate(events) if row.get("event") == name]
        reason = "Exact original runtime seed event required" if name == SEED_EVENT else "Exact original " + name
        require(len(positions) == 1, reason)
        index, = positions
        exact(events[index], dict(event=name, **fields), reason)
        indices[name] = index
    require(indices[INITIAL_EVENT] < min(indices[START_EVENT], indices[SEED_EVENT])
            and max(indices[START_EVENT], indices[SEED_EVENT]) < indices[CLOSE_EVENT] == len(events) - 1,
            "Original initialization, effective seed and short completion order required")
    filtered = [row for row in events if row.get("event") == FILTER_EVENT]
    expected_filtered = [
        dict(event=FILTER_EVENT, epoch=1, batch_idx=row["batch"] - 1, image_id=row["image_id"],
             removed_rows=row["labels"] - row["postfilter_labels"])
        for row in rows if row["labels"] != row["postfilter_labels"]
    ]
    require(len(filtered) == len(expected_filtered), "Original filter events must agree with actual row losses")
    for actual, expected_row in zip(filtered, expected_filtered, strict=True):
        exact(actual, expected_row, "Original filter events must agree with actual row losses")


def validate_arm(closure, report_bytes, visit_bytes, witness_bytes, stdout_bytes, *, seed, mode,
                 repeat_trainer_sha256):
    """Validate one independently OS-closed, seed-qualified 32-visit arm."""
    required = dict(original_os_wait=True, original_os_exit=0, resource_stop=False, worker_closed=True)
    optional = {"stdout_sha256", "terminal_sha256"}
    require(type(closure) is dict and set(required) <= set(closure) <= set(required) | optional
            and all(type(closure[key]) is type(value) and closure[key] == value for key, value in required.items())
            and all(sha256_value(closure[key]) for key in optional & closure.keys()),
            "Exact original short worker OS closure required")
    witness = validate_witness(witness_bytes, seed=seed, mode=mode)
    require(repeat_trainer_sha256 == REPEAT_TRAINER_SHA,
            "Exact independently pinned repeat trainer source required")
    report = parse(report_bytes, dict, 65536)
    rows = validate_report(report, visit_bytes, protocol=PROTOCOL, seed=seed, mode=mode,
                           trainer_sha=repeat_trainer_sha256)
    witness_sha = digest(witness_bytes)
    validate_stdout(stdout_bytes, report, witness, witness_sha, rows)
    stdout_sha = digest(stdout_bytes)
    require(closure.get("stdout_sha256", stdout_sha) == stdout_sha, "Exact independently closed stdout digest required")
    return Window(report, rows, digest(report_bytes), digest(visit_bytes), witness_sha, stdout_sha,
                  closure.get("terminal_sha256"))


def compare_pair(native, center):
    require(inputs(native) == inputs(center), "Same-seed ordered image, normalized pixels and CPU RNG required")
    fields = {
        "differing_prefilter_count_batches": "labels",
        "differing_prefilter_label_batches": "labels_sha256",
        "differing_prefilter_mask_batches": "masks_sha256",
        "differing_prefilter_box_batches": "boxes_sha256",
        "differing_postfilter_count_batches": "postfilter_labels",
        "differing_postfilter_mask_batches": "postfilter_masks_sha256",
        "differing_postfilter_box_batches": "postfilter_boxes_sha256",
    }
    changes = {key: sum(a[field] != b[field] for a, b in zip(native, center, strict=True))
               for key, field in fields.items()}
    require(changes["differing_postfilter_mask_batches"] > 0, "At least one actual supervision mask must change")
    return changes


def trace_digests(rows):
    return dict(ordered_image_ids_sha256=input_digest([row["image_id"] for row in rows]),
                ordered_cpu_rng_sha256=input_digest([row["cpu_rng_sha256"] for row in rows]),
                input_pair_sha256=input_digest(inputs(rows)))


def historical_anchor(pair_bytes, native_bytes, center_bytes):
    pair = parse(pair_bytes, dict, 65536)
    require(digest(pair_bytes) == HISTORICAL_PAIR_SHA, "Exact independently closed seed-5201 pair required")
    report_keys = ("differing_postfilter_count_batches", "differing_postfilter_mask_batches",
                   "differing_postfilter_box_batches")
    expected = dict(status="HRSID_MASK_PHASE_PAIRED_QUALIFICATION_COMPLETE", protocol=HISTORICAL_PROTOCOL,
                    visits_per_arm=VISITS, updates_per_arm=UPDATES, identical_image_order_pixels_and_cpu_rng=True,
                    FIT=True, DEV=False, TEST=False, AP=False)
    exact(pair, expected, "Historical seed-5201 qualification scope", extra=(*report_keys, *MODES))
    native, center = (validate_report(pair[mode], visits, protocol=HISTORICAL_PROTOCOL, seed=5201, mode=mode,
                                     trainer_sha=HISTORICAL_TRAINER_SHA)
                      for mode, visits in zip(MODES, (native_bytes, center_bytes), strict=True))
    changes = compare_pair(native, center)
    require(all(pair[key] == changes[key] for key in report_keys)
            and all(pair["native"][field] == pair["center"][field] for field in LR_FIELDS),
            "Historical pair ledgers must independently reproduce its locked facts")
    return trace_digests(native), {field: pair["native"][field] for field in LR_FIELDS}


def arm_proof(arm):
    return dict(report_sha256=arm.report_sha256, visit_ledger_sha256=arm.visit_ledger_sha256,
                repeat_seed_witness_sha256=arm.seed_witness_sha256, stdout_sha256=arm.stdout_sha256,
                terminal_sha256=arm.terminal_sha256, removed_empty_rows=arm.report["removed_empty_rows"],
                targets_after_filter=arm.report["targets_after_filter"],
                classifier_online_ending_sha256=arm.report["classifier_online_ending_sha256"])


def validate_repeats(arms, historical_pair_bytes, historical_native_visits, historical_center_visits):
    """Compare all closed arms and the independently frozen 5201 reference."""
    require(type(arms) is dict and set(arms) == {(seed, mode) for seed in SEEDS for mode in MODES},
            "Both independently accepted arms are required for each confirmation seed")
    historical, lrs = historical_anchor(historical_pair_bytes, historical_native_visits, historical_center_visits)
    pairs, traces = {}, [historical]
    for seed in SEEDS:
        native, center = (arms[seed, mode] for mode in MODES)
        for mode, arm in zip(MODES, (native, center), strict=True):
            require(type(arm) is Window and arm.report["seed"] == seed and arm.report["mask_resize_mode"] == mode
                    and arm.report["protocol"] == PROTOCOL and arm.report["phase_trainer_sha256"] == REPEAT_TRAINER_SHA
                    and all(arm.report[field] == lrs[field] for field in LR_FIELDS),
                    "Original schedule and independently qualified repeat seed and arm required")
        changes = compare_pair(native.rows, center.rows)
        trace = trace_digests(native.rows)
        traces.append(trace)
        pairs[str(seed)] = dict(seed=seed, visits_per_arm=VISITS, optimizer_updates_per_arm=UPDATES,
                                ema_updates_per_arm=UPDATES, identical_image_order_pixels_and_cpu_rng=True,
                                **trace, **changes, arms=dict(native=arm_proof(native), center=arm_proof(center)))
    require(all(len({trace[field] for trace in traces}) == len(traces)
                for field in ("ordered_image_ids_sha256", "ordered_cpu_rng_sha256", "input_pair_sha256")),
            "Independent ordered image and CPU RNG traces must differ across 5201, 5202 and 5203")
    return dict(status="HRSID_MASK_PHASE_REPEAT_SHORT_QUALIFICATION_ACCEPTED_V1", protocol=PROTOCOL,
                initialization_seed=5201, initial_expanded_sha256=INITIAL_SHA, classifier_initial_sha256=CLASSIFIER_SHA,
                repeat_trainer_sha256=REPEAT_TRAINER_SHA, historical_pair_sha256=HISTORICAL_PAIR_SHA,
                historical_trace=historical, distinct_order_and_cpu_rng_across_seeds=True,
                FIT=True, DEV=False, TEST=False, AP=False, checkpoint_saved=False, pairs=pairs)
