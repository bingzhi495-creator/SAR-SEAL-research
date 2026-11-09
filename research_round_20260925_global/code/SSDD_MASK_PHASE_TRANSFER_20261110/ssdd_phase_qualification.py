"""Verify paired short SSDD FIT qualification from independently closed worker bytes."""

from hashlib import sha256
import json
import math
import re
from typing import NamedTuple


STATUS = "SSDD_MASK_PHASE_SHORT_PAIR_INDEPENDENTLY_ACCEPTED_V1"
REPORT = "SSDD_PHASE_TRAIN_SAFE.json"
VISITS = "QUALIFICATION_VISITS_PRIVATE.json"
Q100_SHA = "40524fc63797d2fcab7b50b6ee1f041f9429df73e505cb9e647a5a8cdf6f2e2f"
COCO_SHA = "a44613a4ecd6b5ba61a62002c600b0b6cb7a9da2936a45317ec4b62c635fb99b"
RECIPE_SHA = "ba9508d9acab5ff68b10db8a1aff5d9001868d6dda00e39f299d6f5cc1b8011b"
FIT_COUNT = 741
PADDED = 744
SHORT_VISITS = 32
UPDATES = 8
WARMUP = 558
PLANNED = 18600
SEED = 5201


class QualificationArm(NamedTuple):
    closure: dict
    report_bytes: bytes
    visit_bytes: bytes
    stdout_bytes: bytes


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return sha256(raw).hexdigest()


def is_hex(value, digits=64):
    return type(value) is str and re.fullmatch(rf"[0-9a-f]{{{digits}}}", value) is not None


def packed(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def encode_receipt(receipt):
    return (json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def unique(pairs):
    value = dict(pairs)
    require(len(value) == len(pairs), "Duplicate short qualification JSON field")
    return value


def reject_constant(value):
    raise ValueError(f"Nonfinite short qualification number: {value}")


def read_json(raw, shape, limit):
    require(type(raw) is bytes and 0 < len(raw) <= limit, "Expected bounded original short qualification bytes")
    value = json.loads(raw, object_pairs_hook=unique, parse_constant=reject_constant)
    require(type(value) is shape, "Unexpected short qualification JSON shape")
    return value


def exact(value, expected, message, *, extra=()):
    require(type(value) is dict and value.keys() == set(expected) | set(extra)
            and all(type(value[key]) is type(item) and value[key] == item for key, item in expected.items()), message)


def finite(value):
    return type(value) is float and math.isfinite(value) or type(value) is int and abs(value) < 10**100


def validate_closure(arm):
    closed = arm.closure
    exact(closed, dict(original_os_wait=True, original_os_exit=0, resource_stop=False, worker_closed=True),
          "Successful independent original worker closure required",
          extra=("terminal_sha256", "stdout_sha256", "output_files"))
    files = closed["output_files"]
    require(type(files) is list and len(files) == 2 and all(type(name) is str for name in files)
            and set(files) == {REPORT, VISITS}, "Qualification can close with only the report and visit ledger")
    require(is_hex(closed["terminal_sha256"]) and is_hex(closed["stdout_sha256"])
            and type(arm.stdout_bytes) is bytes and 0 < len(arm.stdout_bytes) <= 2 * 1024**2
            and digest(arm.stdout_bytes) == closed["stdout_sha256"], "Exact independently closed stdout required")


def visit_rows(raw, allowed_ids):
    rows = read_json(raw, list, 256 * 1024)
    require(len(rows) == SHORT_VISITS, "Exactly 32 original SSDD visits required")
    signatures = ("image_sha256", "cpu_rng_sha256", "labels_sha256", "masks_sha256", "boxes_sha256",
                  "postfilter_masks_sha256", "postfilter_boxes_sha256")
    for cursor, row in enumerate(rows, 1):
        exact(row, dict(batch=cursor), "Exact SSDD short visit cursor and fields required",
              extra=("image_id", "labels", "postfilter_labels", *signatures))
        require(type(row["image_id"]) is int and row["image_id"] in allowed_ids
                and type(row["labels"]) is type(row["postfilter_labels"]) is int
                and 0 <= row["postfilter_labels"] <= row["labels"] <= 100
                and all(is_hex(row[field]) for field in signatures), "Invalid FIT visit identity or target count")
        if row["postfilter_labels"] == row["labels"]:
            require(row["postfilter_masks_sha256"] == row["masks_sha256"]
                    and row["postfilter_boxes_sha256"] == row["boxes_sha256"],
                    "Unfiltered SSDD visit must retain exact mask and box fingerprints")
    require(len(rows) - len({row["image_id"] for row in rows}) <= PADDED - FIT_COUNT,
            "Too many duplicate identities for the original aligned FIT loader")
    return rows


def ordered_inputs(rows):
    return [(row["image_id"], row["image_sha256"], row["cpu_rng_sha256"]) for row in rows]


def validate_telemetry(report):
    begin, end = (report[field] for field in ("classifier_initial_sha256", "classifier_online_ending_sha256"))
    require(is_hex(begin) and is_hex(end) and begin != end, "Updated native online classifier required")
    clocks = report["adam_parameter_clocks"]
    require(type(clocks) is dict and str(UPDATES) in clocks
            and clocks.keys() <= {str(step) for step in range(1, UPDATES + 1)}
            and all(type(count) is int and 0 < count <= 545 for count in clocks.values())
            and sum(clocks.values()) <= 545, "Native AdamW parameter clocks must reach eight updates")
    base, current = report["native_base_lrs"], report["native_next_lrs"]
    require(type(base) is type(current) is list and len(base) == len(current) > 0
            and all(finite(a) and finite(b) and a >= 0 and math.isclose(b, a * (UPDATES / WARMUP),
                                                                       rel_tol=1e-9, abs_tol=1e-12)
                    for a, b in zip(base, current, strict=True)), "Native E100 warmup rates differ")
    loss_min, loss_max, gradient, elapsed = (report[key] for key in
        ("loss_min", "loss_max", "classifier_gradient_norm_max", "elapsed_seconds"))
    allocated, reserved = (report[key] for key in ("peak_allocated_bytes", "peak_reserved_bytes"))
    require(all(finite(value) for value in (loss_min, loss_max, gradient, elapsed))
            and 0 <= loss_min <= loss_max and gradient > 0 and elapsed > 0
            and type(allocated) is type(reserved) is int and 0 < allocated <= reserved,
            "Finite loss, nonzero gradient, positive duration and CUDA memory accounting required")


def validate_stdout(raw, report, rows):
    prefix = "SSDD_MASK_PHASE_"
    actual = []
    for line in raw.splitlines():
        if line.lstrip().startswith(b"{"):
            value = read_json(line, dict, 64 * 1024)
            if type(value.get("event")) is str and value["event"].startswith(prefix):
                actual.append(value)
    expected = [dict(event=prefix + "INITIALIZED", phase=report["phase"], initialization=report["initialization"]),
                dict(event=prefix + "TRAIN_STARTED", seed_witness=report["seed_witness"], visits_per_epoch=PADDED,
                     planned_updates=PLANNED, warmup_updates=WARMUP, updates_per_epoch=186)]
    expected.extend(dict(event=prefix + "EMPTY_FILTER_APPLIED", epoch=1, batch_idx=row["batch"] - 1,
                         image_id=row["image_id"], removed_rows=row["labels"] - row["postfilter_labels"])
                    for row in rows if row["labels"] != row["postfilter_labels"])
    expected.append(dict(event=prefix + "ARM_RETURNED", **report))
    require(packed(actual) == packed(expected), "Original initialization, train, filter and terminal events differ")


def validate_arm(arm, *, mode, protocol, source_manifest_sha256, fit_ids, fit_ids_sha256):
    require(type(arm) is QualificationArm, "Original short qualification arm required")
    validate_closure(arm)
    rows = visit_rows(arm.visit_bytes, fit_ids)
    report = read_json(arm.report_bytes, dict, 64 * 1024)
    expected = dict(status="SSDD_MASK_PHASE_ARM_CLOSED_PENDING_INDEPENDENT_ACCEPTANCE_V1",
                    completed_epochs=0, actual_trainer_epoch_cursor=1, visits=SHORT_VISITS,
                    optimizer_updates=UPDATES, ema_updates=UPDATES, planned_epochs=100,
                    planned_updates=PLANNED, warmup_updates=WARMUP, fit_images=FIT_COUNT, visits_per_epoch=PADDED,
                    removed_empty_rows=sum(row["labels"] - row["postfilter_labels"] for row in rows),
                    precision="bf16-mixed", float32_matmul_precision="high", cudnn_tf32=True,
                    FIT=True, DEV=False, TEST=False, AP=False, checkpoint_saved=False,
                    visit_ledger_sha256=digest(arm.visit_bytes), input_pair_sha256=digest(packed(ordered_inputs(rows))),
                    targets_after_filter=sum(row["postfilter_labels"] for row in rows))
    exact(report, expected, "Exact SSDD FIT-only short report, clocks and consumed ledger required",
          extra=("phase", "initialization", "seed_witness", "elapsed_seconds", "peak_allocated_bytes",
                 "peak_reserved_bytes", "adam_parameter_clocks", "native_base_lrs", "native_next_lrs",
                 "classifier_initial_sha256", "classifier_online_ending_sha256", "loss_min", "loss_max",
                 "classifier_gradient_norm_max"))
    exact(report["phase"], dict(training_protocol=protocol, stage="qualification", mask_resize_mode=mode, seed=SEED,
                                initialization_seed=SEED, num_queries=100, num_select=100, query_groups=13,
                                class_head_slots=2, coco_asset_sha256=COCO_SHA, initial_online_sha256=Q100_SHA,
                                recipe_sha256=RECIPE_SHA, fit_ids_sha256=fit_ids_sha256,
                                source_manifest_sha256=source_manifest_sha256), "Exact prospective phase pins required")
    exact(report["initialization"], dict(online_tensor_sha256=Q100_SHA, state_tensors=545, num_queries=100,
                                        num_select=100, query_groups=13, training_queries_per_image=1300,
                                        class_head_slots=2, initialization_seed=SEED, coco_asset_sha256=COCO_SHA,
                                        coco_asset_bytes=134545398, recipe_sha256=RECIPE_SHA),
          "Exact independent official COCO Q100 initialization required")
    exact(report["seed_witness"], dict(train_config_seed=SEED, torch_cpu_initial_seed=SEED,
                                      torch_cuda_initial_seed=SEED, lightning_global_seed=str(SEED),
                                      lightning_seed_workers="1"), "Exact actual runtime seed witness required")
    validate_telemetry(report)
    validate_stdout(arm.stdout_bytes, report, rows)
    return report, rows


def compare_rows(native, center):
    require(ordered_inputs(native) == ordered_inputs(center), "Ordered image IDs, normalized pixels and CPU RNG differ")
    fields = dict(differing_prefilter_count_visits="labels", differing_prefilter_labels_visits="labels_sha256",
                  differing_prefilter_mask_visits="masks_sha256", differing_prefilter_box_visits="boxes_sha256",
                  differing_postfilter_count_visits="postfilter_labels",
                  differing_postfilter_mask_visits="postfilter_masks_sha256",
                  differing_postfilter_box_visits="postfilter_boxes_sha256")
    pairs = list(zip(native, center, strict=True))
    changes = {key: sum(a[field] != b[field] for a, b in pairs) for key, field in fields.items()}
    changes["same_count_mask_only_visits"] = sum(
        0 < a["labels"] == a["postfilter_labels"] == b["labels"] == b["postfilter_labels"]
        and a["labels_sha256"] == b["labels_sha256"] and a["boxes_sha256"] == b["boxes_sha256"]
        and a["postfilter_masks_sha256"] != b["postfilter_masks_sha256"] for a, b in pairs)
    require(changes["same_count_mask_only_visits"] > 0,
            "At least one observed same-count, unfiltered mask-only visit required")
    return changes


def arm_reference(arm, report):
    return dict(report_sha256=digest(arm.report_bytes), visit_ledger_sha256=digest(arm.visit_bytes),
                stdout_sha256=arm.closure["stdout_sha256"], terminal_sha256=arm.closure["terminal_sha256"],
                original_os_wait=True, original_os_exit=0, resource_stop=False, worker_closed=True,
                removed_empty_rows=report["removed_empty_rows"], targets_after_filter=report["targets_after_filter"],
                classifier_online_ending_sha256=report["classifier_online_ending_sha256"],
                adam_parameter_clocks=report["adam_parameter_clocks"])


def verify_pair(native, center, *, protocol, source_manifest_sha256, expected_fit_ids):
    """Verify pure byte witnesses; the owner independently proves the process closure and file census."""
    require(is_hex(protocol, 40) and is_hex(source_manifest_sha256), "Independent source and protocol pins required")
    fit_ids = tuple(expected_fit_ids)
    require(len(fit_ids) == FIT_COUNT and all(type(value) is int and value >= 0 for value in fit_ids)
            and len(set(fit_ids)) == FIT_COUNT, "Expected the independently admitted FIT741 image roster")
    scope = dict(protocol=protocol, source_manifest_sha256=source_manifest_sha256, fit_ids=set(fit_ids),
                 fit_ids_sha256=digest(packed(sorted(fit_ids))))
    native_report, native_rows = validate_arm(native, mode="native", **scope)
    center_report, center_rows = validate_arm(center, mode="center", **scope)
    require(native.closure["terminal_sha256"] != center.closure["terminal_sha256"],
            "Two independent original terminal records required")
    for field in ("classifier_initial_sha256", "native_base_lrs", "native_next_lrs"):
        require(packed(native_report[field]) == packed(center_report[field]), "Paired initialization or LR path differs")
    changes = compare_rows(native_rows, center_rows)
    return dict(status=STATUS, training_protocol=protocol, source_manifest_sha256=source_manifest_sha256,
                stage="qualification", seed=SEED, fit_images=FIT_COUNT, fit_ids_sha256=scope["fit_ids_sha256"],
                recipe_sha256=RECIPE_SHA, initial_online_sha256=Q100_SHA, coco_asset_sha256=COCO_SHA,
                classifier_initial_sha256=native_report["classifier_initial_sha256"],
                visits_per_arm=SHORT_VISITS, optimizer_updates_per_arm=UPDATES, ema_updates_per_arm=UPDATES,
                ordered_inputs_sha256=native_report["input_pair_sha256"], identical_image_order_pixels_and_cpu_rng=True,
                completed_epochs_per_arm=0, checkpoint_saved=False, FIT=True, DEV=False, TEST=False, AP=False,
                changes=changes, arms=dict(native=arm_reference(native, native_report),
                                           center=arm_reference(center, center_report)))
