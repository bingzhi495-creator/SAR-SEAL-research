"""Independently admit the original closed SSDD phase E24 arms and paired FIT visits."""

from collections import Counter
import math
from pathlib import Path
from typing import NamedTuple

from ssdd_phase_qualification import (
    COCO_SHA, FIT_COUNT, PADDED, PLANNED, Q100_SHA, RECIPE_SHA, SEED, WARMUP, STATUS as SHORT_STATUS,
    digest, encode_receipt, exact, finite, is_hex, packed, read_json, require,
)


ARM_STATUS = "SSDD_MASK_PHASE_E24_ARM_INDEPENDENTLY_ACCEPTED_V1"
PAIR_STATUS = "SSDD_MASK_PHASE_E24_PAIR_INDEPENDENTLY_ACCEPTED_V1"
REPORT_NAME = "SSDD_PHASE_TRAIN_SAFE.json"
EPOCH_NAME = "EPOCH_AUDIT_SAFE.jsonl"
VISIT_NAME = "FIT_VISITS_PRIVATE.jsonl"
CHECKPOINT_NAME = "checkpoint_23.ckpt"
CHECKPOINT_MARKER = "sar_seal_ssdd_mask_resize_phase_v1"
EPOCHS = 24
STEPS_PER_EPOCH = 186
VISITS = EPOCHS * PADDED
UPDATES = EPOCHS * STEPS_PER_EPOCH
MAX_ARCHIVE_BYTES = 1024**3
# CPU torch 2.10.0 stock GradAccumAlignedDataset uses Generator().manual_seed(0).
PAD_INDICES = (224, 327, 209)
CLOCKS = ("adam_parameter_clocks", "native_base_lrs", "native_next_lrs")
BOUNDARY = dict(FIT=True, DEV=False, TEST=False, AP=False)
CLOSURE = dict(original_os_wait=True, original_os_exit=0, resource_stop=False, worker_closed=True)


class ClosedE24Arm(NamedTuple):
    closure: dict
    report_bytes: bytes
    epoch_bytes: bytes
    visit_bytes: bytes
    stdout_bytes: bytes


class ArchiveEvidence(NamedTuple):
    phase: dict
    native_report: dict


def verify_scope(protocol, source_manifest_sha256, expected_fit_ids, qualification_bytes, qualification_sha256):
    require(is_hex(protocol, 40) and is_hex(source_manifest_sha256) and is_hex(qualification_sha256)
            and type(qualification_bytes) is bytes and digest(qualification_bytes) == qualification_sha256,
            "Exact prospective training and accepted short qualification SHA required")
    fit_ids = tuple(expected_fit_ids)
    require(len(fit_ids) == FIT_COUNT and all(type(value) is int and value >= 0 for value in fit_ids)
            and len(set(fit_ids)) == FIT_COUNT, "Exact independent SSDD FIT741 roster required")
    ordered = sorted(fit_ids)
    population = Counter(ordered)
    population.update(ordered[index] for index in PAD_INDICES)
    qualification = read_json(qualification_bytes, dict, 64 * 1024)
    scope = dict(training_protocol=protocol, source_manifest_sha256=source_manifest_sha256, seed=SEED,
                 fit_ids_sha256=digest(packed(ordered)), initial_online_sha256=Q100_SHA,
                 coco_asset_sha256=COCO_SHA, recipe_sha256=RECIPE_SHA)
    exact(qualification, dict(status=SHORT_STATUS, **scope, stage="qualification", fit_images=FIT_COUNT,
                              visits_per_arm=32, optimizer_updates_per_arm=8, ema_updates_per_arm=8,
                              identical_image_order_pixels_and_cpu_rng=True, completed_epochs_per_arm=0,
                              checkpoint_saved=False, **BOUNDARY), "Exact same-scope independently qualified pair required",
          extra=("classifier_initial_sha256", "ordered_inputs_sha256", "changes", "arms"))
    changes, arms = qualification["changes"], qualification["arms"]
    require(encode_receipt(qualification) == qualification_bytes and is_hex(qualification["classifier_initial_sha256"])
            and is_hex(qualification["ordered_inputs_sha256"]) and type(changes) is dict
            and type(changes.get("same_count_mask_only_visits")) is int and 0 < changes["same_count_mask_only_visits"] <= 32
            and type(arms) is dict and set(arms) == {"native", "center"}, "Canonical qualified mask-only pair required")
    for arm in arms.values():
        require(type(arm) is dict and all(type(arm.get(key)) is type(value) and arm[key] == value
                                         for key, value in CLOSURE.items())
                and all(is_hex(arm.get(key)) for key in ("report_sha256", "visit_ledger_sha256", "stdout_sha256",
                                                        "terminal_sha256")), "Independent short arm closure required")
    require(arms["native"]["terminal_sha256"] != arms["center"]["terminal_sha256"],
            "Distinct original short arm terminal records required")
    return scope, population, qualification


def expected_phase(scope, mode):
    require(mode in ("native", "center"), "Expected native or center SSDD arm")
    return dict(**scope, stage="e24", mask_resize_mode=mode, initialization_seed=SEED,
                num_queries=100, num_select=100, query_groups=13, class_head_slots=2)


def read_jsonl(raw, limit, message):
    require(type(raw) is bytes and 0 < len(raw) <= limit, message)
    lines = raw.splitlines(keepends=True)
    require(len(lines) == EPOCHS and all(line.endswith(b"\n") for line in lines), message)
    return [(line, read_json(line, dict, 128 * 1024)) for line in lines]


def read_visit_epochs(raw, population):
    records = []
    for epoch, (line, row) in enumerate(read_jsonl(raw, 2 * 1024**2, "Exactly 24 complete FIT visit rows required"), 1):
        exact(row, dict(epoch=epoch), "Exact original FIT visit epoch cursor required", extra=("visits",))
        visits = row["visits"]
        require(type(visits) is list and len(visits) == PADDED and all(
            type(visit) is list and len(visit) == 2 and type(visit[0]) is type(visit[1]) is int
            and visit[0] >= 0 and 0 <= visit[1] <= 100 for visit in visits), "Invalid original FIT visit ID or target count")
        require(Counter(image_id for image_id, _ in visits) == population,
                "Every epoch must match exact FIT741 and three original alignment identities")
        records.append((line, visits))
    return records


def lr_factor(step):
    if step < WARMUP:
        return step / WARMUP
    progress = (step - WARMUP) / (PLANNED - WARMUP)
    return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress))


def verify_clocks(audit, step, original_base):
    adam, base, current = (audit[key] for key in CLOCKS)
    require(type(adam) is dict and str(step) in adam and all(
        type(key) is str and key.isascii() and key.isdecimal() and len(key) <= len(str(step))
        and 1 <= int(key) <= step and key == str(int(key)) and type(count) is int and 0 < count <= 545
        for key, count in adam.items()) and sum(adam.values()) <= 545, "Native AdamW state does not reach epoch cursor")
    factor = lr_factor(step)
    require(type(base) is type(current) is list and len(base) == len(current) > 0
            and packed(base) == packed(original_base) and all(finite(a) and finite(b) and a >= 0
                and math.isclose(b, a * factor, rel_tol=1e-9, abs_tol=1e-12)
                for a, b in zip(base, current, strict=True)), "Native E100 learning rate path differs")


def validate_epoch_audits(raw, visits):
    rows = read_jsonl(raw, 1024**2, "Exactly 24 complete epoch audits required")
    base = rows[0][1].get("native_base_lrs")
    audits = []
    for epoch, ((_, audit), (line, observed)) in enumerate(zip(rows, visits, strict=True), 1):
        counts = [count for _, count in observed]
        step = epoch * STEPS_PER_EPOCH
        exact(audit, dict(epoch=epoch, visits=PADDED, unique_fit_images=FIT_COUNT, duplicate_fit_visits=PADDED - FIT_COUNT,
                          zero_target_visits=counts.count(0), total_augmented_targets=sum(counts),
                          maximum_augmented_targets=max(counts), optimizer_updates=step, ema_updates=step,
                          visit_row_sha256=digest(line)), "Epoch audit must agree with exact original visits and updates",
              extra=("discarded_empty_mask_rows", "loss_min", "loss_max", "classifier_gradient_norm_max",
                     "epoch_seconds", *CLOCKS))
        removed = audit["discarded_empty_mask_rows"]
        loss_min, loss_max, gradient, elapsed = (audit[key] for key in
            ("loss_min", "loss_max", "classifier_gradient_norm_max", "epoch_seconds"))
        require(type(removed) is int and 0 <= removed <= 100 * PADDED - sum(counts)
                and all(finite(value) for value in (loss_min, loss_max, gradient, elapsed))
                and 0 <= loss_min <= loss_max and gradient > 0 and elapsed > 0,
                "Invalid original epoch filter, loss, gradient or duration accounting")
        verify_clocks(audit, step, base)
        audits.append(audit)
    return audits


def validate_initialization(witness):
    exact(witness, dict(online_tensor_sha256=Q100_SHA, state_tensors=545, num_queries=100, num_select=100,
                        query_groups=13, training_queries_per_image=1300, class_head_slots=2,
                        initialization_seed=SEED, coco_asset_sha256=COCO_SHA,
                        coco_asset_bytes=134545398, recipe_sha256=RECIPE_SHA),
          "Exact official COCO-only Q100 start witness required")


def validate_report(raw, phase, epoch_bytes, visit_bytes, audits):
    report = read_json(raw, dict, 64 * 1024)
    exact(report, dict(status="SSDD_MASK_PHASE_ARM_CLOSED_PENDING_INDEPENDENT_ACCEPTANCE_V1",
                       completed_epochs=EPOCHS, actual_trainer_epoch_cursor=EPOCHS, visits=VISITS,
                       optimizer_updates=UPDATES, ema_updates=UPDATES, planned_epochs=100, planned_updates=PLANNED,
                       warmup_updates=WARMUP, fit_images=FIT_COUNT, visits_per_epoch=PADDED,
                       removed_empty_rows=sum(audit["discarded_empty_mask_rows"] for audit in audits),
                       precision="bf16-mixed", float32_matmul_precision="high", cudnn_tf32=True, **BOUNDARY,
                       fit_visit_ledger_sha256=digest(visit_bytes), epoch_audit_sha256=digest(epoch_bytes), checkpoint_saved=True),
          "Exact original SSDD E24 report, FIT scope and ledger SHA values required",
          extra=("phase", "initialization", "seed_witness", "elapsed_seconds", "peak_allocated_bytes",
                 "peak_reserved_bytes", "archive", *CLOCKS))
    exact(report["phase"], phase, "Exact prospective own-arm SSDD E24 phase marker required")
    validate_initialization(report["initialization"])
    exact(report["seed_witness"], dict(train_config_seed=SEED, torch_cpu_initial_seed=SEED,
                                      torch_cuda_initial_seed=SEED, lightning_global_seed=str(SEED),
                                      lightning_seed_workers="1"), "Exact effective runtime seed required")
    allocated, reserved, elapsed = (report[key] for key in ("peak_allocated_bytes", "peak_reserved_bytes", "elapsed_seconds"))
    require(type(allocated) is type(reserved) is int and 0 < allocated <= reserved and finite(elapsed) and elapsed > 0,
            "Invalid SSDD GPU memory or duration accounting")
    require(all(packed(report[key]) == packed(audits[-1][key]) for key in CLOCKS),
            "Final native optimizer and schedule observations must match E24 audit")
    return report


def validate_events(raw, report, visits, audits):
    prefix = "SSDD_MASK_PHASE_"
    actual = []
    for line in raw.splitlines():
        if line.lstrip().startswith(b"{"):
            row = read_json(line, dict, 64 * 1024)
            if type(row.get("event")) is str and row["event"].startswith(prefix):
                actual.append(row)
    filters = {}
    for row in actual:
        if row["event"] != prefix + "EMPTY_FILTER_APPLIED":
            continue
        exact(row, dict(event=prefix + "EMPTY_FILTER_APPLIED"), "Invalid original SSDD empty-filter event fields",
              extra=("epoch", "batch_idx", "image_id", "removed_rows"))
        epoch, batch, image_id, removed = (row[key] for key in ("epoch", "batch_idx", "image_id", "removed_rows"))
        require(all(type(value) is int for value in (epoch, batch, image_id, removed))
                and 1 <= epoch <= EPOCHS and 0 <= batch < PADDED and (epoch, batch) not in filters,
                "Invalid or repeated original SSDD empty-filter event cursor")
        original_id, retained = visits[epoch - 1][1][batch]
        require(image_id == original_id and 0 < removed <= 100 - retained,
                "Original filter event must match its exact visit and Q100 target count")
        filters[(epoch, batch)] = row
    expected = [dict(event=prefix + "INITIALIZED", phase=report["phase"], initialization=report["initialization"]),
                dict(event=prefix + "TRAIN_STARTED", seed_witness=report["seed_witness"], visits_per_epoch=PADDED,
                     planned_updates=PLANNED, warmup_updates=WARMUP, updates_per_epoch=STEPS_PER_EPOCH)]
    for epoch, audit in enumerate(audits, 1):
        removed = 0
        for batch in range(PADDED):
            row = filters.get((epoch, batch))
            if row is not None:
                expected.append(row)
                removed += row["removed_rows"]
            step = (epoch - 1) * STEPS_PER_EPOCH + (batch + 1) // 4
            if (batch + 1) % 4 == 0 and step % 100 == 0:
                expected.append(dict(event=prefix + "PROGRESS", epoch=epoch, visits_in_epoch=batch + 1, optimizer_updates=step))
        require(removed == audit["discarded_empty_mask_rows"], "Filter log and independently reconciled epoch totals differ")
        expected.append(dict(event=prefix + "EPOCH_OBSERVED", **audit))
    expected.append(dict(event=prefix + "ARM_RETURNED", **report))
    require(packed(actual) == packed(expected), "Original start, progress, epoch and closure events differ")


def validate_archive(archive, expected_path, phase, checkpoint_reader):
    exact(archive, dict(weights_only_cpu_readback=True), "Exact original native CPU archive report required",
          extra=("checkpoint_ref", "classifier_raw_max_abs_change", "classifier_raw_ema_max_abs_difference"))
    movement, distance = (archive[key] for key in ("classifier_raw_max_abs_change", "classifier_raw_ema_max_abs_difference"))
    require(finite(movement) and movement > 0 and finite(distance) and distance >= 0,
            "Native raw classifier movement and EMA distance required")
    ref = archive["checkpoint_ref"]
    exact(ref, {}, "Exact original checkpoint reference required", extra=("path", "bytes", "sha256"))
    path = Path(expected_path)
    require(path.is_absolute() and ".." not in path.parts and path.name == CHECKPOINT_NAME and type(ref["path"]) is str
            and ref["path"] == path.as_posix() and type(ref["bytes"]) is int and 0 < ref["bytes"] <= MAX_ARCHIVE_BYTES
            and is_hex(ref["sha256"]), "Exact original E24 checkpoint path, size and SHA required")
    require(callable(checkpoint_reader), "Independent native weights-only CPU checkpoint reader required")
    evidence = checkpoint_reader(ref.copy(), phase.copy())
    require(type(evidence) is ArchiveEvidence and packed(evidence.native_report) == packed(archive),
            "Independent original native archive readback differs")
    exact(evidence.phase, phase, "Independent checkpoint top-level phase marker differs")
    return ref


def make_checkpoint_reader(*, torch, native_inspector, e1_api, boundary_api,
                           initial_classifier, initialization, classifier_initial_sha256):
    """Wrap a pinned native inspector around one stable CPU weights-only deserialization."""
    validate_initialization(initialization)
    require(callable(native_inspector) and is_hex(classifier_initial_sha256)
            and initial_classifier.dtype == torch.float32 and initial_classifier.device.type == "cpu"
            and len(initial_classifier.shape) == 2 and initial_classifier.shape[0] == 2,
            "Expected the independently qualified initial classifier on CPU")
    classifier = initial_classifier.detach().contiguous().clone()
    header = repr((str(classifier.dtype), tuple(classifier.shape))).encode()
    require(digest(header + classifier.numpy().tobytes(order="C")) == classifier_initial_sha256,
            "CPU tensor must match the independently qualified initial classifier")

    def read(ref, phase):
        require(type(phase) is dict and phase.get("stage") == "e24" and phase.get("seed") == SEED
                and phase.get("initialization_seed") == SEED and phase.get("initial_online_sha256") == Q100_SHA
                and phase.get("recipe_sha256") == RECIPE_SHA, "Expected fixed SSDD E24 phase and Q100 seed")
        observed = None

        class CPUWeightsOnly:
            def __init__(self):
                self.float32 = torch.float32
                self.isfinite = torch.isfinite

            def load(self, stream, *, map_location, weights_only):
                nonlocal observed
                require(observed is None and map_location == "cpu" and weights_only is True,
                        "Exactly one native CPU weights-only archive deserialization required")
                checkpoint = torch.load(stream, map_location=map_location, weights_only=weights_only)
                marker = checkpoint.get(CHECKPOINT_MARKER) if type(checkpoint) is dict else None
                exact(marker, phase, "Checkpoint must carry the exact E24 phase marker at the top level")
                observed = marker.copy()
                return checkpoint

        native = native_inspector(Path(ref["path"]), epoch=EPOCHS, torch=CPUWeightsOnly(), e1_api=e1_api,
                                  boundary_api=boundary_api, recipe_sha256=RECIPE_SHA, initial_witness=classifier)
        require(observed is not None, "Native inspector must use the admitted CPU weights-only archive load")
        return ArchiveEvidence(observed, native)

    return read


def validate_arm(arm, *, mode, expected_checkpoint_path, checkpoint_reader, protocol, source_manifest_sha256,
                 expected_fit_ids, qualification_bytes, qualification_sha256):
    """Return canonical arm receipt bytes after independent control and native archive proof."""
    scope, population, short = verify_scope(protocol, source_manifest_sha256, expected_fit_ids,
                                            qualification_bytes, qualification_sha256)
    phase = expected_phase(scope, mode)
    require(type(arm) is ClosedE24Arm, "Expected an original closed SSDD E24 arm")
    closure = arm.closure
    exact(closure, CLOSURE, "Successful independent original E24 worker closure required",
          extra=("terminal_sha256", "stdout_sha256", "output_files"))
    names = closure["output_files"]
    require(type(names) is list and len(names) == 4 and all(type(name) is str for name in names)
            and set(names) == {REPORT_NAME, EPOCH_NAME, VISIT_NAME, CHECKPOINT_NAME}, "Exact fixed E24 arm file census required")
    require(is_hex(closure["terminal_sha256"]) and is_hex(closure["stdout_sha256"])
            and type(arm.stdout_bytes) is bytes and 0 < len(arm.stdout_bytes) <= 8 * 1024**2
            and digest(arm.stdout_bytes) == closure["stdout_sha256"], "Exact independent original E24 stdout required")
    require(all(closure[key] != short["arms"][mode][key] for key in ("stdout_sha256", "terminal_sha256")),
            "Long arm cannot reuse original short arm controls")
    visits = read_visit_epochs(arm.visit_bytes, population)
    audits = validate_epoch_audits(arm.epoch_bytes, visits)
    report = validate_report(arm.report_bytes, phase, arm.epoch_bytes, arm.visit_bytes, audits)
    validate_events(arm.stdout_bytes, report, visits, audits)
    ref = validate_archive(report["archive"], expected_checkpoint_path, phase, checkpoint_reader)
    receipt = dict(status=ARM_STATUS, phase=phase, qualification_pair_sha256=qualification_sha256, **CLOSURE,
                   completed_epochs=EPOCHS, visits=VISITS, optimizer_updates=UPDATES, ema_updates=UPDATES,
                   checkpoint_sha256=ref["sha256"], checkpoint_bytes=ref["bytes"], result_sha256=digest(arm.report_bytes),
                   epoch_audit_sha256=digest(arm.epoch_bytes), visit_ledger_sha256=digest(arm.visit_bytes),
                   stdout_sha256=closure["stdout_sha256"], terminal_sha256=closure["terminal_sha256"],
                   filtered_empty_rows_total=report["removed_empty_rows"], **BOUNDARY)
    raw = encode_receipt(receipt)
    return raw, digest(raw)


def read_accepted_arm(raw, visits, *, mode, scope, qualification_sha256):
    arm = read_json(raw, dict, 64 * 1024)
    fields = ("checkpoint_sha256", "result_sha256", "epoch_audit_sha256", "stdout_sha256", "terminal_sha256")
    exact(arm, dict(status=ARM_STATUS, qualification_pair_sha256=qualification_sha256, **CLOSURE,
                    completed_epochs=EPOCHS, visits=VISITS, optimizer_updates=UPDATES, ema_updates=UPDATES,
                    visit_ledger_sha256=digest(visits), **BOUNDARY), "Exact independently accepted E24 arm and ledger required",
          extra=("phase", "checkpoint_bytes", "filtered_empty_rows_total", *fields))
    exact(arm["phase"], expected_phase(scope, mode), "Accepted E24 arm scope or mode differs")
    size, filtered = (arm[key] for key in ("checkpoint_bytes", "filtered_empty_rows_total"))
    require(encode_receipt(arm) == raw and all(is_hex(arm[key]) for key in fields)
            and type(size) is int and 0 < size <= MAX_ARCHIVE_BYTES and type(filtered) is int
            and 0 <= filtered <= 100 * VISITS, "Invalid accepted E24 arm digest, size or filter accounting")
    return arm


def compare_ordered_visits(native, center):
    order, different, native_total, center_total, native_zero, center_zero = [], 0, 0, 0, 0, 0
    for (_, left), (_, right) in zip(native, center, strict=True):
        epoch_ids = []
        for (left_id, left_count), (right_id, right_count) in zip(left, right, strict=True):
            require(left_id == right_id, "Ordered SSDD FIT image identities differ")
            epoch_ids.append(left_id)
            different += left_count != right_count
            native_total += left_count
            center_total += right_count
            native_zero += left_count == 0
            center_zero += right_count == 0
        order.append(epoch_ids)
    return dict(ordered_fit_visits_sha256=digest(packed(order)), differing_target_count_visits=different,
                native_retained_targets=native_total, center_retained_targets=center_total,
                native_zero_target_visits=native_zero, center_zero_target_visits=center_zero)


def create_pair_receipt(native_arm_bytes, center_arm_bytes, native_visit_bytes, center_visit_bytes, *,
                        protocol, source_manifest_sha256, expected_fit_ids, qualification_bytes, qualification_sha256):
    """Reconcile two independently accepted arm receipts without reopening their archives."""
    scope, population, _ = verify_scope(protocol, source_manifest_sha256, expected_fit_ids,
                                        qualification_bytes, qualification_sha256)
    native = read_accepted_arm(native_arm_bytes, native_visit_bytes, mode="native", scope=scope,
                               qualification_sha256=qualification_sha256)
    center = read_accepted_arm(center_arm_bytes, center_visit_bytes, mode="center", scope=scope,
                               qualification_sha256=qualification_sha256)
    require(all(native[key] != center[key] for key in ("checkpoint_sha256", "result_sha256", "stdout_sha256", "terminal_sha256")),
            "Distinct original closed E24 arm receipts and checkpoints required")
    stats = compare_ordered_visits(read_visit_epochs(native_visit_bytes, population),
                                   read_visit_epochs(center_visit_bytes, population))
    receipt = dict(status=PAIR_STATUS, **scope, qualification_pair_sha256=qualification_sha256,
                   completed_epochs=EPOCHS, visits_per_arm=VISITS, optimizer_updates_per_arm=UPDATES,
                   ema_updates_per_arm=UPDATES, ordered_fit_visits_match=True,
                   ordered_fit_visits_sha256=stats["ordered_fit_visits_sha256"],
                   native_arm_receipt_sha256=digest(native_arm_bytes), center_arm_receipt_sha256=digest(center_arm_bytes),
                   native_checkpoint_sha256=native["checkpoint_sha256"], center_checkpoint_sha256=center["checkpoint_sha256"],
                   **BOUNDARY)
    raw = encode_receipt(receipt)
    return raw, digest(raw), stats
