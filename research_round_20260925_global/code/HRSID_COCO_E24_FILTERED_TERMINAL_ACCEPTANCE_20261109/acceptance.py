"""Independent validation of the closed filtered HRSID E24 training endpoint."""

from collections import Counter
from hashlib import sha256
import json
import math
import re


PROTOCOL = "cff45e6e91975e571278a0cf689f180e6b2a652e"
FILTER_SHA = "3b164cbf8c104539eabee8775c8e1f297d9a1607e37473c4d7a4a9b561f93471"
FIT_INDEX_SHA = "e95cf26cc60a2c6fb651e907541380a256f2df7b1e153244a6c0f803b0b19f8f"
COCO_SHA = "a44613a4ecd6b5ba61a62002c600b0b6cb7a9da2936a45317ec4b62c635fb99b"
INITIAL_SHA = "63ff93afdd977ba79b5b5a46ba8b4839ef6c40eb0722b29e4667d4932404fbc2"
RECIPE_SHA = "ba9508d9acab5ff68b10db8a1aff5d9001868d6dda00e39f299d6f5cc1b8011b"
MARKER = "sar_seal_hrsid_coco_fit_native_endpoint_v1"
EPOCHS, ENDPOINTS = 24, (1, 8, 24)
FIT, VISITS, UPDATES = 2937, 2940, 735
WARMUP, SCHEDULE, WIDTH, GROUPS, SLOTS = 2205, 73500, 164, 28, 543
CAP = 6 * 1024**3


def require(condition, message):
    if not condition:
        raise ValueError(message)


def is_sha256(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def exact_fields(value, expected, message):
    require(type(value) is dict and all(
        type(value.get(key)) is type(item) and value[key] == item for key, item in expected.items()
    ), message)


def integer(value, low, high):
    return type(value) is int and low <= value <= high


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def json_object(raw, label):
    def unique(fields):
        out = dict(fields)
        require(len(out) == len(fields), "Duplicate JSON key")
        return out

    def invalid_constant(_):
        raise ValueError("Nonfinite JSON constant")

    try:
        value = json.loads(raw, object_pairs_hook=unique, parse_constant=invalid_constant)
    except (ValueError, UnicodeError) as exc:
        raise ValueError("Invalid " + label) from exc
    require(type(value) is dict, "Expected JSON object for " + label)
    return value


def validate_closure(closure, stdout_bytes):
    exact_fields(closure, dict(original_os_wait=True, original_os_exit=0, resource_stop=False,
                               owner_closed=True, worker_closed=True),
                 "Verified closed original owner and worker required")
    require(is_sha256(closure.get("terminal_sha256"))
            and closure.get("stdout_sha256") == sha256(stdout_bytes).hexdigest(),
            "Verified original terminal and stdout digests required")


def schedule_factor(step):
    if step < WARMUP:
        return step / WARMUP
    return .05 + .95 * .5 * (1 + math.cos(math.pi * (step - WARMUP) / (SCHEDULE - WARMUP)))


def check_lrs(base, next_lrs, step):
    require(type(base) is list and type(next_lrs) is list and len(base) == len(next_lrs) == GROUPS
            and all(finite(a) and finite(b) and a >= 0 and math.isclose(a * schedule_factor(step), b,
                    rel_tol=1e-9, abs_tol=1e-12) for a, b in zip(base, next_lrs, strict=True)),
            "Native E100 scheduler groups and learning rates")


def validate_result(result_bytes, epoch_audit_bytes, fit_visit_bytes):
    """Validate fixed training configuration and the three declared archive references."""
    result = json_object(result_bytes, "training result")
    exact_fields(result, dict(
        status="NATIVE_HRSID_COCO_FIT_E24_COMPLETE_PENDING_INDEPENDENT_ACCEPTANCE",
        protocol=PROTOCOL, native_index_sha256=FIT_INDEX_SHA, original_recipe_sha256=RECIPE_SHA,
        visited_fit_images=FIT, total_visits=VISITS * EPOCHS, total_updates=UPDATES * EPOCHS,
        starting_optimizer_updates=0, starting_ema_updates=0, ema_updates=UPDATES * EPOCHS,
        completed_epochs=EPOCHS, scheduled_epochs=100, scheduled_updates=SCHEDULE,
        warmup_updates=WARMUP, precision="bf16-mixed", batch_size=1, grad_accumulation=4,
        gpu_allocator_limit_bytes=CAP, FIT_only=True, DEV=False, TEST=False, AP=False,
        exact_restart_trajectory_verified=False, target_filter_sha256=FILTER_SHA,
        coco_asset_sha256=COCO_SHA, coco_asset_bytes=134545398, native_original_width=100,
        expanded_online_sha256=INITIAL_SHA, query_width=WIDTH, query_groups=13,
        class_head_slots=2, initialized_from_SSDD=False,
        epoch_audit_sha256=sha256(epoch_audit_bytes).hexdigest(),
        fit_visit_ledger_sha256=sha256(fit_visit_bytes).hexdigest(),
    ), "Exact closed FIT-only training result and ledger digests")
    peak = result.get("gpu_peak_allocated_bytes")
    reserved = result.get("gpu_peak_reserved_bytes")
    elapsed = result.get("elapsed_seconds")
    require(is_sha256(result.get("width100_loaded_sha256")) and finite(elapsed) and elapsed > 0
            and integer(peak, 1, CAP) and integer(reserved, peak, CAP)
            and integer(result.get("filtered_empty_rows_total"), 1, WIDTH * VISITS * EPOCHS),
            "Bounded closed training result")
    refs = result.get("archive_refs")
    require(type(refs) is list and len(refs) == len(ENDPOINTS), "Three fixed endpoint archives required")
    for epoch, ref in zip(ENDPOINTS, refs, strict=True):
        step = epoch * UPDATES
        exact_fields(ref, dict(epoch=epoch, archive_name=f"checkpoint_{epoch - 1}.ckpt", rng_saved=True,
                               adam_parameter_clocks={str(step): SLOTS}), "Fixed native archive reference")
        require(all(is_sha256(ref.get(key)) for key in
                    ("archive_sha256", "ending_online_sha256", "ending_ema_sha256"))
                and ref["ending_online_sha256"] != ref["ending_ema_sha256"]
                and integer(ref.get("archive_bytes"), 1, 1024**3), "Bounded native archive and distinct state digests")
        check_lrs(ref.get("native_base_lrs"), ref.get("native_next_lrs"), step)
    require(len({ref["archive_sha256"] for ref in refs}) == len(ENDPOINTS), "Endpoint archive identities differ")
    return result


def validate_epochs(epoch_audit_bytes, fit_visit_bytes, stdout_bytes, result, expected_fit_ids):
    """Read back all FIT visits and match original log events to the closed ledgers."""
    fit_ids = tuple(expected_fit_ids)
    require(len(fit_ids) == FIT and all(type(iid) is int for iid in fit_ids)
            and len(set(fit_ids)) == FIT, "Exact previously qualified FIT image roster")
    fit_ids = set(fit_ids)
    audit_lines = epoch_audit_bytes.splitlines(keepends=True)
    visit_lines = fit_visit_bytes.splitlines(keepends=True)
    require(len(audit_lines) == len(visit_lines) == EPOCHS
            and all(line.endswith(b"\n") for line in audit_lines + visit_lines),
            "Exactly 24 complete epoch and visit ledger rows")
    stdout_lines = stdout_bytes.rstrip().splitlines()
    require(bool(stdout_lines) and json_object(stdout_lines[-1], "final original stdout") == result,
            "Original worker final stdout agrees with training result")
    audit_events = []
    filter_events = {}
    for line in stdout_lines[:-1]:
        if not line.startswith(b"{"):
            continue
        event = json_object(line, "original stdout event")
        name = event.pop("event", None)
        if name == "HRSID_COCO_NATIVE_EPOCH_OBSERVED":
            audit_events.append(event)
        if name == "HRSID_NATIVE_EMPTY_MASK_FILTER_APPLIED":
            epoch, batch = event.get("epoch"), event.get("batch_idx")
            require(integer(epoch, 1, EPOCHS) and integer(batch, 0, VISITS - 1)
                    and type(event.get("image_id")) is int
                    and integer(event.get("removed_rows"), 1, WIDTH)
                    and (epoch, batch) not in filter_events, "Valid distinct original empty-mask filter events")
            filter_events[epoch, batch] = event
        require(name != "HRSID_NATIVE_HARD_TARGET_FAILURE", "No original hard target failure")
    require(len(audit_events) == EPOCHS, "Original worker logged exactly 24 epoch audits")
    population = None
    cumulative = 0
    endpoints = {}
    for epoch, audit_line, visit_line, observed in zip(
            range(1, EPOCHS + 1), audit_lines, visit_lines, audit_events, strict=True):
        audit = json_object(audit_line, "epoch audit")
        ledger = json_object(visit_line, "FIT visit row")
        step = epoch * UPDATES
        exact_fields(audit, dict(epoch=epoch, visits=VISITS, unique_fit_images=FIT,
                                 duplicate_fit_visits=VISITS - FIT, optimizer_updates=step, ema_updates=step,
                                 adam_parameter_clocks={str(step): SLOTS},
                                 visit_row_sha256=sha256(visit_line).hexdigest()), "Exact FIT epoch counts and visit SHA")
        require(audit == observed, "Closed epoch audit equals its original stdout event")
        exact_fields(ledger, {"epoch": epoch}, "Exact visit ledger epoch")
        visits = ledger.get("visits")
        require(type(visits) is list and len(visits) == VISITS and all(
            type(row) is list and len(row) == 2 and type(row[0]) is int
            and integer(row[1], 0, WIDTH) for row in visits), "Exact native visits and bounded augmented target counts")
        current_population = Counter(iid for iid, _ in visits)
        if population is None:
            population = current_population
        require(set(current_population) == fit_ids and current_population == population,
                "Consistent native padded FIT population")
        counts = [count for _, count in visits]
        removed = 0
        for batch, (iid, retained) in enumerate(visits):
            event = filter_events.get((epoch, batch))
            if event is not None:
                require(event["image_id"] == iid and retained + event["removed_rows"] <= WIDTH,
                        "Empty-mask filter event matches the original FIT visit")
                removed += event["removed_rows"]
        exact_fields(audit, dict(zero_target_visits=counts.count(0), maximum_augmented_targets=max(counts),
                                 total_augmented_targets=sum(counts), discarded_empty_mask_rows=removed),
                     "FIT target and filtered-row accounting")
        min_loss = audit.get("lightning_accumulation_scaled_loss_min")
        mean_loss = audit.get("lightning_accumulation_scaled_loss_mean")
        max_loss = audit.get("lightning_accumulation_scaled_loss_max")
        elapsed = audit.get("epoch_seconds")
        gradient = audit.get("classifier_gradient_norm_max")
        require(all(finite(v) for v in (min_loss, mean_loss, max_loss, elapsed, gradient))
                and min_loss <= mean_loss <= max_loss and elapsed > 0 and gradient > 0
                and integer(audit.get("peak_allocated_bytes"), 1, result["gpu_peak_allocated_bytes"]),
                "Finite FIT loss and classifier gradient records")
        check_lrs(audit.get("native_base_lrs"), audit.get("native_next_lrs"), step)
        cumulative += removed
        if epoch in ENDPOINTS:
            endpoints[epoch] = (cumulative, audit)
    require(cumulative == result["filtered_empty_rows_total"], "Cumulative repaired filter row count")
    return endpoints


def validate_closed_run(closure, result_bytes, epoch_audit_bytes, fit_visit_bytes,
                        stdout_bytes, checkpoint_reader, expected_fit_ids):
    """Return the exact DEV705 bridge receipt bytes and their SHA-256.

    The caller independently verifies the original terminal and normalizes its
    closure evidence before reading the closed result and ledgers. The reader
    receives each validated archive reference and returns the CPU checkpoint,
    observed byte count and independently measured SHA-256.
    """
    validate_closure(closure, stdout_bytes)
    result = validate_result(result_bytes, epoch_audit_bytes, fit_visit_bytes)
    endpoints = validate_epochs(epoch_audit_bytes, fit_visit_bytes, stdout_bytes, result, expected_fit_ids)
    for ref in result["archive_refs"]:
        cumulative, audit = endpoints[ref["epoch"]]
        require(all(ref[key] == audit[key] for key in ("adam_parameter_clocks", "native_base_lrs", "native_next_lrs")),
                "Endpoint and epoch optimizer evidence agree")
        saved, size, digest = checkpoint_reader(ref)
        require(type(size) is int and size == ref["archive_bytes"] and digest == ref["archive_sha256"],
                "Independent checkpoint byte evidence agrees with original report")
        validate_checkpoint(saved, ref, cumulative)
        del saved
    return _receipt_for(result)




def tensor_digest(state, torch):
    digest = sha256()
    for name, value in state.items():
        value = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, list(value.shape), str(value.dtype)]).encode() + b"\n")
        digest.update(memoryview(value.reshape(-1).view(torch.uint8).numpy()))
    return digest.hexdigest()


def check_archive_loop(saved, epoch):
    step = epoch * UPDATES
    exact_fields(saved, {"pytorch-lightning_version": "2.6.6", "epoch": epoch - 1, "global_step": step},
                 "Exact serialized Lightning checkpoint boundary")
    loop = saved["loops"]["fit_loop"]
    batch = loop["epoch_loop.batch_progress"]
    optimizer = loop["epoch_loop.automatic_optimization.optim_progress"]["optimizer"]["step"]
    exact_fields(batch, {"is_last_batch": True}, "Complete final native training batch")
    for scope, scale in (("current", 1), ("total", epoch)):
        exact_fields(loop["epoch_progress"][scope],
                     dict(ready=epoch, started=epoch, processed=epoch, completed=epoch - 1),
                     "Exact serialized native epoch boundary")
        exact_fields(batch[scope], {field: VISITS * scale for field in ("ready", "started", "processed", "completed")},
                     "Exact serialized native batch clock")
        exact_fields(optimizer[scope], dict(ready=UPDATES * scale, completed=UPDATES * scale),
                     "Exact serialized native optimizer clock")


def check_archive_marker(saved, epoch, cumulative, torch):
    marker = saved[MARKER]
    expected = dict(protocol=PROTOCOL, epoch=epoch, actual_updates=epoch * UPDATES,
                    native_index_sha256=FIT_INDEX_SHA, coco_sha256=COCO_SHA, original_recipe_sha256=RECIPE_SHA,
                    initial_expanded_sha256=INITIAL_SHA, FIT_only=True, exact_restart_trajectory_verified=False,
                    target_filter_sha256=FILTER_SHA, filtered_empty_rows_total=cumulative)
    exact_fields(marker, expected, "Exact repaired checkpoint provenance and filter count")
    require(set(marker) == expected.keys() | {"rng_state"}, "Exact repaired checkpoint marker schema")
    rng = marker["rng_state"]
    require(type(rng) is dict and set(rng) == {"python", "numpy", "torch_cpu", "torch_cuda"}
            and type(rng["python"]) is tuple and type(rng["numpy"]) is tuple
            and type(rng["torch_cuda"]) is tuple and len(rng["torch_cuda"]) == 1
            and all(isinstance(value, torch.Tensor) and value.device.type == "cpu" and value.dtype == torch.uint8
                    and value.ndim == 1 and value.numel() > 0 for value in (rng["torch_cpu"], rng["torch_cuda"][0])),
            "Complete native RNG envelope")
    config = saved["model_config"]
    exact_fields(config, dict(resolution=576, num_queries=WIDTH, group_detr=13, num_select=100, num_classes=1,
                              pretrain_weights=None, segmentation_head=True), "Exact native saved model configuration")


def finite_tensor(value, torch):
    return (isinstance(value, torch.Tensor) and value.device.type == "cpu"
            and (not value.is_floating_point() or bool(torch.isfinite(value).all())))


def check_archive_optimizer(saved, epoch, ref, torch):
    step = epoch * UPDATES
    optimizer, = saved["optimizer_states"]
    scheduler, = saved["lr_schedulers"]
    exact_fields(scheduler, {"last_epoch": step, "base_lrs": ref["native_base_lrs"],
                             "_last_lr": ref["native_next_lrs"]}, "Exact serialized E100 schedule")
    groups = optimizer["param_groups"]
    slots = [slot for group in groups for slot in group["params"]]
    states = optimizer["state"]
    require(len(groups) == GROUPS and len(slots) == len(set(slots))
            and len(states) == SLOTS and set(states) <= set(slots)
            and [group["lr"] for group in groups] == ref["native_next_lrs"],
            "Exact serialized AdamW slots and group learning rates")
    for state in states.values():
        first, second = state["exp_avg"], state["exp_avg_sq"]
        require(float(state["step"]) == step and finite_tensor(first, torch) and finite_tensor(second, torch)
                and first.shape == second.shape and first.dtype == second.dtype
                and first.is_floating_point() and first.numel() > 0,
                "Full finite AdamW moments and exact update clock")


def check_archive_model(saved, epoch, ref, torch):
    online = saved["state_dict"]
    callback = saved["callbacks"]["RFDETREMACallback"]
    average = callback["average_model_state_dict"]
    require(float(callback["latest_update_step"]) == float(average["n_averaged"]) == epoch * UPDATES,
            "Exact serialized EMA state keys and update clock")
    raw = {key.removeprefix("model."): value for key, value in online.items() if key.startswith("model.")}
    averaged = {key.removeprefix("module.model."): value for key, value in average.items()
                if key.startswith("module.model.")}
    require(raw.keys() == averaged.keys() and all(
        finite_tensor(value, torch) and finite_tensor(averaged[name], torch)
        and value.shape == averaged[name].shape and value.dtype == averaged[name].dtype
        for name, value in raw.items()), "Matching finite native online and EMA tensors")
    require(all(raw[name].shape[0] == WIDTH * 13 for name in ("refpoint_embed.weight", "query_feat.weight"))
            and raw["class_embed.weight"].shape[0] == 2, "Exact native model query tables and class slots")
    require(tensor_digest(raw, torch) == ref["ending_online_sha256"]
            and tensor_digest(averaged, torch) == ref["ending_ema_sha256"],
            "Exact independent online and EMA tensor digests")


def validate_checkpoint(saved, ref, cumulative):
    """Recompute endpoint loop, provenance, optimizer, EMA and tensor evidence."""
    import torch

    epoch = ref["epoch"]
    try:
        check_archive_loop(saved, epoch)
        check_archive_marker(saved, epoch, cumulative, torch)
        check_archive_optimizer(saved, epoch, ref, torch)
        check_archive_model(saved, epoch, ref, torch)
    except (KeyError, TypeError, IndexError, OverflowError) as exc:
        raise ValueError("Incomplete native checkpoint evidence") from exc


def read_closed_checkpoint(path, ref):
    """Read one canonical closed path in a caller's admitted checkpoint reader."""
    from os import fstat
    from pathlib import Path

    import torch

    def identity(stat):
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns

    path = Path(path)
    require(path.is_absolute() and path.resolve(strict=True) == path and path.name == ref["archive_name"],
            "Exact canonical native checkpoint path")
    digest = sha256()
    with path.open("rb") as stream:
        before = fstat(stream.fileno())
        require(0 < before.st_size <= 1024**3 and before.st_size == ref["archive_bytes"],
                "Exact bounded native checkpoint size")
        for chunk in iter(lambda: stream.read(1024**2), b""):
            digest.update(chunk)
        observed_sha = digest.hexdigest()
        require(observed_sha == ref["archive_sha256"], "Exact closed native checkpoint bytes")
        stream.seek(0)
        saved = torch.load(stream, map_location="cpu", weights_only=True)
        require(identity(before) == identity(fstat(stream.fileno())) == identity(path.stat()),
                "Stable closed native checkpoint identity")
    return saved, before.st_size, observed_sha


def _receipt_for(result):
    receipt = dict(
        status="HRSID_FILTERED_E24_DEV705_READY_V1", training_protocol=PROTOCOL,
        original_os_wait=True, original_os_exit=0, resource_stop=False,
        completed_epochs=EPOCHS, optimizer_updates=UPDATES * EPOCHS, ema_updates=UPDATES * EPOCHS,
        target_filter_sha256=FILTER_SHA, filtered_empty_rows_total=result["filtered_empty_rows_total"],
        checkpoint_sha256=result["archive_refs"][-1]["archive_sha256"],
    )
    raw = (json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    return raw, sha256(raw).hexdigest()



