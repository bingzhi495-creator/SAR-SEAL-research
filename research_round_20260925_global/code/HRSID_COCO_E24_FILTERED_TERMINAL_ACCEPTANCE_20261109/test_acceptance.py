from hashlib import sha256
import json
import unittest

import torch

from acceptance import (
    CAP, COCO_SHA, FILTER_SHA, FIT, FIT_INDEX_SHA, INITIAL_SHA, MARKER, PROTOCOL, RECIPE_SHA,
    schedule_factor, validate_closed_run,
)


class TerminalAcceptanceTests(unittest.TestCase):
    def test_live_worker_never_reaches_the_archive_reader(self):
        stdout = b"synthetic stdout\n"
        closure = dict(
            original_os_wait=True,
            original_os_exit=0,
            resource_stop=False,
            owner_closed=True,
            worker_closed=False,
            terminal_sha256="a" * 64,
            stdout_sha256=sha256(stdout).hexdigest(),
        )

        def unexpected_read(_):
            self.fail("A live worker must not permit checkpoint access")

        with self.assertRaisesRegex(ValueError, "closed original"):
            validate_closed_run(closure, b"", b"", b"", stdout, unexpected_read, ())

    def test_closed_but_missing_result_cannot_read_an_archive(self):
        stdout = b"synthetic stdout\n"
        closure = dict(
            original_os_wait=True,
            original_os_exit=0,
            resource_stop=False,
            owner_closed=True,
            worker_closed=True,
            terminal_sha256="a" * 64,
            stdout_sha256=sha256(stdout).hexdigest(),
        )

        def unexpected_read(_):
            self.fail("Missing training result must not permit checkpoint access")

        with self.assertRaisesRegex(ValueError, "training result"):
            validate_closed_run(closure, b"", b"", b"", stdout, unexpected_read, ())


    def test_complete_independent_readback_returns_the_exact_dev_receipt(self):
        args, reader, calls = synthetic_run()
        raw, digest = validate_closed_run(*args, reader, range(FIT))
        self.assertEqual(calls, [1, 8, 24])
        self.assertEqual(digest, sha256(raw).hexdigest())
        self.assertEqual(json.loads(raw), {
            "status": "HRSID_FILTERED_E24_DEV705_READY_V1",
            "training_protocol": PROTOCOL,
            "original_os_wait": True,
            "original_os_exit": 0,
            "resource_stop": False,
            "completed_epochs": 24,
            "optimizer_updates": 17640,
            "ema_updates": 17640,
            "target_filter_sha256": FILTER_SHA,
            "filtered_empty_rows_total": 2,
            "checkpoint_sha256": "3" * 64,
        })




    def test_filter_event_identity_mismatch_prevents_all_checkpoint_reads(self):
        args, reader, calls = synthetic_run()
        closure, result, audit, visits, stdout = args
        stdout = stdout.replace(b'"image_id": 100, "removed_rows": 1', b'"image_id": 101, "removed_rows": 1', 1)
        closure["stdout_sha256"] = sha256(stdout).hexdigest()
        with self.assertRaisesRegex(ValueError, "matches the original FIT visit"):
            validate_closed_run(closure, result, audit, visits, stdout, reader, range(FIT))
        self.assertEqual(calls, [])

    def test_unrelated_lightning_state_does_not_change_detector_digest(self):
        args, reader, _ = synthetic_run()

        def detector_only_ema(ref):
            archive, size, digest = reader(ref)
            archive["state_dict"]["criterion.synthetic_aux"] = torch.tensor([0.])
            return archive, size, digest

        raw, _ = validate_closed_run(*args, detector_only_ema, range(FIT))
        self.assertEqual(json.loads(raw)["checkpoint_sha256"], "3" * 64)

    def test_fractional_ema_clock_cannot_qualify_the_endpoint(self):
        args, reader, _ = synthetic_run()

        def wrong_ema(ref):
            archive, size, digest = reader(ref)
            if ref["epoch"] == 24:
                archive["callbacks"]["RFDETREMACallback"]["latest_update_step"] += 0.5
            return archive, size, digest

        with self.assertRaisesRegex(ValueError, "EMA state keys and update clock"):
            validate_closed_run(*args, wrong_ema, range(FIT))

    def test_changed_online_tensor_cannot_qualify_the_endpoint(self):
        args, reader, _ = synthetic_run()

        def changed_state(ref):
            archive, size, digest = reader(ref)
            if ref["epoch"] == 24:
                archive["state_dict"]["model.class_embed.weight"][0, 0] += 1
            return archive, size, digest

        with self.assertRaisesRegex(ValueError, "online and EMA tensor digests"):
            validate_closed_run(*args, changed_state, range(FIT))





def row(value):
    return (json.dumps(value, sort_keys=True) + "\n").encode()


def state_digest(state):
    h = sha256()
    for name, tensor in state.items():
        h.update(json.dumps([name, list(tensor.shape), str(tensor.dtype)]).encode() + b"\n")
        h.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def synthetic_archive(epoch, removed, base, current_lrs, archive_sha):
    step = epoch * 735
    counter = {"ready": epoch, "started": epoch, "processed": epoch, "completed": epoch - 1}
    batch = {field: 2940 for field in ("ready", "started", "processed", "completed")}
    loop = {
        "epoch_progress": {"current": counter, "total": counter.copy()},
        "epoch_loop.batch_progress": {"is_last_batch": True, "current": batch,
                                      "total": {field: value * epoch for field, value in batch.items()}},
        "epoch_loop.automatic_optimization.optim_progress": {
            "optimizer": {"step": {"current": {"ready": 735, "completed": 735},
                                   "total": {"ready": step, "completed": step}}}},
    }
    raw = {"refpoint_embed.weight": torch.zeros((2132, 2)), "query_feat.weight": torch.zeros((2132, 2)),
           "class_embed.weight": torch.zeros((2, 2))}
    average = {key: value + 1 for key, value in raw.items()}
    moment = {"step": torch.tensor(float(step)), "exp_avg": torch.zeros(1), "exp_avg_sq": torch.zeros(1)}
    groups = [dict(lr=lr, params=list(range(group, 543, 28))) for group, lr in enumerate(current_lrs)]
    archive = {
        "pytorch-lightning_version": "2.6.6", "epoch": epoch - 1, "global_step": step,
        "loops": {"fit_loop": loop},
        "model_config": dict(resolution=576, num_queries=164, group_detr=13, num_select=100,
                             num_classes=1, pretrain_weights=None, segmentation_head=True),
        "state_dict": {"model." + key: value for key, value in raw.items()},
        "callbacks": {"RFDETREMACallback": dict(latest_update_step=step, average_model_state_dict={
            **{"module.model." + key: value for key, value in average.items()}, "n_averaged": torch.tensor(step),
        })},
        "optimizer_states": [dict(state={i: moment for i in range(543)}, param_groups=groups)],
        "lr_schedulers": [dict(last_epoch=step, base_lrs=base, _last_lr=current_lrs)],
        MARKER: dict(
            protocol=PROTOCOL, epoch=epoch, actual_updates=step, native_index_sha256=FIT_INDEX_SHA,
            coco_sha256=COCO_SHA, original_recipe_sha256=RECIPE_SHA, initial_expanded_sha256=INITIAL_SHA,
            FIT_only=True, exact_restart_trajectory_verified=False, target_filter_sha256=FILTER_SHA,
            filtered_empty_rows_total=removed,
            rng_state=dict(python=(), numpy=(), torch_cpu=torch.zeros(4, dtype=torch.uint8),
                           torch_cuda=(torch.zeros(4, dtype=torch.uint8),)),
        ),
    }
    ref = dict(
        epoch=epoch, archive_name=f"checkpoint_{epoch-1}.ckpt", archive_sha256=archive_sha, archive_bytes=100,
        ending_online_sha256=state_digest(raw), ending_ema_sha256=state_digest(average), rng_saved=True,
        adam_parameter_clocks={str(step): 543}, native_base_lrs=base, native_next_lrs=current_lrs,
    )
    return ref, archive


def synthetic_run():
    base = [.0001] * 28
    visits = [[iid, 1] for iid in range(FIT)] + [[iid, 1] for iid in range(3)]
    audit_lines, visit_lines, stdout_lines, refs = [], [], [], []
    archives = {}
    count = 0
    for epoch in range(1, 25):
        step = 735 * epoch
        rates = [lr * schedule_factor(step) for lr in base]
        visit_line = row(dict(epoch=epoch, visits=visits))
        visit_lines.append(visit_line)
        removed = int(epoch in (5, 11))
        count += removed
        audit = dict(
            epoch=epoch, visits=2940, unique_fit_images=2937, duplicate_fit_visits=3,
            zero_target_visits=0, maximum_augmented_targets=1, total_augmented_targets=2940,
            discarded_empty_mask_rows=removed, optimizer_updates=step, ema_updates=step,
            adam_parameter_clocks={str(step): 543}, visit_row_sha256=sha256(visit_line).hexdigest(),
            lightning_accumulation_scaled_loss_min=1., lightning_accumulation_scaled_loss_mean=2.,
            lightning_accumulation_scaled_loss_max=3., classifier_gradient_norm_max=1.,
            epoch_seconds=1., peak_allocated_bytes=1000, native_base_lrs=base, native_next_lrs=rates,
        )
        audit_lines.append(row(audit))
        if removed:
            stdout_lines.append(row(dict(event="HRSID_NATIVE_EMPTY_MASK_FILTER_APPLIED", epoch=epoch,
                                         batch_idx=100, image_id=100, removed_rows=1)))
        stdout_lines.append(row(dict(event="HRSID_COCO_NATIVE_EPOCH_OBSERVED", **audit)))
        if epoch in (1, 8, 24):
            ref, archive = synthetic_archive(epoch, count, base, rates, str(len(refs) + 1) * 64)
            refs.append(ref)
            archives[epoch] = archive
    audit_bytes, visit_bytes = b"".join(audit_lines), b"".join(visit_lines)
    result = dict(
        status="NATIVE_HRSID_COCO_FIT_E24_COMPLETE_PENDING_INDEPENDENT_ACCEPTANCE",
        protocol=PROTOCOL, native_index_sha256=FIT_INDEX_SHA, original_recipe_sha256=RECIPE_SHA,
        visited_fit_images=2937, total_visits=70560, total_updates=17640, starting_optimizer_updates=0,
        starting_ema_updates=0, ema_updates=17640, completed_epochs=24, scheduled_epochs=100,
        scheduled_updates=73500, warmup_updates=2205, precision="bf16-mixed", batch_size=1,
        grad_accumulation=4, gpu_allocator_limit_bytes=CAP, gpu_peak_allocated_bytes=1000,
        gpu_peak_reserved_bytes=1000, elapsed_seconds=30., FIT_only=True, DEV=False, TEST=False, AP=False,
        exact_restart_trajectory_verified=False, target_filter_sha256=FILTER_SHA,
        coco_asset_sha256=COCO_SHA, coco_asset_bytes=134545398, native_original_width=100,
        width100_loaded_sha256="e"*64, expanded_online_sha256=INITIAL_SHA, query_width=164, query_groups=13,
        class_head_slots=2, initialized_from_SSDD=False, epoch_audit_sha256=sha256(audit_bytes).hexdigest(),
        fit_visit_ledger_sha256=sha256(visit_bytes).hexdigest(), filtered_empty_rows_total=count,
        archive_refs=refs,
    )
    result_bytes = row(result)
    stdout = b"".join(stdout_lines + [result_bytes])
    closure = dict(original_os_wait=True, original_os_exit=0, resource_stop=False, owner_closed=True,
                   worker_closed=True, terminal_sha256="a"*64, stdout_sha256=sha256(stdout).hexdigest())
    calls = []

    def reader(ref):
        calls.append(ref["epoch"])
        return archives[ref["epoch"]], ref["archive_bytes"], ref["archive_sha256"]

    return (closure, result_bytes, audit_bytes, visit_bytes, stdout), reader, calls



