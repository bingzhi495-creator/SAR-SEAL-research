"""Synthetic checks for independent repeat qualification."""
import importlib.util
from hashlib import sha256
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location("repeat_qualification", Path(__file__).with_name("repeat_qualification.py"))
QUAL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = QUAL
SPEC.loader.exec_module(QUAL)
PROTOCOL = "a49361b92977a605267025b1e7798420936cbe53"
REPEAT_SHA = "42a174573a04c529dfbae943bcc88ef4ed5d793d7eda6766511396034fda321f"


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def witness(seed, mode):
    return dict(status="HRSID_MASK_PHASE_REPEAT_SEED_WITNESS_V1", protocol=PROTOCOL,
                mask_resize_mode=mode, stage="qualification", initialization_seed=5201,
                training_seed=seed, global_rank=0, train_config_seed=seed, torch_cpu_initial_seed=seed,
                torch_cuda_initial_seed=seed, lightning_global_seed=seed, lightning_seed_workers=True)


def fingerprint(text):
    return sha256(text.encode()).hexdigest()


def arm_fixture(seed, mode, *, filtered=False, input_seed=None):
    if input_seed is None:
        input_seed = seed
    rows = []
    for i in range(32):
        labels = 1
        post = 0 if filtered and i == 0 else labels
        masks = fingerprint(f"mask-{mode}-{i}")
        boxes = fingerprint(f"box-{i}")
        rows.append(dict(batch=i + 1, image_id=input_seed * 100 + i, image_sha256=fingerprint(f"image-{input_seed}-{i}"),
                         cpu_rng_sha256=fingerprint(f"rng-{input_seed}-{i}"), labels=labels, postfilter_labels=post,
                         labels_sha256=fingerprint(f"labels-{i}"), masks_sha256=masks, boxes_sha256=boxes,
                         postfilter_masks_sha256=masks if post else fingerprint("empty-masks"),
                         postfilter_boxes_sha256=boxes if post else fingerprint("empty-boxes")))
    visits = encoded(rows)
    identity = [(r["image_id"], r["image_sha256"], r["cpu_rng_sha256"]) for r in rows]
    report = dict(status="NATIVE_HRSID_MASK_RESIZE_QUALIFICATION_COMPLETE", protocol=PROTOCOL,
                  mask_resize_mode=mode, stage="qualification", seed=seed,
                  phase_trainer_sha256=REPEAT_SHA, phase_patch_sha256=QUAL.PATCH_SHA,
                  native_trainer_sha256=QUAL.NATIVE_SHA, initial_expanded_sha256=QUAL.INITIAL_SHA,
                  initial_online_sha256=QUAL.INITIAL_SHA, classifier_initial_sha256=QUAL.CLASSIFIER_SHA,
                  classifier_online_ending_sha256=fingerprint(f"ending-{seed}-{mode}"),
                  adam_parameter_clocks={"8": 543}, native_base_lrs=[.0001] * 28,
                  native_next_lrs=[.0001 * (8 / 2205)] * 28, visits=32, optimizer_updates=8, ema_updates=8,
                  completed_epochs=0, actual_trainer_epoch_cursor=1, planned_updates=73500, warmup_updates=2205,
                  loss_min=.001, loss_max=4., classifier_gradient_norm_max=1.,
                  removed_empty_rows=sum(r["labels"] - r["postfilter_labels"] for r in rows),
                  targets_after_filter=sum(r["postfilter_labels"] for r in rows),
                  peak_allocated_bytes=10, peak_reserved_bytes=20, seconds=3.,
                  FIT=True, DEV=False, TEST=False, AP=False, checkpoint_saved=False,
                  visit_ledger_sha256=sha256(visits).hexdigest(),
                  input_pair_sha256=sha256(json.dumps(identity, separators=(",", ":")).encode()).hexdigest())
    observed = witness(seed, mode)
    observed_bytes = encoded(observed)
    events = [
        dict(event="HRSID_COCO_NATIVE_INITIALIZATION", class_head_slots=2, coco_asset_bytes=134545398,
             coco_asset_sha256=QUAL.COCO_SHA, expanded_online_sha256=QUAL.INITIAL_SHA,
             initialized_from_SSDD=False, native_original_width=100, query_groups=13, query_width=164,
             training=False, width100_loaded_sha256=QUAL.WIDTH100_SHA),
        dict(event="HRSID_COCO_NATIVE_E24_STARTED", planned_epochs=100, stop_after_epoch=24,
             updates_per_epoch=735, visits_per_epoch=2940),
        dict(event="HRSID_MASK_PHASE_REPEAT_SEED_OBSERVED", seed_witness_sha256=sha256(observed_bytes).hexdigest(),
             **observed),
    ]
    if filtered:
        events.append(dict(event="HRSID_NATIVE_EMPTY_MASK_FILTER_APPLIED", epoch=1, batch_idx=0,
                           image_id=rows[0]["image_id"], removed_rows=1))
    events.append(dict(event="HRSID_MASK_PHASE_SHORT_WINDOW_CLOSED", **report))
    stdout = b"[INFO] synthetic stdout\n" + b"".join((json.dumps(event) + "\n").encode() for event in events)
    closure = dict(original_os_wait=True, original_os_exit=0, worker_closed=True, resource_stop=False,
                   stdout_sha256=sha256(stdout).hexdigest(), terminal_sha256=fingerprint("terminal"))
    return closure, encoded(report), visits, observed_bytes, stdout


def historical_fixture():
    pair = dict(status="HRSID_MASK_PHASE_PAIRED_QUALIFICATION_COMPLETE", protocol=QUAL.HISTORICAL_PROTOCOL,
                visits_per_arm=32, updates_per_arm=8, identical_image_order_pixels_and_cpu_rng=True,
                differing_postfilter_box_batches=0, differing_postfilter_count_batches=0,
                differing_postfilter_mask_batches=32, FIT=True, DEV=False, TEST=False, AP=False)
    visits = []
    for mode in ("native", "center"):
        _, raw, rows, _, _ = arm_fixture(5201, mode)
        report = json.loads(raw)
        report.update(protocol=QUAL.HISTORICAL_PROTOCOL, phase_trainer_sha256=QUAL.HISTORICAL_TRAINER_SHA)
        pair[mode] = report
        visits.append(rows)
    return encoded(pair), *visits


def accepted_arm(seed, mode, **options):
    return QUAL.validate_arm(*arm_fixture(seed, mode, **options), seed=seed, mode=mode,
                             repeat_trainer_sha256=REPEAT_SHA)


class QualificationTests(unittest.TestCase):
    def test_fit_start_must_use_the_requested_seed(self):
        actual = witness(5202, "native")
        self.assertEqual(QUAL.validate_witness(encoded(actual), seed=5202, mode="native"), actual)
        actual["train_config_seed"] = 5201
        with self.assertRaisesRegex(ValueError, "runtime repeat seed"):
            QUAL.validate_witness(encoded(actual), seed=5202, mode="native")

    def test_closed_arm_requires_runtime_witness_and_filter_trace(self):
        args = arm_fixture(5202, "center", filtered=True)
        accepted = QUAL.validate_arm(*args, seed=5202, mode="center", repeat_trainer_sha256=REPEAT_SHA)
        self.assertEqual(accepted.report["removed_empty_rows"], 1)
        closure, report, visits, observed, stdout = args
        for event, reason in (("HRSID_MASK_PHASE_REPEAT_SEED_OBSERVED", "runtime seed event"),
                              ("HRSID_NATIVE_EMPTY_MASK_FILTER_APPLIED", "filter events")):
            changed = b"".join(line for line in stdout.splitlines(keepends=True) if event.encode() not in line)
            changed_closure = dict(closure, stdout_sha256=sha256(changed).hexdigest())
            with self.subTest(event=event), self.assertRaisesRegex(ValueError, reason):
                QUAL.validate_arm(changed_closure, report, visits, observed, changed,
                                  seed=5202, mode="center", repeat_trainer_sha256=REPEAT_SHA)

    def test_both_seeds_are_paired_and_distinct_from_the_screen(self):
        historical = historical_fixture()
        arms = {(seed, mode): accepted_arm(seed, mode, filtered=(seed, mode) == (5202, "center"))
                for seed in (5202, 5203) for mode in ("native", "center")}
        with patch.object(QUAL, "HISTORICAL_PAIR_SHA", sha256(historical[0]).hexdigest()):
            result = QUAL.validate_repeats(arms, *historical)
            self.assertEqual(result["status"], "HRSID_MASK_PHASE_REPEAT_SHORT_QUALIFICATION_ACCEPTED_V1")
            self.assertEqual(result["pairs"]["5202"]["differing_postfilter_count_batches"], 1)
            wrong_within = dict(arms)
            wrong_within[5202, "center"] = accepted_arm(5202, "center", input_seed=5203)
            with self.assertRaisesRegex(ValueError, "Same-seed"):
                QUAL.validate_repeats(wrong_within, *historical)
            reused = dict(arms)
            for mode in ("native", "center"):
                reused[5202, mode] = accepted_arm(5202, mode, input_seed=5201)
            with self.assertRaisesRegex(ValueError, "Independent ordered"):
                QUAL.validate_repeats(reused, *historical)


if __name__ == "__main__":
    unittest.main()
