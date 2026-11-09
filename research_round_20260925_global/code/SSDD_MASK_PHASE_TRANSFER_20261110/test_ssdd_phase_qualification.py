import hashlib
import json
import unittest

from ssdd_phase_qualification import QualificationArm, encode_receipt, verify_pair


PROTOCOL = "1" * 40
MANIFEST = "2" * 64
RECIPE = "ba9508d9acab5ff68b10db8a1aff5d9001868d6dda00e39f299d6f5cc1b8011b"
COCO = "a44613a4ecd6b5ba61a62002c600b0b6cb7a9da2936a45317ec4b62c635fb99b"
Q100 = "40524fc63797d2fcab7b50b6ee1f041f9429df73e505cb9e647a5a8cdf6f2e2f"
FIT_IDS = tuple(range(1, 742))
OUTPUT_FILES = ["QUALIFICATION_VISITS_PRIVATE.json", "SSDD_PHASE_TRAIN_SAFE.json"]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def fingerprint(value):
    return sha(str(value).encode())


def packed(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def fixture(mode, edit=None):
    rows = []
    for index in range(32):
        boxes = fingerprint(f"boxes:{index}")
        masks = fingerprint("center mask" if index == 0 and mode == "center" else f"masks:{index}")
        row = dict(batch=index + 1, image_id=FIT_IDS[index], image_sha256=fingerprint(f"image:{index}"),
                   cpu_rng_sha256=fingerprint(f"rng:{index}"), labels=1, labels_sha256=fingerprint("label"),
                   masks_sha256=masks, boxes_sha256=boxes, postfilter_labels=1,
                   postfilter_masks_sha256=masks, postfilter_boxes_sha256=boxes)
        if mode == "center" and index == 1:
            row.update(postfilter_labels=0, postfilter_masks_sha256=fingerprint("no masks"),
                       postfilter_boxes_sha256=fingerprint("no boxes"))
        rows.append(row)
    if edit:
        edit(rows)
    phase = dict(training_protocol=PROTOCOL, stage="qualification", mask_resize_mode=mode, seed=5201,
                 initialization_seed=5201, num_queries=100, num_select=100, query_groups=13, class_head_slots=2,
                 coco_asset_sha256=COCO, initial_online_sha256=Q100, recipe_sha256=RECIPE,
                 fit_ids_sha256=sha(packed(FIT_IDS)), source_manifest_sha256=MANIFEST)
    init = dict(online_tensor_sha256=Q100, state_tensors=545, num_queries=100, num_select=100, query_groups=13,
                training_queries_per_image=1300, class_head_slots=2, initialization_seed=5201,
                coco_asset_sha256=COCO, coco_asset_bytes=134545398, recipe_sha256=RECIPE)
    seed = dict(train_config_seed=5201, torch_cpu_initial_seed=5201, torch_cuda_initial_seed=5201,
                lightning_global_seed="5201", lightning_seed_workers="1")
    identity = [(row["image_id"], row["image_sha256"], row["cpu_rng_sha256"]) for row in rows]
    visits = packed(rows) + b"\n"
    report = dict(status="SSDD_MASK_PHASE_ARM_CLOSED_PENDING_INDEPENDENT_ACCEPTANCE_V1", phase=phase,
                  initialization=init, seed_witness=seed, completed_epochs=0, actual_trainer_epoch_cursor=1,
                  visits=32, optimizer_updates=8, ema_updates=8, planned_epochs=100, planned_updates=18600,
                  warmup_updates=558, fit_images=741, visits_per_epoch=744,
                  removed_empty_rows=sum(row["labels"] - row["postfilter_labels"] for row in rows),
                  precision="bf16-mixed", float32_matmul_precision="high", cudnn_tf32=True,
                  elapsed_seconds=15.0, peak_allocated_bytes=1000, peak_reserved_bytes=2000,
                  FIT=True, DEV=False, TEST=False, AP=False, adam_parameter_clocks={"8": 543},
                  native_base_lrs=[0.0001] * 28, native_next_lrs=[0.0001 * 8 / 558] * 28,
                  visit_ledger_sha256=sha(visits), checkpoint_saved=False, input_pair_sha256=sha(packed(identity)),
                  classifier_initial_sha256=fingerprint("initial"),
                  classifier_online_ending_sha256=fingerprint(f"ending:{mode}"),
                  targets_after_filter=sum(row["postfilter_labels"] for row in rows),
                  loss_min=0.1, loss_max=2.0, classifier_gradient_norm_max=0.3)
    events = [dict(event="SSDD_MASK_PHASE_INITIALIZED", phase=phase, initialization=init),
              dict(event="SSDD_MASK_PHASE_TRAIN_STARTED", seed_witness=seed, visits_per_epoch=744,
                   planned_updates=18600, warmup_updates=558, updates_per_epoch=186)]
    events.extend(dict(event="SSDD_MASK_PHASE_EMPTY_FILTER_APPLIED", epoch=1, batch_idx=row["batch"] - 1,
                       image_id=row["image_id"], removed_rows=row["labels"] - row["postfilter_labels"])
                  for row in rows if row["labels"] != row["postfilter_labels"])
    events.append(dict(event="SSDD_MASK_PHASE_ARM_RETURNED", **report))
    stdout = b"Lightning ordinary log\n" + b"\n".join(packed(event) for event in events) + b"\n"
    closure = dict(original_os_wait=True, original_os_exit=0, resource_stop=False, worker_closed=True,
                   terminal_sha256=fingerprint(f"terminal:{mode}"), stdout_sha256=sha(stdout),
                   output_files=OUTPUT_FILES[:])
    return QualificationArm(closure, packed(report), visits, stdout)


class QualificationTests(unittest.TestCase):
    def pair(self, native=None, center=None, **overrides):
        pins = dict(protocol=PROTOCOL, source_manifest_sha256=MANIFEST, expected_fit_ids=FIT_IDS)
        return verify_pair(native or fixture("native"), center or fixture("center"), **(pins | overrides))

    def test_equal_inputs_keep_both_mask_only_change_and_natural_row_loss(self):
        receipt = self.pair()
        self.assertEqual(receipt["status"], "SSDD_MASK_PHASE_SHORT_PAIR_INDEPENDENTLY_ACCEPTED_V1")
        self.assertEqual(receipt["visits_per_arm"], 32)
        self.assertEqual(receipt["optimizer_updates_per_arm"], 8)
        self.assertEqual(receipt["ema_updates_per_arm"], 8)
        self.assertEqual(receipt["changes"]["same_count_mask_only_visits"], 1)
        self.assertEqual(receipt["changes"]["differing_postfilter_count_visits"], 1)
        self.assertEqual(receipt["changes"]["differing_postfilter_mask_visits"], 2)
        self.assertEqual(json.loads(encode_receipt(receipt)), receipt)
        self.assertEqual(encode_receipt(receipt), encode_receipt(self.pair()))

    def test_worker_must_have_independent_original_successful_closure(self):
        center = fixture("center")
        for key, value in (("original_os_wait", False), ("original_os_exit", 1),
                           ("original_os_exit", False), ("resource_stop", True), ("worker_closed", False)):
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, "original worker closure"):
                self.pair(center=center._replace(closure=center.closure | {key: value}))
        same = center._replace(closure=center.closure | {"terminal_sha256": fixture("native").closure["terminal_sha256"]})
        with self.assertRaisesRegex(ValueError, "Two independent original terminal records"):
            self.pair(center=same)

    def test_archive_in_short_output_is_rejected(self):
        center = fixture("center")
        files = OUTPUT_FILES + ["checkpoint_0.ckpt"]
        with self.assertRaisesRegex(ValueError, "only the report and visit ledger"):
            self.pair(center=center._replace(closure=center.closure | {"output_files": files}))

    def test_pair_rejects_any_order_pixel_or_rng_drift(self):
        for field, value in (("image_id", 741), ("image_sha256", fingerprint("changed image")),
                             ("cpu_rng_sha256", fingerprint("changed rng"))):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Ordered image IDs"):
                self.pair(center=fixture("center", lambda rows: rows[0].update({field: value})))

    def test_pair_refuses_data_and_source_pins_outside_original_scope(self):
        for changed in ({"protocol": "3" * 40}, {"source_manifest_sha256": "4" * 64},
                        {"expected_fit_ids": FIT_IDS[:-1] + (9999,)}):
            with self.subTest(changed=next(iter(changed))), self.assertRaisesRegex(ValueError, "prospective phase pins"):
                self.pair(**changed)

    def test_bad_filter_tensor_provenance_fails_even_with_consistent_report_digests(self):
        center = fixture("center", lambda rows: rows[0].update(postfilter_boxes_sha256=fingerprint("replaced")))
        with self.assertRaisesRegex(ValueError, "Unfiltered SSDD visit must retain exact"):
            self.pair(center=center)

    def test_target_loss_alone_is_insufficient_as_a_mask_only_visit(self):
        def change_boxes(rows):
            boxes = fingerprint("also different center boxes")
            rows[0].update(boxes_sha256=boxes, postfilter_boxes_sha256=boxes)

        with self.assertRaisesRegex(ValueError, "same-count, unfiltered mask-only visit"):
            self.pair(center=fixture("center", change_boxes))

    def test_original_stdout_digest_and_order_are_both_required(self):
        center = fixture("center")
        log = center.stdout_bytes.splitlines()
        log[1], log[2] = log[2], log[1]
        tampered = b"\n".join(log) + b"\n"
        with self.assertRaisesRegex(ValueError, "Exact independently closed stdout"):
            self.pair(center=center._replace(stdout_bytes=tampered))
        with self.assertRaisesRegex(ValueError, "initialization, train, filter and terminal events differ"):
            self.pair(center=center._replace(stdout_bytes=tampered, closure=center.closure | {"stdout_sha256": sha(tampered)}))

    def test_an_ap_flag_or_wrong_ema_clock_invalidates_even_a_matching_stdout(self):
        center = fixture("center")
        for key, value in (("AP", True), ("ema_updates", 7)):
            report = json.loads(center.report_bytes) | {key: value}
            log = center.stdout_bytes.splitlines()
            log[-1] = packed(dict(event="SSDD_MASK_PHASE_ARM_RETURNED", **report))
            stdout = b"\n".join(log) + b"\n"
            changed = center._replace(report_bytes=packed(report), stdout_bytes=stdout,
                                      closure=center.closure | {"stdout_sha256": sha(stdout)})
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "FIT-only short report"):
                self.pair(center=changed)
