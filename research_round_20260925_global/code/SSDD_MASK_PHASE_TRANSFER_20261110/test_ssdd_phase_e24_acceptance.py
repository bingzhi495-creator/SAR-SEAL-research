import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

from ssdd_phase_e24_acceptance import ArchiveEvidence, ClosedE24Arm, create_pair_receipt, make_checkpoint_reader, validate_arm
from ssdd_phase_qualification import encode_receipt, verify_pair
from test_ssdd_phase_qualification import FIT_IDS, MANIFEST, PROTOCOL, fingerprint, fixture, packed, sha


def lr_factor(step):
    if step < 558:
        return step / 558
    return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * (step - 558) / (18600 - 558)))


def long_arm(mode, short, edit=None):
    marker = dict(json.loads(short.report_bytes)["phase"], stage="e24")
    init = json.loads(short.report_bytes)["initialization"]
    seed = json.loads(short.report_bytes)["seed_witness"]
    report_path = (Path(__file__).resolve().parent / f"SYNTHETIC_{mode}" / "checkpoint_23.ckpt").as_posix()
    archive = dict(checkpoint_ref=dict(path=report_path, bytes=543000000, sha256=fingerprint(f"checkpoint:{mode}")),
                   weights_only_cpu_readback=True, classifier_raw_max_abs_change=0.1,
                   classifier_raw_ema_max_abs_difference=0.05)
    events = [dict(event="SSDD_MASK_PHASE_INITIALIZED", phase=marker, initialization=init),
              dict(event="SSDD_MASK_PHASE_TRAIN_STARTED", seed_witness=seed, visits_per_epoch=744,
                   planned_updates=18600, warmup_updates=558, updates_per_epoch=186)]
    rows, audit_rows = [], []
    removed_total = 0
    for epoch in range(1, 25):
        ids = FIT_IDS + (225, 328, 210)
        visits = [[image_id, image_id % 3 + 1] for image_id in ids]
        if mode == "center" and epoch == 1:
            visits[0][1] -= 1
        if edit:
            edit(epoch, visits)
        row = packed(dict(epoch=epoch, visits=visits)) + b"\n"
        rows.append(row)
        removed = int(mode == "center" and epoch == 1)
        removed_total += removed
        for batch, (image_id, _) in enumerate(visits, 1):
            if batch == 1 and removed:
                events.append(dict(event="SSDD_MASK_PHASE_EMPTY_FILTER_APPLIED", epoch=epoch,
                                   batch_idx=batch - 1, image_id=image_id, removed_rows=removed))
            step = (epoch - 1) * 186 + batch // 4
            if batch % 4 == 0 and step % 100 == 0:
                events.append(dict(event="SSDD_MASK_PHASE_PROGRESS", epoch=epoch,
                                   visits_in_epoch=batch, optimizer_updates=step))
        counts = [n for _, n in visits]
        steps = epoch * 186
        audit = dict(epoch=epoch, visits=744, unique_fit_images=741, duplicate_fit_visits=3,
                     zero_target_visits=counts.count(0), total_augmented_targets=sum(counts),
                     maximum_augmented_targets=max(counts), discarded_empty_mask_rows=removed,
                     optimizer_updates=steps, ema_updates=steps, visit_row_sha256=sha(row), loss_min=0.3,
                     loss_max=2.4, classifier_gradient_norm_max=1.1, epoch_seconds=0.7,
                     adam_parameter_clocks={str(steps): 543}, native_base_lrs=[0.0001] * 28,
                     native_next_lrs=[0.0001 * lr_factor(steps)] * 28)
        audit_rows.append(packed(audit) + b"\n")
        events.append(dict(event="SSDD_MASK_PHASE_EPOCH_OBSERVED", **audit))
    visit_bytes, epoch_bytes = b"".join(rows), b"".join(audit_rows)
    report = dict(status="SSDD_MASK_PHASE_ARM_CLOSED_PENDING_INDEPENDENT_ACCEPTANCE_V1", phase=marker,
                  initialization=init, seed_witness=seed, completed_epochs=24, actual_trainer_epoch_cursor=24,
                  visits=17856, optimizer_updates=4464, ema_updates=4464, planned_epochs=100, planned_updates=18600,
                  warmup_updates=558, fit_images=741, visits_per_epoch=744, removed_empty_rows=removed_total,
                  precision="bf16-mixed", float32_matmul_precision="high", cudnn_tf32=True, elapsed_seconds=23.0,
                  peak_allocated_bytes=1024, peak_reserved_bytes=2048, FIT=True, DEV=False, TEST=False, AP=False,
                  fit_visit_ledger_sha256=sha(visit_bytes), epoch_audit_sha256=sha(epoch_bytes), checkpoint_saved=True,
                  archive=archive, **{key: audit[key] for key in ("adam_parameter_clocks", "native_base_lrs",
                                                               "native_next_lrs")})
    events.append(dict(event="SSDD_MASK_PHASE_ARM_RETURNED", **report))
    stdout = b"".join(packed(event) + b"\n" for event in events)
    closure = dict(original_os_wait=True, original_os_exit=0, resource_stop=False, worker_closed=True,
                   terminal_sha256=fingerprint(f"terminal:e24:{mode}"), stdout_sha256=sha(stdout),
                   output_files=["SSDD_PHASE_TRAIN_SAFE.json", "FIT_VISITS_PRIVATE.jsonl", "EPOCH_AUDIT_SAFE.jsonl",
                                 "checkpoint_23.ckpt"])
    return ClosedE24Arm(closure, packed(report), epoch_bytes, visit_bytes, stdout)


class E24Tests(unittest.TestCase):
    def setUp(self):
        self.short = {mode: fixture(mode) for mode in ("native", "center")}
        qualification = encode_receipt(verify_pair(self.short["native"], self.short["center"], protocol=PROTOCOL,
                                                   source_manifest_sha256=MANIFEST, expected_fit_ids=FIT_IDS))
        self.scope = dict(protocol=PROTOCOL, source_manifest_sha256=MANIFEST, expected_fit_ids=FIT_IDS,
                          qualification_bytes=qualification, qualification_sha256=sha(qualification))

    def arm(self, mode, edit=None):
        return long_arm(mode, self.short[mode], edit)

    def check(self, mode, arm, **overrides):
        report = json.loads(arm.report_bytes)
        arguments = dict(mode=mode, expected_checkpoint_path=report["archive"]["checkpoint_ref"]["path"],
                         checkpoint_reader=lambda ref, phase: ArchiveEvidence(report["phase"], report["archive"]))
        return validate_arm(arm, **(self.scope | arguments | overrides))

    def test_e24_pair_checks_exact_population_and_allows_natural_target_difference(self):
        native, center = self.arm("native"), self.arm("center")
        a, a_sha = self.check("native", native)
        b, b_sha = self.check("center", center)
        pair_raw, pair_sha, stats = create_pair_receipt(a, b, native.visit_bytes, center.visit_bytes, **self.scope)
        pair = json.loads(pair_raw)
        self.assertEqual(pair["status"], "SSDD_MASK_PHASE_E24_PAIR_INDEPENDENTLY_ACCEPTED_V1")
        self.assertEqual(pair["completed_epochs"], 24)
        self.assertEqual(pair["visits_per_arm"], 17856)
        self.assertEqual(pair["optimizer_updates_per_arm"], pair["ema_updates_per_arm"])
        self.assertEqual(pair["optimizer_updates_per_arm"], 4464)
        self.assertEqual(pair["native_arm_receipt_sha256"], a_sha)
        self.assertEqual(pair["center_arm_receipt_sha256"], b_sha)
        self.assertEqual(pair_sha, sha(pair_raw))
        self.assertEqual(stats["differing_target_count_visits"], 1)
        self.assertEqual(stats["native_retained_targets"] - stats["center_retained_targets"], 1)

    def test_wrong_duplicate_roster_is_rejected_before_checkpoint_access(self):
        def wrong_padding(epoch, visits):
            if epoch == 10:
                visits[-1][0] = 500

        reader = lambda *_: self.fail("Opened checkpoint before rejecting wrong visit roster")
        with self.assertRaisesRegex(ValueError, "three original alignment identities"):
            self.check("native", self.arm("native", wrong_padding), checkpoint_reader=reader)

    def test_two_valid_arms_with_different_visit_orders_do_not_pair(self):
        def reorder(epoch, visits):
            if epoch == 4:
                visits[0], visits[1] = visits[1], visits[0]

        native, center = self.arm("native"), self.arm("center", reorder)
        a, _ = self.check("native", native)
        b, _ = self.check("center", center)
        with self.assertRaisesRegex(ValueError, "Ordered SSDD FIT image identities differ"):
            create_pair_receipt(a, b, native.visit_bytes, center.visit_bytes, **self.scope)

    def test_archive_reader_must_return_exact_independent_native_evidence_and_own_marker(self):
        arm = self.arm("native")
        report = json.loads(arm.report_bytes)
        archive = report["archive"]
        wrong_native_report = dict(archive, checkpoint_ref=dict(archive["checkpoint_ref"], bytes=12))
        with self.assertRaisesRegex(ValueError, "native archive readback differs"):
            self.check("native", arm, checkpoint_reader=lambda *_: ArchiveEvidence(report["phase"], wrong_native_report))
        wrong_marker = dict(report["phase"], mask_resize_mode="center")
        with self.assertRaisesRegex(ValueError, "checkpoint top-level phase marker differs"):
            self.check("native", arm, checkpoint_reader=lambda *_: ArchiveEvidence(wrong_marker, archive))
        wrong_path = Path(archive["checkpoint_ref"]["path"]).parent / "checkpoint_22.ckpt"
        with self.assertRaisesRegex(ValueError, "checkpoint path, size and SHA"):
            self.check("native", arm, expected_checkpoint_path=wrong_path)

    def test_original_successful_os_exit_is_required_before_archive_access(self):
        arm = self.arm("native")
        reader = lambda *_: self.fail("Opened checkpoint before original worker closure")
        for key, value in (("original_os_wait", False), ("original_os_exit", 3),
                           ("resource_stop", True), ("worker_closed", False)):
            bad = arm._replace(closure=arm.closure | {key: value})
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "original E24 worker closure"):
                self.check("native", bad, checkpoint_reader=reader)

    def test_independently_accepted_short_scope_is_pinned(self):
        arm = self.arm("native")
        with self.assertRaisesRegex(ValueError, "accepted short qualification SHA"):
            self.check("native", arm, qualification_sha256="8" * 64)
        with self.assertRaisesRegex(ValueError, "same-scope independently qualified pair"):
            self.check("native", arm, source_manifest_sha256="9" * 64)

    def test_original_filter_event_must_join_to_exact_epoch_visit(self):
        center = self.arm("center")
        lines = [json.loads(line) for line in center.stdout_bytes.splitlines()]
        filtered, = (row for row in lines if row["event"] == "SSDD_MASK_PHASE_EMPTY_FILTER_APPLIED")
        filtered["image_id"] = 2
        log = b"".join(packed(row) + b"\n" for row in lines)
        bad = center._replace(stdout_bytes=log, closure=center.closure | {"stdout_sha256": sha(log)})
        with self.assertRaisesRegex(ValueError, "filter event must match its exact visit"):
            self.check("center", bad)

    def test_original_report_cannot_replace_final_update_clock(self):
        native = self.arm("native")
        report = json.loads(native.report_bytes)
        report["ema_updates"] = 4463
        bad = native._replace(report_bytes=packed(report))
        with self.assertRaisesRegex(ValueError, "Exact original SSDD E24 report"):
            self.check("native", bad)

    def test_reusable_reader_intercepts_exactly_one_native_cpu_load_and_checks_phase_seed_q100(self):
        report = json.loads(self.arm("native").report_bytes)
        phase, init, archive = (report[key] for key in ("phase", "initialization", "archive"))

        class Classifier:
            dtype = "torch.float32"
            shape = (2, 2)
            device = SimpleNamespace(type="cpu")

            def detach(self):
                return self

            contiguous = clone = detach

            def numpy(self):
                return SimpleNamespace(tobytes=lambda order: bytes(16))

        calls = []
        classifier = Classifier()
        classifier_sha = sha(repr((classifier.dtype, classifier.shape)).encode() + bytes(16))
        stored_phase = phase.copy()

        def load(stream, *, map_location, weights_only):
            calls.append((stream, map_location, weights_only))
            return {"sar_seal_ssdd_mask_resize_phase_v1": stored_phase}

        def inspect(path, *, epoch, torch, e1_api, boundary_api, recipe_sha256, initial_witness):
            self.assertEqual(path, Path(archive["checkpoint_ref"]["path"]))
            self.assertEqual(epoch, 24)
            self.assertEqual(initial_witness.shape, classifier.shape)
            torch.load("synthetic stream", map_location="cpu", weights_only=True)
            return archive

        dependencies = dict(torch=SimpleNamespace(float32="torch.float32", isfinite=lambda _: True, load=load),
                            native_inspector=inspect, e1_api=object(), boundary_api=object(),
                            initial_classifier=classifier, initialization=init, classifier_initial_sha256=classifier_sha)
        reader = make_checkpoint_reader(**dependencies)
        self.assertEqual(reader(archive["checkpoint_ref"], phase), ArchiveEvidence(phase, archive))
        self.assertEqual(calls, [("synthetic stream", "cpu", True)])
        for key, value in (("mask_resize_mode", "center"), ("seed", 5202), ("initial_online_sha256", "a" * 64)):
            stored_phase = phase | {key: value}
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "exact E24 phase marker"):
                reader(archive["checkpoint_ref"], phase)
        with self.assertRaisesRegex(ValueError, "official COCO-only Q100 start witness"):
            make_checkpoint_reader(**(dependencies | {"initialization": init | {"online_tensor_sha256": "a" * 64}}))
        with self.assertRaisesRegex(ValueError, "independently qualified initial classifier"):
            make_checkpoint_reader(**(dependencies | {"classifier_initial_sha256": "a" * 64}))
