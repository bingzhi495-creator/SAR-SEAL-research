"""Synthetic checks for the stock SSDD schedule and phase boundary."""
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from ssdd_phase_train import (
    COCO_SHA, FitProgress, PHASE_MARKER, Q100_SHA, check_start_witness, mark_e24_archive,
    retained_target, run_fit_arm,
)


E100 = runpy.run_path(str(Path(__file__).parent.parent /
                        "RFDETR_NATIVE_UNINTERRUPTED_E100_WORKER_CANDIDATE_20261107/e100_worker.py"))
IDS = tuple(range(741))
PADDING = (1, 7, 21)


class PhaseTrainTests(unittest.TestCase):
    def test_real_window_accounting_exits_without_a_completed_epoch(self):
        progress = FitProgress("qualification", IDS, PADDING)
        for i, image in enumerate(IDS[:32]):
            progress.consume(epoch=1, batch=i, image_id=image, retained=1, removed=0)
            self.assertEqual(progress.after_batch(i, (i + 1) // 4), i == 31)
        self.assertIsNone(progress.finish_epoch(1, 8, should_stop=True))
        self.assertEqual(progress.completed, 0)
        self.assertEqual(progress.total_visits, 32)
        self.assertEqual(progress.removed_empty_rows, 0)


    def test_qualification_rejects_wrong_update_missing_visit_or_partial_archive(self):
        progress = FitProgress("qualification", IDS, PADDING)
        progress.consume(epoch=1, batch=0, image_id=5, retained=1, removed=0)
        with self.assertRaisesRegex(RuntimeError, "AdamW"):
            progress.after_batch(0, 1)
        with self.assertRaisesRegex(RuntimeError, "without completing an epoch"):
            progress.finish_epoch(1, 8, should_stop=True)
        gate = E100["ArchiveEndpoints"](SimpleNamespace(), stop_epoch=24, recipe_sha256="a" * 64)
        with self.assertRaisesRegex(RuntimeError, "E24 phase archive"):
            mark_e24_archive(gate, {}, {"stage": "qualification"})
        self.assertEqual(gate.saved, [])
        self.assertEqual(gate.archive.save_top_k, 0)

    def test_native_24_epoch_and_phase_markers_share_original_boundary(self):
        progress = FitProgress("e24", IDS, PADDING)
        recipe = "a" * 64
        gate = E100["ArchiveEndpoints"](SimpleNamespace(), stop_epoch=24, recipe_sha256=recipe)
        trainer = SimpleNamespace(should_stop=False)
        phase = dict(stage="e24", mask_resize_mode="center", seed=5201, num_queries=100,
                     source_manifest_sha256="d" * 64)
        for epoch in range(1, 25):
            ordered = IDS[epoch:] + IDS[:epoch] + PADDING
            for batch, image in enumerate(ordered):
                progress.consume(epoch=epoch, batch=batch, image_id=image, retained=2, removed=0)
                updates = (epoch - 1) * 186 + (batch + 1) // 4
                self.assertFalse(progress.after_batch(batch, updates))
            self.assertEqual(progress.finish_epoch(epoch, updates, should_stop=False),
                             [(image, 2) for image in ordered])
            gate.complete_epoch(trainer, epoch)
            self.assertEqual(gate.archive.save_top_k, -1 if epoch == 24 else 0)
        checkpoint = dict(epoch=23, global_step=4464)
        mark_e24_archive(gate, checkpoint, phase)
        self.assertEqual(checkpoint[E100["BOUNDARY_KEY"]], E100["endpoint_marker"](24, recipe))
        self.assertEqual(checkpoint[PHASE_MARKER], phase)
        self.assertEqual(gate.verify_completion(), (24,))
        self.assertEqual(progress.total_visits, 17856)
        self.assertTrue(trainer.should_stop)

    def test_full_epoch_refuses_wrong_padding_even_with_same_length_and_unique_count(self):
        progress = FitProgress("e24", IDS, PADDING)
        for batch, image in enumerate(IDS + (1, 7, 30)):
            progress.consume(epoch=1, batch=batch, image_id=image, retained=0, removed=2)
        with self.assertRaisesRegex(RuntimeError, "padded FIT population"):
            progress.finish_epoch(1, 186, should_stop=False)
        self.assertEqual(progress.completed, 0)

    def test_exact_q100_witness_binds_complete_tensor_and_recipe(self):
        recipe = "a" * 64
        witness = dict(online_tensor_sha256=Q100_SHA, state_tensors=545, num_queries=100, num_select=100,
                       query_groups=13, training_queries_per_image=1300, class_head_slots=2,
                       initialization_seed=5201, coco_asset_sha256=COCO_SHA, coco_asset_bytes=134545398,
                       recipe_sha256=recipe)
        check_start_witness(witness, Q100_SHA, recipe)
        for field, value in (("num_queries", 164), ("online_tensor_sha256", "b" * 64),
                             ("recipe_sha256", "b" * 64), ("state_tensors", 544)):
            with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, "Q100 start witness"):
                check_start_witness(dict(witness, **{field: value}), Q100_SHA, recipe)

    def test_wrong_start_request_is_rejected_before_imports_or_output(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            output = root / "fresh"
            with self.assertRaisesRegex(RuntimeError, "owner-admitted SSDD stage"):
                run_fit_arm(stage="qualification", mode="center", output=output, fit_layout=root / "fit",
                            coco_path=root / "weights", expected_fit_ids=IDS,
                            expected_initial_sha256="a" * 64, protocol="b" * 40, recipe_sha256="c" * 64,
                            source_manifest_sha256="d" * 64, dependencies=None)
            self.assertFalse(output.exists())

    def test_shared_original_filter_removes_only_empty_mask_rows(self):
        import torch

        repair = runpy.run_path(str(Path(__file__).parent.parent /
                                   "HRSID_NATIVE_EMPTY_MASK_FILTER_20261109/filter_empty_masks.py"))
        image = SimpleNamespace(tensors=SimpleNamespace(
            shape=(1, 3, 576, 576), dtype=torch.float32, device=SimpleNamespace(type="cuda")))
        masks = torch.zeros(2, 576, 576, dtype=torch.bool)
        masks[1, 12, 21] = True
        target = dict(labels=torch.zeros(2, dtype=torch.int64), masks=masks,
                      boxes=torch.tensor([[.1, .1, .1, .1], [.2, .2, .1, .1]]), image_id=torch.tensor(6))
        iid, retained, removed, record = retained_target(
            torch, image, (target,), set(IDS), repair["drop_empty_augmented_masks"], None)
        self.assertEqual((iid, retained, removed, record), (6, 1, 1, None))
        self.assertEqual(target["labels"].numel(), 1)
        self.assertTrue(torch.equal(target["masks"], masks[1:]))
        self.assertEqual(target["boxes"].shape, (1, 4))


if __name__ == "__main__":
    unittest.main()

