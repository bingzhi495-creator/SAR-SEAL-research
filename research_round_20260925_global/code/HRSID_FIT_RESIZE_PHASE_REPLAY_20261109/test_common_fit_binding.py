import io
import json
import random
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from common_fit_binding import ConsumedFitLedger, bind_common_observer, common_fit_dataset


def fetched_row(image_id, token, *, common=1):
    return dict(
        image_id=image_id, raw_rows=3, native_present=3, center_present=3,
        native_nonempty=2, center_nonempty=2, common_nonempty=common,
        native_only_nonempty=2 - common, center_only_nonempty=2 - common,
        common_changed_masks=common, crop=True, flipped=False,
        image_sha256=token * 64, trace_sha256="1" * 64, rng_before_sha256="2" * 64,
        rng_after_sha256="3" * 64, common_row_ids_sha256="4" * 64,
        common_nonmask_sha256="5" * 64, native_common_masks_sha256="6" * 64,
        center_common_masks_sha256="7" * 64,
    )


class ConsumedFitLedgerTests(unittest.TestCase):
    def test_fifo_counts_only_consumed_views_and_records_unconsumed_prefetch(self):
        output = io.StringIO()
        ledger = ConsumedFitLedger(output)
        ledger.fetched(fetched_row(3, "a"))
        ledger.fetched(fetched_row(3, "b", common=0))
        ledger.consume(image_id=3, common_count=1, epoch=1, batch=1)
        ledger.fetched(fetched_row(4, "c"))
        ledger.consume(image_id=3, common_count=0, epoch=1, batch=2)
        report = ledger.finish(2)
        rows = [json.loads(row) for row in output.getvalue().splitlines()]
        self.assertEqual([row["image_sha256"] for row in rows], ["a" * 64, "b" * 64])
        self.assertEqual([row["ordinal"] for row in rows], [1, 2])
        self.assertEqual((report["consumed_visits"], report["fetched_visits"], report["prefetched_unconsumed"]), (2, 3, 1))
        self.assertEqual((report["common_targets_total"], report["common_empty_visits"], report["changed_mask_visits"]), (1, 1, 1))
        self.assertEqual(len(report["unconsumed_prefetch_sha256"]), 64)


    def test_mismatch_is_rejected_before_native_callback_or_consumption(self):
        output = io.StringIO()
        ledger = ConsumedFitLedger(output)
        ledger.fetched(fetched_row(8, "a"))
        calls = []
        observer = SimpleNamespace(on_train_batch_start=lambda *args: calls.append(args))
        bind_common_observer(observer, ledger)
        active = SimpleNamespace(current_epoch=0)
        target = {"image_id": torch.tensor([9]), "labels": torch.tensor([0])}
        with self.assertRaisesRegex(RuntimeError, "next prepared visit"):
            observer.on_train_batch_start(active, object(), (None, [target]), 0)
        self.assertFalse(calls)
        self.assertEqual(output.getvalue(), "")
        target["image_id"] = torch.tensor([8])
        observer.on_train_batch_start(active, object(), (None, [target]), 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(ledger.finish(1)["common_targets_total"], 1)

    def test_native_guard_removal_is_not_a_completed_common_visit(self):
        output = io.StringIO()
        ledger = ConsumedFitLedger(output)
        ledger.fetched(fetched_row(8, "a"))
        target = {"image_id": torch.tensor([8]), "labels": torch.tensor([0])}

        def drop(_active, _module, batch, _idx):
            batch[1][0]["labels"] = torch.empty(0, dtype=torch.long)

        observer = SimpleNamespace(on_train_batch_start=drop)
        bind_common_observer(observer, ledger)
        with self.assertRaisesRegex(RuntimeError, "already be nonempty"):
            observer.on_train_batch_start(SimpleNamespace(current_epoch=0), object(), (None, [target]), 0)
        self.assertEqual(output.getvalue(), "")

    def test_non_fit_split_is_refused_before_original_dataset_builder(self):
        calls = []

        def build(*args, **kwargs):
            calls.append((args, kwargs))
            raise AssertionError("No original build allowed here")

        module = SimpleNamespace(build_dataset=build)
        with common_fit_dataset(module, object, Path.cwd(), 1, object(), "native",
                                ConsumedFitLedger(io.StringIO())):
            with self.assertRaisesRegex(RuntimeError, "FIT dataset"):
                module.build_dataset("val")
        self.assertIs(module.build_dataset, build)
        self.assertEqual(calls, [])


    def test_stock_fit_builder_delivers_the_selected_common_target(self):
        from PIL import Image
        from rfdetr.datasets import _torchvision as native, coco
        from rfdetr.training import module_data

        from common_mask_supervision import ROW_ID
        from paired_fit_transform import PairedFitTransform

        with TemporaryDirectory() as temp:
            dataset_root = Path(temp)
            train_root = dataset_root / "train"
            train_root.mkdir()
            Image.new("RGB", (19, 13), (13, 17, 29)).save(train_root / "ship.png")
            annotation = {
                "images": [{"id": 7, "file_name": "ship.png", "width": 19, "height": 13}],
                "categories": [{"id": 0, "name": "ship"}],
                "annotations": [
                    {"id": 1, "image_id": 7, "category_id": 0, "bbox": [5, 4, 7, 4],
                     "segmentation": [[5, 4, 12, 4, 12, 8, 5, 8]], "area": 28, "iscrowd": 0},
                ],
            }
            (train_root / "_annotations.coco.json").write_text(json.dumps(annotation), encoding="utf8")
            config = SimpleNamespace(
                dataset_file="roboflow", dataset_dir=dataset_root, segmentation_head=True,
                square_resize_div_64=True, multi_scale=False, expanded_scales=False,
                patch_size=16, num_windows=2, augmentation_backend="torchvision", scale_jitter=True,
            )
            original = module_data.build_dataset
            images, targets, ledgers = [], [], []
            for mode in ("native", "center"):
                output = io.StringIO()
                ledger = ConsumedFitLedger(output)
                random.seed(5219)
                np.random.seed(5219)
                torch.random.default_generator.manual_seed(5219)
                with common_fit_dataset(module_data, coco.CocoDetection, train_root.resolve(), 1,
                                        native, mode, ledger):
                    dataset = module_data.build_dataset("train", args=config, resolution=576)
                    image, target = dataset[0]
                    self.assertNotIn(ROW_ID, target)
                    self.assertEqual(target["labels"].tolist(), [0])
                    observer = SimpleNamespace(on_train_batch_start=lambda *args: None)
                    bind_common_observer(observer, ledger)
                    observer.on_train_batch_start(SimpleNamespace(current_epoch=0), None,
                                                  (None, [target]), 0)
                    with self.assertRaisesRegex(RuntimeError, "Only one original FIT dataset"):
                        module_data.build_dataset("train", args=config, resolution=576)
                self.assertIs(module_data.build_dataset, original)
                self.assertNotIsInstance(dataset._transforms, PairedFitTransform)
                self.assertEqual(ledger.finish(1)["changed_mask_visits"], 1)
                images.append(image)
                targets.append(target)
                ledgers.append(output.getvalue())
            self.assertTrue(torch.equal(*images))
            self.assertEqual(*ledgers)
            self.assertFalse(torch.equal(targets[0]["masks"], targets[1]["masks"]))
            self.assertTrue(torch.equal(targets[0]["boxes"], targets[1]["boxes"]))


if __name__ == "__main__":
    unittest.main()
