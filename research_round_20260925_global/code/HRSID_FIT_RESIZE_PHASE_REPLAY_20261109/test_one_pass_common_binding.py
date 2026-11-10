from hashlib import sha256
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch
from rfdetr.datasets import _torchvision as native

from common_fit_binding import encode
from one_pass_common_binding import ConsumedFitLedger, bind_common_observer, common_fit_dataset
from one_pass_common_pair_transform import OnePassCommonPairTransform
from paired_fit_transform import _training_mask_digest
from test_paired_fit_transform import compose, fixture, set_seed


def crop(image, target):
    return native.crop(image, target, top=1, left=1, height=2, width=2)


def changed_fixture():
    image, raw = fixture()
    raw["masks"][0].zero_()
    raw["masks"][0, 3:6, 3:6] = True
    raw["boxes"][0] = torch.tensor([3., 3., 6., 6.])
    raw["area"][0] = 9
    return image, raw


class OnePassCommonBindingTest(unittest.TestCase):
    def test_actual_selected_digests_and_crop_counts_close_per_arm(self):
        image, raw = changed_fixture()
        stock = (
            compose(native.Resize((5, 8)), native.RandomHorizontalFlip(p=.5)),
            compose(native.Resize((4, 4)), crop, native.Resize((3, 3))),
        )
        shared_streams = []
        selected_digests = []
        for mode in ("stepwise_center", "one_pass"):
            with self.subTest(mode=mode):
                set_seed()
                stream = StringIO()
                ledger = ConsumedFitLedger(stream)
                ledger.bind_mode(mode)
                native_visits = []
                observer = SimpleNamespace(on_train_batch_start=lambda *args: native_visits.append(args[-1]))
                bind_common_observer(observer, ledger)
                selected = sha256()
                for batch, augmentation in enumerate(stock, 1):
                    transform = OnePassCommonPairTransform(native, augmentation, mode, audit=ledger.fetched)
                    augmented, target = transform(image, raw)
                    observer.on_train_batch_start(SimpleNamespace(current_epoch=0), None, (augmented, [target]), batch - 1)
                    selected.update(encode(dict(ordinal=batch, image_id=31, epoch=1, batch=batch,
                                                mask_sha256=_training_mask_digest(target["masks"]))).encode("utf-8"))
                result = ledger.finish(2)
                self.assertEqual(native_visits, [0, 1])
                self.assertEqual(result["verified_selected_mask_visits"], 2)
                self.assertEqual(result["verified_selected_mask_stream_sha256"], selected.hexdigest())
                self.assertEqual(result["consumed_direct_visits"], 1)
                self.assertEqual(result["consumed_crop_visits"], 1)
                self.assertEqual(result["changed_crop_visits"], 1)
                shared_streams.append(stream.getvalue())
                selected_digests.append(selected.hexdigest())
        self.assertEqual(shared_streams[0], shared_streams[1])
        self.assertNotEqual(selected_digests[0], selected_digests[1])



    def test_wrong_arm_bytes_fail_before_native_and_guard_mutation_fails_after(self):
        image, raw = changed_fixture()
        stock = compose(native.Resize((4, 4)), crop, native.Resize((3, 3)))
        for fault in ("before", "after"):
            with self.subTest(fault=fault):
                stream = StringIO()
                ledger = ConsumedFitLedger(stream)
                ledger.bind_mode("stepwise_center")
                stepwise = OnePassCommonPairTransform(native, stock, "stepwise_center", audit=ledger.fetched)
                augmented, target = stepwise(image, raw)
                if fault == "before":
                    _, target = OnePassCommonPairTransform(native, stock, "one_pass")(image, raw)

                visited = []

                def guard(_trainer, _model, batch, _index):
                    visited.append(True)
                    masks = batch[1][0]["masks"]
                    masks[0, 0, 0] = ~masks[0, 0, 0]

                observer = SimpleNamespace(on_train_batch_start=guard)
                bind_common_observer(observer, ledger)
                reason = ("Actual selected FIT mask digest" if fault == "before"
                          else "Native FIT guard changed the selected target masks")
                with self.assertRaisesRegex(RuntimeError, reason):
                    observer.on_train_batch_start(SimpleNamespace(current_epoch=0), None, (augmented, [target]), 0)
                self.assertEqual(visited, [] if fault == "before" else [True])
                self.assertEqual(ledger.consumed_visits, 0)
                self.assertEqual(stream.getvalue(), "")

    def test_direct_cannot_hide_a_changed_crop_and_b_rows_cannot_diverge(self):
        image, raw = changed_fixture()
        stock = compose(native.Resize((4, 4)), crop, native.Resize((3, 3)))
        records = []
        OnePassCommonPairTransform(native, stock, "stepwise_center", audit=records.append)(image, raw)
        ledger = ConsumedFitLedger(StringIO())
        ledger.bind_mode("one_pass")
        for row, reason in (
            (dict(records[0], crop=False), "Direct FIT visits must have identical masks and support"),
            (dict(records[0], center_present=records[0]["native_present"] + 1), "Fixed B crop rows"),
        ):
            with self.assertRaisesRegex(RuntimeError, reason):
                ledger.fetched(row)
        self.assertEqual(ledger.fetched_visits, 0)

    def test_stock_fit_builder_restores_the_dataset_and_delivers_empty_selected_masks(self):
        image, raw = fixture("empty")
        stock = compose(native.Resize((5, 8)))
        with TemporaryDirectory() as root:
            fit_root = Path(root).resolve()

            class Detection:
                def __init__(self):
                    self.root = fit_root
                    self.ids = [31]
                    self.include_masks = True
                    self.include_keypoints = False
                    self._transforms = stock

                def __len__(self):
                    return len(self.ids)

            dataset = Detection()
            original_build = lambda image_set: dataset
            module_data = SimpleNamespace(build_dataset=original_build)
            stream = StringIO()
            ledger = ConsumedFitLedger(stream)
            observer = SimpleNamespace(on_train_batch_start=lambda *_args: None)
            bind_common_observer(observer, ledger)
            with common_fit_dataset(module_data, Detection, fit_root, 1, native, "one_pass", ledger):
                with self.assertRaisesRegex(RuntimeError, "Only one original FIT dataset"):
                    module_data.build_dataset("val")
                self.assertIs(module_data.build_dataset("train"), dataset)
                self.assertIsNot(dataset._transforms, stock)
                augmented, selected = dataset._transforms(image, raw)
                observer.on_train_batch_start(SimpleNamespace(current_epoch=0), None, (augmented, [selected]), 0)
            self.assertIs(module_data.build_dataset, original_build)
            self.assertIs(dataset._transforms, stock)
            self.assertEqual(selected["masks"].shape, (0, 5, 8))
            self.assertEqual(ledger.finish(1)["common_empty_visits"], 1)


if __name__ == "__main__":
    unittest.main()
