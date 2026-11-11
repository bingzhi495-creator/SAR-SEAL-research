from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from srnet_dev187_export import (
    AUTHOR_COMMIT, FIT_INDEX_SHA, STOCK_TEST, TRAIN_SOURCE_SHA, TRAIN_STATUS, WAIT_STATUS,
    export_test_config, make_image_only_index, validate_closed_training, write_export,
)


class ExportTests(unittest.TestCase):
    def test_image_only_index_preserves_logical_aliases_and_order(self):
        manifest = dict(expected_ids=[701, 109], items=[
            dict(image_id=701, original_hw=[12, 19],
                 image_ref=dict(path="/old/valid/a.jpg", bytes=10, sha256="a"*64)),
            dict(image_id=109, original_hw=[12, 19],
                 image_ref=dict(path="/old/valid/b.jpg", bytes=10, sha256="a"*64)),
        ])
        self.assertEqual(make_image_only_index(manifest), dict(
            images=[dict(id=701, file_name="a.jpg", height=12, width=19),
                    dict(id=109, file_name="b.jpg", height=12, width=19)],
            categories=[dict(id=1, name="ship")], annotations=[]))

    def test_thresholds_differ_in_only_one_author_setting(self):
        stock = deepcopy(STOCK_TEST)
        primary = export_test_config(stock, "stock")
        sensitivity = export_test_config(stock, "low_threshold")
        self.assertEqual(primary, stock)
        self.assertEqual(sensitivity, dict(stock, score_thr=.001))
        self.assertEqual(stock["score_thr"], .3)
        with self.assertRaisesRegex(ValueError, "author test settings"):
            export_test_config(dict(stock, max_per_img=100), "stock")

    def test_training_eligibility_binds_original_wait_clocks_and_report(self):
        checkpoint = Path("synthetic") / "epoch_24.pth"
        digest = "a"*64
        report_sha, stdout_sha = "b"*64, "c"*64
        report = dict(status=TRAIN_STATUS, stage="e24", worker_source_sha256=TRAIN_SOURCE_SHA,
                      author_commit=AUTHOR_COMMIT, fit_index_sha256=FIT_INDEX_SHA,
                      actual_batch_size=4, fit_images=741, visits_per_full_epoch=744,
                      completed_epochs=24, optimizer_updates=4464, model_forward_calls=4464,
                      consumed_visits=17856, stock_author_source=True,
                      FIT=True, DEV=False, TEST=False, AP=False,
                      checkpoint=dict(path=str(checkpoint), bytes=17, sha256=digest))
        wait = dict(status=WAIT_STATUS, original_os_wait=True, exit_code=0, job_assigned=True,
                    active_before_drain=0, active_after_wait=0, resource_stop=None,
                    source_sha256=TRAIN_SOURCE_SHA, log_sha256={"ORIGINAL.stdout": stdout_sha})
        stdout = b"prior diagnostic line\n" + json.dumps(dict(status=TRAIN_STATUS,
                                                            result_sha256=report_sha)).encode() + b"\n"
        args = (stdout, report_sha, stdout_sha, checkpoint, digest)
        self.assertEqual(validate_closed_training(report, wait, *args), report["checkpoint"])
        with self.assertRaisesRegex(ValueError, "original P349 OS wait"):
            validate_closed_training(report, dict(wait, active_before_drain=1), *args)
        with self.assertRaisesRegex(ValueError, "training clocks"):
            validate_closed_training(dict(report, optimizer_updates=4463), wait, *args)
        with self.assertRaisesRegex(ValueError, "exact completed training report"):
            validate_closed_training(report, wait, stdout, "d"*64, stdout_sha, checkpoint, digest)

    def test_export_stays_image_grouped_with_empty_image_and_separate_boxes(self):
        items = [dict(image_id=701), dict(image_id=109)]
        native = [([3, 4], ["mask-a", "mask-b"]), ([], [])]

        def map_native(item, boxes, masks):
            predictions = [dict(mask=mask) for mask in masks]
            rotated = [dict(rbox=box) for box in boxes]
            return predictions, rotated

        with TemporaryDirectory() as output:
            predictions, boxes = (Path(output)/name for name in ("predictions.jsonl", "boxes.jsonl"))
            result = write_export(native, items, lambda data, i, item: data[i], map_native, predictions, boxes)
            self.assertEqual([json.loads(line) for line in predictions.read_bytes().splitlines()], [
                dict(image_id=701, predictions=[dict(mask="mask-a"), dict(mask="mask-b")]),
                dict(image_id=109, predictions=[]),
            ])
            self.assertEqual([json.loads(line) for line in boxes.read_bytes().splitlines()], [
                dict(image_id=701, rotated_boxes=[dict(rbox=3), dict(rbox=4)]),
                dict(image_id=109, rotated_boxes=[]),
            ])
            self.assertEqual(result["zero_candidate_images"], 1)
            self.assertEqual(result["predictions_sha256"], sha256(predictions.read_bytes()).hexdigest())
            self.assertEqual(result["rotated_boxes_sha256"], sha256(boxes.read_bytes()).hexdigest())
