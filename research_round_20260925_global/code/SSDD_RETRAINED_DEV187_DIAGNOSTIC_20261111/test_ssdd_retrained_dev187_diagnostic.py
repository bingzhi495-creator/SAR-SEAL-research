"""Closed-input admission and score decomposition regression fixtures."""

from contextlib import redirect_stdout
from hashlib import sha256
from io import StringIO
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from pycocotools import mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import ssdd_retrained_dev187_diagnostic as diagnostic


def fixture():
    images = [dict(id=image, width=12, height=12) for image in range(1, 188)]
    annotations, native, center = [], [], []
    for image in images:
        positions = [(1, 1), (5, 1)]
        if image["id"] <= 6:
            positions.append((1, 6))
        for y, x in positions:
            bitmap = np.zeros((12, 12), dtype=np.uint8, order="F")
            bitmap[y:y + 3, x:x + 3] = 1
            raw = mask.encode(bitmap)
            rle = dict(raw, counts=raw["counts"].decode("ascii"))
            iid = image["id"]
            annotations.append(dict(id=len(annotations) + 1, image_id=iid, category_id=1, iscrowd=0,
                                    area=9, bbox=[x, y, 3, 3], segmentation=rle))
            row = dict(image_id=iid, category_id=1, score=.90, segmentation=rle)
            native.append(row)
            center.append(dict(row, score=.80))
    gt = dict(info={}, images=images, categories=[dict(id=1, name="ship")], annotations=annotations)
    return gt, native, center


def independent_metrics(gt, rows):
    coco = COCO()
    coco.dataset = gt
    with redirect_stdout(StringIO()):
        coco.createIndex()
        evaluation = COCOeval(coco, coco.loadRes(rows), "segm")
        evaluation.evaluate()
        evaluation.accumulate()
        evaluation.summarize()
    values = evaluation.stats.tolist()
    precision = evaluation.eval["precision"][8, :, :, 0, -1]
    return dict(AP=values[0], AP50=values[1], AP75=values[2], AP90=float(precision[precision > -1].mean()),
                APs=values[3], AR100=values[8], all12=values)


def write_json(path, value, size=None):
    raw = json.dumps(value).encode("utf-8")
    if size is not None:
        raw = raw.ljust(size)
    path.write_bytes(raw)
    return sha256(raw).hexdigest()


class AcceptedPairTest(unittest.TestCase):
    def test_rejects_unaccepted_receipt_before_reading_scientific_inputs(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            receipt = root / "acceptance.json"
            receipt.write_text('{}', encoding="utf-8")
            missing = root / "unreadable.json"
            with self.assertRaisesRegex(ValueError, "Accepted P295 receipt digest differs"):
                diagnostic.diagnose(root, receipt, missing, missing, missing)

    def test_reproduces_official_scores_and_rejects_swapped_original_arms(self):
        code = Path(os.environ["P299_TEST_CODE"]).resolve(strict=True)
        gt, native_rows, center_rows = fixture()
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            gt_path = root / "integer.json"
            gt_sha = write_json(gt_path, gt, 118319)
            prediction_paths, arm_reports, primary = [], {}, {}
            for arm, rows in (("native", native_rows), ("center", center_rows)):
                parent = root / arm
                parent.mkdir()
                path = parent / "PREDICTIONS.json"
                prediction_paths.append(path)
                arm_reports[arm] = dict(prediction_sha256=write_json(path, rows), prediction_count=len(rows),
                                        nonempty_masks=len(rows), all_zero_masks=0)
                primary[arm] = independent_metrics(gt, rows)
            primary["center_minus_native_points"] = {
                name: 0. for name in ("AP", "AP50", "AP75", "AP90", "APs", "AR100")
            }
            receipt = dict(status="P286_SSDD_DEV187_INDEPENDENTLY_ACCEPTED", coordinator_closed=True,
                           coordinator_original_os_wait=True, coordinator_original_os_exit=0,
                           primary_GT="INTEGER_PRIMARY", DEV=True, TEST=False, FIT=False, AP=True, training=False,
                           floating_gt_sha256=None, integer_gt_sha256=gt_sha, images=187, distinct_jpeg=183,
                           annotations=380, arms=arm_reports, primary=primary,
                           training_protocol="fixture", evaluation_protocol="fixture")
            receipt_path = root / "acceptance.json"
            receipt_sha = write_json(receipt_path, receipt)
            with patch.object(diagnostic, "ACCEPTED_SHA256", receipt_sha):
                result = diagnostic.diagnose(code, receipt_path, gt_path, *prediction_paths)
                with self.assertRaisesRegex(ValueError, "Accepted native predictions digest differs"):
                    diagnostic.diagnose(code, receipt_path, gt_path, *reversed(prediction_paths))
            self.assertTrue(result["full_primary_scores_exact"])
            self.assertEqual(result["primary"], primary)
            self.assertEqual(result["equal_nonempty_candidate_analysis"]["prefix_counts"]["retained_per_arm"], 380)
            self.assertEqual(result["fixed_official_iou50_common_gt_geometry"]["strata"]["all"]["paired"]
                             ["center"]["mean_iou"], 1.)


if __name__ == "__main__":
    unittest.main()
