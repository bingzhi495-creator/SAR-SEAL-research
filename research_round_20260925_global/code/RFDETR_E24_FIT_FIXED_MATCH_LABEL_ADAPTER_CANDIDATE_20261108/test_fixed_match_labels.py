"""Synthetic official-COCO contract tests. No project data."""
from copy import deepcopy
import unittest

import numpy as np

from fixed_match_labels import build_fixed_labels
from pycocotools import mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def synthetic_coco(annotations):
    coco = COCO()
    coco.dataset = {
        "images": [{"id": 10, "height": 4, "width": 8}],
        "annotations": annotations,
        "categories": [{"id": 1, "name": "ship"}],
    }
    coco.createIndex()
    return coco


def capture(predictions):
    rows = {
        "image_id": 10, "image_sha256": "a" * 64, "gray_sha256": "b" * 64,
        "original_hw": [4, 8], "records": [],
    }
    offsets, indices, logits = [0], [], []
    for i, prediction in enumerate(predictions):
        plane = np.zeros((4, 8), dtype=np.uint8)
        plane[1, prediction] = 1
        rle = mask.encode(np.asfortranarray(plane))
        rows["records"].append({
            "native_rank": 2 * i + 1, "query_id": i + 4,
            "score": 0.75 if i < 2 else 0.5,
            "segmentation": {"size": rle["size"], "counts": rle["counts"].decode("ascii")},
        })
        for y, x in np.ndindex(plane.shape):
            window = plane[max(0, y - 2):y + 3, max(0, x - 2):x + 3]
            if np.min(window) != np.max(window):
                indices.append(y * plane.shape[1] + x)
                logits.append(1.0 if plane[y, x] else -1.0)
        offsets.append(len(indices))
    arrays = {
        "offsets": np.array(offsets, dtype=np.int64),
        "positions": np.array(indices, dtype=np.uint32),
        "logits": np.array(logits, dtype=np.float32),
    }
    return rows, arrays


def annotation(crowd=0):
    return {"id": 100, "image_id": 10, "category_id": 1, "area": 3,
            "iscrowd": crowd, "bbox": [2, 1, 3, 1],
            "segmentation": [[2, 1, 5, 1, 5, 2, 2, 2]]}


class MatchLabelTest(unittest.TestCase):
    def build(self, ground_truth, rows, arrays):
        called = []

        def witness(np_api, codec, actual_rows, actual_arrays):
            self.assertIs(np_api, np)
            self.assertIs(codec, mask)
            self.assertIs(actual_rows, rows)
            self.assertIs(actual_arrays, arrays)
            called.append(True)

        result = build_fixed_labels(np, mask, COCOeval, integer_coco=ground_truth,
                                    rows_document=rows, arrays=arrays, verify_capture=witness)
        self.assertEqual(called, [True])
        return result

    def test_tied_first_native_detection_matches_at_half_iou_and_labels_both_flip_signs(self):
        ground_truth = synthetic_coco([annotation()])
        original = deepcopy(ground_truth.dataset)
        rows, arrays = capture([[2, 3, 5], [2, 3, 5], []])
        captured = deepcopy(rows)
        result = self.build(ground_truth, rows, arrays)
        self.assertEqual(ground_truth.dataset, original)
        self.assertEqual(rows, captured)
        match, duplicate, empty = result["rows"]
        self.assertEqual([r["status"] for r in (match, duplicate, empty)],
                         ["matched", "unmatched", "unmatched"])
        self.assertEqual((match["native_rank"], match["query_id"], match["score"], match["GT_annotation_id"]),
                         (1, 4, 0.75, 100))
        self.assertEqual(match["flip_correct"].dtype, np.bool_)
        self.assertEqual(arrays["positions"][match["start"]:match["stop"]][match["flip_correct"]].tolist(),
                         [12, 13])
        self.assertIsNone(duplicate["flip_correct"])
        self.assertIsNone(empty["flip_correct"])
        self.assertTrue(empty["mask_empty"])
        self.assertEqual(empty["start"], empty["stop"])


    def test_crowd_matches_remain_ignored_and_unlabeled(self):
        ground_truth = synthetic_coco([annotation(crowd=1)])
        rows, arrays = capture([[2, 3, 5], [2, 3, 5], []])
        result = self.build(ground_truth, rows, arrays)
        self.assertEqual([(r["status"], r["GT_annotation_id"]) for r in result["rows"]],
                         [("ignored", 100), ("ignored", 100), ("unmatched", None)])
        self.assertTrue(all(r["flip_correct"] is None for r in result["rows"]))
        self.assertEqual(result["summary"]["ignored_GT"], 1)
        self.assertEqual(result["summary"]["ignored"], 2)

    def test_empty_detections_and_empty_ground_truth_are_separate_cases(self):
        for gt_present, dt_present in ((False, False), (True, False), (False, True)):
            with self.subTest(GT=gt_present, detections=dt_present):
                ground_truth = synthetic_coco([annotation()] if gt_present else [])
                rows, arrays = capture([[2, 3, 5]] if dt_present else [])
                result = self.build(ground_truth, rows, arrays)
                self.assertEqual(result["summary"]["unmatched_GT"], int(gt_present))
                self.assertEqual(result["summary"]["unmatched"], int(dt_present))
                self.assertEqual(len(result["rows"]), int(dt_present))
                self.assertTrue(all(r["flip_correct"] is None for r in result["rows"]))

    def test_matched_full_mask_stays_matched_with_no_labels_on_empty_support(self):
        ann = annotation()
        ann.update(area=32, bbox=[0, 0, 8, 4], segmentation=[[0, 0, 8, 0, 8, 4, 0, 4]])
        ground_truth = synthetic_coco([ann])
        rows, arrays = capture([[]])
        rle = mask.encode(np.ones((4, 8), dtype=np.uint8, order="F"))
        rows["records"][0]["segmentation"] = {"size": rle["size"], "counts": rle["counts"].decode("ascii")}
        result = self.build(ground_truth, rows, arrays)
        row = result["rows"][0]
        self.assertEqual(row["status"], "matched")
        self.assertFalse(row["mask_empty"])
        self.assertEqual(row["flip_correct"].size, 0)
        self.assertEqual(result["summary"]["empty_bands"], 1)


if __name__ == "__main__":
    unittest.main()
