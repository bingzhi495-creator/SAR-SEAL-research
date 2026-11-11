import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from pycocotools import mask

from dfine_dev705 import (ACCEPTED_E24, FIT_MAP_SHA, FIT_ORDER_SHA, TRAIN_PROTOCOL, admitted_e24,
                          encoded, pack_candidates, run_score, validate_rows)


class DfineDevExportTest(unittest.TestCase):
    def test_original_order_preserves_empty_masks_and_excludes_coco_boxes(self):
        masks = np.zeros((2, 8, 8), dtype=np.uint8)
        masks[1, 1:3, 1:3] = 1
        labels = np.array([0, 0], dtype=np.int64)
        scores = np.array([.9, .8], dtype=np.float32)
        boxes = np.array([[0, 0, 8, 8], [1, 1, 6, 6]], dtype=np.float32)
        predictions, audit = pack_candidates(7, (8,8), labels, scores, boxes, masks)
        self.assertEqual([r["score"] for r in predictions["coco"]], scores.tolist())
        self.assertEqual([r["native_rank"] for r in audit["candidates"]], [1,2])
        self.assertEqual([int(mask.area(r["segmentation"])) for r in predictions["coco"]], [0,4])
        self.assertTrue(all("bbox" not in row for row in predictions["coco"]))
        self.assertEqual(audit["candidates"][0]["box_xyxy"], [0,0,8,8])
        self.assertEqual(validate_rows(7,(8,8),predictions,audit), (2,1))
        audit["candidates"][1]["score"] = .7
        with self.assertRaisesRegex(ValueError, "audit"):
            validate_rows(7,(8,8),predictions,audit)

    def test_zero_candidates_has_complete_image_witness(self):
        pred, audit = pack_candidates(7,(8,8), np.empty(0,np.int64), np.empty(0,np.float32),
                                      np.empty((0,4),np.float32), None)
        self.assertEqual(validate_rows(7,(8,8),pred,audit), (0,0))
        self.assertEqual(pred["image_id"], 7)


    def test_independent_e24_receipt_preserves_budget_and_role(self):
        receipt = dict(status=ACCEPTED_E24, protocol_sha256=TRAIN_PROTOCOL, fit_order_sha256=FIT_ORDER_SHA,
                       fit_id_map_sha256=FIT_MAP_SHA,FIT=True, DEV=False, TEST=False, AP=False,
                       original_exit=0, job_drained=True, completed_epochs=24, fit_visits=70560,
                       optimizer_updates=17640, ema_updates=17640, scheduler_updates=17640,
                       ema_checkpoint_bytes=48119167)
        for k in ("worker_source_sha256","result_sha256","original_tool_wait_sha256",
                  "worker_terminal_sha256","ema_checkpoint_sha256"):
            receipt[k] = "a"*64
        self.assertEqual(admitted_e24(receipt), receipt)
        for wrong in (dict(receipt, completed_epochs=23),dict(receipt, DEV=True),
                      dict(receipt, optimizer_updates=17639),dict(receipt, job_drained=False)):
            with self.assertRaisesRegex(ValueError, "independent FIT-only E24"):
                admitted_e24(wrong)

    def test_scoring_rejects_missing_independent_export_before_manifest_or_labels(self):
        args = SimpleNamespace(export_acceptance="unused", export_acceptance_sha="a"*64)
        with patch("dfine_dev705.cpu_runtime"), patch("dfine_dev705.export_receipt",return_value={}), \
             patch("dfine_dev705.pinned",return_value=encoded(dict(status="INCOMPLETE"))), \
             patch("dfine_dev705.read_manifest") as manifest:
            with self.assertRaisesRegex(ValueError, "independently closed and label-free"):
                run_score(args)
            manifest.assert_not_called()


if __name__ == "__main__":
    unittest.main()

