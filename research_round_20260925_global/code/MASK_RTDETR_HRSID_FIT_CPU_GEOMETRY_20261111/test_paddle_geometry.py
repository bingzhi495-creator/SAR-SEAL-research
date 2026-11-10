"""Synthetic-only tests for the fixed Paddle geometry operator replay."""
from pathlib import Path
import random
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paddle_geometry import center_masks, load_stock, transform_batch


def fixture(ordinal):
    height, width = 157 + 11 * ordinal, 251 + 7 * ordinal
    y, x = np.mgrid[:height, :width]
    gray = ((x + 3 * y + ordinal) % 190).astype(np.uint8)
    ok, encoded = cv2.imencode(".png", np.repeat(gray[..., None], 3, axis=2))
    if not ok:
        raise RuntimeError("Synthetic PNG encoding failed")
    bounds = [[10., 11., 25., 30.], [90., 63., 120., 91.], [175., 94., 197., 112.]]
    polygons = [[[x0, y0, x1, y0, x1, y1, x0, y1]] for x0, y0, x1, y1 in bounds]
    return dict(image=encoded.tobytes(), gt_bbox=np.array(bounds, np.float32),
                gt_class=np.zeros((3, 1), np.int32), gt_poly=polygons,
                is_crowd=np.zeros((3, 1), np.int32), gt_read_order=np.arange(3))


class ProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv2.setNumThreads(1)
        cls.stock = load_stock(Path(__file__).resolve().parent / "UPSTREAM")

    def test_center_has_explicit_half_open_tie(self):
        row = np.arange(4, dtype=np.uint8)[None, None, :]
        self.assertEqual(center_masks(row, (1, 7)).ravel().tolist(), [0, 0, 1, 2, 2, 3, 3])

    def test_official_crop_filters_rasters_with_boxes_without_polygons(self):
        samples = [fixture(i) for i in range(4)]
        views = transform_batch(self.stock, samples, seed=5204)
        self.assertTrue(any(view.cropped for view in views))
        self.assertTrue(any(view.expanded for view in views))
        self.assertTrue(any(view.cropped and len(view.before_resize["gt_segm"]) == 1 for view in views))
        for view in views:
            self.assertNotIn("gt_poly", view.before_resize)
            self.assertEqual(view.output["image"].shape, (3, *view.center_masks.shape[-2:]))
            self.assertEqual(view.output["gt_segm"].shape, view.center_masks.shape)
            np.testing.assert_array_equal(view.output["gt_read_order"], view.before_resize["gt_read_order"])
            for row, source_id in enumerate(view.before_resize["gt_read_order"]):
                ys, xs = np.nonzero(view.before_resize["gt_segm"][row])
                self.assertTrue(len(xs))
                np.testing.assert_array_equal(
                    [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], view.before_resize["gt_bbox"][row])
                self.assertIn(int(source_id), [0, 1, 2])
        self.assertTrue(all("gt_poly" in sample for sample in samples))

    def test_one_native_resize_and_replay_are_exact(self):
        py_before = random.getstate()
        np_before = np.random.get_state()
        samples = [fixture(i) for i in range(4)]
        first = transform_batch(self.stock, samples, seed=5202)
        second = transform_batch(self.stock, samples, seed=5202)
        self.assertEqual(py_before, random.getstate())
        self.assertEqual(np_before[0], np.random.get_state()[0])
        np.testing.assert_array_equal(np_before[1], np.random.get_state()[1])
        self.assertEqual(np_before[2:], np.random.get_state()[2:])
        self.assertTrue(all("gt_poly" in sample for sample in samples))
        self.assertTrue(any(np.any(view.output["gt_segm"] != view.center_masks) for view in first))
        for a, b in zip(first, second, strict=True):
            h, w = a.before_resize["gt_segm"].shape[-2:]
            oh, ow = a.center_masks.shape[-2:]
            witness = np.stack([
                cv2.resize(mask, None, fx=ow/w, fy=oh/h, interpolation=cv2.INTER_NEAREST)
                for mask in a.before_resize["gt_segm"]
            ])
            np.testing.assert_array_equal(a.output["gt_segm"], witness)
            for key in ("image", "gt_bbox", "gt_class", "gt_read_order", "gt_segm"):
                np.testing.assert_array_equal(a.output[key], b.output[key])
            np.testing.assert_array_equal(a.center_masks, b.center_masks)

    def test_rejects_nonstock_batch_and_external_input_reference(self):
        with self.assertRaises(ValueError):
            transform_batch(self.stock, [fixture(0)], seed=1)
        samples = [fixture(i) for i in range(4)]
        samples[0]["image"] = "external"
        with self.assertRaises(ValueError):
            transform_batch(self.stock, samples, seed=1)


if __name__ == "__main__":
    unittest.main()
