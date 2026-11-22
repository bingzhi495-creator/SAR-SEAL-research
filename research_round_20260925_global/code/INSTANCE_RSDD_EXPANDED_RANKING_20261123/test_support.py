"""Synthetic 512-pixel native feature, tie-policy and fixed-count regressions."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CODE = Path(__file__).resolve().parents[1]
if __name__ == "__main__" and len(sys.argv) > 1 and Path(sys.argv[1]).is_dir():
    CODE = Path(sys.argv.pop(1)).resolve()
support = load(Path(__file__).with_name("support.py"))


class ExpandedSupport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        old = CODE / "DFINE_SEG_EXTERNAL_BASELINE_FIT32_E24_20261112/e24"
        canvas = load(CODE / "DFINE_HRSID_TO_SSDD_POINT_TRANSFER_20261120/native_points.py")
        pair, cls.rank = canvas.load_old(old / "fit109_paired_point/pair_math.py",
                                         old / "fit109_probability_rank/rank_math.py")
        cls.pair, _ = canvas.make_native(pair, cls.rank, 512, 512)
        cls.sampler = SimpleNamespace(editable_pools=cls.rank.editable_pools,
                                      sample_probability=lambda plane, idx, width: plane.ravel()[idx])
        cls.free = load(old / "fit109_free_pairing/free_pairing.py")
        cls.trainer = load(old / "fit109_paired_point/train_point.py")
        cls.fit_step = load(CODE / "RFDETR_CORE36_MATCHED_POINT_SPOT_20261120/fit_step.py")
        rng = np.random.default_rng(5217)
        cls.stock = np.zeros((512, 512), dtype=bool)
        cls.stock[483:499, 479:512] = True
        cls.stock[489:496, 494:498] = False
        cls.box = np.array([474.125, 481.25, 512, 503.75], dtype=np.float32)
        cls.area, cls.score = int(cls.stock.sum()), .8181872824430453
        probs = rng.integers(0, 16, size=cls.stock.shape).astype(np.float32) / 64
        cls.prob = np.where(cls.stock, .5+probs, .5-probs)
        cls.image = pair.image_features(rng.integers(0, 256, size=(*cls.stock.shape, 3), dtype=np.uint8))
        ref = cls.pair.paired_points(cls.stock, cls.box, cls.area, cls.score, cls.prob, cls.image, cls.sampler)
        cls.native, cls.legacy = ref[:2]
        cls.coords, cls.features = support.expanded_features(cls.rank, cls.stock, cls.box, cls.area, cls.score,
                                                            cls.prob, cls.image, cls.native)

    def test_original_features_and_complete_two_layer_coverage(self):
        n = len(self.native)
        self.assertEqual(n, 32)
        self.assertNotEqual(len(self.coords[0]), len(self.coords[1]))
        reference_pools, _ = self.rank.editable_pools(self.stock, self.box)
        for side, name in enumerate(("add", "drop")):
            indices, features = self.coords[side], self.features[side]
            np.testing.assert_array_equal(indices[:n], self.native[:, side])
            self.assertEqual(features[:n].tobytes(), self.legacy[:, side].tobytes())
            self.assertGreater(len(indices), n)
            self.assertTrue(np.all(np.diff(indices[n:]) > 0))
            for layer in (1, 2):
                np.testing.assert_array_equal(np.sort(indices[features[:, 2] == layer]),
                                              np.sort(reference_pools[name + str(layer)][0]))
        # Dropping a native point makes the claimed prefix ineligible.
        with self.assertRaisesRegex(ValueError, "prefix length"):
            support.expanded_features(self.rank, self.stock, self.box, self.area, self.score,
                                      self.prob, self.image, self.native[:-1])

    def test_original_head_output_and_fixed_count_actions(self):
        torch.manual_seed(5217)
        model = self.trainer.PointHead(14).eval()
        with torch.no_grad():
            model.network[-1].weight.normal_(0, .04)
        center = self.legacy.mean(axis=(0, 1))
        scale = self.legacy.std(axis=(0, 1)) + .1
        calls = []

        def predict(batch):
            calls.append(batch.shape)
            return self.fit_step.predict(model, batch, center, scale)

        qa, qd = support.predict_sides(predict, self.features)
        n = len(self.native)
        self.assertEqual(calls, [(n, 2, 14), (len(qa)+len(qd)-2*n, 1, 14)])
        reference = predict(self.legacy)
        self.assertEqual(np.column_stack((qa[:n], qd[:n])).tobytes(), reference.tobytes())
        for k in (0, 1, 8):
            actions = self.free.choose_free(reference, 8, fixed_count=k)
            selected = support.select(tuple(self.native.T), (qa[:n], qd[:n]), k, tie="legacy")
            for side in (0, 1):
                expected = [self.native[a[side], side] for a in actions]
                np.testing.assert_array_equal(selected[side], expected)
            adds, drops = support.select(self.coords, (qa, qd), k)
            self.assertEqual((len(adds), len(drops)), (k, k))
            mask = self.stock.copy()
            mask.flat[adds], mask.flat[drops] = True, False
            self.assertEqual(int(mask.sum()), self.area)
            self.assertEqual(int((mask ^ self.stock).sum()), 2*k)

    def test_tied_scores_keep_legacy_rows_or_native_pixel_order(self):
        scores = np.full(self.native.shape, .5, np.float32)
        k = 8
        actions = self.free.choose_free(scores, k, fixed_count=k)
        legacy = support.select(tuple(self.native.T), tuple(scores.T), k, tie="legacy")
        natural = support.select(tuple(self.native.T), tuple(scores.T), k)
        for side in (0, 1):
            np.testing.assert_array_equal(legacy[side], [self.native[a[side], side] for a in actions])
            np.testing.assert_array_equal(natural[side], np.sort(self.native[:, side])[:k])
        self.assertFalse(np.array_equal(legacy[1], natural[1]))

    def test_one_empty_side_has_no_native_prefix_and_zero_budget(self):
        box = np.array([479, 483, 480, 484], dtype=np.float32)
        native = np.empty((0, 2), np.int32)
        coords, features = support.expanded_features(self.rank, self.stock, box, self.area, self.score,
                                                     self.prob, self.image, native)
        self.assertEqual(tuple(map(len, coords)), (0, 1))
        scores = support.predict_sides(lambda batch: batch[:, :, 0], features)
        self.assertEqual(tuple(map(len, scores)), (0, 1))
        self.assertEqual(tuple(map(len, support.select(coords, scores, 0))), (0, 0))
        with self.assertRaisesRegex(ValueError, "swap count"):
            support.select(coords, scores, 1)


if __name__ == "__main__":
    unittest.main()
