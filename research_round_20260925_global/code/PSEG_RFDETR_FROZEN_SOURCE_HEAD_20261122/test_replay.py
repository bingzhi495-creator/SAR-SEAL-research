"""Synthetic-only tests for native rectangular RF features and matched readout."""
import importlib.util
from collections import Counter
from hashlib import sha256
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image
from pycocotools import mask as codec
import torch

source = Path(__file__).with_name("replay.py")
spec = importlib.util.spec_from_file_location("rf_replay", source)
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
CODE = Path(sys.argv.pop(1))


class FrozenRFReplay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.m = r.methods(CODE)
        cls.original = r.read_source(CODE / r.SPOT / "feature_adapter.py",
                                     cls.m.readout.PINS["feature_adapter.py"])

    def test_rectangular_embedding_reproduces_source_feature_schema(self):
        hw = (36, 73)
        rng = np.random.default_rng(11)
        stock = np.zeros(hw, bool)
        stock[8:20, 19:31] = True
        stock[8:12, 23:26] = False
        prob = rng.random(hw, dtype=np.float32)
        rgb = rng.integers(256, size=(*hw, 3), dtype=np.uint8)
        p800 = np.zeros((800, 800), np.float32)
        s800 = np.zeros((800, 800), bool)
        i800 = np.zeros((800, 800, 3), np.uint8)
        p800[:36, :73], s800[:36, :73], i800[:36, :73] = prob, stock, rgb
        box = np.array([11.25, 3., 37., 28.125], dtype=np.float32)
        ref = self.original.extract_candidate(np, self.m.pair, self.m.rank, s800, box, .91, p800,
                                              self.m.pair.image_features(i800))
        pair, sampler = r.canvas(self.m, hw)
        coords, features, layers, _, _ = pair.paired_points(stock, box, int(stock.sum()), .91, prob,
                                                            self.m.pair.image_features(rgb), sampler)
        for actual, expected in zip(np.divmod(coords, hw[1]), np.divmod(ref["coordinates"], 800)):
            np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(features, ref["features"])
        np.testing.assert_array_equal(layers, ref["layers"])
        q = np.clip(features[:, :, 9] * .7 + features[:, :, 0] * .3, 0, 1)
        decisions = r.decisions(self.m, stock, coords, features, lambda _: q)
        original = self.original.readouts_sparse(np, stock, coords, q, q, features[:, :, 0], 8)
        for arm, source_arm in (("fixed", "SF"), ("free_matched_count", "SU=F")):
            chosen = np.array([v for a, d in decisions[arm] for v in (coords[a, 0], coords[d, 1])])
            np.testing.assert_array_equal(chosen, original[source_arm]["flipped"])

    def test_native_box_pixels_clip_at_actual_borders(self):
        pair, _ = r.canvas(self.m, (37, 81))
        self.assertEqual(pair.box_pixels([-4.5, -1, 100, 800]), 37 * 81)

    def test_complete_fixture_image_and_wrong_count(self):
        hw = (42, 75)
        plane = np.full((144, 144), -2, dtype=np.float32)
        plane[18:70, 24:65] = 2
        plane[42:51, 35:47] = -2
        plane[95:99, 93:98] = 2
        values = torch.nn.functional.interpolate(torch.from_numpy(plane)[None, None], size=hw,
                                                mode="bilinear", align_corners=False)[0, 0].numpy()
        stock = values > 0
        native = codec.encode(np.asfortranarray(stock, dtype=np.uint8))
        native["counts"] = native["counts"].decode()
        rgb = np.random.default_rng(91).integers(256, size=(*hw, 3), dtype=np.uint8)
        with TemporaryDirectory() as name:
            root = Path(name)
            Image.fromarray(rgb).save(root / "example.png")
            np.save(root / "logits.npy", plane[None])
            row = dict(image_id=5, source_image_name="example.png", original_hw=list(hw),
                       rgb_sha256=sha256(rgb.tobytes()).hexdigest(),
                       image_sha256=sha256((root / "example.png").read_bytes()).hexdigest(),
                       native_boxes_xyxy=[[0, 0, hw[1], hw[0]]], native_ranks=[4], query_ids=[42],
                       logit_ranks=[4], logit_query_ids=[42], posterior_overrides=[],
                       logits_ref=dict(path="logits.npy", bytes=(root / "logits.npy").stat().st_size,
                                       sha256=sha256((root / "logits.npy").read_bytes()).hexdigest()))
            pred = [dict(image_id=5, category_id=1, score=.501, segmentation=native)]
            count = Counter()
            result = r.replay_image(self.m, row, pred, root, count, prepare=True, image_root=root,
                                    predict=lambda f: (1 - f[:, :, 0]).astype(np.float32), stored=None)
            self.assertGreater(count["attraction_dynamic_swaps"], 0)
            self.assertGreater(count["fixed_swaps"], 0)
            verified = Counter()
            r.replay_image(self.m, row, pred, root, verified, prepare=False, image_root=None,
                           predict=None, stored=result)
            self.assertEqual(count, verified)
            result["changes"]["4"]["attraction_dynamic"]["swaps"] += 1
            with self.assertRaisesRegex(ValueError, "budgets"):
                r.replay_image(self.m, row, pred, root, Counter(), prepare=False, image_root=None,
                               predict=None, stored=result)

    def test_reject_unverified_raster_and_invalid_override(self):
        plane = np.full((144, 144), -2, np.float32)
        stock = np.zeros((33, 51), bool)
        stock[5, 6] = True
        with self.assertRaisesRegex(ValueError, "stock parity"):
            r.native_probability(plane, stock, None)
        with self.assertRaisesRegex(ValueError, "off ties"):
            r.native_probability(plane, stock, np.zeros(stock.shape, np.float32))


if __name__ == "__main__":
    unittest.main()
