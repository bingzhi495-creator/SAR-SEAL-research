"""Exercise the equal-budget overlay with small synthetic masks."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import unittest

import numpy as np
from pycocotools import mask

spec = spec_from_file_location("_target_point_policy", Path(__file__).with_name("target.py"))
target = module_from_spec(spec)
spec.loader.exec_module(target)
CODE = Path(sys.argv.pop(1))


def swap(mask_, add, drop):
    edited = mask_.copy(order="C")
    edited.flat[add] = True
    edited.flat[drop] = False
    item = mask.encode(np.asfortranarray(edited, dtype=np.uint8))
    item["counts"] = item["counts"].decode("ascii")
    return {"rle": item, "swaps": len(add)}


class TargetPolicy(unittest.TestCase):
    def test_overlay_keeps_old_masks_and_requires_every_dynamic_edit(self):
        original = np.zeros((24, 40), dtype=bool)
        original[8:16, 6:14] = True
        coords = np.array([[325, 326], [334, 333]])
        dynamic = swap(original, [325], [326])
        adapted = swap(original, [334], [333])
        prior = {"image_id": 3, "rgb_sha256": "a" * 64, "points": [],
                 "changes": {"2": {target.DYNAMIC: dynamic}}}
        overlay = {"image_id": 3, "changes": {"2": adapted}}
        result = target.join_overlay(prior, overlay)
        self.assertEqual(result["changes"]["2"][target.DYNAMIC], dynamic)
        self.assertEqual(result["changes"]["2"][target.TARGET], adapted)
        self.assertNotIn(target.TARGET, prior["changes"]["2"])
        target.verify_xor(original, coords, adapted, 1)
        with self.assertRaisesRegex(ValueError, "Exact dynamic-count coverage"):
            target.join_overlay(prior, {"image_id": 3, "changes": {}})
        with self.assertRaisesRegex(ValueError, "Endpoint XOR"):
            target.verify_xor(original, coords, swap(original, [0], [333]), 1)

    def test_independent_readback_breaks_ties_with_prior_endpoint_index(self):
        q = np.array([[.6, .4], [.8, .1], [.8, .1], [.9, .7]], dtype=np.float32)
        self.assertEqual(target.readback_actions(q, 2), [(3, 1), (1, 2)])

    def test_reads_native_adapter_coordinates_and_features(self):
        r = target.read_module(CODE / target.OLD_ROOT / "replay.py", target.REPLAY_SHA)
        modules = r.methods(CODE)
        stock = np.zeros((36, 73), dtype=bool)
        stock[8:20, 19:31] = True
        rng = np.random.default_rng(11)
        prob = rng.random(stock.shape, dtype=np.float32)
        image = modules.pair.image_features(rng.integers(256, size=(*stock.shape, 3), dtype=np.uint8))
        box = np.array([11.25, 3., 37., 28.125], dtype=np.float32)
        pair, sampler = r.canvas(modules, stock.shape)
        expected = pair.paired_points(stock, box, int(stock.sum()), .91, prob, image, sampler)
        self.assertEqual(len(expected), 5)
        coords, features = target.original_point_features(pair, sampler, stock, box, .91, prob, image, expected[0])
        np.testing.assert_array_equal(coords, expected[0])
        np.testing.assert_array_equal(features, expected[1])


if __name__ == "__main__":
    unittest.main()
