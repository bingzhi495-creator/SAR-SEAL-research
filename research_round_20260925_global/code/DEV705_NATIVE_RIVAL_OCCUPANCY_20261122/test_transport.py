import argparse
from collections import Counter
from pathlib import Path
from runpy import run_path
from types import SimpleNamespace
import unittest

import numpy as np
from pycocotools import mask as codec


cli = argparse.ArgumentParser()
for key in ("code", "rival", "fit-worker"):
    cli.add_argument("--" + key, type=Path, required=True)
args, rest = cli.parse_known_args()
transport = run_path(str(Path(__file__).with_name("transport.py")))
score = run_path(str(Path(__file__).with_name("score.py")))


def fixture_rle(points):
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask.ravel()[list(points)] = 1
    packed = codec.encode(np.asfortranarray(mask))
    return dict(size=packed["size"], counts=packed["counts"].decode("ascii"))


class FrozenTransport(unittest.TestCase):
    def test_original_seam_and_matched_drop_checks_on_generated_masks(self):
        m = transport["sources"](args)
        baseline = m.old.image_edits
        proposed = transport["image_replay"](args, m)
        old = fixture_rle(range(32))
        coco = [dict(image_id=101, score=0.95, segmentation=old),
                dict(image_id=101, score=0.95, segmentation=fixture_rle([32]))]
        details = [{"mask_area": 32, "box_xyxy": [0, 0, 8, 8]}, {"mask_area": 1}]
        coordinates = np.stack((np.arange(32, 40), np.arange(8)), axis=1)
        p = np.stack((np.linspace(.9, .2, 8), np.array([.1] + [.99] * 7)), axis=1)
        features = np.full((8, 2, 2), .5)
        pair = SimpleNamespace(paired_points=lambda *x: (coordinates, features, None, 0, 0),
                               choose_pairs=lambda gains, cap: np.flatnonzero(gains > 0)[:cap].tolist())
        free = m.old.load(args.code / transport["E24"] / "fit109_free_pairing/free_pairing.py",
                          m.old.FREE_SOURCE, "test_free")
        learner = SimpleNamespace(probabilities=lambda *x: p)
        models = pair, None, free, learner, None, None, None
        raw = baseline(coco, details, [None, None], None, models, Counter())
        changes, coverage = proposed(coco, details, [None, None], None, models, Counter())
        row = dict(image_id=101, changes=changes, coverage=coverage)
        verify = transport["verify_image"]
        verify(m, {"image_id": 101}, coco, details, raw, row)
        self.assertTrue(coverage["1"]["native_rival"])
        self.assertTrue(coverage["1"]["true_changed"])
        self.assertEqual(changes["1"][m.control]["swaps"], 1)
        corrupt = dict(swaps=1, rle=fixture_rle([i for i in range(32) if i != 1] + [33]))
        changes["1"][m.arms[0]] = corrupt
        with self.assertRaises(ValueError):
            verify(m, {"image_id": 101}, coco, details, raw, row)


if __name__ == "__main__":
    unittest.main(argv=[__file__, *rest])
