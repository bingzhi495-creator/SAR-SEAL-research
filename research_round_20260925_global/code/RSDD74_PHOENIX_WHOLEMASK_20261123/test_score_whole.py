"""Synthetic native-order and coverage checks; no study inputs."""
import importlib.util
from pathlib import Path
import argparse
import sys
from types import SimpleNamespace
import unittest

import numpy as np
from pycocotools import mask as codec


def mask(area):
    bitmap = np.zeros(512 * 512, dtype=np.uint8)
    bitmap[:area] = 1
    out = codec.encode(np.asfortranarray(bitmap.reshape(512, 512)))
    return {"size": out["size"], "counts": out["counts"].decode("ascii")}


def load():
    spec = importlib.util.spec_from_file_location("whole", Path(__file__).with_name("score_whole.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NativePolicyTest(unittest.TestCase):
    def test_full_order_and_strict_one_percent_boundary(self):
        whole = load()
        native = [mask(n) for n in (0, 1, 2621, 2622)]
        candidates = [{"position":i, "segmentation":m} for i,m in enumerate(native, 1)]
        outputs = {"ordinal":1, "lvis":[dict(candidates[1], segmentation=native[0]), *candidates[2:]],
                   "coco":candidates[3:]}
        changes = {2: {"phoenix":{"rle":native[1]}}}
        count = whole.attach_image(1, native, candidates, outputs, changes)
        self.assertEqual(count["lvis"]["decoded_candidates"], 3)
        self.assertEqual(count["lvis"]["empty_outputs"], 1)
        self.assertEqual(count["lvis"]["unchanged_outputs"], 2)
        self.assertEqual(count["lvis"]["drops"], 1)
        self.assertEqual(count["coco"]["decoded_candidates"], 1)
        self.assertEqual(changes[2]["phoenix"]["rle"], native[1])
        self.assertEqual(changes[2]["lvis"]["rle"], native[0])
        self.assertNotIn("coco", changes[2])
        self.assertNotIn(1, changes)

    def test_geometry_uses_each_written_mask_area(self):
        import shapely
        whole = load()
        # One true pixel in a two-pixel output: IoU .5, normalized FP 1, FN 0.
        truth = shapely.box(0., 0., 1., 1.)
        np.testing.assert_allclose(whole.geometry_vector(.5, 2, truth.area), [50,100,0,100,1])
        np.testing.assert_allclose(whole.geometry_vector(0., 0, truth.area), [0,0,100,0,0])

    def test_mirror_gate_and_indeterminate(self):
        whole = load()
        self.assertTrue(whole.decision(-.30, [-.5,-.1], 20, 8)["supported_loss"])
        self.assertFalse(whole.decision(.31, [.1,1.], 20, 7)["supported_advantage"])
        self.assertFalse(whole.decision(.30, [-.1,1.], 20, 8)["supported_advantage"])

    def test_malformed_and_reordered_masks_rejected(self):
        whole = load()
        with self.assertRaises(ValueError):
            whole.native_area({"size":[512,512],"counts":"1"})
        native = [mask(77), mask(2622)]
        members = [{"position":i,"segmentation":m} for i,m in enumerate(native,1)]
        out = {"ordinal":1,"lvis":members[::-1],"coco":members[1:]}
        with self.assertRaises(ValueError):
            whole.attach_image(1, native, members, out, {})

    def test_full_coco_and_geometry_on_one_synthetic_ship(self):
        if not CODE:
            self.skipTest("Pass --code to exercise pinned COCO evaluator")
        import shapely
        whole = load()
        _, deps = whole.frozen_sources(SimpleNamespace(code=CODE))
        _, _, cfa, _, _, _, evaluator, _, _ = deps
        # 64 pixels along row zero, two identical detections at different scores. The
        # whole writer erases the original match but recovers it through the second.
        stock, empty = mask(64), mask(0)
        poly = [0.,0.,64.,0.,64.,1.,0.,1.]
        gt = {"info":{},"images":[{"id":1,"width":512,"height":512}],
              "categories":[{"id":1,"name":"ship"}],
              "annotations":[{"id":1,"image_id":1,"category_id":1,"iscrowd":0,
                              "area":64.,"bbox":[0.,0.,64.,1.],"segmentation":[poly]}]}
        detect = {"image_id":1,"category_id":1,"score":.9,"segmentation":stock}
        blocks = {1:{"stock":[detect,dict(detect,score=.8)],"group":"a","ranks":[1,2],
                     "changes":{1:{"lvis":{"rle":empty}}}}}
        out, _ = evaluator.score_one(gt,blocks,"raw",anchors=True)
        prev = {k:(v if v>=0 else None) for k,v in out.items()}
        ref = dict(images=1, annotations=1, source_groups=1, metrics_fraction={a:prev for a in whole.MATCHED},
                   geometry={"counts":dict(native_matches=1,eligible_matches=1,invalid_polygon_matches=0,area_excluded_matches=0),
                             "represented_groups":1,"sensor_date_blocks":1,"group_equal_arms":{
                                 a:dict(zip(cfa.FIELDS,[100.,0.,0.,100.,64.])) for a in whole.MATCHED}})
        report = whole.score_view({1},gt,blocks,{1:shapely.Polygon(np.array(poly).reshape(-1,2))},
                                  {"a":("sensor","date")}, ref, deps)
        self.assertEqual(report["coverage"]["lvis"]["matched_hulls_50"], 1)
        self.assertEqual(report["coverage"]["lvis"]["original_matched_retained_50"], 0)
        self.assertEqual(report["geometry"]["paired"]["lvis_minus_phoenix"]["continuous_iou_points"]["interval_95"], [-100.,-100.])


CODE = None
if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--code", type=Path)
    opts, rest = parser.parse_known_args()
    CODE = opts.code
    sys.argv[1:] = rest
    unittest.main()
