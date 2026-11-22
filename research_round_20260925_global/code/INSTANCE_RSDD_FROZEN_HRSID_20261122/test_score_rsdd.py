"""Synthetic sensor and acquisition checks without opening Instance-RSDD."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import unittest

import numpy as np

spec = spec_from_file_location("synthetic_rsdd_scorer", Path(__file__).with_name("score_rsdd.py"))
score = module_from_spec(spec)
spec.loader.exec_module(score)
CODE = Path(sys.argv.pop(1))
sys.path.append(sys.argv.pop(1))


class TestScores(unittest.TestCase):
    def test_dedup_preserves_lowest_id_within_sensor_view(self):
        data = {9: {"rgb": "same", "sensor": "GF-3"}, 3: {"rgb": "same", "sensor": "GF-3"},
                4: {"rgb": "another", "sensor": "TerraSAR-X"}}
        self.assertEqual(score.representatives(data, {9,3}), {3})
        self.assertEqual(score.representatives(data, {9,3,4}), {3,4})

    def test_scene_weighting_one_scene_one_vote(self):
        x = np.zeros((8,3),dtype=float)
        x[score.ARMS.index("free_natural"),0] = .02
        z = np.zeros_like(x)
        got = score.geometry_summary({1:[x],2:[z,z,z]})
        row = got["paired_contrasts_points"]["free_natural minus raw"]["continuous_iou_points"]
        self.assertAlmostEqual(row["delta"], .01)
        self.assertEqual(row["interval_95"], [0.,.02])

    def test_coco_rekeys_annotation_zero_and_keeps_image_zero_and_geometry(self):
        gt = {"images":[{"id":0}], "annotations":[{"id":0,"image_id":0,"segmentation":[[0,0,1,0,1,1]]}]}
        normalized, changed = score.coco_annotation_ids(gt)
        self.assertTrue(changed)
        self.assertEqual(normalized["annotations"][0]["id"], 1)
        self.assertEqual(normalized["annotations"][0]["image_id"], 0)
        self.assertEqual(normalized["annotations"][0]["segmentation"], gt["annotations"][0]["segmentation"])
        self.assertEqual(gt["annotations"][0]["id"], 0)

    def test_zero_ids_with_released_coco_and_continuous_backends(self):
        import shapely
        reference = score.module(CODE/"PSEG_RFDETR_FROZEN_SOURCE_HEAD_20261122/score.py",score.SCORE_SOURCE)
        adapter = score.module(CODE/"UEIKAP_RFDETR_CARRIER_20261122/score_rf.py",score.GEOMETRY_SOURCE)
        geometry = adapter.load_geometry(CODE/"UEIKAP_RESERVED_HULL_FROZEN_20261122/score.py")
        raster = np.zeros((34,51),dtype=np.uint8)
        raster[5:13,5:13] = 1
        rle = geometry.codec.encode(np.asfortranarray(raster))
        rle["counts"] = rle["counts"].decode("ascii")
        polygon = [(5,5),(13,5),(13,13),(5,13)]
        annotation = {"id":0,"image_id":0,"category_id":1,"area":64,"bbox":[5,5,8,8],"iscrowd":0,
                      "segmentation":[[v for xy in polygon for v in xy]]}
        gt,_ = score.coco_annotation_ids({"info":{},"images":[{"id":0,"height":34,"width":51}],
                                         "annotations":[annotation],"categories":[{"id":0,"name":"background"},{"id":1,"name":"ship"}]})
        stock = {"image_id":0,"category_id":1,"score":.7,"segmentation":rle}
        view = score.result_view(gt,{0},{0:{"stock":[stock],"ranks":[1],"changes":{},"group":3}},
                                 reference,geometry,{1:shapely.Polygon(polygon)},shapely)
        self.assertAlmostEqual(view["metrics_fraction"]["raw"]["AP"],1.0)
        self.assertEqual(view["geometry"]["original_anchors"],1)
        self.assertEqual(view["geometry"]["stats"]["group_equal_arms"]["raw"]["continuous_iou_points"],100.0)


if __name__ == "__main__":
    unittest.main()
