"""Known perfect boxes must remain perfect after COCO80-to-native label mapping."""
import unittest
from common_metrics import evaluate_common


class MappingIntegration(unittest.TestCase):
    def test_official_evaluator_sees_perfect_car_and_bus_after_mapping(self):
        gt = {"info": {}, "images": [{"id": 1, "width": 100, "height": 100}],
              "categories": [{"id": i + 1, "name": n} for i, n in enumerate(["vehicle", "bus", "bicycle", "pedestrian"])],
              "annotations": [
                  {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0},
                  {"id": 2, "image_id": 1, "category_id": 2, "bbox": [60, 60, 20, 20], "area": 400, "iscrowd": 0}]}
        raw = [{"image_id": 1, "class_id": 2, "bbox": [10, 10, 20, 20], "score": 0.9},
               {"image_id": 1, "class_id": 5, "bbox": [60, 60, 20, 20], "score": 0.8}]
        result = evaluate_common(gt, raw, kind="coco80")
        self.assertAlmostEqual(result["map50"], 1.0, places=6)
        self.assertAlmostEqual(result["map50_95"], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
