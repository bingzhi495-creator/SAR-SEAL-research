import hashlib
import json
import unittest

from bridge import group_canonical_rows


def fixture():
    source = []
    manifest = []
    preds = [{"image_id": 100, "category_id": 1, "score": .42, "segmentation": {"size": [2, 3], "counts": "6"}}]
    for i in range(187):
        image_id = i + 100
        count = int(i == 0)
        rows = preds[:count]
        raw = json.dumps(rows, separators=(",", ":"), allow_nan=False).encode()
        source.append(dict(image_id=image_id, original_hw=[2, 3], image_ref={"sha256": "a"*64}))
        manifest.append(dict(image_id=image_id, original_hw=[2, 3], image_sha256="a"*64, prediction_start=min(i, 1),
            prediction_count=count, raw_query_count=100, native_count=100, unmapped_slot1_count=100-count,
            retained_native_ranks=[7] if count else [], coco_rows=count, codec_roundtrip_masks=count,
            nonempty_masks=count, all_zero_masks=0, canonical_json_sha256=hashlib.sha256(raw).hexdigest()))
    result = dict(schema="RFDETR-E24-canonical-DEV-image-index-v1", checkpoint_sha256="b"*64, recipe_sha256="c"*64,
        prediction_offset_base=0, native_rank_base=1, predictions_ref={"sha256": "d"*64}, images=manifest,
        image_count=187, native_count_total=18700, unmapped_slot1_total=18699, prediction_count=1,
        zero_prediction_images=186, codec_roundtrip_masks=1, nonempty_masks=1, all_zero_masks=0)
    return result, preds, source


class BridgeTests(unittest.TestCase):
    def test_order_score_and_zero_image_survive(self):
        m, preds, source = fixture()
        g = group_canonical_rows(m, preds, source)
        self.assertEqual(g["images"][0]["native_ranks"], [7])
        self.assertEqual(g["images"][0]["coco"][0]["score"], .42)
        self.assertEqual(g["images"][1]["coco"], [])
        self.assertEqual(len(g["images"]), 187)

    def test_manifest_requires_bound_images_and_unchanged_rows(self):
        m, preds, source = fixture()
        m["images"][1]["image_sha256"] = "f"*64
        with self.assertRaises(ValueError):
            group_canonical_rows(m, preds, source)
        m, preds, source = fixture()
        preds[0]["score"] = .43
        with self.assertRaises(ValueError):
            group_canonical_rows(m, preds, source)


if __name__ == "__main__":
    unittest.main()
