from copy import deepcopy
import unittest

import numpy as np
from pycocotools import mask

from score_phoenix_fit import BUNDLE_SHA, REQUEST_STATUS, evaluate


def rle(x0, x1, y0, y1):
    bitmap = np.zeros((800, 800), np.uint8, order="F")
    bitmap[y0:y1, x0:x1] = 1
    out = mask.encode(bitmap)
    return dict(size=out["size"], counts=out["counts"].decode("ascii"))


def inputs():
    request = dict(status=REQUEST_STATUS, provenance={"fit_bundle_sha256": BUNDLE_SHA}, images=[])
    index = dict(images=[], annotations=[], categories=[dict(id=1)])
    for image in (21, 22):
        request["images"].append(dict(image_id=image, image_file=f"image_{image-20:03}.png",
            encoded_sha256=f"{image:064x}", decoded_rgb_sha256="b"*64, candidate_rows=[
                dict(native_rank=1, query_id=7, score=.9, category_id=1, segmentation=rle(0, 8, 0, 10)),
                dict(native_rank=2, query_id=2, score=.8, category_id=1, segmentation=rle(0, 0, 0, 0)),
            ]))
        index["images"].append(dict(id=image, height=800, width=800))
        index["annotations"].append(dict(id=image, image_id=image, category_id=1,
            iscrowd=0, segmentation=rle(0, 10, 0, 10)))
    sha = "c"*64
    variants = {}
    for rule in ("stock_coco", "all_nonempty"):
        variants[rule] = dict(rule=rule, request_sha256=sha, images=[
            dict(image_id=im["image_id"], candidate_rows=deepcopy(im["candidate_rows"])) for im in request["images"]])
    variants["all_nonempty"]["images"][0]["candidate_rows"][0]["segmentation"] = rle(0, 10, 0, 10)
    return request, sha, variants, index


class TestPhoenixFit(unittest.TestCase):
    def test_fixed_match_preserves_original_pair_and_recovers_at_point_nine(self):
        result = evaluate(*inputs())
        self.assertEqual(result["original_050_matches"], 2)
        self.assertEqual(result["metrics"]["original"]["mean_fixed_iou"], .8)
        self.assertEqual(result["metrics"]["all_nonempty"]["mean_fixed_iou"], .9)
        self.assertEqual(result["metrics"]["all_nonempty"]["fixed_thresholds"]["0.90"]["recovered"], 1)
        self.assertEqual(result["metrics"]["all_nonempty"]["fixed_thresholds"]["0.90"]["lost"], 0)

    def test_no_edit_to_unselected_or_score(self):
        request, sha, variants, index = inputs()
        variants["stock_coco"]["images"][0]["candidate_rows"][0]["segmentation"] = rle(0, 10, 0, 10)
        with self.assertRaisesRegex(ValueError, "unselected"):
            evaluate(request, sha, variants, index)
        request, sha, variants, index = inputs()
        variants["all_nonempty"]["images"][0]["candidate_rows"][0]["score"] = .901
        with self.assertRaisesRegex(ValueError, "identity"):
            evaluate(request, sha, variants, index)


if __name__ == "__main__":
    unittest.main()
