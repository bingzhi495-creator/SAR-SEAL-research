import os
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import numpy as np

parent = Path(__file__).parent
spec = spec_from_file_location("score", parent / "score.py")
score = module_from_spec(spec)
spec.loader.exec_module(score)
code = Path(os.environ["SRNET_PUBLIC_CODE"])
reserve = score.load(code / score.RESERVE, score.RESERVE_SHA)
old = score.load(code / score.HISTORIC, reserve.HISTORIC)
evaluator = old.module(code / "PSEG_RFDETR_FROZEN_SOURCE_HEAD_20261122/score.py", old.SCORE_SOURCE)
adapter = old.module(code / "UEIKAP_RFDETR_CARRIER_20261122/score_rf.py", old.GEOMETRY_SOURCE)
core = adapter.load_geometry(code / "UEIKAP_RESERVED_HULL_FROZEN_20261122/score.py")
sys.path.append(os.environ["SRNET_SHAPELY_SITE"])
import shapely


def test_score_preserves_empty_images_and_tracks_changed_native_anchors():
    reserve.ARMS = ("raw", "fixed", "free_matched_count")
    reserve.PAIRS = (("free_matched_count", "fixed"),)
    score.support_empty_predictions(evaluator)
    vertices = [5, 5, 11, 5, 11, 11, 5, 11]
    ref = {"id": 1, "image_id": 1, "category_id": 1, "iscrowd": 0, "segmentation": [vertices],
           "area": 36, "bbox": [5, 5, 6, 6]}
    gt = dict(info={}, images=[{"id": i, "height": 16, "width": 16} for i in (1, 2)],
              categories=[{"id": 1, "name": "ship"}], annotations=[ref, dict(ref, id=2, image_id=2)])
    stock = np.zeros((16, 16), dtype=np.uint8, order="F")
    stock[5:11, 5:11] = 1

    def rle(array):
        data = core.codec.encode(np.asfortranarray(array))
        return dict(data, counts=data["counts"].decode())

    original = dict(image_id=1, category_id=1, score=.9, segmentation=rle(stock))
    fixed = stock.copy()
    fixed[5, 5], fixed[5, 4] = 0, 1
    free = stock.copy()
    free[10, 10], free[11, 10] = 0, 1
    blocks = {1: dict(stock=[original], ranks=[1], group=1,
                     changes={1: dict(fixed=dict(swaps=1, rle=rle(fixed)),
                                      free_matched_count=dict(swaps=1, rle=rle(free)))}),
              2: dict(stock=[], ranks=[], group=2, changes={})}
    polygons = {i: shapely.Polygon(np.asarray(vertices).reshape(-1,2)) for i in (1, 2)}
    dates = {1: ("GF-3", "2020-01-12"), 2: ("GF-3", "2020-01-13")}
    full = score.score_view(reserve, evaluator, core, shapely, gt, {1, 2}, blocks, polygons, dates)
    assert full["images"] == 2 and full["annotations"] == 2
    assert full["geometry"]["actual_changed_eligible_native_matches"] == 1
    assert full["geometry"]["actual_changed_parent_groups"] == 1
    assert full["geometry"]["counts"]["eligible_matches"] == 1
    assert full["metrics_fraction"]["raw"]["AP"] > 0
    empty = score.score_view(reserve, evaluator, core, shapely, gt, {2}, blocks, polygons, dates)
    assert empty["metrics_fraction"]["raw"]["AP"] == 0
    assert empty["geometry"]["counts"]["original_matches"] == 0


if __name__ == "__main__":
    test_score_preserves_empty_images_and_tracks_changed_native_anchors()
    print("COCO empty-view compatibility, untouched image denominator and native changed-anchor count: PASS")
