from importlib.util import module_from_spec, spec_from_file_location
from itertools import combinations
from pathlib import Path
import os

import numpy as np

spec = spec_from_file_location("oracle", Path(__file__).with_name("oracle.py"))
oracle = module_from_spec(spec)
spec.loader.exec_module(oracle)


def test_fractional_cells_maximise_fixed_area_overlap_at_exact_k():
    coords = np.array([[10, 21], [12, 23], [14, 25], [16, 27]], dtype=np.int32)
    occupancies = np.array([[.1, .1], [.75, .9], [.7, .45], [.75, .2]], dtype=np.float64)
    selected = oracle.best_choices(coords, occupancies, 2)
    assert selected == [(1, 0), (3, 3)]
    value = oracle.selected_mass(occupancies, selected)
    feasible = [occupancies[list(a), 0].sum()-occupancies[list(d), 1].sum()
                for a in combinations(range(4), 2) for d in combinations(range(4), 2)]
    assert abs(value[0]-value[1]-max(feasible)) < 1e-12
    assert oracle.best_choices(coords, occupancies, 0) == []
    assert oracle.selected_mass(occupancies, []) == (0.0, 0.0)


def test_fixed_native_matching_matches_original_mask_evaluator():
    from pycocotools import mask

    code = Path(os.environ["SRNET_PUBLIC_CODE"])
    evaluator = oracle.load(code / "PSEG_RFDETR_FROZEN_SOURCE_HEAD_20261122/score.py",
                            "77a3fbfaa04ac08287eff4ec7bb827ce3914e61aa20dd4f1e6f9c1e3b8ccc9fa")
    truth = [[4, 4, 10, 4, 10, 10, 4, 10]]
    annotation = dict(id=1, image_id=1, category_id=1, iscrowd=0, segmentation=truth, area=36, bbox=[4,4,6,6])
    gt = dict(info={}, images=[dict(id=1, height=16, width=16)], categories=[dict(id=1, name="ship")],
              annotations=[annotation])
    bits = np.zeros((16, 16), dtype=np.uint8, order="F")
    bits[4:10, 4:10] = 1
    rle = mask.encode(bits)
    orig = dict(image_id=1, category_id=1, score=.9, segmentation=dict(rle, counts=rle["counts"].decode()))
    blocks = {1: dict(stock=[orig, dict(orig)], ranks=[1, 2], changes={})}
    _, known = evaluator.score_one(gt, blocks, "raw", anchors=True)
    new = oracle.native_anchors(evaluator, gt, blocks)
    assert new == known
    assert new[1][1] == 1


if __name__ == "__main__":
    test_fractional_cells_maximise_fixed_area_overlap_at_exact_k()
    test_fixed_native_matching_matches_original_mask_evaluator()
    print("fractional exact budget, stable ties, zero budget, and native matching parity: PASS")

