from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
from pycocotools import mask

spec = spec_from_file_location("score_transverse", Path(__file__).with_name("score_transverse.py"))
score = module_from_spec(spec)
spec.loader.exec_module(score)


def rle(array):
    result = mask.encode(np.asfortranarray(array, dtype=np.uint8))
    return dict(result, counts=result["counts"].decode("ascii"))


def test_independent_replay_checks_unbalanced_signed_positions_and_total():
    stock = np.zeros((12,12), dtype=bool)
    stock[3:9,3:9] = True
    coords = np.array([[12, 39], [14, 40], [16,41], [18,42]], dtype=np.int32)
    changed = stock.copy()
    changed.flat[coords[[0,1,2],0]] = True
    changed.flat[coords[[1],1]] = False
    encoded = rle(changed)
    spec = dict(action=[[0,1,2],[1]],counts=[3,1],rle=encoded)
    assert score.reconstruct(stock, coords, spec, 2) == encoded
    try:
        score.reconstruct(stock, coords, dict(spec, action=[[0,1,1],[1]]), 2)
    except ValueError:
        pass
    else:
        raise AssertionError("Repeated signed position was accepted")

def test_rank_one_native_mask_is_admitted_and_all_controls_hold_the_quota():
    s = np.zeros((64,64), dtype=bool)
    s[5, 4:40] = True
    pts = np.array([[3*64+8, 5*64+4], [3*64+9, 5*64+5]], dtype=np.int32)
    edit = s.copy()
    edit.flat[pts[0,0]], edit.flat[pts[0,1]] = True, False
    payload = dict(action=[[0],[0]],counts=[1,1],rle=rle(edit))
    opts = {name: payload for name in score.ARMS}
    points = np.column_stack(edit.nonzero())
    widths = np.var(points, axis=0)
    layer = dict(eligible=True, disposition="eligible", axis_ratio=None,
                 feasible_quotas=3, admitted_counts={a:1 for a in (score.T,score.A,score.L)},
                 admitted_quotas={a:[1] for a in (score.T,score.A,score.L)},
                 width_by_arm={a:widths[0] for a in opts},
                 longitudinal_variance_by_arm={a:widths[1] for a in opts})
    assert score.verify_layer(s, pts, 1, layer, opts) == (True, True)


if __name__ == "__main__":
    test_independent_replay_checks_unbalanced_signed_positions_and_total()
    test_rank_one_native_mask_is_admitted_and_all_controls_hold_the_quota()
    print("signed RLE and independent rank-one eligibility/area and spread control: PASS")

