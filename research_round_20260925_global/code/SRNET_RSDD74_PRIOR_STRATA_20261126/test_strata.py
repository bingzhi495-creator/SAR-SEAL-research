from importlib.util import module_from_spec, spec_from_file_location
from itertools import combinations
from pathlib import Path

import numpy as np

spec = spec_from_file_location("strata", Path(__file__).with_name("strata.py"))
strata = module_from_spec(spec)
spec.loader.exec_module(strata)


def test_constrained_selection_matches_exhaustive_stratum_quota():
    p = np.array([[0., 1.], [1e-4, .9999], [.12, .6], [.4, .7]], dtype=np.float32)
    tail, exact = strata.clip_strata(p)
    assert tail.tolist() == [[True, True], [True, True], [False, False], [False, False]]
    assert exact.tolist() == [[True, True], [False, False], [False, False], [False, False]]
    occ = np.array([[.05, .65], [.12, .35], [.7, .1], [.45, .4]])
    choices_i = [(0, 0), (1, 3)]
    observed = strata.optimal_in_strata(occ, tail, choices_i)
    assert observed == [(1, 1), (0, 2)]
    want_a, want_d = [sum(tail[pick[side], side] for pick in choices_i) for side in (0, 1)]
    best = max(sum(occ[i, 0] for i in a) - sum(occ[j, 1] for j in d)
               for a in combinations(range(4), 2) for d in combinations(range(4), 2)
               if sum(tail[i, 0] for i in a) == want_a and sum(tail[j, 1] for j in d) == want_d)
    actual = sum(occ[i, 0]-occ[j, 1] for i, j in observed)
    assert abs(actual - best) < 1e-12
    assert strata.optimal_in_strata(occ, tail, []) == []


def test_regret_geometry_separates_within_tail_and_across_tail_changes():
    from pycocotools import mask
    import os
    import sys

    sys.path.append(os.environ["SRNET_SHAPELY_SITE"])
    import shapely

    code = Path(os.environ["SRNET_PUBLIC_CODE"])
    last = strata.load(Path(os.environ["SRNET_ORACLE_SOURCE"]), strata.ORACLE_WORKER)
    historical = strata.load(code / last.PAST, last.PAST_SHA)
    source = code / "UEIKAP_RESERVED_HULL_FROZEN_20261122/score.py"
    core = strata.load(source, strata.digest(source.read_bytes()))
    base_path = code / "PSEG_RFDETR_FROZEN_SOURCE_HEAD_20261122/replay.py"
    base = strata.load(base_path, strata.digest(base_path.read_bytes()))
    stock = np.zeros((10, 10), dtype=np.uint8, order="F")
    stock[2:8, 2:8] = 1
    coords = np.array([[21, 22], [11, 27], [18, 33], [81, 47]], dtype=np.int32)
    fixed, incumbent = [(0, 0)], [(1, 0)]
    options = dict(fixed=base.mask_edit(stock, coords, fixed),
                   free_matched_count=base.mask_edit(stock, coords, incumbent))
    raw = mask.encode(stock)
    raw["counts"] = raw["counts"].decode()
    p = np.array([[.2, 1.], [0., 1.], [0., .9], [.2, .9]], dtype=np.float32)
    polygon = shapely.box(1, 2, 7, 8)
    values, counts, _ = strata.anchor_geometry(last, core, base, historical.iou, raw, options,
                                               (coords, fixed, incumbent), strata.clip_strata(p),
                                               polygon, shapely)
    assert values[0, 0] < values[1, 0] < values[2, 0]
    assert np.allclose(np.diff(values[:, 3]), [100/36, 100/36])
    assert counts["incumbent"]["add_clip_tail"] == counts["stratified_oracle"]["add_clip_tail"] == 1
    assert counts["oracle"]["add_clip_tail"] == 0


if __name__ == "__main__":
    test_constrained_selection_matches_exhaustive_stratum_quota()
    test_regret_geometry_separates_within_tail_and_across_tail_changes()
    print("FP32 clipping edges, exact stratum quota and signed geometry decomposition: PASS")

