"""Check canonical immutable point/target joins without experiment inputs."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np

source = Path(__file__).with_name("prepare_source.py")
spec = spec_from_file_location("prepare_source", source)
module = module_from_spec(spec)
spec.loader.exec_module(module)


def test_candidate_indices_offsets_and_missing_target_roles():
    feature = np.full((1, 2, 14), .25, dtype=np.float32)
    coord = np.array([[80, 81]], dtype=np.int32)
    roster = [
        dict(image_id=1, group="y"), dict(image_id=2, group="x"), dict(image_id=3, group="z")]
    flat = [
        dict(image_id=1, rank=1, query_id=62, area=33), dict(image_id=1, rank=3, query_id=7, area=32),
        dict(image_id=2, rank=1, query_id=69, area=33)]
    row = dict(image_id=1, group="y", rank=3, query_id=7, coordinates=coord, features=feature,
               role="valid", matched_gt_id=17, truth=np.array([[.375, .75]]),
               raw_continuous=dict(predicted_area=32, truth_area=31.5, intersection=30.1))
    missing = dict(row, image_id=2, group="x", rank=1, query_id=69, role="unmatched",
                   matched_gt_id=None, truth=None, raw_continuous=None)
    x = module.pack_points(roster, flat, [row, missing], {"x": 3, "y": 1, "z": 0})
    assert x["candidate_index"].tolist() == [1, 2]
    assert x["offset"].tolist() == [0, 1, 2]
    assert x["fold"].tolist() == [1, 3]
    assert x["roster_fold"].tolist() == [1, 3, 0]
    assert x["role"].tolist() == [2, 0]
    assert x["matched_gt_id"].tolist() == [17, -1]
    assert x["features"].dtype == np.float32
    assert x["truth"].dtype == np.float64 and np.array_equal(x["truth"][0], [.375, .75])
    assert np.isnan(x["truth"][1]).all() and np.isnan(x["truth_area"][1])


if __name__ == "__main__":
    test_candidate_indices_offsets_and_missing_target_roles()
    print("1 synthetic canonical source packing test passed")

