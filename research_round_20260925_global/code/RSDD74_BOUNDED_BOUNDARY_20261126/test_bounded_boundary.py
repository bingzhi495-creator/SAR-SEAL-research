from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import os

import numpy as np
spec = spec_from_file_location("score_bounded_boundary", Path(__file__).with_name("score_bounded_boundary.py"))
score = module_from_spec(spec)
spec.loader.exec_module(score)


def test_same_budget_boundary_difference_and_original_parity():
    code = Path(os.environ["BOUNDARY_PUBLIC_CODE"])
    b = code / score.BOUNDARY_DIR
    metric = score.load(b / "boundary_metric.py", score.METRIC_SHA)
    stats = score.load(b / "boundary_stats.py", score.STATS_SHA)
    prior = stats.Audit(metric, ("SF", "SU_equal"), {"SU_equal_minus_SF":(("SU_equal","SF"),)})
    new = stats.Audit(metric, score.ARMS, score.CONTRASTS)
    dates = {1:("GF3","same"), 2:("GF3","same"), 3:("GF3","other")}
    authors = {1:2, 2:1, 3:1}
    for p in (1, 2, 3):
        gt = np.zeros((16, 16), dtype=bool)
        gt[3:10, 4:11] = True
        native = np.roll(gt, 1, axis=1)
        fixed = np.roll(gt, p, axis=1)
        bounded = np.roll(gt, max(0, p-1), axis=1)
        prior.add(p, gt, native, {"SF":fixed, "SU_equal":gt})
        new.add(p, gt, native, {"F":fixed, "L":bounded, "I":gt})
    old, current = prior.report(authors, dates), new.report(authors, dates)
    score.verify_boundary_parity(current, old)
    primary = current["widths"]["1"]["contrasts"]["I_minus_L"]
    assert primary["mean_points"] > 0
    assert primary["resample_blocks"] == 2
    assert len(primary["paired_95_interval_points"]) == len(primary["block_95_interval_points"]) == 2


if __name__ == "__main__":
    test_same_budget_boundary_difference_and_original_parity()
    print("old Boundary IoU parity and parent/sensor-date bootstrap with new L: PASS")

