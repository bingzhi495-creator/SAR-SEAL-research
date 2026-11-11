import unittest

from score_seed_area_confirm import labeled_cases, score, validate_input


class AreaSignalConfirmationTest(unittest.TestCase):
    def test_held_group_scores_and_same_coverage(self):
        rows = []
        for ordinal in range(18):
            for offset in (4, -4):
                rows.append(dict(ordinal=ordinal, group=f"g{ordinal}", area=64,
                                 truth=64+offset, match_status="valid", covered=True,
                                 cap=12, mean_delta=0, median_delta=offset, soft_delta=0, size_delta=0))
            rows.append(dict(ordinal=ordinal, group=f"g{ordinal}", area=64,
                             truth=64, match_status="valid", covered=False,
                             cap=12, mean_delta=4, median_delta=4, soft_delta=4, size_delta=4))
            rows.append(dict(ordinal=ordinal, group=f"g{ordinal}", area=64,
                             truth=None, match_status="unmatched", covered=True,
                             cap=12, mean_delta=0, median_delta=4, soft_delta=0, size_delta=0))
        result = score(rows)
        self.assertEqual(result["gate"], "MEDIAN_AREA_SIGNAL_LEAD")
        self.assertEqual(result["counts"]["valid_matches"], 54)
        self.assertEqual(result["counts"]["covered_valid_matches"], 36)
        self.assertEqual(result["metrics"]["median"]["equal_source_error"], 0)
        self.assertGreater(result["metrics"]["mean"]["equal_source_error"], 2)
        self.assertEqual(result["metrics"]["median"]["actions"]["unmatched"], 18)
        self.assertEqual(result["metrics"]["median"]["actions"]["uncovered"], 0)
        self.assertEqual(result["positive_folds"], dict(zero=3, mean=3, soft=3, size=3))
        self.assertEqual([f["calibration"]["median"] for f in result["folds"]],
                         [{"a": 0, "b": 1}]*3)
        tied_mean = [dict(row, mean_delta=row["median_delta"]) for row in rows]
        tied = score(tied_mean)
        self.assertEqual(tied["gains"]["mean"], 0)
        self.assertEqual(tied["gate"], "NO_MEDIAN_AREA_SIGNAL_LEAD")

    def test_prelabel_rank_and_source_join(self):
        import numpy as np
        from pycocotools import mask
        rle = mask.encode(np.ones((8, 8), dtype=np.uint8, order="F"))
        rle["counts"] = rle["counts"].decode("ascii")
        old = dict(ordinal=0, group="g0", basename="synthetic.png",
                   candidates=[dict(rank=2, score=.9, area=64, rle=rle)])
        signals = dict(ordinal=0, group="g0", basename="synthetic.png",
                       signals=[dict(rank=2, donor_ranks={"5201": 4, "5203": 2},
                                     cap=12, mean_delta=2, median_delta=3, soft_delta=-2, size_delta=1)])
        rows, counts = validate_input([old], [signals])
        self.assertEqual(rows[0]["signals"][2]["median_delta"], 3)
        self.assertEqual(counts["eligible_native_candidates"], 1)
        with self.assertRaisesRegex(ValueError, "native candidate"):
            validate_input([old], [dict(signals, signals=[dict(signals["signals"][0], rank=3)])])
        with self.assertRaisesRegex(ValueError, "source"):
            validate_input([old], [dict(signals, group="different")])


    def test_no_gt_yields_unmatched_without_matcher_edge_failure(self):
        from types import SimpleNamespace
        point = dict(rank=2, area=64, rle={"size": [8, 8], "counts": "synthetic"})
        input_row = dict(ordinal=0, group="g0", basename="synthetic.png",
                         candidates=[point], eligible={2: 64}, signals={})
        labels = dict(images=[dict(id=9, file_name="synthetic.png", height=8, width=8)],
                      annotations=[], categories=[dict(id=1, name="ship")])
        geom = SimpleNamespace(geometry=SimpleNamespace(polygon_support=lambda *_: ({}, set(), {})), vector=None)

        def fail(*_):
            self.fail("The original stock matcher adapter requires at least one GT for a nonempty pool")

        cases = labeled_cases([input_row], labels, SimpleNamespace(freeze_coco=fail), geom)
        self.assertEqual(cases[0]["match_status"], "unmatched")
        self.assertIsNone(cases[0]["truth"])


if __name__ == "__main__":
    unittest.main()


