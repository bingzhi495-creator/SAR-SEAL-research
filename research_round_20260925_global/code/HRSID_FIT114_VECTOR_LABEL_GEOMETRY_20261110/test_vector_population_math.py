import unittest

import numpy as np
import torch

from vector_label_geometry import final_polygons, polygon_raster, vector_footprint_metrics
from run_fit114_vector_population import finish_census, observed_projection
from vector_population_math import bootstrap_draws, paired_cluster_stats, run_footprint_metrics


class VectorPopulationMathTest(unittest.TestCase):
    def test_run_intersection_matches_the_original_unit_cell_integration(self):
        segmentation = [[.2, .4, 7.7, .3, 8.4, 4.8, 6.4, 7.9, .2, 7.1]]
        steps = [dict(kind="crop", before=[9, 11], after=[8, 9], top=1, left=1),
                 dict(kind="resize", before=[8, 9], after=[7, 11]),
                 dict(kind="hflip", before=[7, 11], after=[7, 11])]
        parts, hw = final_polygons(segmentation, (9, 11), steps)
        raster = polygon_raster(parts, hw)
        masks = {
            "vector_first": raster, "shifted": np.roll(raster, 1, axis=1),
            "holes": raster & (np.indices(hw).sum(axis=0) % 3 != 0),
            "empty": np.zeros(hw, dtype=np.bool_),
        }
        before, after = vector_footprint_metrics(parts, hw, masks), run_footprint_metrics(parts, hw, masks)
        self.assertAlmostEqual(before["visible_polygon_area"], after["visible_polygon_area"])
        for arm, metrics in before["by_arm"].items():
            for key, value in metrics.items():
                self.assertAlmostEqual(value, after["by_arm"][arm][key], places=12)



    def test_image_clusters_keep_repeated_rows_together_for_both_effect_estimates(self):
        values = [(0, 1.), (0, 1.), (1, -1.)]
        counts = np.asarray([[1, 1, 1], [3, 0, 0], [0, 3, 0], [0, 0, 3]])
        report = paired_cluster_stats(values, counts)
        self.assertEqual((report["paired_rows"], report["source_images"], report["undefined_draws"]), (3, 2, 1))
        self.assertAlmostEqual(report["row_weighted_mean_delta"], 1/3)
        self.assertEqual(report["image_equal_mean_delta"], 0.)
        np.testing.assert_allclose(report["row_weighted_image_cluster_percentile95"],
                                   np.quantile([1/3, 1, -1], [.025, .975]))
        np.testing.assert_allclose(report["image_equal_image_cluster_percentile95"],
                                   np.quantile([0, 1, -1], [.025, .975]))
        empty = paired_cluster_stats([], counts)
        self.assertIsNone(empty["row_weighted_image_cluster_percentile95"])
        self.assertEqual(empty["undefined_draws"], len(counts))
        only_one = paired_cluster_stats([(0, 1.), (0, 2.)], counts)
        self.assertIsNone(only_one["image_equal_image_cluster_percentile95"])
        self.assertEqual(only_one["source_images"], 1)
        drawn = bootstrap_draws(3, draws=8)
        np.testing.assert_array_equal(drawn, bootstrap_draws(3, draws=8))
        np.testing.assert_array_equal(drawn.sum(axis=1), np.full(8, 3))

    def test_fixed_common_scope_retains_vector_empty_and_reports_rle_exclusion(self):
        source = [dict(area=10, segmentation=[[0, 0, 1, 0, 0, 1]]),
                  dict(area=2048, segmentation={"size": [1, 2], "counts": "1"}),
                  dict(area=11)]
        key = "source_row"
        masks = torch.tensor([[[True, False]], [[True, True]], [[False, False]]])
        raw = dict(masks=masks)

        def target(ids):
            return {key: torch.tensor(ids), "masks": masks[ids]}

        def project(*_targets):
            common = target([0, 1])
            return common, common, common

        observe, tally, calls = observed_projection(project, source, raw, key,
                                                     ("stock", "stepwise_center", "one_pass_center"))
        a = target([0, 1, 2])
        b = target([0, 1])
        a_common, b_common, c_common = observe(a, b, b)
        self.assertEqual(a_common[key].tolist(), b_common[key].tolist())
        self.assertEqual(c_common[key].tolist(), [0, 1])
        finish_census(tally, calls, [{"small": True, "vector_empty": True}],
                      dict(common=2, polygon_eligible=1, skipped={"not a polygon-list annotation": 1}))
        self.assertEqual(tally["small"]["polygon_list"]["vector_final_empty"], 1)
        self.assertEqual(tally["other"]["RLE"]["vector_ineligible"], 1)
        self.assertEqual(tally["small"]["missing"]["stock_empty"], 1)


if __name__ == "__main__":
    unittest.main()
