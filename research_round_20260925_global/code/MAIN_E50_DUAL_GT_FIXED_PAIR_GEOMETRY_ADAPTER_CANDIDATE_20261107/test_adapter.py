"""Small fabricated counts for the geometry aggregation interface."""

import unittest
from types import SimpleNamespace

import numpy as np

from adapter import summarize_pairs


def row(image, annotation, plain_fn, full_fn, area=5):
    names = ("shared_foreground_FN", "shared_background_FP", "integer_band_FN",
             "integer_band_FP", "float_band_FN", "float_band_FP")

    def method(fn):
        errors = dict.fromkeys(names, 0)
        errors["shared_foreground_FN"] = fn
        iou = {gt: dict(intersection=area-fn, union=area, at_least_0_75=(4*(area-fn) >= 3*area))
               for gt in ("integer", "float")}
        return dict(errors=errors, iou=iou)

    methods = dict(native=method(0), PLAIN=method(plain_fn), FULL=method(full_fn))
    contrasts = {}
    for left, right in (("PLAIN", "native"), ("FULL", "native"), ("FULL", "PLAIN")):
        contrasts[left + "_minus_" + right] = dict(
            errors={key: methods[left]["errors"][key] - methods[right]["errors"][key] for key in names},
            iou75_crossing={gt: int(methods[left]["iou"][gt]["at_least_0_75"])
                            - int(methods[right]["iou"][gt]["at_least_0_75"]) for gt in ("integer", "float")})
    return dict(image_id=image, GT_annotation_id=annotation,
                geometry=dict(regions=dict(shared_foreground=area, shared_background=3,
                                           integer_only=0, float_only=0, GT_union=area),
                              methods=methods, contrasts=contrasts))


class AggregationTests(unittest.TestCase):
    def test_paired_macro_with_image_cluster_resampling(self):
        rows = [row(11, 101, 1, 0), row(11, 102, 1, 0), row(22, 103, 0, 1)]
        control = SimpleNamespace(check=lambda: None)
        summary = summarize_pairs(np, control, image_ids=[11, 22], all_annotation_ids=[101, 102, 103, 104], rows=rows)
        primary = summary["primary_FULL_minus_PLAIN"]
        self.assertAlmostEqual(primary["point_pp"], -20/3)
        self.assertEqual(primary["CI95_pp"], [-20.0, 20.0])
        self.assertEqual(primary["sampling_unit"], "image")
        self.assertEqual(primary["draws"], 10000)
        self.assertEqual(primary["margin_decision"], "DOES_NOT_CLEAR_MARGIN")
        self.assertEqual(summary["scope"]["unpaired_GT"], 1)

    def test_pixel_pooling_and_only_predeclared_subgroups(self):
        rows = [row(11, 101, 1, 0), row(22, 102, 0, 1, area=10)]
        control = SimpleNamespace(check=lambda: None)
        summary = summarize_pairs(np, control, image_ids=[11, 22], all_annotation_ids=[101, 102, 103],
                                  rows=rows, predeclared_subgroups={"synthetic_group": [101, 103]})
        contrast = summary["descriptive"]["contrasts"]["FULL_minus_PLAIN"]
        self.assertEqual(contrast["shared_error_total"], dict(pixels=0, instance_mean_pp=-5.0, pixel_pooled_pp=0.0))
        self.assertEqual(set(summary["subgroups"]), {"synthetic_group"})
        subgroup = summary["subgroups"]["synthetic_group"]
        self.assertEqual(subgroup["scope"], dict(total_GT=2, paired_GT=1, unpaired_GT=1))
        self.assertEqual(subgroup["descriptive"]["contrasts"]["FULL_minus_PLAIN"]["shared_error_total"],
                         dict(pixels=-1, instance_mean_pp=-20.0, pixel_pooled_pp=-20.0))

    def test_native_pair_must_match_saved_query_and_detection(self):
        from adapter import select_fixed_pairs

        ids = [11]
        native = [dict(image_id=11, query_id=q, category_id=1, score=0.5) for q in range(100)]
        annotations = {idx: dict(record, id=idx) for idx, record in enumerate(native, 1)}
        params = SimpleNamespace(imgIds=ids, catIds=[1], maxDets=[1, 10, 100], iouType="segm",
                                 iouThrs=np.linspace(.5, .95, 10), areaRngLbl=["all"], areaRng=[[0, 1e10]])
        evaluations = [dict(aRng=[0, 1e10], maxDet=100, category_id=1, image_id=11, gtIds=[101], gtIgnore=[0])]
        evaluation = SimpleNamespace(params=params, _paramsEval=params, evalImgs=evaluations,
                                     cocoDt=SimpleNamespace(anns=annotations))
        pair = dict(image_id=11, GT_annotation_id=101, dt_id=1, query_id=0)
        accepted = dict(dataset_index=0, image_id=11, GT_annotation_id=101, native_dt_id=1, query_id=0)
        selector = lambda *args: [pair]
        result, ignored = select_fixed_pairs(np, evaluation, native, ids, [101], [accepted], selector)
        self.assertEqual((result, ignored), ([pair], 0))
        with self.assertRaisesRegex(ValueError, "accepted native pair identities"):
            select_fixed_pairs(np, evaluation, native, ids, [101], [dict(accepted, query_id=1)], selector)

    def test_complete_scene_mapping_and_strict_margin(self):
        control = SimpleNamespace(check=lambda: None)
        ids = [11, 22, 33]
        scene_map = {11: "scene_a", 22: "scene_a", 33: "scene_b"}
        for area, decision in ((100, "DOES_NOT_CLEAR_MARGIN"), (50, "FEWER_COMMON_ERRORS_BEYOND_MARGIN")):
            with self.subTest(area=area):
                rows = [row(image, annotation, 1, 0, area) for image, annotation in zip(ids, [101, 102, 103])]
                summary = summarize_pairs(np, control, image_ids=ids, all_annotation_ids=[101, 102, 103],
                                          rows=rows, verified_scene_by_image=scene_map)
                primary = summary["primary_FULL_minus_PLAIN"]
                self.assertEqual((primary["sampling_unit"], primary["clusters"]), ("scene", 2))
                self.assertEqual(primary["margin_decision"], decision)
        with self.assertRaisesRegex(ValueError, "Complete externally verified scene mapping"):
            summarize_pairs(np, control, image_ids=ids, all_annotation_ids=[101, 102, 103], rows=rows,
                            verified_scene_by_image={11: "scene_a", 33: "scene_b"})

    def test_zero_pair_clusters_are_retained_and_undefined_draws_reported(self):
        control = SimpleNamespace(check=lambda: None)
        rows = [row(11, 101, 1, 0), row(22, 102, 0, 1)]
        summary = summarize_pairs(np, control, image_ids=[11, 22, 33], all_annotation_ids=[101, 102, 103], rows=rows)
        primary = summary["primary_FULL_minus_PLAIN"]
        self.assertEqual((primary["clusters"], primary["clusters_with_pairs"]), (3, 2))
        self.assertEqual(primary["draws_executed"], 10000)
        self.assertLess(primary["valid_draws"], 10000)
        self.assertIsNone(primary["CI95_pp"])
        self.assertEqual(primary["margin_decision"], "UNRESOLVED_UNDEFINED_REPLICATES")
