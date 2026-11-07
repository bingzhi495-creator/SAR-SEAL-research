"""Fabricated metadata and overlap matrices for the RF-DETR DEV adapter."""

import unittest
from types import SimpleNamespace

from adapter import EXPORT_SCHEMA, normalize_export


def prediction(image, score, counts):
    return dict(image_id=image, category_id=1, score=score, segmentation=dict(size=[2, 3], counts=counts))


def image_block(image, ranks, rows):
    return dict(image_id=image, raw_query_count=100, native_count=100,
                unmapped_slot1_count=100-len(rows), native_ranks=ranks, coco=rows)


class ExportTests(unittest.TestCase):
    def test_retained_native_order_includes_an_empty_image(self):
        images = [dict(id=5, height=2, width=3), dict(id=8, height=2, width=3)]
        rows = [prediction(5, .7, "a"), prediction(5, .01, "b")]
        document = dict(schema=EXPORT_SCHEMA, images=[image_block(5, [2, 99], rows), image_block(8, [], [])])
        flat, blocks = normalize_export(document, images)
        self.assertEqual(flat, rows)
        self.assertEqual([(item["start"], item["native_ranks"]) for item in blocks], [(0, [2, 99]), (2, [])])

    def test_bad_rank_order_and_global_empty_fail_without_repair(self):
        images = [dict(id=5, height=2, width=3)]
        rows = [prediction(5, .7, "a"), prediction(5, .01, "b")]
        with self.assertRaisesRegex(ValueError, "retained stock all-slot ranks"):
            normalize_export(dict(schema=EXPORT_SCHEMA, images=[image_block(5, [99, 2], rows)]), images)
        with self.assertRaisesRegex(ValueError, "Globally empty mapped export"):
            normalize_export(dict(schema=EXPORT_SCHEMA, images=[image_block(5, [], [])]), images)

    def test_assignment_retains_the_actual_integer_coco_detection_row(self):
        from adapter import assigned_rows

        thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        np_stub = SimpleNamespace(linspace=lambda *args: thresholds, array_equal=lambda a, b: a == b)
        params = SimpleNamespace(iouThrs=thresholds, iouType="segm", imgIds=[5], catIds=[1],
                                 maxDets=[1, 10, 100], areaRngLbl=["all"], areaRng=[[0, 1e10]])
        rows = [prediction(5, .8, "a"), prediction(5, .3, "b")]
        detections = {index: dict(row, id=index) for index, row in enumerate(rows, 1)}
        evaluation = SimpleNamespace(params=params, _paramsEval=params, cocoDt=SimpleNamespace(anns=detections),
                                     evalImgs=[dict(aRng=[0, 1e10], maxDet=100, category_id=1, image_id=5,
                                                    gtIds=[7, 8, 9], gtIgnore=[0, 0, 1],
                                                    gtMatches=[[2, 0, 1]], dtIds=[1, 2])])
        self.assertEqual(assigned_rows(np_stub, evaluation, rows, [5], [7, 8, 9]),
                         {7: dict(ignored=False, dt_id=2), 8: dict(ignored=False, dt_id=None),
                          9: dict(ignored=True, dt_id=None)})
        detections[2]["segmentation"] = rows[0]["segmentation"]
        with self.assertRaisesRegex(ValueError, "original mapped detection identity"):
            assigned_rows(np_stub, evaluation, rows, [5], [7, 8, 9])

    def test_mapped_oracle_and_same_assigned_row_are_separate(self):
        from adapter import FLOAT, INTEGER, collect_geometry, summarize_geometry

        class Matrix:
            def __init__(self, values):
                self.values = values
                self.shape = (len(values), len(values[0]))

            def __getitem__(self, indices):
                row, column = indices
                return self.values[row][column]

        images = {5: dict(height=2, width=3), 8: dict(height=2, width=3)}
        def coco(prefix):
            annotations = {aid: dict(image_id=iid, category_id=1, iscrowd=0,
                                     segmentation=dict(size=[2, 3], counts=prefix+str(aid)))
                           for aid, iid in ((7, 5), (8, 5), (9, 8))}
            return SimpleNamespace(imgs=images, anns=annotations,
                                   getAnnIds=lambda imgIds, catIds: [aid for aid, ann in annotations.items()
                                                                    if ann["image_id"] in imgIds],
                                   annToRLE=lambda ann: ann["segmentation"])
        blocks = [dict(image_id=5, start=0, native_ranks=[2, 99], coco=[prediction(5, .8, "a"), prediction(5, .3, "b")]),
                  dict(image_id=8, start=2, native_ranks=[], coco=[])]
        assignments = {7: dict(ignored=False, dt_id=2), 8: dict(ignored=False, dt_id=None),
                       9: dict(ignored=False, dt_id=None)}
        mask = SimpleNamespace(iou=lambda dt, gt, crowd: Matrix([[.9, .2], [.55, .1]])
                               if gt[0]["counts"] == "i7" else Matrix([[.8, .2], [.45, .1]]))
        result = collect_geometry(SimpleNamespace(check=lambda: None), {INTEGER: coco("i"), FLOAT: coco("f")},
                                  blocks, assignments, mask)
        self.assertEqual(result[0]["assigned_native_rank"], 99)
        self.assertEqual(result[0]["by_GT"][INTEGER], dict(mapped_oracle_max_IoU=.9, mapped_oracle_best_native_rank=2,
                                                          fixed_assigned_IoU=.55))
        self.assertEqual(result[0]["by_GT"][FLOAT]["fixed_assigned_IoU"], .45)
        self.assertEqual(result[2]["by_GT"][FLOAT], dict(mapped_oracle_max_IoU=0.0, mapped_oracle_best_native_rank=None,
                                                        fixed_assigned_IoU=None))
        summary = summarize_geometry(result)
        self.assertEqual(summary["mapped_export_only_oracle"][FLOAT]["all_GT"]["threshold_counts"]["0.75"], 1)
        self.assertEqual(summary["integer_assigned_same_rows"][FLOAT]["threshold_counts"]["0.50"], 0)
        self.assertEqual(summary["GT_without_mapped_candidates"], 1)
        self.assertNotIn("image_id", str(summary))
