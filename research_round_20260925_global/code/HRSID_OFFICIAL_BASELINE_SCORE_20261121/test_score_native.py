"""Generated COCO examples only. Point NATIVE_SCORE_ADAPTER at the ordinary published adapter."""
from contextlib import redirect_stdout
from copy import deepcopy
from hashlib import sha256
from io import BytesIO, StringIO
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from pycocotools import mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import score_native as score


SHAPE = (8, 8)
IDS = [20, 5, 11]


def rle(bitmap):
    value = mask.encode(np.asfortranarray(bitmap, dtype=np.uint8))
    return dict(size=value["size"], counts=value["counts"].decode("ascii"))


def fixture():
    poly = [[1, 1, 4, 1, 4, 4, 1, 4]]
    def annotation(i, image_id, points=poly):
        return dict(id=i, image_id=image_id, category_id=1, bbox=[1,1,3,3], area=9, iscrowd=0, segmentation=deepcopy(points))
    gt = dict(info={}, images=[dict(id=i, width=8, height=8) for i in IDS], categories=[dict(id=1, name="ship")],
              annotations=[annotation(10,5), annotation(11,5,[[1,2,2,2,4,2]]), annotation(12,11)])
    views = dict(official=dict(images=IDS, annotations=[10,11,12]),
                 empty_raster_sensitivity=dict(images=IDS, annotations=[10,12]),
                 image_identity_sensitivity=dict(images=IDS[:2], annotations=[10,11]))
    bitmap = np.zeros(SHAPE, dtype=np.uint8)
    bitmap[1:4, 1:4] = 1
    encoded = rle(bitmap)
    zero = rle(np.zeros_like(bitmap))
    def prediction(i, confidence, segmentation):
        return dict(image_id=i, category_id=1, score=confidence, segmentation=segmentation)
    def block(i, ranks, rows):
        return dict(image_id=i, raw_query_count=164, native_count=100, unmapped_slot1_count=100-len(rows),
                    native_ranks=ranks, coco=rows)
    return gt, views, [block(20,[4],[prediction(20,.99,encoded)]),
                       block(5,[1,99],[prediction(5,.81,encoded),prediction(5,.81,zero)]), block(11,[],[])]


def parse(blocks, expected=None):
    raw = b"".join((json.dumps(b) + "\n").encode() for b in blocks)
    stats = dict(predictions_sha256=sha256(raw).hexdigest(), mapped_rows=sum(len(b["coco"]) for b in blocks), prediction_bytes=len(raw))
    stats.update(expected or {})
    return score.parse_native(BytesIO(raw), IDS, *SHAPE, stats)


class ScoreChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = score.adapter_from(Path(os.environ["NATIVE_SCORE_ADAPTER"]))

    def test_metrics_match_direct_stock_evaluator_and_preserve_inputs(self):
        gt, views, blocks = fixture()
        rows, digests = parse(blocks)
        original = deepcopy((gt, rows))
        measured = score.evaluate_views(gt, views, rows, digests, self.adapter)
        self.assertEqual((gt, rows), original)
        for name, view in views.items():
            local = deepcopy(gt)
            local["images"] = [i for i in local["images"] if i["id"] in view["images"]]
            local["annotations"] = [a for a in local["annotations"] if a["id"] in view["annotations"]]
            expected = COCO()
            expected.dataset = local
            with redirect_stdout(StringIO()):
                expected.createIndex()
                pred = expected.loadRes(deepcopy([r for i in view["images"] for r in rows[i]]))
                ev = COCOeval(expected, pred, "segm")
                ev.evaluate()
                ev.accumulate()
                ev.summarize()
            direct = list(ev.stats)
            at90 = ev.eval["precision"][8,:,:,0,2]
            metrics = measured[name]["metrics"]
            self.assertEqual([metrics[m] for m in score.METRICS], direct)
            self.assertEqual(metrics["AP90"], at90[at90 > -1].mean())
            self.assertEqual(measured[name]["images_without_annotations"], 1)
        primary = measured["official"]
        self.assertEqual(primary["images_without_predictions"], 1)
        self.assertEqual(primary["mapped_predictions"], 3)
        self.assertEqual(primary["ordered_prediction_block_digests_sha256"], measured["empty_raster_sensitivity"]["ordered_prediction_block_digests_sha256"])
        self.assertGreater(measured["empty_raster_sensitivity"]["metrics"]["AP"], primary["metrics"]["AP"])
        self.assertGreater(measured["image_identity_sensitivity"]["metrics"]["AP"], primary["metrics"]["AP"])

    def test_negative_image_false_positive_lowers_ap(self):
        gt, views, blocks = fixture()
        rows, digests = parse(blocks)
        score_with_fp = score.evaluate_views(gt, views, rows, digests, self.adapter)
        blocks[0].update(unmapped_slot1_count=100, native_ranks=[], coco=[])
        rows, digests = parse(blocks)
        without_fp = score.evaluate_views(gt, views, rows, digests, self.adapter)
        self.assertGreater(without_fp["official"]["metrics"]["AP"], score_with_fp["official"]["metrics"]["AP"])

    def test_globally_empty_export_retains_entire_population(self):
        gt, views, blocks = fixture()
        for b in blocks:
            b.update(unmapped_slot1_count=100, native_ranks=[], coco=[])
        rows, digests = parse(blocks)
        result = score.evaluate_views(gt, views, rows, digests, self.adapter)
        self.assertEqual(result["official"]["images_without_predictions"], 3)
        self.assertEqual(result["official"]["metrics"]["AP"], 0)

    def test_malformed_and_random_compressed_rles(self):
        rng = np.random.RandomState(5217)
        for shape in [(8,8), (39,42), (800,800)]:
            for density in (0, .01, 1):
                bitmap = rng.rand(*shape) < density
                score.validate_rle(rle(bitmap), *shape)
        for text in ("", "o", "1", "!", "ooooooo"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                score.validate_rle(dict(size=[8,8], counts=text), 8,8)

    def test_sealed_order_bytes_and_sparse_slots(self):
        _, _, original = fixture()
        for key, val in (("predictions_sha256","3"*64), ("mapped_rows",12), ("prediction_bytes",7)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                parse(original, {key:val})
        for field, value in (("native_ranks",[1,1]), ("native_ranks",[99,1]), ("native_ranks",[True,2]), ("unmapped_slot1_count",97)):
            blocks = deepcopy(original)
            blocks[1][field] = value
            with self.subTest(field=field,value=value), self.assertRaises(ValueError):
                parse(blocks)
        for corrupt in (list(reversed(original)), original[:-1], original + original[:1]):
            with self.assertRaises(ValueError):
                parse(corrupt)
        blocks = deepcopy(original)
        blocks[1]["coco"][1]["score"] = .9
        with self.assertRaises(ValueError):
            parse(blocks)

    def test_sensitivities_cannot_change_unrelated_annotations_or_drop_negative_image(self):
        gt, views, blocks = fixture()
        rows, digests = parse(blocks)
        for name, field in (("empty_raster_sensitivity", "images"), ("image_identity_sensitivity", "annotations")):
            broken = deepcopy(views)
            broken[name][field] = broken[name][field][1:]
            with self.assertRaises(ValueError):
                score.evaluate_views(gt, broken, rows, digests, self.adapter)

    def test_wrong_phase_and_wrong_cohort_reject_before_any_payload_open(self):
        with TemporaryDirectory() as work:
            path = Path(work)
            admission = dict(phase=score.PHASE, status=score.ADMITTED, endpoint_sha256=score.ENDPOINT,
                             cohorts_sha256=score.COHORT, gt_sha256=score.GT, adapter_sha256=score.ADAPTER,
                             score_source_sha256=score.digest(Path(score.__file__).read_bytes()), runtime_versions=score.runtime(),
                             export_readback_sha256="1"*64, predictions_sha256="2"*64, prediction_bytes=100, mapped_rows=0)
            meta, cohorts = path / "admit", path / "cohorts"
            cohorts.write_text("{}")
            base = ["--admission", str(meta), "--cohorts", str(cohorts), "--export-readback", str(path/"absent"),
                    "--gt", str(path/"absent-gt"), "--predictions", str(path/"absent-preds"), "--adapter", str(path/"absent-adapter"),
                    "--output", str(path/"absent-out")]
            for phase, pattern in (("input_only","explicit native"), (score.PHASE,"Pinned file bytes")):
                raw = json.dumps(dict(admission, phase=phase)).encode()
                meta.write_bytes(raw)
                with self.assertRaisesRegex(ValueError, pattern):
                    score.main(base + ["--admission-sha256", score.digest(raw)])
                self.assertFalse((path/"absent-out").exists())

    def test_export_closure_requires_observed_exit_and_exact_prediction_digest(self):
        adm = dict(endpoint_sha256=score.ENDPOINT, predictions_sha256="1"*64, mapped_rows=100, prediction_bytes=80000)
        valid = dict(adm, status=score.EXPORT_ACCEPTED, original_process_terminal=True, original_exit=0,
                     images=1962, GT_opened=False, AP=False)
        score.validate_export_closure(valid, adm)
        for key, value in (("original_process_terminal",False), ("original_exit",None), ("original_exit",False), ("GT_opened",True),
                           ("AP",True), ("images",1961), ("predictions_sha256","2"*64)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                score.validate_export_closure(dict(valid, **{key:value}), adm)


if __name__ == "__main__":
    unittest.main()
