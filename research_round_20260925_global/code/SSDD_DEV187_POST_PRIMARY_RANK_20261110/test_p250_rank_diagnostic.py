"""Synthetic post-primary diagnostic with original COCO and coverage helpers."""
from copy import deepcopy
from os import environ
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from pycocotools import mask

from p250_rank_diagnostic import analyze_pair, bound_prediction_refs, load_support


def rle(x=None):
    canvas = np.zeros((12, 12), dtype=np.uint8, order="F")
    if x is not None:
        canvas[1:3, x:x + 2] = 1
    value = mask.encode(canvas)
    return dict(size=value["size"], counts=value["counts"].decode("ascii"))


def fixture():
    images = [dict(id=image, height=12, width=12) for image in (7, 3, 11)]
    gt = dict(info={}, images=images, categories=[dict(id=1, name="ship")], annotations=[
        dict(id=index, image_id=image, category_id=1, area=4, bbox=[1, 1, 2, 2],
             iscrowd=0, segmentation=rle(1)) for index, image in enumerate((7, 3), 1)
    ])

    def row(image, score, segmentation):
        return dict(image_id=image, category_id=1, score=score, segmentation=segmentation)

    hit, miss, empty = rle(1), rle(8), rle()
    a = [row(7, .99, empty), row(7, .90, hit), row(7, .05, miss), row(3, .10, miss), row(11, .70, empty)]
    b = [row(7, .10, miss), row(3, .90, hit), row(11, .80, miss)]
    return gt, a, b


class P250RankDiagnosticTest(unittest.TestCase):
    def test_exact_score_donor_invariance_and_same_equal_nonempty_coverage(self):
        support = load_support(Path(environ["P255_TEST_CODE"]).resolve(strict=True))
        inputs = fixture()
        before = deepcopy(inputs)
        result = analyze_pair(*inputs, support)
        self.assertEqual(inputs, before)
        self.assertEqual(result["prefix_counts"]["retained_per_arm"], 2)
        self.assertEqual(result["prefix_counts"]["native"]["mapped_empty"], 2)
        self.assertEqual(result["prefix_counts"]["center"]["discarded_nonempty_after_count_matching"], 1)
        self.assertEqual(result["matching_witness"]["native"]["all_area_dt_matches_cells"], 20)
        for mode, other in (("native", "center"), ("center", "native")):
            cells = result["cells"][mode]
            self.assertAlmostEqual(cells[mode]["AP"], 51 / 101, delta=1e-12)
            self.assertAlmostEqual(cells[other]["AP"], 51 / 202, delta=1e-12)
            self.assertEqual(cells[mode]["recall90"], cells[other]["recall90"])
            self.assertEqual(
                result["score_blind_coverage"]["arms"][mode]["0.90"]["maximum_one_to_one"], 1,
            )


    def test_maximum_matching_does_not_double_count_a_shared_candidate(self):
        support = load_support(Path(environ["P255_TEST_CODE"]).resolve(strict=True))
        gt, a, b = fixture()
        gt["images"] = gt["images"][:1]
        gt["annotations"][1]["image_id"] = 7
        native = a[1:3]
        center = [dict(b[1], image_id=7), dict(b[1], image_id=7)]
        result = analyze_pair(gt, native, center, support)["score_blind_coverage"]
        self.assertEqual(result["eligible_gt"], 2)
        for threshold in ("0.75", "0.90"):
            self.assertEqual(result["arms"]["native"][threshold],
                             {"any_candidate": 2, "maximum_one_to_one": 1, "maximum_recall": .5})
            self.assertEqual(result["arms"]["center"][threshold],
                             {"any_candidate": 2, "maximum_one_to_one": 2, "maximum_recall": 1.})

    def test_fails_if_per_image_ignore_changes_even_when_summary_recalls_match(self):
        support = load_support(Path(environ["P255_TEST_CODE"]).resolve(strict=True))
        template = support[1]
        original = template.evaluate_cell
        calls = 0

        def tamper(*args):
            nonlocal calls
            calls += 1
            metrics, matching, layout = original(*args)
            if calls == 2:
                item = next(row for row in matching if row is not None and row["dtIgnore"].size)
                item["dtIgnore"][0, 0] = not item["dtIgnore"][0, 0]
            return metrics, matching, layout

        with patch.object(template, "evaluate_cell", side_effect=tamper):
            with self.assertRaisesRegex(ValueError, "matching evidence changed"):
                analyze_pair(*fixture(), support)

    def test_plan_prediction_ref_must_match_independently_accepted_digest(self):
        p250 = load_support(Path(environ["P255_TEST_CODE"]).resolve(strict=True))[0]
        entries, accepted = [], []
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            for seed in p250.TRAIN_PROTOCOLS:
                for mode in p250.MODES:
                    index = len(entries) + 1
                    identity = dict(seed=seed, mode=mode, source_commit="a" * 40,
                                    checkpoint_sha256="b" * 64, source_pair_receipt_sha256="c" * 64,
                                    source_arm_receipt_sha256="d" * 64)
                    ref = dict(path=str(root / str(index) / p250.PREDICTION_NAME), bytes=1, sha256=f"{index:064x}")
                    entries.append(dict(identity, terminal={"sha256": "e" * 64}, report={"sha256": "f" * 64},
                                        predictions=ref))
                    metrics = dict.fromkeys(p250.METRICS, 0.)
                    metrics["all12"] = [0.] * 12
                    accepted.append(dict(identity, metrics=metrics, source_training_protocol=p250.TRAIN_PROTOCOLS[seed],
                                         mapped_rows=0, empty_image_predictions=187,
                                         export_terminal_sha256="e" * 64, export_report_sha256="f" * 64,
                                         predictions_sha256=ref["sha256"]))
            plan, score = {"arms": entries}, {"arms": accepted}
            refs, _ = bound_prediction_refs(p250, plan, score)
            self.assertEqual(len(refs), 6)
            entries[0]["predictions"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "digests differ"):
                bound_prediction_refs(p250, plan, score)


if __name__ == "__main__":
    unittest.main()
