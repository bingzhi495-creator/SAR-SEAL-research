"""Check the reviewer return CLI with generated polygons only."""

import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


PROGRAM = Path(__file__).with_name("validate_reviews.py")
VALID_BOX = [[1.125, 1], [9.875, 1], [9.875, 9], [1.125, 9]]


def entry(number, status, shapes=None):
    return {
        "image_id": f"UMB-{number:03d}",
        "width": 1024,
        "height": 1024,
        "status": status,
        "notes": "",
        "modes": ["native"],
        "shapes": [] if shapes is None else shapes,
    }


def shape(identifier, label, points=VALID_BOX):
    return {"id": identifier, "label": label, "points": points}


def box(x):
    return [[x, 10], [x + 10, 10], [x + 10, 20], [x, 20]]


class ReviewReturnTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def fixture(self, name, data):
        path = self.root / name
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def roster(self, count):
        images = [
            dict(image_id=row["image_id"], width=row["width"], height=row["height"],
                 png_sha256="a" * 64)
            for row in (entry(i, "unreviewed") for i in range(1, count + 1))
        ]
        return self.fixture("roster.json", {"schema": "sar-hull-roster-v1", "images": images})

    def review(self, role, rows):
        return self.fixture(f"{role}.json", {
            "schema": "sar-hull-review-v1", "reviewer": role, "images": rows
        })

    def cli(self, *args):
        return subprocess.run([sys.executable, "-B", str(PROGRAM), *map(str, args)],
                              capture_output=True, text=True, encoding="utf-8")

    def test_one_return_preserves_empty_incomplete_and_ignored_counts(self):
        review = self.review("A", [
            entry(1, "complete", [shape(1, "uncertain")]),
            entry(2, "complete_empty"),
            entry(3, "incomplete", [shape(1, "ship_hull"), shape(2, "truncated")]),
            entry(4, "unreviewed"),
        ])
        result = self.cli("validate", "--roster", self.roster(4), "--annotation", review)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["review"]["images_by_status"], {
            "complete": 1, "complete_empty": 1, "incomplete": 1, "unreviewed": 1,
        })
        self.assertEqual(report["review"]["shapes_by_label"], {
            "ship_hull": 1, "truncated": 1, "uncertain": 1,
        })
        self.assertEqual(report["review"]["complete_windows_with_hulls"], 0)

    def test_agreement_uses_maximum_accepted_pairs_before_similarity(self):
        a = self.review("A", [
            entry(1, "complete", [shape(1, "ship_hull", box(3)), shape(2, "ship_hull", box(6)),
                                  shape(3, "uncertain", box(0))]),
            entry(2, "complete_empty"),
            entry(3, "incomplete", [shape(1, "ship_hull")]),
            entry(4, "complete", [shape(1, "uncertain")]),
        ])
        b = self.review("B", [
            entry(1, "complete", [shape(1, "ship_hull", box(3)), shape(2, "ship_hull", box(0)),
                                  shape(3, "truncated", box(3))]),
            entry(2, "complete_empty"),
            entry(3, "complete", [shape(1, "ship_hull")]),
            entry(4, "complete", [shape(1, "ship_hull")]),
        ])
        result = self.cli("agreement", "--roster", self.roster(4), "--a", a, "--b", b)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        agreement = report["agreement"]
        self.assertEqual(agreement["jointly_completed_windows"], 3)
        self.assertEqual(agreement["accepted_pairs"], 2)
        self.assertEqual(agreement["a_unmatched_hulls"], 0)
        self.assertEqual(agreement["b_unmatched_hulls"], 1)
        self.assertAlmostEqual(agreement["symmetric_instance_f1"], 0.8)
        self.assertAlmostEqual(agreement["matched_continuous_iou_mean"], 7 / 13)
        self.assertEqual(
            [(item["a_shape_id"], item["b_shape_id"]) for item in report["windows"][0]["pairs"]],
            [(1, 2), (2, 1)],
        )
        self.assertIsNone(report["windows"][2]["pairs"])

    def test_invalid_outlines_fail_before_a_report_is_written(self):
        roster = self.roster(1)
        cases = [
            ("self-intersection", [[1, 1], [8, 8], [1, 8], [8, 1]]),
            ("collinear", [[1, 1], [2, 2], [3, 3]]),
            ("negative x", [[-0.125, 1], [9, 1], [9, 9], [1, 9]]),
            ("too-large y", [[1, 1], [9, 1], [9, 1024.125], [1, 9]]),
            ("nonfinite", [[1, 1], [9, 1], [9, float("nan")], [1, 9]]),
        ]
        for outcome, points in cases:
            with self.subTest(outcome=outcome):
                review = self.review("A", [entry(1, "complete", [shape(1, "ship_hull", points)])])
                output = self.root / "should_not_exist.json"
                result = self.cli("validate", "--roster", roster, "--annotation", review, "--output", output)
                self.assertEqual(result.returncode, 2)
                self.assertIn("validation failed", result.stderr)
                self.assertFalse(output.exists())

    def test_missing_and_duplicate_roster_ids_and_inconsistent_statuses_fail(self):
        roster = self.roster(2)
        cases = [
            [entry(1, "complete_empty")],
            [entry(1, "complete_empty"), entry(1, "complete_empty")],
            [entry(1, "complete_empty"), entry(3, "complete_empty")],
            [entry(1, "complete"), entry(2, "complete_empty")],
            [entry(1, "complete_empty"), entry(2, "complete_empty", [shape(1, "ship_hull")])],
            [entry(1, "unreviewed"), entry(2, "unreviewed", [shape(1, "ship_hull"), shape(1, "uncertain")])],
        ]
        for i, rows in enumerate(cases):
            with self.subTest(case=i):
                result = self.cli("validate", "--roster", roster, "--annotation", self.review("A", rows))
                self.assertEqual(result.returncode, 2)
                self.assertIn("validation failed", result.stderr)
                self.assertFalse(result.stdout)

    def test_inclusive_iou_threshold_and_native_edge_coordinates(self):
        # On 1024^2, these 9-by-8 boxes overlap in a 6-by-8 rectangle.
        a_box = [[1015, 1016], [1024, 1016], [1024, 1024], [1015, 1024]]
        b_box = [[1012, 1016], [1021, 1016], [1021, 1024], [1012, 1024]]
        a = self.review("A", [entry(1, "complete", [shape(1, "ship_hull", a_box)])])
        b = self.review("B", [entry(1, "complete", [shape(1, "ship_hull", b_box)])])
        result = self.cli("agreement", "--roster", self.roster(1), "--a", a, "--b", b)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["agreement"]["accepted_pairs"], 1)
        self.assertEqual(report["agreement"]["matched_continuous_iou_mean"], 0.50)


if __name__ == "__main__":
    unittest.main()
