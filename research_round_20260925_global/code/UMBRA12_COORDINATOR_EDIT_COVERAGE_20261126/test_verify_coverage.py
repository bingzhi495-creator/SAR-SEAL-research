"""Generated coordinator coverage test; no detector or review files."""
from pathlib import Path
import json
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_coverage import aggregate


def slot(swaps, counts):
    return {"swaps": swaps, "rle": {"size": [1024, 1024], "counts": counts}}


def ledger(changes):
    rows = [{"image_id": i, "changes": changes.get(i, {})} for i in range(1, 13)]
    return "\n".join(json.dumps(row) for row in rows).encode()


class CoverageTest(unittest.TestCase):
    def test_counts_exact_equal_budget_differences_without_disclosing_acquisitions(self):
        a, b = slot(1, "ABC"), slot(1, "ABD")
        payload = ledger({
            1: {"5": {"fixed": a, "free_matched_count": b},
                "8": {"fixed": a, "free_matched_count": a}},
            7: {"3": {"fixed": b, "free_matched_count": a}},
            9: {"1": {"free_natural": b}},
        })
        self.assertEqual(aggregate(payload), {
            "positive_fixed_pair_acquisitions": 2, "positive_fixed_pair_slots": 3,
            "different_equal_count_acquisitions": 2, "different_equal_count_slots": 2,
        })

    def test_different_realized_budgets_fail(self):
        one, two = slot(1, "ABC"), slot(2, "ABC")
        with self.assertRaisesRegex(ValueError, "per-slot signed budgets differ"):
            aggregate(ledger({2: {"3": {"fixed": one, "free_matched_count": two}}}))

    def test_missing_equal_count_arm_fails(self):
        with self.assertRaisesRegex(ValueError, "both equal-count arms"):
            aggregate(ledger({2: {"3": {"fixed": slot(1, "ABC")}}}))

    def test_duplicate_slots_fail(self):
        raw = ledger({1: {"5": {"free_natural": slot(1, "ABC")}}})
        double_slot = raw.replace(b'"5": {', b'"5": {}, "5": {', 1)
        with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
            aggregate(double_slot)

    def test_missing_acquisition_fails(self):
        with self.assertRaisesRegex(ValueError, "twelve frozen acquisition rows"):
            aggregate(ledger({}).split(b"\n", 1)[1])

    def test_pinned_input_bytes_fail_when_tampered(self):
        from tempfile import TemporaryDirectory
        from verify_coverage import digest, pin

        with TemporaryDirectory() as folder:
            artifact = Path(folder).resolve() / "generated.json"
            artifact.write_bytes(b"altered")
            with self.assertRaisesRegex(ValueError, "Pinned bytes differ"):
                pin(artifact, digest(b"expected"))


if __name__ == "__main__":
    unittest.main()
