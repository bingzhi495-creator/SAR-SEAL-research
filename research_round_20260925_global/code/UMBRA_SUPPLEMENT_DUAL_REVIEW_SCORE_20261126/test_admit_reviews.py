import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from admit_reviews import freeze_originals


class AdmissionTest(TestCase):
    def test_preserves_original_bytes_and_roles_with_skipped_ids(self):
        ids = [f"UMB-{n:03d}" for n in range(101, 120) if n != 110]
        roster = {"schema": "sar-hull-roster-v1", "images": [
            {"image_id": key, "width": 1024, "height": 1024, "png_sha256": "0" * 64}
            for key in ids
        ]}
        entries = [
            {"image_id": key, "width": 1024, "height": 1024, "status": "complete_empty",
             "notes": "", "modes": ["native"], "shapes": []} for key in ids
        ]
        encode = lambda doc: json.dumps(doc).encode()
        returns = {
            role: encode({"schema": "sar-hull-review-v1", "reviewer": role, "images": entries})
            for role in ("A", "B")
        }
        validator = Path(os.environ.get("UMBRA_TEST_CODE", Path(__file__).resolve().parents[1])) / "UMBRA_PROSPECTIVE_GEC_ANNOTATION_20261122/validate_reviews.py"
        with TemporaryDirectory() as directory:
            output = Path(directory) / "admission"
            receipt = freeze_originals(validator, encode(roster), returns, output, {"test_only": True})
            self.assertEqual((output / "REVIEWER_A_ORIGINAL.json").read_bytes(), returns["A"])
            self.assertEqual(receipt["agreement"]["jointly_completed_windows"], 18)
            self.assertEqual(receipt["scope_windows"], 18)
            with self.assertRaises(ValueError):
                freeze_originals(validator, encode(roster), {"A": returns["B"], "B": returns["A"]},
                                 Path(directory) / "reversed", {"test_only": True})
