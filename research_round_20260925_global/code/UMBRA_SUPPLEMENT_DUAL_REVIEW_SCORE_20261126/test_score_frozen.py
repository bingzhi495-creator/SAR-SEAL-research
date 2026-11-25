import importlib.util
import os
from pathlib import Path
from unittest import TestCase

from score_frozen import build_frames, evaluate, evaluators


class SupplementalScoreTest(TestCase):
    def test_original_metric_uses_supplemental_group_count(self):
        code = Path(os.environ.get("UMBRA_TEST_CODE", Path(__file__).resolve().parents[1]))
        path = code / "UMBRA12_DUAL_REVIEW_SCORE_20261123/test_score.py"
        spec = importlib.util.spec_from_file_location("legacy_fixture", path)
        fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)
        hw = (24, 48)
        mask_rows, edits, hulls, _ = fixture.balanced_fixture(hw)
        core, metrics = evaluators(code)
        edits = core.verify_edits(edits, mask_rows, *hw)
        ids = [f"UMB-{n:03d}" for n in range(101, 114) if n != 105]
        for count, decision in ((10, "inconclusive"), (12, "positive")):
            frames = {key: (mask_rows, edits) for key in ids[:count]}
            labels = {key: hulls for key in frames}
            scores, combined = evaluate(core, metrics, {role: labels for role in ("A", "B")}, frames, hw)
            self.assertEqual(combined["decision"], decision)
            self.assertEqual(scores["A"]["labeled_hulls_and_coverage"]["sf_positive_source_groups"], count)

    def test_preserves_gap_in_verified_blind_streams(self):
        code = Path(os.environ.get("UMBRA_TEST_CODE", Path(__file__).resolve().parents[1]))
        core, _ = evaluators(code)
        roster = ("UMB-101", "UMB-103")
        preds, audits, edits = [], [], []
        for key in roster:
            image_id = int(key[4:])
            preds.append(dict(image_id=image_id, raw_query_count=164, native_count=100,
                              unmapped_slot1_count=100, native_ranks=[], coco=[]))
            audits.append(dict(image_id=image_id, acquisition_group=key, source_image_name=key + ".png",
                               original_hw=[1024, 1024], sensor="Umbra", rgb_sha256=f"{image_id:064x}",
                               native_ranks=[], candidates=[]))
            edits.append(dict(image_id=image_id, changes={}))
        streams = preds, audits, edits
        self.assertEqual(build_frames(core, roster, streams), dict.fromkeys(roster, ([], {})))
        with self.assertRaises(ValueError):
            build_frames(core, tuple(reversed(roster)), streams)