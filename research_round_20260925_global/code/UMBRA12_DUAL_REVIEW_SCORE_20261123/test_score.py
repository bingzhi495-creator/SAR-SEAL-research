"""Synthetic integration checks for the twelve-window scorer. This file never opens real labels or predictions."""
import argparse
from contextlib import redirect_stdout
from importlib.util import module_from_spec, spec_from_file_location
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from pycocotools import mask as codec
from shapely.geometry import Polygon, box


HERE = Path(__file__).resolve().parent
CODE = None


def load_score():
    spec = spec_from_file_location("umbra_score_under_test", HERE / "score.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_geometry(score):
    return score.source_module(CODE / "UEIKAP_RESERVED_HULL_FROZEN_20261122/score.py", score.CORE_SHA)


def rle(bitmap):
    result = codec.encode(np.asfortranarray(bitmap, dtype=np.uint8))
    result["counts"] = result["counts"].decode("ascii")
    return result


def balanced_fixture(hw):
    masks, amendments, truth_a, truth_b = [], {}, [], []
    for slot, x in enumerate((1, 13, 25), 1):
        stock = np.zeros(hw, dtype=np.uint8)
        stock[1:5, x:x + 8] = 1
        stock[1, x] = 0
        stock[1, x + 8] = 1
        fixed, free = stock.copy(), stock.copy()
        fixed[1, x + 8], fixed[2, x + 8] = 0, 1
        free[1, x + 8], free[1, x] = 0, 1
        masks.append(rle(stock))
        amendments[str(slot)] = {
            "fixed": {"rle": rle(fixed), "swaps": 1},
            "free_matched_count": {"rle": rle(free), "swaps": 1},
        }
        truth_a.append(box(x, 1, x + 8, 5))
        truth_b.append(Polygon([(x+1, 1), (x+8, 1), (x+8, 2), (x+9, 2), (x+9, 3),
                                (x+8, 3), (x+8, 5), (x, 5), (x, 2), (x+1, 2)]))
    return masks, amendments, truth_a, truth_b


def original_review(role, ids, positive, negative):
    rows = []
    for ordinal, key in enumerate(ids):
        geometry = positive if role == "A" else negative
        shapes = [
            dict(id=index, label="ship_hull", points=[list(xy) for xy in polygon.exterior.coords[:-1]])
            for index, polygon in enumerate(geometry, 1)
        ] if ordinal < 8 else []
        # This completed-empty to complete disagreement must remain in the joint denominator.
        if role == "B" and ordinal == 8:
            shapes = [dict(id=1, label="ship_hull", points=[[60, 20], [68, 20], [68, 24], [60, 24]])]
        rows.append(dict(image_id=key, width=1024, height=1024, status="complete" if shapes else "complete_empty",
                         notes="", modes=["native"], shapes=shapes))
    return dict(schema="sar-hull-review-v1", reviewer=role, images=rows)


def write_fixture(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


class TestUmbraScoreIntegration(unittest.TestCase):
    def test_admission_failure_precedes_any_blind_payload_reads(self):
        score = load_score()
        with tempfile.TemporaryDirectory() as folder:
            # Absent freeze receipt. Blind files are sentinels and must never be accessed.
            parent = Path(folder)
            args = argparse.Namespace(code=CODE, protocol=CODE.parent / score.PROTOCOL_NAME,
                                      review_run=parent / "no_review_return", blind_run=parent / "secret",
                                      blind_receipt=parent / "secret_receipt.json", output=parent / "score")
            with patch.object(score, "read_blind_frames", side_effect=AssertionError("opened blind capture early")):
                with self.assertRaisesRegex(FileNotFoundError, "observed_request"):
                    score.execute(args)
            self.assertFalse(args.output.exists())

    def test_native_rank_gaps_do_not_change_edit_slot(self):
        score = load_score()
        core = source_geometry(score)
        masks, amendments, _, _ = balanced_fixture(score.HW)
        pred, audit, edits = [], [], []
        for ordinal, key in enumerate(score.IDS, 1):
            pred.append(dict(image_id=ordinal, raw_query_count=164, native_count=100,
                             unmapped_slot1_count=98, native_ranks=[4, 97], coco=[
                                 dict(image_id=ordinal, category_id=1, score=.9, segmentation=masks[0]),
                                 dict(image_id=ordinal, category_id=1, score=.1, segmentation=masks[1])]))
            audit.append(dict(image_id=ordinal, acquisition_group=key, source_image_name=key + ".png",
                              original_hw=list(score.HW), sensor="Umbra", rgb_sha256=f"{ordinal:064x}",
                              native_ranks=[4, 97], candidates=[
                                  dict(position=1, native_rank=4, score=.9, mask_area=32),
                                  dict(position=2, native_rank=97, score=.1, mask_area=32)]))
            edits.append(dict(image_id=ordinal, changes={"2": amendments["2"]} if ordinal == 1 else {}))
        frames, identities = score.build_frames(core, (pred, audit, edits))
        self.assertEqual(len(identities), 12)
        self.assertEqual(frames["UMB-001"][1][2]["fixed"], (1, amendments["2"]["fixed"]["rle"]))
        pred[0]["native_ranks"] = [4, 96]
        with self.assertRaisesRegex(ValueError, "ranks"):
            score.build_frames(core, (pred, audit, edits))

    def test_blind_ledger_requires_exact_observed_receipts_and_payload(self):
        score = load_score()
        core = source_geometry(score)
        with tempfile.TemporaryDirectory(prefix="umbra-synthetic-ledger-") as folder:
            root = Path(folder)
            for phase in score.PHASES:
                (root / phase).mkdir()
                (root / f"observed_{phase}").mkdir()
            predictions, audits, edits = [], [], []
            for ordinal, key in enumerate(score.IDS, 1):
                predictions.append(dict(image_id=ordinal, raw_query_count=164, native_count=100,
                                       unmapped_slot1_count=100, native_ranks=[], coco=[]))
                audits.append(dict(image_id=ordinal, acquisition_group=key, source_image_name=key + ".png",
                                   original_hw=list(score.HW), sensor="Umbra", rgb_sha256=f"{ordinal:064x}",
                                   native_ranks=[], candidates=[]))
                edits.append(dict(image_id=ordinal, changes={}))
            paths = {"prediction_ledger_sha256": root / "capture/PREDICTIONS_PRIVATE.jsonl",
                     "candidate_audit_sha256": root / "capture/CANDIDATE_AUDIT_PRIVATE.jsonl",
                     "edit_ledger_sha256": root / "prepare-edits/EDITS_PRIVATE.jsonl"}
            for path, rows in zip(paths.values(), (predictions, audits, edits), strict=True):
                path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            hashes = {k: score.digest(v.read_bytes()) for k, v in paths.items()}
            observed = {}
            for phase in score.PHASES:
                result = dict(worker_sha256="fixture-worker")
                if phase == "verify-capture":
                    result.update(capture_sha256=observed["capture"]["result_sha256"], counts={"mapped": 0})
                if phase == "verify-edits":
                    result.update(prepared_result_sha256=observed["prepare-edits"]["result_sha256"],
                                  capture_readback_result_sha256=observed["verify-capture"]["result_sha256"],
                                  edits_sha256=hashes["edit_ledger_sha256"])
                result_path = write_fixture(root / phase / "RESULT_SAFE.json", result)
                receipt = dict(phase=phase, actual_exit=0, output_footer_verified=True,
                               worker_sha256=result["worker_sha256"], result_sha256=score.digest(result_path.read_bytes()))
                terminal = write_fixture(root / f"observed_{phase}" / "TERMINAL.json", receipt)
                observed[phase] = dict(receipt, terminal_sha256=score.digest(terminal.read_bytes()))
            blind = write_fixture(root / "fixture_qualified_blind.json",
                                  dict(status="FROZEN_BLIND_CANDIDATES_AND_EDIT_LEDGER_VERIFIED",
                                       protocol_sha256=score.PROTOCOL_SHA, anonymous_roster_sha256=score.ROSTER_SHA,
                                       images=12, detector_forwards=12, optimizer_updates=0, target_labels_opened=False,
                                       scoring_executed=False, exact_native_parity=True, exact_support_and_xor=True,
                                       exact_independent_replay=True, observed=observed, **hashes))
            args = argparse.Namespace(blind_receipt=blind, blind_run=root)
            with patch.object(score, "BLIND_RECEIPT_SHA", score.digest(blind.read_bytes())):
                frames, _ = score.read_blind_frames(core, args)
                self.assertEqual(len(frames), 12)
                self.assertTrue(all(frame == ([], {}) for frame in frames.values()))
                paths["edit_ledger_sha256"].write_text("{}", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "EDITS_PRIVATE"):
                    score.read_blind_frames(core, args)

    def test_group_weighting_and_positive_match_coverage_are_independent(self):
        score = load_score()
        core = source_geometry(score)
        metrics = score.source_module(HERE / "metrics.py", score.METRICS_SHA)
        hw = (24, 48)
        masks, amendments, hulls, _ = balanced_fixture(hw)
        frames = dict(A=(masks[:1], core.verify_edits({"1": amendments["1"]}, masks[:1], *hw)),
                      B=(masks[:1], core.verify_edits({"1": amendments["1"]}, masks[:1], *hw)), C=([], {}))
        result = metrics.score_reference(core, dict(A=hulls[:1], B=[hulls[0], box(42, 10, 46, 14)], C=[]), frames, hw)
        main = result["primary_all_labeled_hulls"]["contrasts"][core.PRIMARY]
        delta = 200 / 33
        self.assertAlmostEqual(main["mean_points"], delta * .75)
        np.testing.assert_allclose(main["paired_95_interval_points"], [delta * .5, delta])
        self.assertEqual(result["labeled_hulls_and_coverage"]["native_unmatched"], 1)
        self.assertFalse(result["coverage_gate_met"])

    def test_both_references_are_needed_for_advancement(self):
        score = load_score()
        core = source_geometry(score)
        metrics = score.source_module(HERE / "metrics.py", score.METRICS_SHA)
        hw = (24, 48)
        masks, amendments, a, b = balanced_fixture(hw)
        updates = core.verify_edits(amendments, masks, *hw)
        ids = score.IDS[:8]
        frames = {i: (masks, updates) for i in ids}
        positive = metrics.score_reference(core, dict.fromkeys(ids, a), frames, hw)
        negative = metrics.score_reference(core, dict.fromkeys(ids, b), frames, hw)
        reduced = metrics.score_reference(core, dict.fromkeys(ids[:7], a), frames, hw)
        self.assertTrue(positive["frozen_geometry_advance"])
        self.assertTrue(negative["frozen_geometry_adverse"])
        self.assertFalse(reduced["coverage_gate_met"])
        self.assertEqual(metrics.combine(positive, positive)["decision"], "positive")
        self.assertEqual(metrics.combine(negative, negative)["decision"], "adverse")
        self.assertEqual(metrics.combine(positive, negative)["decision"], "inconclusive")
        self.assertEqual(metrics.combine(positive, reduced)["decision"], "inconclusive")

    def test_empty_confirmed_hull_reference_yields_no_effect(self):
        score = load_score()
        core = source_geometry(score)
        metrics = score.source_module(HERE / "metrics.py", score.METRICS_SHA)
        reference = metrics.score_reference(core, {"UMB-001": []}, {"UMB-001": ([], {})}, score.HW)
        self.assertIsNone(reference["primary_all_labeled_hulls"])
        self.assertFalse(reference["coverage_gate_met"])
        self.assertEqual(reference["labeled_hulls_and_coverage"]["source_groups_without_reference_hulls"], 1)
        json.dumps(reference, allow_nan=False)

    def test_admitted_originals_feed_both_distinct_references(self):
        score = load_score()
        core = source_geometry(score)
        protocol = CODE.parent / score.PROTOCOL_NAME
        spec = spec_from_file_location("synthetic_test_umbra_admission", HERE / "admit_reviews.py")
        admission = module_from_spec(spec)
        with patch.dict(sys.modules, {spec.name: admission}):
            spec.loader.exec_module(admission)
        masks, amendments, a, b = balanced_fixture(score.HW)
        updates = core.verify_edits(amendments, masks, *score.HW)
        frames = {key: (masks, updates) if i < 8 else ([], {}) for i, key in enumerate(score.IDS)}
        synthetic_receipt = dict(prediction_ledger_sha256="fixture-only", edit_ledger_sha256="fixture-only",
                                 candidate_audit_sha256="fixture-only")
        with tempfile.TemporaryDirectory(prefix="umbra-synthetic-test-") as folder:
            root = Path(folder)
            roster = write_fixture(root / "fixture_roster.json", dict(schema="sar-hull-roster-v1", images=[
                dict(image_id=key, width=1024, height=1024, png_sha256=f"{i+1:064x}")
                for i, key in enumerate(score.IDS)]))
            fixture_roster_sha = score.digest(roster.read_bytes())
            review_a = write_fixture(root / "fixture_a.json", original_review("A", score.IDS, a, b))
            review_b = write_fixture(root / "fixture_b.json", original_review("B", score.IDS, a, b))
            argv = ["admit_reviews.py", "--validator", str(CODE / "UMBRA_PROSPECTIVE_GEC_ANNOTATION_20261122/validate_reviews.py"),
                    "--protocol", str(protocol), "--roster", str(roster),
                    "--a", str(review_a), "--a-sha256", score.digest(review_a.read_bytes()),
                    "--b", str(review_b), "--b-sha256", score.digest(review_b.read_bytes()),
                    "--output", str(root / "request")]
            with patch.object(admission, "ROSTER_SHA256", fixture_roster_sha), patch.object(sys, "argv", argv):
                with redirect_stdout(StringIO()):
                    admission.main()
            admitted = root / "request" / "RESULT_SAFE.json"
            (root / "observed_request").mkdir()
            # A simulated observer envelope is confined to this explicitly synthetic fixture.
            write_fixture(root / "observed_request" / "TERMINAL.json",
                          dict(phase="request", actual_exit=0, output_footer_verified=True,
                               worker_sha256=score.ADMISSION_WORKER_SHA, result_sha256=score.digest(admitted.read_bytes())))
            args = argparse.Namespace(code=CODE, protocol=protocol, review_run=root, output=root / "unused",
                                      blind_run=root / "fixture_only", blind_receipt=root / "fixture_only.json")
            with patch.object(score, "ROSTER_SHA", fixture_roster_sha):
                with patch.object(score, "read_blind_frames", return_value=(frames, synthetic_receipt)) as reader:
                    report = score.execute(args)
                    reader.assert_called_once()
                self.assertEqual(report["jointly_completed_windows"], 12)
                self.assertEqual(report["agreement"]["a_ship_hulls"], 24)
                self.assertEqual(report["agreement"]["b_ship_hulls"], 25)
                self.assertTrue(report["references"]["A"]["frozen_geometry_advance"])
                self.assertTrue(report["references"]["B"]["frozen_geometry_adverse"])
                self.assertEqual(report["references"]["B"]["labeled_hulls_and_coverage"]["native_unmatched"], 1)
                self.assertEqual(report["joint_decision"]["decision"], "inconclusive")
                json.dumps(report, allow_nan=False)
                (root / "request" / "REVIEWER_B_ORIGINAL.json").write_text("{}", encoding="utf-8")
                with patch.object(score, "read_blind_frames", side_effect=AssertionError("review bytes changed")):
                    with self.assertRaisesRegex(ValueError, "REVIEWER_B_ORIGINAL"):
                        score.execute(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code", type=Path, required=True)
    args = parser.parse_args()
    CODE = args.code.resolve(strict=True)
    unittest.main(argv=[__file__], verbosity=2)
