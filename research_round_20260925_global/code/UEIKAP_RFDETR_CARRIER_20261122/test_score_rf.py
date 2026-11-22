"""Synthetic cross-carrier adapter checks. No external image or label is required."""

from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from pycocotools import mask as codec
import shapely

import score_rf as adapter


DEFAULT_CORE = Path(__file__).resolve().parent.parent / "UEIKAP_RESERVED_HULL_FROZEN_20261122" / "score.py"
core = adapter.load_geometry(Path(os.environ.get("UEIKAP_REFERENCE_SCORER", DEFAULT_CORE)))


def write(path, obj, stream=False):
    data = (b"".join(json.dumps(item).encode() + b"\n" for item in obj) if stream else json.dumps(obj).encode())
    path.write_bytes(data)
    return sha256(data).hexdigest()


def encode(bitmap):
    rle = codec.encode(np.asfortranarray(bitmap))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def fixture(root):
    expected_ids = [1, 2]
    groups = ["scene-a", "scene-b"]
    original = np.zeros((8, 8), np.uint8)
    original[:6, :6] = 1
    original[0, 0], original[6, 6] = 0, 1
    native = encode(original)
    better = original.copy()
    better[0, 0], better[6, 6] = 1, 0
    worse = original.copy()
    worse[0, 1], worse[7, 7] = 0, 1
    images, labels, members, records = [], [], [], []
    for i, group in zip(expected_ids, groups):
        path = root / f"lab{i}"
        raw = b"0 0 0 .75 0 .75 .75 0 .75\n1 0 0 .2 0 .2 .2\n"
        path.write_bytes(raw)
        digest = sha256(raw).hexdigest()
        images.append({"image_id": i, "original_hw": [8, 8]})
        labels.append({"image_id": i, "acquisition_group": group, "count_by_class": {"0": 1, "1": 1},
                       "label_ref": {"path": path.name, "bytes": len(raw), "sha256": digest}})
        members.extend([{"image_id": i, "zip_member": f"images/{i}.png"},
                        {"image_id": i, "zip_member": f"x/labels/{i}.txt", "bytes": len(raw), "sha256": digest}])
        records.append({"acquisition_group": group, "class0_candidate_polygons": 1,
                        "tiff_zip_name": f"x/images_tifs/{i}.tif"})
    ih = write(root / "images_roster.json", {"expected_ids": expected_ids, "items": images})
    th = write(root / "intake_terminal.json", {
        "status": "completed_verified", "actual_exit": 0, "images_roster_sha256": ih,
        "labels_roster_sha256": write(root / "labels_roster.json", {"expected_ids": expected_ids, "items": labels}),
        "member_provenance_sha256": write(root / "member_provenance.json", members)})
    oh = write(root / "reservation", {"held_group_count": 2, "held_image_count": 2,
                                     "held_class0_candidate_polygon_count": 2, "held_groups": groups, "items": records})
    stream_files = [root / name for name in ("DEV_MASK_PREDICTIONS_PRIVATE.jsonl",
                    "DEV_CANDIDATE_AUDIT_PRIVATE.jsonl", "EDITS_PRIVATE.jsonl")]
    native_row = {"image_id": 1, "category_id": 1, "score": 0.7, "segmentation": native}
    stocks = [{"image_id": 1, "coco": [native_row]}, {"image_id": 2, "coco": []}]
    audits = [{"image_id": 1, "candidates": [{"score": 0.7, "position": 1, "native_rank": 34, "mask_area": 36}]},
              {"image_id": 2, "candidates": []}]
    updates = [{"image_id": 1, "changes": {"1": {"fixed": {"swaps": 1, "rle": encode(worse)},
                       "free_matched_count": {"swaps": 1, "rle": encode(better)}}}}, {"image_id": 2, "changes": {}}]
    receipt = dict(status=adapter.REPLAY_STATUS, manifest_sha256=ih, exact_support_xor_area=True,
                   exact_replayed_policy_choices=True, author_TEST=False, model_forwards=0, optimizer_updates=0)
    for name, path, values in zip(("predictions_sha256", "candidate_audit_sha256", "edits_sha256"),
                                  stream_files, (stocks, audits, updates)):
        receipt[name] = write(path, values, stream=True)
    args = SimpleNamespace(intake=root, intake_terminal_sha=th, reservation=root / "reservation",
                           readback=root / "receipt", readback_sha=write(root / "receipt", receipt),
                           predictions=stream_files[0], audit=stream_files[1], edits=stream_files[2])
    return args, oh, receipt, stocks, audits


class TestRFAdapter(unittest.TestCase):
    def test_equal_family_and_original_score_agree_with_gaps_and_empty_image(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            args, oh, receipt, stocks, audits = fixture(root)
            with patch.multiple(core, IMAGES=2, LABELS=2, GROUPS=2, HEIGHT=8, WIDTH=8, RESERVATION_SHA=oh):
                result = adapter.execute(core, args, shapely)
                self.assertFalse(result["coverage_gate_met"])
                self.assertEqual(result["images_without_ship_candidate"], 1)
                self.assertEqual(result["labeled_hulls_and_coverage"]["sf_positive_source_groups"], 1)
                self.assertEqual(result["labeled_hulls_and_coverage"]["native_unmatched"], 1)
                self.assertEqual(result["primary_all_labeled_hulls"]["arm_mean_iou_points"]["free_matched_count"], 50.)
                self.assertGreater(result["primary_all_labeled_hulls"]["contrasts"][core.PRIMARY]["mean_points"], 0.)
                audits[0]["candidates"][0]["native_rank"] = 1
                receipt["candidate_audit_sha256"] = write(args.audit, audits, True)
                receipt["status"] = "UEIKAP_DFINE_RESERVED_EDIT_REPLAY_ACCEPTED_V1"
                h = write(args.readback, receipt)
                ref = core.execute(root, args.intake_terminal_sha, args.reservation, root, root, args.readback, h, shapely)
                for k in ("primary_all_labeled_hulls", "originally_matched_hulls_sensitivity",
                          "continuous_iou_threshold_transitions_on_original_matches", "labeled_hulls_and_coverage",
                          "edited_among_original_matches_by_arm", "frozen_geometry_advance"):
                    self.assertEqual(result[k], ref[k])

    def test_bad_labels_never_open_on_foreign_or_corrupted_receipt(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            args, _, receipt, _, _ = fixture(root)
            (root / "labels_roster.json").unlink()
            receipt["status"] = "UEIKAP_DFINE_RESERVED_EDIT_REPLAY_ACCEPTED_V1"
            args.readback_sha = write(args.readback, receipt)
            with self.assertRaisesRegex(ValueError, "RF-DETR replay"):
                adapter.execute(core, args, shapely)
            receipt["status"] = adapter.REPLAY_STATUS
            args.readback_sha = write(args.readback, receipt)
            args.edits.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                adapter.execute(core, args, shapely)

    def test_rejects_duplicate_and_nonascending_joint_ranks_and_wrong_compact_position(self):
        with TemporaryDirectory() as folder:
            _, _, _, stocks, audits = fixture(Path(folder))
            pred = deepcopy(stocks[0])
            pred["coco"] *= 2
            first = audits[0]["candidates"][0]
            for pos, rank in [(2, 34), (2, 33), (2, 101), (4, 97)]:
                audit = {"image_id": 1, "candidates": [first, dict(first, position=pos, native_rank=rank)]}
                with self.assertRaisesRegex(ValueError, "(ranks|positions)"):
                    adapter.native_masks(core, pred, audit, 1)
            valid = {"image_id": 1, "candidates": [first, dict(first, position=2, native_rank=97)]}
            with patch.multiple(core, HEIGHT=8, WIDTH=8):
                self.assertEqual(len(adapter.native_masks(core, pred, valid, 1)), 2)


if __name__ == "__main__":
    unittest.main()
