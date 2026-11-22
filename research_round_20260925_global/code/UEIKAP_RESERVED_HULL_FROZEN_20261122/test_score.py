"""Generated tests for the native RLE and scene-equal scoring contract."""

from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from pycocotools import mask as codec
import shapely


spec = spec_from_file_location("ueikap_score", Path(__file__).with_name("score.py"))
score = module_from_spec(spec)
spec.loader.exec_module(score)


def rectangle(x0, y0, x1, y1):
    mask = np.zeros((8, 8), dtype=np.uint8, order="F")
    mask[y0:y1, x0:x1] = 1
    return mask


def encode(mask):
    row = codec.encode(np.asfortranarray(mask))
    row["counts"] = row["counts"].decode("ascii")
    return row


def write(path, data):
    path.write_bytes(data)
    return sha256(data).hexdigest()


def write_json(path, obj):
    return write(path, json.dumps(obj).encode("utf-8"))


def write_stream(path, rows):
    return write(path, b"".join(json.dumps(row).encode("utf-8") + b"\n" for row in rows))


class TestGeometry(unittest.TestCase):
    def test_exact_fractional_polygon_area(self):
        mask = encode(rectangle(2, 2, 4, 3))
        polygon = shapely.box(2.25, 2, 4.25, 3)
        self.assertAlmostEqual(score.geometric_iou(mask, polygon, shapely), 1.75 / 2.25)

    def test_coco_tie_rule_and_single_matching(self):
        left = encode(rectangle(0, 0, 4, 4))
        right = encode(rectangle(4, 0, 8, 4))
        half = encode(rectangle(0, 0, 2, 4))
        self.assertEqual(score.match_native([half, left, left, right], [left, left, right]), {1: 1, 0: 2, 2: 4})

    def test_rejects_xor_or_exact_pair_count_mismatch(self):
        stock = rectangle(0, 0, 6, 6)
        changed = stock.copy()
        changed[0, 0], changed[7, 7] = 0, 1
        move = {"rle": encode(changed), "swaps": 1}
        self.assertEqual(score.verify_edits({"1": {"fixed": move, "free_matched_count": move}}, [encode(stock)], 8, 8)[1]["fixed"][0], 1)
        with self.assertRaisesRegex(ValueError, "Paired edit counts"):
            score.verify_edits({"1": {"fixed": move}}, [encode(stock)], 8, 8)
        malformed = dict(move, swaps=2)
        with self.assertRaisesRegex(ValueError, "balanced XOR"):
            score.verify_edits({"1": {"fixed": malformed, "free_matched_count": malformed}}, [encode(stock)], 8, 8)

    def test_all_labels_include_the_unmatched_scene(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = b"0 0 0 .75 0 .75 .75 0 .75\n1 .1 .1 .3 .1 .3 .4\n"
            label_rows = []
            image_rows = []
            members = []
            old_rows = []
            for index in (1, 2):
                path = root / f"sample_{index}.txt"
                h = write(path, raw)
                label_rows.append({"image_id": index, "acquisition_group": f"scene{index}",
                                   "count_by_class": {"0": 1, "1": 1},
                                   "label_ref": {"path": path.name, "bytes": len(raw), "sha256": h}})
                image_rows.append({"image_id": index, "original_hw": [8, 8]})
                members.extend([{"image_id": index, "zip_member": f"UEIKAP_SAR/images/train/chip{index}.png"},
                                {"image_id": index, "zip_member": f"UEIKAP_SAR/labels/train/chip{index}.txt",
                                 "bytes": len(raw), "sha256": h}])
                old_rows.append({"acquisition_group": f"scene{index}", "class0_candidate_polygons": 1,
                                 "tiff_zip_name": f"UEIKAP_SAR/images_tifs/train/chip{index}.tif"})
            reservation = root / "reserved.json"
            oh = write_json(reservation, {"held_group_count": 2, "held_image_count": 2,
                                          "held_class0_candidate_polygon_count": 2,
                                          "held_groups": ["scene1", "scene2"], "items": old_rows})
            mh = write_json(root / "member_provenance.json", members)
            lh = write_json(root / "labels_roster.json", {"items": label_rows, "expected_ids": [1, 2]})
            ih = write_json(root / "images_roster.json", {"items": image_rows, "expected_ids": [1, 2]})
            th = write_json(root / "intake_terminal.json", {"status": "completed_verified", "actual_exit": 0,
                                                            "labels_roster_sha256": lh, "images_roster_sha256": ih,
                                                            "member_provenance_sha256": mh})
            stock = encode(rectangle(0, 0, 6, 6))
            ph = write_stream(root / "DEV_MASK_PREDICTIONS_PRIVATE.jsonl", [{"image_id": 1, "coco": [
                {"image_id": 1, "category_id": 1, "score": 0.75, "segmentation": stock}]},
                {"image_id": 2, "coco": []}])
            ah = write_stream(root / "DEV_CANDIDATE_AUDIT_PRIVATE.jsonl", [
                {"image_id": 1, "candidates": [{"native_rank": 1, "score": 0.75, "mask_area": 36}]},
                {"image_id": 2, "candidates": []}])
            eh = write_stream(root / "EDITS_PRIVATE.jsonl", [{"image_id": index, "changes": {}} for index in (1, 2)])
            receipt = root / "accepted.json"
            rh = write_json(receipt, {"status": "UEIKAP_DFINE_RESERVED_EDIT_REPLAY_ACCEPTED_V1",
                                      "exact_replayed_policy_choices": True, "exact_support_xor_area": True,
                                      "model_forwards": 0, "optimizer_updates": 0, "manifest_sha256": ih,
                                      "author_TEST": False, "predictions_sha256": ph, "candidate_audit_sha256": ah,
                                      "edits_sha256": eh})
            with patch.multiple(score, IMAGES=2, LABELS=2, GROUPS=2, HEIGHT=8, WIDTH=8, RESERVATION_SHA=oh):
                result = score.execute(root, th, reservation, root, root, receipt, rh, shapely)
                shifted_labels = [dict(label_rows[0], acquisition_group="scene2"), label_rows[1]]
                with self.assertRaisesRegex(ValueError, "old reservation"):
                    score.verify_reservation({"held_group_count": 2, "held_image_count": 2,
                                              "held_class0_candidate_polygon_count": 2,
                                              "held_groups": ["scene1", "scene2"], "items": old_rows}, shifted_labels, members)
            self.assertEqual(result["labeled_hulls_and_coverage"]["wake_rows_excluded"], 2)
            self.assertEqual(result["primary_all_labeled_hulls"]["arm_mean_iou_points"]["raw"], 50.0)
            self.assertEqual(result["originally_matched_hulls_sensitivity"]["arm_mean_iou_points"]["raw"], 100.0)
            self.assertEqual(result["primary_all_labeled_hulls"]["contrasts"][score.PRIMARY]["mean_points"], 0.0)
            self.assertEqual(result["native_matched_mask_area_px_five_number_summary"]["median"], 36.0)
            thresholds = result["continuous_iou_threshold_transitions_on_original_matches"]
            self.assertEqual(thresholds["0.9"]["raw"], {"retained": 1, "lost": 0, "gained": 0, "both_below": 0})
            self.assertFalse(result["frozen_geometry_advance"])


if __name__ == "__main__":
    unittest.main()
