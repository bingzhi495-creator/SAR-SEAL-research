"""Synthetic admission and acquisition-equal center-ship scoring tests."""

from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from pycocotools import mask as coco_mask

import score_schwegmann as subject


PROTOCOL = "a" * 40
SOURCE = "b" * 40
WORKER = subject.EXPORT_WORKER_SHA256
ZIP = "d" * 64


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def write(path, value):
    raw = value if type(value) is bytes else encode(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return dict(path=path.as_posix(), bytes=len(raw), sha256=sha256(raw).hexdigest())


def rle(points):
    mask = np.zeros((101, 101), np.uint8, order="F")
    for y, x in points:
        mask[y, x] = 1
    encoded = coco_mask.encode(mask)
    return {"size": encoded["size"], "counts": encoded["counts"].decode("ascii")}


def candidate(segmentation):
    return dict(native_rank=3, score=.75, bbox_xyxy=[48., 48., 53., 53.], segmentation=segmentation)


def row(image_id, value):
    return dict(image_id=image_id, raw_query_count=164, native_count=100, mapped_rows=80,
                eligible_count=int(value is not None), candidate=value)


def fixture():
    center = rle([(50, 50), (50, 51)])
    target = dict(status="P254_SCHWEGMANN_SCORER_ONLY_GT_V1", images=[
        dict(image_id=i, acquisition="acq_001" if i < 3 else "acq_002", segmentation=center) for i in (1, 2, 3)])
    native = [row(1, candidate(center)), row(2, None), row(3, candidate(center))]
    updated = [row(1, candidate(center)), row(2, candidate(center)), row(3, candidate(rle([])))]
    arms = {(seed, mode): values for seed in (5201, 5202, 5203)
            for mode, values in (("native", native), ("center", updated))}
    return [1, 2, 3], target, arms


def closed_fixture(root, *, stopped=False):
    ids, gt, arms = fixture()
    roster = [dict(image_id=i, basename=f"chip_{i:04d}.png", bytes=1200, sha256=f"{i:064x}") for i in ids]
    image_manifest = write(root / "target" / "images.json", dict(status="P254_SCHWEGMANN_IMAGE_ONLY_V1", images=roster))
    ground_truth = write(root / "score_only" / "truth.json", gt)
    entries = []
    for (seed, mode), rows in arms.items():
        stage = root / f"{seed}_{mode}"
        pred = write(stage / "PREDICTIONS_PRIVATE.jsonl", b"".join(encode(row) + b"\n" for row in rows))
        lock = sha256(f"lock {seed} {mode}".encode()).hexdigest()
        training, pair, checkpoint = subject.HRSID_SOURCE[seed]
        identity = dict(seed=seed, mode=mode, checkpoint_sha256=checkpoint[mode],
                        source_pair_receipt_sha256=pair, source_arm_receipt_sha256=sha256(f"arm {seed} {mode}".encode()).hexdigest(),
                        export_lock_sha256=lock)
        report_data = dict(status="P254_SCHWEGMANN_FROZEN_EXPORT_COMPLETE_PENDING_INDEPENDENT_ACCEPTANCE_V1",
            evaluation_protocol=PROTOCOL, source_commit=SOURCE, source_training_protocol=training, source_seed=seed,
            source_mode=mode, checkpoint_sha256=checkpoint[mode], source_pair_receipt_sha256=pair,
            source_arm_receipt_sha256=identity["source_arm_receipt_sha256"], target_image_manifest_sha256=image_manifest["sha256"],
            target_image_zip_sha256=ZIP, lock_sha256=lock, worker_sha256=WORKER, predictions_sha256=pred["sha256"],
            images=3, raw_queries=492, native_top_rows=300, mapped_rows=240, eligible_rows=2 if mode=="native" else 3,
            candidate_images=2 if mode=="native" else 3, missing_candidate_images=int(mode=="native"),
            selected_empty_masks=int(mode=="center"), model_forward_calls=3, precision="FP32_TF32_disabled", device="cpu",
            query_width=164, num_select=100, checkpoint_ema_updates=17640, roi_xyxy=[40, 40, 61, 61],
            selection_rule="native_top100_then_slot0_highest_rank_box_center_in_fixed_roi", training=False,
            target_adaptation=False, GT_opened=False, official_TEST_opened=False, AP=False, elapsed_seconds=10.)
        report = write(stage / "EXPORT_PRIVATE.json", report_data)
        stdout = write(stage / "WORKER.stdout", b"progress\n" + encode(report_data) + b"\n")
        stderr = write(stage / "WORKER.stderr", b"")
        terminal = write(stage / "TERMINAL_PRIVATE.json", dict(original_os_wait=True, original_os_exit=0,
            resource_stop=stopped and (seed,mode)==(5203,"center"), worker_closed=True, seed=seed, mode=mode,
            export_lock_sha256=lock, stdout_sha256=stdout["sha256"], stderr_sha256=stderr["sha256"],
            elapsed_seconds=10.5, peak_rss_bytes=3000000))
        entries.append(dict(**identity, terminal=terminal, report=report, predictions=pred, stdout=stdout, stderr=stderr))
    plan = write(root / "plan.json", dict(status="P254_SCHWEGMANN_CLOSED_SCORE_PLAN_V1", evaluation_protocol=PROTOCOL,
                 source_commit=SOURCE, export_worker_sha256=WORKER, image_zip_sha256=ZIP,
                 image_manifest=image_manifest, ground_truth=ground_truth, arms=entries))
    return plan, ground_truth


class CenterScoreTests(unittest.TestCase):
    def test_fixed_positive_misses_are_zero_and_acquisitions_have_equal_weight(self):
        report = subject.score_population(*fixture())
        result = report["seeds"]["5201"]
        self.assertEqual(result["native"]["acquisition_equal_iou"], .75)
        self.assertEqual(result["center"]["acquisition_equal_iou"], .5)
        self.assertEqual(result["native"]["missing_candidates"], 1)
        self.assertEqual(result["center"]["selected_empty_masks"], 1)
        self.assertEqual(result["native"]["recovery"]["0.90"]["count"], 2)
        self.assertEqual(result["common_nonempty"]["images"], 1)
        self.assertEqual(report["pooled"]["mean_acquisition_equal_iou_delta"], -.25)
        self.assertFalse(report["AP"])

    def test_all_six_closed_reports_bind_to_independent_score(self):
        with TemporaryDirectory() as directory:
            plan, truth = closed_fixture(Path(directory).resolve())
            result = subject.score_closed_cohort(Path(plan["path"]), plan["sha256"], PROTOCOL)
        self.assertEqual(result["ground_truth_sha256"], truth["sha256"])
        self.assertEqual(result["pooled"]["mean_acquisition_equal_iou_delta"], -.25)
        self.assertEqual(result["decision"], "INSUFFICIENT_PHYSICAL_ACQUISITIONS")

    def test_stopped_export_refuses_before_opening_gt(self):
        with TemporaryDirectory() as directory:
            plan, truth = closed_fixture(Path(directory).resolve(), stopped=True)
            Path(truth["path"]).unlink()
            with self.assertRaisesRegex(ValueError, "Original successful export closure"):
                subject.score_closed_cohort(Path(plan["path"]), plan["sha256"], PROTOCOL)

    def test_half_open_center_selection_is_a_strict_roi(self):
        ids, gt, arms = fixture()
        arms[5201, "native"][0]["candidate"]["bbox_xyxy"] = [60., 48., 62., 53.]
        with self.assertRaisesRegex(ValueError, "half-open ROI"):
            subject.score_population(ids, gt, arms)

    def test_bounded_rle_matches_author_codec_on_dense_runs(self):
        y, x = np.indices((101, 101))
        checkerboard = (y+x) % 2 == 0
        original = coco_mask.encode(np.asfortranarray(checkerboard.astype(np.uint8)))
        encoded = dict(size=original["size"], counts=original["counts"].decode("ascii"))
        np.testing.assert_array_equal(subject.decode_rle(encoded), checkerboard)
        with self.assertRaises(ValueError):
            subject.decode_rle(dict(encoded, counts=encoded["counts"][:-1]))

    def test_sixteen_groups_stay_descriptive_even_with_perfect_positive_effect(self):
        ids = list(range(1, 33))
        mask = rle([(50, 50)])
        gt = dict(status="P254_SCHWEGMANN_SCORER_ONLY_GT_V1", images=[
            dict(image_id=i, acquisition=f"acq_{(i+1)//2:03d}", segmentation=mask) for i in ids])
        arms = {(seed, mode): [row(i, None if mode == "native" else candidate(mask)) for i in ids]
                for seed, mode in subject.ROSTER}
        result = subject.score_population(ids, gt, arms)
        self.assertEqual(result["acquisitions"], 16)
        self.assertEqual(result["pooled"]["mean_acquisition_equal_iou_delta"], 1.)
        self.assertEqual(result["decision"], "INSUFFICIENT_PHYSICAL_ACQUISITIONS")


if __name__ == "__main__":
    unittest.main()
