"""Exercise frozen-source admission and actual official SSDD scoring on synthetic masks."""

from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from pycocotools import mask

import score_frozen_hrsid_ssdd as subject


PROTOCOL = "a" * 40


def encoded(value):
    return json.dumps(value, separators=(",", ":"), allow_nan=False).encode("utf-8")


def write(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return dict(path=path.as_posix(), bytes=len(raw), sha256=sha256(raw).hexdigest())


def digest(value):
    return sha256(value.encode()).hexdigest()


class Fixture:
    def __init__(self, root):
        self.root = root
        self.ids = list(range(1, 188))
        items = [dict(image_id=i, original_hw=[16, 16], image_ref=dict(
            path=f"/synthetic/valid/{i}.jpg", bytes=10, sha256=digest(str(min(i, 183))))) for i in self.ids]
        self.manifest = write(root / "source.json", encoded(dict(
            schema="Root-RFDETR-DEV187-image-only-manifest-v1", population="internal_DEV187",
            model_execution=False, AP=False, TEST=False, expected_ids=self.ids, items=items)))
        gt_images = [dict(id=i, height=16, width=16, file_name=f"{i}.jpg") for i in self.ids]
        annotations = [dict(id=i, image_id=(i - 1) % 187 + 1, category_id=1, iscrowd=0,
                            bbox=[1, 1, 2, 2], area=4, segmentation=[[1, 1, 1, 3, 3, 3, 3, 1]])
                       for i in range(1, 381)]
        self.gt = write(root / "integer_gt.json", encoded(dict(
            info={}, images=gt_images, annotations=annotations, categories=[dict(id=1, name="ship")])))
        self.zip_digest = digest("target zip")

    def arm(self, seed=5201, mode="native"):
        folder = self.root / f"{seed}_{mode}"
        checkpoint = digest(f"checkpoint {seed} {mode}")
        arm_digest = digest(f"arm {seed} {mode}")
        pair_digest = digest(f"pair {seed}")
        source = "d66baf3c57fafdba4242ad1b55e08cf76bfd77f7" if seed == 5201 else "a49361b92977a605267025b1e7798420936cbe53"
        source_commit = source
        data = np.zeros((16, 16), np.uint8, order="F")
        if mode == "center" or seed == 5201:
            data[1:3, 1:3] = 1
        else:
            data[10:12, 10:12] = 1
        rle = mask.encode(data)
        segmentation = dict(size=rle["size"], counts=rle["counts"].decode("ascii"))
        rows = []
        for i in self.ids:
            count = int(i > 1)
            coco = [dict(image_id=i, category_id=1, score=.55, segmentation=segmentation)] if count else []
            rows.append(dict(image_id=i, raw_query_count=164, native_count=100,
                             unmapped_slot1_count=100 - count, native_ranks=[2] if count else [], coco=coco))
        predictions = write(folder / "DEV_NATIVE_PREDICTIONS_PRIVATE.jsonl",
                            b"".join(encoded(row) + b"\n" for row in rows))
        report = write(folder / "DEV_NATIVE_EXPORT_SAFE.json", encoded(dict(
            status="FROZEN_HRSID_TO_SSDD_DEV187_EXPORT_COMPLETE_PENDING_INDEPENDENT_ACCEPTANCE_V1",
            evaluation_protocol=PROTOCOL, source_training_protocol=source, source_commit=source_commit,
            source_seed=seed, source_mode=mode, checkpoint_sha256=checkpoint,
            source_pair_receipt_sha256=pair_digest, source_arm_receipt_sha256=arm_digest,
            target_image_manifest_sha256=self.manifest["sha256"], target_image_zip_sha256=self.zip_digest,
            predictions_sha256=predictions["sha256"], images=187, raw_queries=30668,
            native_top_rows=18700, mapped_rows=186, unmapped_slot1_rows=18514, empty_image_predictions=1,
            model_forward_calls=187, precision="FP32_TF32_disabled", device="cpu", query_width=164,
            num_select=100, checkpoint_ema_updates=17640, training=False, target_adaptation=False,
            GT_opened=False, official_TEST_opened=False, AP=False, elapsed_seconds=1.25,
            peak_allocated_bytes=0, peak_reserved_bytes=0)))
        terminal = write(folder / "TERMINAL_PRIVATE.json", encoded(dict(
            original_os_wait=True, original_os_exit=0, resource_stop=False, worker_closed=True,
            seed=seed, mode=mode, export_lock_sha256=digest(f"lock {seed} {mode}"),
            stdout_sha256=digest(f"stdout {seed} {mode}"), stderr_sha256=digest("empty stderr"),
            elapsed_seconds=1.5, peak_rss_bytes=12345)))
        return dict(seed=seed, mode=mode, source_commit=source_commit, checkpoint_sha256=checkpoint,
                    source_pair_receipt_sha256=pair_digest, source_arm_receipt_sha256=arm_digest,
                    terminal=terminal, report=report, predictions=predictions)

    def evaluate(self, arms, **selector):
        plan = write(self.root / "plan.json", encoded(dict(
            status="FROZEN_HRSID_TO_SSDD_DEV187_CLOSED_PLAN_V1", evaluation_protocol=PROTOCOL,
            image_manifest=self.manifest, integer_gt=self.gt, arms=arms)))
        scoring, official = subject.load_scoring()
        with patch.multiple(subject, IMAGE_MANIFEST_SHA=self.manifest["sha256"],
                            INTEGER_GT_SHA=self.gt["sha256"], INTEGER_GT_BYTES=self.gt["bytes"]), \
                patch.object(scoring, "IMAGE_MANIFEST_SHA", self.manifest["sha256"]), \
                patch.object(subject, "load_scoring", return_value=(scoring, official)):
            return subject.evaluate_plan(Path(plan["path"]), plan["sha256"], PROTOCOL, **selector)


class FrozenScoreTests(unittest.TestCase):
    def test_selected_arm_scores_original_grid_including_zero_candidate_image(self):
        with TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory).resolve())
            result = fixture.evaluate([fixture.arm()], seed=5201, mode="native")
        self.assertEqual((result["images"], result["annotations"], result["distinct_jpeg_sha256"]), (187, 380, 183))
        self.assertFalse(result["official_TEST_opened"])
        self.assertEqual(result["arms"][0]["empty_image_predictions"], 1)
        self.assertGreater(result["arms"][0]["metrics"]["AP90"], .45)

    def test_all_six_preserve_arms_and_compute_paired_seed_deltas(self):
        with TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory).resolve())
            keys = [(seed, mode) for seed in (5201, 5202, 5203) for mode in ("native", "center")]
            result = fixture.evaluate([fixture.arm(*key) for key in reversed(keys)])
        self.assertEqual([(arm["seed"], arm["mode"]) for arm in result["arms"]], keys)
        paired = result["paired_summary"]
        deltas = [row["center_minus_native_points"]["AP"] for row in paired["by_seed"]]
        self.assertEqual(deltas[0], 0.)
        self.assertGreater(deltas[1], 45.)
        self.assertEqual(deltas[1], deltas[2])
        self.assertAlmostEqual(paired["mean_center_minus_native_points"]["AP"], sum(deltas) / 3)

    def test_full_plan_rejects_mixed_source_pairs(self):
        with TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory).resolve())
            arms = [fixture.arm(seed, mode) for seed in (5201, 5202, 5203) for mode in ("native", "center")]
            arms[1]["source_pair_receipt_sha256"] = digest("other pair")
            report = json.loads(Path(arms[1]["report"]["path"]).read_bytes())
            report["source_pair_receipt_sha256"] = arms[1]["source_pair_receipt_sha256"]
            arms[1]["report"] = write(Path(arms[1]["report"]["path"]), encoded(report))
            with self.assertRaisesRegex(ValueError, "same accepted source pair"):
                fixture.evaluate(arms)

    def test_rejects_nonoriginal_terminal_before_reading_predictions(self):
        with TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory).resolve())
            arm = fixture.arm()
            terminal_path = Path(arm["terminal"]["path"])
            terminal = json.loads(terminal_path.read_bytes())
            terminal["worker_closed"] = False
            arm["terminal"] = write(terminal_path, encoded(terminal))
            Path(arm["predictions"]["path"]).unlink()
            with self.assertRaisesRegex(ValueError, "independently closed original export"):
                fixture.evaluate([arm], seed=5201, mode="native")

    def test_rejects_reordered_native_rank_and_score_even_if_owner_pins_match(self):
        with TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory).resolve())
            arm = fixture.arm()
            pred_path = Path(arm["predictions"]["path"])
            rows = [json.loads(line) for line in pred_path.read_bytes().splitlines()]
            rows[1]["coco"].append(dict(rows[1]["coco"][0], score=.8))
            rows[1]["unmapped_slot1_count"] -= 1
            rows[1]["native_ranks"].append(3)
            arm["predictions"] = write(pred_path, b"".join(encoded(row) + b"\n" for row in rows))
            report_path = Path(arm["report"]["path"])
            report = json.loads(report_path.read_bytes())
            report["predictions_sha256"] = arm["predictions"]["sha256"]
            report["mapped_rows"] += 1
            report["unmapped_slot1_rows"] -= 1
            arm["report"] = write(report_path, encoded(report))
            with self.assertRaisesRegex(ValueError, "descending order"):
                fixture.evaluate([arm], seed=5201, mode="native")


if __name__ == "__main__":
    unittest.main()
