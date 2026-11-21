"""Independent resumed-export readback tests using synthetic masks and image IDs."""

from hashlib import sha256
import importlib.util
from io import BytesIO
import json
import os
from pathlib import Path
import unittest

from verify_resumed import COMPLETE, OWNER_STATUS, RESUME, check_terminal, verify_stream


class ResumedStreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        work = Path(__file__).resolve().parent.parent
        default = work / "HRSID_OFFICIAL_BASELINE_READBACK_WORK_20261121_01/verify_export.py"
        path = Path(os.environ.get("TEST_READBACK_SOURCE", default))
        if sha256(path.read_bytes()).hexdigest() != "27c3f2685c8aa2c6e6443b5eeb003a7da20f2cdf6391167bc386e3e097b11784":
            raise AssertionError("Unexpected synthetic test verifier dependency")
        spec = importlib.util.spec_from_file_location("_synthetic_original_verifier", path)
        cls.verify = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.verify)

    def setUp(self):
        self.manifest = [dict(image_id=i, height=2, width=3, decoded_rgb_sha256=h * 64)
                         for i, h in ((8, "a"), (1, "b"), (9, "c"), (3, "a"))]
        self.lines = []
        for row in self.manifest:
            i = row["image_id"]
            predictions = [] if i == 1 else [
                dict(image_id=i, category_id=1, score=0.625, segmentation=dict(size=[2, 3], counts="6"))
            ]
            block = dict(image_id=i, raw_query_count=164, native_count=100,
                         unmapped_slot1_count=100-len(predictions), native_ranks=[4] if predictions else [],
                         coco=predictions)
            self.lines.append((json.dumps(block, separators=(",", ":")) + "\n").encode())
        self.raw = b"".join(self.lines)
        self.prefix = b"".join(self.lines[:2])
        self.result = dict(prediction_bytes=len(self.raw), predictions_sha256=sha256(self.raw).hexdigest(),
                           mapped_rows=3, prefix_mapped_rows=1)

    def scan(self, raw=None, result=None, prefix_images=2, prefix_bytes=None, prefix_digest=None, duplicates=1):
        return verify_stream(BytesIO(self.raw if raw is None else raw), self.manifest,
                             self.result if result is None else result, self.verify, prefix_images,
                             len(self.prefix) if prefix_bytes is None else prefix_bytes,
                             sha256(self.prefix).hexdigest() if prefix_digest is None else prefix_digest, duplicates)

    def test_complete_independent_census(self):
        metrics = self.scan()
        self.assertEqual(metrics["predictions_sha256"], sha256(self.raw).hexdigest())
        self.assertEqual(metrics["mapped_rows"], 3)
        self.assertEqual(metrics["unmapped_slots"], 397)
        self.assertEqual(metrics["images_without_predictions"], 1)
        self.assertEqual(metrics["identical_duplicate_prediction_pairs"], 1)

    def test_changed_valid_prefix_and_updated_full_digest(self):
        # JSON whitespace preserves row semantics and duplicate geometry.
        changed = b" " + self.raw
        rebound = dict(self.result, prediction_bytes=len(changed), predictions_sha256=sha256(changed).hexdigest())
        with self.assertRaises(ValueError):
            self.scan(raw=changed, result=rebound, prefix_bytes=1+len(self.prefix))

    def test_prefix_cut_must_end_on_exact_manifest_record(self):
        for stop in (len(self.prefix)-1, len(self.prefix)+1, len(self.lines[0])):
            with self.subTest(stop=stop), self.assertRaises(ValueError):
                self.scan(prefix_bytes=stop, prefix_digest=sha256(self.raw[:stop]).hexdigest())

    def test_changed_suffix_full_hash(self):
        suffix = self.lines[2].replace(b"0.625", b"0.526")
        changed = self.prefix + suffix + self.lines[3]
        with self.assertRaises(ValueError):
            self.scan(raw=changed)

    def test_reordered_suffix_even_with_consistent_full_digest(self):
        changed = self.prefix + self.lines[3] + self.lines[2]
        with self.assertRaises(ValueError):
            self.scan(raw=changed, result=dict(self.result, predictions_sha256=sha256(changed).hexdigest()))

    def test_incorrect_full_prefix_and_duplicate_counts(self):
        for name in ("prefix_mapped_rows", "mapped_rows", "prediction_bytes"):
            result = {**self.result, name: self.result[name] + 1}
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.scan(result=result)
        with self.assertRaises(ValueError):
            self.scan(duplicates=0)

    def test_distinct_worker_success_and_unavailable_original_exit(self):
        launch = "d" * 64
        receipt = dict(status=OWNER_STATUS, actual_process_terminal=True, actual_exit=0,
                       original_interrupted_exit=None, footer_status=COMPLETE, resume_source_sha256=RESUME,
                       launch_request_sha256=launch, result_sha256="e" * 64, owner_pid=101, worker_pid=102)
        check_terminal(receipt, launch)
        for key, bad in (("status", "original"), ("actual_process_terminal", False), ("actual_exit", False),
                         ("actual_exit", 1), ("original_interrupted_exit", 0), ("footer_status", "unknown"),
                         ("resume_source_sha256", "a" * 64), ("launch_request_sha256", "a" * 64),
                         ("result_sha256", "bad"), ("owner_pid", -1), ("worker_pid", 101), ("worker_pid", True)):
            with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                check_terminal({**receipt, key: bad}, launch)
        del receipt["original_interrupted_exit"]
        with self.assertRaises(ValueError):
            check_terminal(receipt, launch)


if __name__ == "__main__":
    unittest.main()
