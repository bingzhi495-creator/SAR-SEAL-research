"""Integration check: actual video frames must each receive detector inference."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import cv2

ROOT = Path(__file__).resolve().parent


class TrafficBenchmarkIntegration(unittest.TestCase):
    def test_every_requested_frame_is_inferred_and_accounted_for(self):
        with tempfile.TemporaryDirectory(prefix="traffic-test-") as directory:
            target = Path(directory)
            subprocess.run([
                sys.executable, str(ROOT / "traffic_benchmark.py"),
                "--model", str(ROOT / "weights/yolov8n.pt"),
                "--source", str(ROOT / "input/vehicles.mp4"),
                "--output", str(target), "--frames", "6", "--warmup", "1",
                "--device", "0", "--half",
            ], check=True, timeout=120)
            summary = json.loads((target / "summary.json").read_text(encoding="utf-8"))
            records = [json.loads(line) for line in (target / "detections.jsonl").read_text().splitlines()]
            self.assertEqual(summary["frames_inferred"], 6)
            self.assertEqual([r["frame_index"] for r in records], list(range(6)))
            self.assertEqual(summary["skipped_frames"], 0)
            self.assertFalse(summary["ebike_supported"])
            self.assertEqual(summary["accuracy_evaluation"], "not_evaluated_no_ground_truth")
            self.assertGreater(summary["pipeline_fps_without_encoding"], 0)
            video = cv2.VideoCapture(str(target / "annotated.mp4"))
            self.assertEqual(int(video.get(cv2.CAP_PROP_FRAME_COUNT)), 6)
            video.release()


if __name__ == "__main__":
    unittest.main()
