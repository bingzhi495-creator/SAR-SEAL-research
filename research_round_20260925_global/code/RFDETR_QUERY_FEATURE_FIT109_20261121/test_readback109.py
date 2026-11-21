import importlib.util
from io import BytesIO
from pathlib import Path
import unittest

import numpy as np


class ReadbackTest(unittest.TestCase):
    def setUp(self):
        script = Path(__file__).with_name("verify_capture109.py")
        spec = importlib.util.spec_from_file_location("verify_capture109", script)
        self.v = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.v)

    def test_supported_and_empty_archives_and_bitwise_old14(self):
        import runpy
        from types import SimpleNamespace
        code = Path("D:/SAR_SEAL_private_execution_20261021_01/github_release_20261011/research_round_20260925_global/code")
        residual = SimpleNamespace(**runpy.run_path(str(code/"RFDETR_QUERY_CHANNEL_RESIDUAL_20261120/channel_residual.py")))
        for count in (0, 2):
            coords = np.arange(2*count, dtype=np.int32).reshape(count,2)
            old = np.zeros((count,2,14), np.float32)
            layers = np.ones((count,2), np.int8)
            points = [dict(coordinates=coords, features=old, layers=layers, rank=3, query_id=8)] if count else []
            rng = np.random.default_rng(33)
            arrays = dict(image_id=np.asarray(42), ranks=np.asarray([3] if count else [], int),
                          query_ids=np.asarray([8] if count else [], int), old14=old.copy(), coords=coords, layers=layers,
                          offsets=np.r_[0,np.cumsum([count])] if count else np.array([0],np.int64),
                          spatial=rng.standard_normal((count,2,256)).astype(np.float32),
                          projected_queries=rng.standard_normal((len(points),256)).astype(np.float32))
            def serialized():
                out = BytesIO()
                np.savez(out, **arrays)
                return out.getvalue()
            totals = self.v.replay_image(serialized(), 42, points, residual, np)
            self.assertEqual(totals["point_pairs"], count)
            self.assertEqual(totals["candidate_sets"], len(points))
            self.assertEqual(totals["no_support_images"], int(not points))
            if count:
                self.assertEqual((totals["moved_points"], totals["informative_points"]), (4,4))
                arrays["old14"][0,0,0] = -0.0
                with self.assertRaisesRegex(ValueError, "old14"):
                    self.v.replay_image(serialized(), 42, points, residual, np)
            else:
                arrays["image_id"] = np.asarray(44)
                with self.assertRaisesRegex(ValueError, "image_id"):
                    self.v.replay_image(serialized(), 42, points, residual, np)

    def test_original_terminal_must_bind_session_footer_and_scientific_summary(self):
        digest = "b"*64
        evidence = dict(launch=dict(session_id=10), terminal=dict(exit_code=0),
                        observed_footer=dict(result_sha256=digest,status="RFDETR_FIT109_QUERY_BASIS_CAPTURE_COMPLETE"),
                        result_sha256=digest)
        self.v.closed_tool(evidence, 10, digest)
        with self.assertRaises(ValueError):
            self.v.closed_tool(evidence, 11, digest)
        evidence["terminal"]["exit_code"] = 1
        with self.assertRaises(ValueError):
            self.v.closed_tool(evidence, 10, digest)


if __name__ == "__main__":
    unittest.main()
