"""Allocator initialization ordering fixture for the V2 SSDD export stage."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ssdd_dev187_stage_worker as worker


class LazyCUDA:
    def __init__(self, count=1):
        self.count = count
        self.initialized = False
        self.calls = []

    def is_available(self):
        self.calls.append("is_available")
        return True

    def device_count(self):
        self.calls.append("device_count")
        return self.count

    def init(self):
        self.calls.append("init")
        self.initialized = True

    def reset_peak_memory_stats(self, device):
        self.calls.append("reset")
        if not self.initialized or device != "cuda:0":
            raise RuntimeError("Invalid device argument 0: did you call init?")


def fake_torch(cuda):
    return SimpleNamespace(cuda=cuda, set_num_threads=lambda _: None, set_num_interop_threads=lambda _: None,
                           set_float32_matmul_precision=lambda _: None,
                           backends=SimpleNamespace(cudnn=SimpleNamespace(allow_tf32=True)))


class ExportConfigurationTest(unittest.TestCase):
    def test_configuration_initializes_allocator_before_original_exporter_reset(self):
        cuda = LazyCUDA()
        torch = fake_torch(cuda)
        with patch.dict(sys.modules, {"torch": torch}):
            worker.configure_export()
        cuda.reset_peak_memory_stats("cuda:0")
        self.assertEqual(cuda.calls, ["is_available", "device_count", "init", "reset"])
        self.assertFalse(torch.backends.cudnn.allow_tf32)

    def test_non_single_device_is_refused_before_init(self):
        cuda = LazyCUDA(count=2)
        with patch.dict(sys.modules, {"torch": fake_torch(cuda)}):
            with self.assertRaisesRegex(ValueError, "Exactly one preadmitted"):
                worker.configure_export()
        self.assertEqual(cuda.calls, ["is_available", "device_count"])
        self.assertFalse(cuda.initialized)


if __name__ == "__main__":
    unittest.main()
