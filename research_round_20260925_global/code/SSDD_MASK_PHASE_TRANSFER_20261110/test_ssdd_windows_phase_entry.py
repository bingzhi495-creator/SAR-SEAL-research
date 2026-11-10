"""Prospective Windows-only source, replay closure and resource guards."""

from hashlib import sha256
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import ssdd_windows_phase_entry as subject


def packed(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


class WindowsSSDDAdmissionTest(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.lock = dict(
            protocol="1" * 40, implementation_commit="2" * 40, source_manifest_sha256="3" * 64,
            worker={"sha256": "4" * 64}, fit_readback={"sha256": "5" * 64}, python={"sha256": "6" * 64},
        )
        self.stock = dict(
            status="SSDD_FIT_NATIVE_CENTER_REPLAY_COMPLETE", images=741, views=741, seed_base=5311,
            exact_shared_geometry_rng_images=True, exact_common_nonmask_fields=True, forward_calls=0, optimizer_updates=0,
            FIT=True, DEV=False, TEST=False, AP=False, capacity=dict(limit=100, native_max_present=29, center_max_present=29),
            membership=dict(common_nonempty=1445),
            by_path=dict(direct=dict(views=354, small=dict(masks_changed=30), other=dict(masks_changed=1)),
                         crop=dict(views=387, small=dict(masks_changed=32), other=dict(masks_changed=2))),
        )
        self.report = dict(subject.replay_identity(self.lock), status=subject.REPLAY_STATUS, FIT=True, DEV=False,
                           TEST=False, AP=False, cuda_initialized=False, torch_threads=1, stock_replay=self.stock)
        self.rewrite()

    def ref(self, name, data):
        path = self.root / name
        path.write_bytes(data)
        return dict(path=path.as_posix(), bytes=len(data), sha256=sha256(data).hexdigest())

    def rewrite(self, **changes):
        report = self.ref("FIT_REPLAY_SAFE.json", packed(self.report))
        final = dict(status=subject.REPLAY_STATUS, protocol=self.lock["protocol"], result_sha256=report["sha256"])
        stdout = self.ref("WORKER.stdout", (json.dumps(final) + "\n").encode())
        stderr = self.ref("WORKER.stderr", b"")
        terminal = dict(
            status=subject.TERMINAL_STATUS, original_os_wait=True, original_os_exit=0, worker_closed=True,
            resource_stop=False, stop_reason=None, pid=42, birth_utc="2026-11-10T00:00:00Z", elapsed_seconds=5.,
            peak_rss_bytes=1000, protocol=self.lock["protocol"], worker_source_sha256=self.lock["worker"]["sha256"],
            result_sha256=report["sha256"], stdout_sha256=stdout["sha256"], stderr_sha256=stderr["sha256"],
        )
        terminal.update(changes)
        self.lock["cpu_replay"] = dict(
            report=report, terminal=self.ref("TERMINAL_PRIVATE.json", packed(terminal)), stdout=stdout, stderr=stderr,
        )

    def test_exact_original_cpu_replay_is_required_before_any_torch_import(self):
        receipt = subject.verify_replay_receipt(self.lock)
        self.assertEqual(receipt["stock_replay"]["views"], 741)
        self.assertNotIn("torch", sys.modules)
        self.rewrite(worker_closed=False)
        with self.assertRaisesRegex(ValueError, "Original independently closed"):
            subject.verify_replay_receipt(self.lock)

    def test_replay_closure_wall_and_cross_protocol_guards(self):
        self.rewrite(elapsed_seconds=301.)
        with self.assertRaisesRegex(ValueError, "resource evidence"):
            subject.verify_replay_receipt(self.lock)
        self.rewrite()
        self.lock["implementation_commit"] = "9" * 40
        with self.assertRaisesRegex(ValueError, "same exact Windows source|share exact Windows source"):
            subject.verify_replay_receipt(self.lock)

    def test_cuda_short_has_hard_free_floor_and_six_gib_allocator_cap(self):
        applied = []
        cuda = SimpleNamespace(
            is_available=lambda: True, device_count=lambda: 1, is_bf16_supported=lambda **kwargs: True,
            mem_get_info=lambda device: (12 * subject.GIB - 1, 24 * subject.GIB),
            set_per_process_memory_fraction=lambda fraction, device: applied.append((fraction, device)),
        )
        with self.assertRaisesRegex(ValueError, "12 GiB"):
            subject.admit_cuda_short(cuda)
        self.assertEqual(applied, [])
        cuda.mem_get_info = lambda device: (12 * subject.GIB, 24 * subject.GIB)
        self.assertEqual(subject.admit_cuda_short(cuda), (12 * subject.GIB, 24 * subject.GIB, .25))
        self.assertEqual(applied, [(.25, 0)])

    def test_source_manifest_is_legacy_canonical_reconstruction(self):
        self.assertEqual(len(subject.WORKER_MANIFEST_SHA), 64)
        self.assertNotEqual(subject.WORKER_MANIFEST_SHA, subject.RECIPE_SHA)

    def test_windows_host_query_needs_no_uninstalled_psutil(self):
        self.assertGreater(subject.windows_available_bytes(), 0)
        self.assertNotIn("psutil", sys.modules)
        with patch.object(subject, "windows_available_bytes", return_value=16 * subject.GIB - 1):
            with self.assertRaisesRegex(ValueError, "below 16 GiB"):
                subject.admit_host()


if __name__ == "__main__":
    unittest.main()
