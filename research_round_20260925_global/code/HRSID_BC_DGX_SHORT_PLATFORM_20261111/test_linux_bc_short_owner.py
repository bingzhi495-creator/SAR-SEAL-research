"""Exercise the isolated request handoff and original OS wait without radar data."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

from linux_bc_short_owner import make_arm_lock, validate_document, supervise_child


class ShortOwnerTest(unittest.TestCase):
    def test_each_arm_uses_fresh_output_and_shared_science(self):
        root = Path("/study")
        platform = {"training_protocol": "a" * 40, "implementation_commit": "b" * 40}
        sources = {"one_pass_common_qualification.py": {"sha256": "c" * 64}}
        native = make_arm_lock(root, "stepwise_center", platform, sources)
        center = make_arm_lock(root, "one_pass", platform, sources)
        self.assertEqual(native["sources"], center["sources"])
        self.assertEqual(native["training_protocol"], center["training_protocol"])
        self.assertEqual(native["native_output"], "/study/stepwise_center/ACTUAL")
        self.assertEqual(center["audit_output"], "/study/one_pass/AUDIT")
        self.assertEqual(center["training_seed"], 5202)
        self.assertIsNone(center["qualification_pair_ref"])

    def test_duplicate_json_keys_refused(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_document(b'{"status":"first","status":"second"}')

    @unittest.skipUnless(sys.platform == "linux", "Linux original wait")
    def test_real_child_wait_records_original_exit_and_disjoint_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            stage = Path(folder)
            command = [sys.executable, "-I", "-B", "-X", "utf8", "-c",
                       "import sys; print('fixture'); sys.exit(3)"]
            terminal = supervise_child(stage, command, {}, 0, 256 * 1024**2, 15)
            self.assertEqual(terminal["original_os_exit"], 3)
            self.assertTrue(terminal["worker_closed"])
            self.assertFalse(terminal["resource_stop"])
            self.assertEqual((stage / "WORKER.stdout").read_text().strip(), "fixture")
            self.assertEqual(json.loads((stage / "TERMINAL_PRIVATE.json").read_bytes()), terminal)


if __name__ == "__main__":
    unittest.main()

