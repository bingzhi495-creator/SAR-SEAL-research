from pathlib import Path
from runpy import run_path
from types import ModuleType
import unittest


entry = run_path(str(Path(__file__).with_name("launch_repeat5202_center.py")))


class RepeatE24CenterGateTest(unittest.TestCase):
    def test_requires_independent_native_e24_and_matching_authority(self):
        a = dict(
            status=entry["ARM_STATUS"], qualification_pair_sha256=entry["SHORT_PAIR_SHA"],
            initialization_seed=5201, training_seed=5202, completed_epochs=24, visits=17856,
            optimizer_updates=4464, ema_updates=4464, original_os_wait=True, original_os_exit=0,
            resource_stop=False, worker_closed=True, FIT=True, DEV=False, TEST=False, AP=False,
            phase=dict(stage="e24", mask_resize_mode="native", seed=5202, initialization_seed=5201,
                       training_protocol=entry["PROTOCOL"], source_manifest_sha256=entry["MANIFEST"]),
            terminal_sha256="a"*64, launch_sha256="b"*64,
        )
        authority = dict(status=entry["ARM_READY"], mode="native", resource_closed=True, active_attempt=None,
                         failure=None, terminal_ref={"sha256":"a"*64}, launch_ref={"sha256":"b"*64})
        entry["verify_native"](a, authority)
        for bad in (dict(a, training_seed=5201), dict(a, optimizer_updates=4463), dict(a, DEV=True),
                    dict(a, phase=dict(a["phase"], mask_resize_mode="center"))):
            with self.assertRaises(ValueError):
                entry["verify_native"](bad, authority)
        with self.assertRaises(ValueError):
            entry["verify_native"](a, dict(authority, resource_closed=False))

    def test_serial_coordinator_replaces_only_the_native_gate(self):
        calls = []
        namespace = ModuleType("fake_native")
        namespace.__file__ = __file__
        namespace.admit_center = lambda mode, legacy: calls.append((mode, legacy))
        stock = """
def run_owner(mode):
    legacy = "original guards"
    need(mode == "native", "This original launch admits only native A; center is after independent A acceptance")
    return "original owner body"
"""
        result = entry["serial_owner"](namespace, stock)("center")
        self.assertEqual(result, "original owner body")
        self.assertEqual(calls, [("center", "original guards")])


if __name__ == "__main__":
    unittest.main()
