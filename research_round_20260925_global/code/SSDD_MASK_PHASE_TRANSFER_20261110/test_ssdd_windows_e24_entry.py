from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ssdd_windows_e24_entry import make_formal_lock, validate_stage_binding


class FormalScopeTest(unittest.TestCase):
    def test_long_arm_cannot_change_the_qualified_short_mode_or_protocol(self):
        scope = dict(mode="center", protocol="1" * 40, operation="short")
        lock = make_formal_lock("center", "2" * 40, {"scope": 1}, {"qualification": 1}, {"worker": 1})
        validate_stage_binding(lock, scope)
        with self.assertRaisesRegex(ValueError, "same qualified Windows mode"):
            validate_stage_binding(lock, dict(scope, mode="native"))
        with self.assertRaisesRegex(ValueError, "distinct E24 operational"):
            validate_stage_binding(dict(lock, operation_commit=scope["protocol"]), scope)


if __name__ == "__main__":
    unittest.main()
