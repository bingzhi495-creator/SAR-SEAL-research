from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import windows_float_common_short as subject


class FloatWindowsShortGuards(unittest.TestCase):
    def test_validated_float_layout_is_handed_to_original_worker_and_entry_restored(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            integer, floating = root / "int", root / "float"
            integer.mkdir()
            floating.mkdir()
            seen = []
            scope = object()

            def original(value):
                self.assertIs(value, scope)
                seen.append("stock")
                return integer, [12, 17]

            entry = SimpleNamespace(read_fit_population=original)

            def validate(value, source, ids):
                self.assertEqual((value, source, ids), (scope, integer, [12, 17]))
                seen.append("float")
                return floating

            with subject.float_population(entry, validate):
                self.assertEqual(entry.read_fit_population(scope), (floating, [12, 17]))
            self.assertIs(entry.read_fit_population, original)
            self.assertEqual(seen, ["stock", "float"])

    def test_floating_population_alias_fails_closed(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            def original(_):
                return root, [7]
            entry = SimpleNamespace(read_fit_population=original)
            with self.assertRaisesRegex(ValueError, "separate float FIT"):
                with subject.float_population(entry, lambda *_: root):
                    entry.read_fit_population(object())
            self.assertIs(entry.read_fit_population, original)

    def test_wait_uses_two_ordered_original_workers_and_clean_return(self):
        events = subject.original_owner_events()
        subject.verify_original_events(events)
        with self.assertRaisesRegex(ValueError, "ordered"):
            subject.verify_original_events([events[1], events[0], events[2]])
        with self.assertRaisesRegex(ValueError, "ordered"):
            subject.verify_original_events([*events[:2], dict(events[2], resource_closed=False)])


if __name__ == "__main__":
    unittest.main()
