import shlex
import unittest

from session_route import build_launch_argv


class SessionRouteTests(unittest.TestCase):
    def test_selected_overlay_precedes_the_same_direct_exec(self):
        logical = ["exec", "/opt/venv/python", "-u", "/owner/main.py",
                   "--spec", "/work/spec.json", "--spec-sha", "1" * 64]
        original = ["ssh", "-T", "research-host", shlex.join(logical)]
        observed = build_launch_argv(original, logical, "/safe/overlay path")
        self.assertEqual(observed[:-1], original[:-1])
        self.assertEqual(observed[-1], "export PYTHONPATH='/safe/overlay path' && " + original[-1])
        self.assertEqual(original[-1], shlex.join(logical))

    def test_rejects_multiple_search_roots_and_a_non_owner_exec(self):
        logical = ["exec", "/opt/venv/python", "-u", "/owner/main.py",
                   "--spec", "/work/spec.json", "--spec-sha", "1" * 64]
        original = ["ssh", "-T", "research-host", shlex.join(logical)]
        with self.assertRaises(ValueError):
            build_launch_argv(original, logical, "/one:/two")
        with self.assertRaises(ValueError):
            build_launch_argv(original[:-1] + ["exec /usr/bin/true"], logical, "/safe/overlay")


if __name__ == "__main__":
    unittest.main()
