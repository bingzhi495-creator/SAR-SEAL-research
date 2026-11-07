"""Synthetic stdlib fixtures only: no audit-hook installation or socket creation."""
import copy
import socket
import unittest

import offline_audit as audit


SOURCE = {
    "path": "/root-verified/site-packages/urllib3/util/connection.py",
    "bytes": 4444,
    "sha256": "2633bbdb69731e5ccb5cf4e4afd65605d86c7979cc5633126f50c92d5ad74a74",
}


def caller():
    return {
        "module": "urllib3.util.connection",
        "function": "_has_ipv6",
        "line": 125,
        "source_ref": copy.deepcopy(SOURCE),
        "code_identity_matches_export": True,
    }


class ClassificationTests(unittest.TestCase):
    def test_exact_capability_probe_is_classified_but_never_allowed(self):
        record = audit.classify_attempt(
            "socket.bind", family=int(socket.AF_INET6), address=("::1", 0),
            callers=[caller()], expected_source_ref=SOURCE,
        )
        self.assertEqual(record["classification"], "expected_blocked_capability_probe")
        self.assertIs(record["allowed"], False)
        self.assertEqual(record["caller"]["source_ref"], SOURCE)
        self.assertEqual(record["address"], ["::1", 0])

    def test_near_matches_and_deep_stack_appearance_are_unexpected(self):
        base = dict(family=int(socket.AF_INET6), address=("::1", 0),
                    callers=[caller()], expected_source_ref=SOURCE)
        wrong_source = copy.deepcopy(SOURCE)
        wrong_source["sha256"] = "0" * 64
        no_path = dict(SOURCE, path="")
        cases = [
            dict(address=("::1", False)), dict(address=("::1", 1)),
            dict(address=("::1", 0, 0, 0)), dict(address=("127.0.0.1", 0)),
            dict(family=float(socket.AF_INET6)), dict(family=int(socket.AF_INET)),
            dict(callers=[dict(caller(), code_identity_matches_export=False)]),
            dict(callers=[dict(caller(), module="somewhere_else")]),
            dict(callers=[dict(caller(), source_ref=wrong_source)]),
            dict(callers=[dict(caller(), source_ref=no_path)], expected_source_ref=no_path),
            dict(callers=[dict(caller(), function="not_the_direct_caller"), caller()]),
            dict(callers=[]),
        ]
        for changed in cases:
            with self.subTest(changed=changed):
                record = audit.classify_attempt("socket.bind", **dict(base, **changed))
                self.assertEqual(record["classification"], "unexpected_blocked")
                self.assertIs(record["allowed"], False)
        for event in ("socket.connect", "socket.getaddrinfo", "subprocess.Popen", "os.system"):
            with self.subTest(event=event):
                record = audit.classify_attempt(event, **base)
                self.assertEqual(record["classification"], "unexpected_blocked")

    def test_summary_never_upgrades_unknown_or_repeated_expected_events(self):
        expected = audit.classify_attempt(
            "socket.bind", family=int(socket.AF_INET6), address=("::1", 0),
            callers=[caller()], expected_source_ref=SOURCE)
        self.assertTrue(audit.assess_events([], SOURCE)["passed"])
        one = audit.assess_events([expected], SOURCE)
        self.assertTrue(one["passed"])
        self.assertEqual(one["expected_count"], 1)
        self.assertIs(one["qualified"], False)
        for records in ([expected, expected],
                        [dict(expected, classification="unknown")],
                        [dict(expected, event="socket.connect")],
                        [dict(expected, allowed=True)],
                        [dict(expected, capture_error="stack unavailable")]):
            with self.subTest(records=records):
                self.assertFalse(audit.assess_events(records, SOURCE)["passed"])

    def test_manual_hook_denies_effect_events_without_creating_socket_or_child(self):
        observer = audit.OfflineAudit(SOURCE)
        for event in ("socket.bind", "socket.connect", "socket.getnameinfo",
                      "socket.sendmsg", "subprocess.Popen", "os.exec", "os.posix_spawn"):
            with self.subTest(event=event):
                with self.assertRaises(OSError):
                    observer.hook(event, ())
        receipt = observer.receipt()
        self.assertEqual(receipt["unexpected_count"], 7)
        self.assertFalse(receipt["passed"])
        self.assertFalse(receipt["installed"])
        self.assertEqual(receipt["events"][0]["caller"]["module"], __name__)
        self.assertEqual(receipt["events"][0]["caller"]["function"],
                         "test_manual_hook_denies_effect_events_without_creating_socket_or_child")
        self.assertIsNone(observer.hook("socket.__new__", ()))

    def test_malformed_missing_evidence_cannot_pass(self):
        for records in (None, {}, [None]):
            with self.subTest(records=records):
                self.assertFalse(audit.assess_events(records, SOURCE)["passed"])
        malformed = audit.classify_attempt(
            "socket.bind", family=int(socket.AF_INET6), address=("::1", 0),
            callers={"fake": caller()}, expected_source_ref=SOURCE)
        self.assertEqual(malformed["classification"], "unexpected_blocked")
        self.assertFalse(audit.assess_events([], dict(SOURCE, sha256="0" * 64))["passed"])


if __name__ == "__main__":
    unittest.main()
