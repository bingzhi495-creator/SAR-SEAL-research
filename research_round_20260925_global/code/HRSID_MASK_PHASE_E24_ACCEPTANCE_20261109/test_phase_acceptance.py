from hashlib import sha256
import json
from pathlib import Path
import runpy
import sys
from tempfile import TemporaryDirectory
import unittest

from phase_acceptance import (
    PHASE_MARKER,
    compare_ordered_visits,
    create_pair_receipt,
    load_native_validator,
    split_native_stdout,
    validate_phase_run,
    validate_phase_sidecar,
    validate_result,
)


PROTOCOL = "d66baf3c57fafdba4242ad1b55e08cf76bfd77f7"
NATIVE_DIR = Path(__file__).resolve().parent.parent / "HRSID_COCO_E24_FILTERED_TERMINAL_ACCEPTANCE_20261109"
NATIVE_SOURCE = NATIVE_DIR / "acceptance.py"
PHASE_COMPLETE = "HRSID_MASK_PHASE_E24_OUTPUT_COMPLETE"
PHASE_SIDECAR_STATUS = "NATIVE_HRSID_MASK_RESIZE_E24_CLOSED_PENDING_INDEPENDENT_ACCEPTANCE"


def encoded(value, ending=b"\n"):
    return json.dumps(value, sort_keys=True).encode("utf-8") + ending


def phase_event(*, mode="center", protocol=PROTOCOL, ending=b"\n"):
    return encoded(dict(event=PHASE_COMPLETE, mask_resize_mode=mode, protocol=protocol), ending)


def expected_marker():
    return dict(protocol=PROTOCOL, mask_resize_mode="center", stage="e24", seed=5201,
                phase_trainer_sha256="a" * 64, phase_patch_sha256="b" * 64,
                native_trainer_sha256="c" * 64,
                initial_expanded_sha256="63ff93afdd977ba79b5b5a46ba8b4839ef6c40eb0722b29e4667d4932404fbc2")


def visit_ledger(changes=None, *, swap_padding=False):
    lines = []
    changes = changes or {}
    for epoch in range(1, 25):
        visits = [[iid, 1] for iid in range(2937)] + [[iid, 1] for iid in range(3)]
        for batch, row in enumerate(visits):
            row[1] = changes.get((epoch, batch), 1)
        if epoch == 24 and swap_padding:
            visits[-2], visits[-1] = visits[-1], visits[-2]
        lines.append(encoded(dict(epoch=epoch, visits=visits)))
    return b"".join(lines)


def native_fixture(native):
    prior_module = sys.modules.get("acceptance")
    try:
        sys.modules["acceptance"] = native
        fixture = runpy.run_path(str(NATIVE_DIR / "test_acceptance.py"))["synthetic_run"]
        return fixture()
    finally:
        if prior_module is None:
            del sys.modules["acceptance"]
        else:
            sys.modules["acceptance"] = prior_module


def zero_phase_fixture(native, mode, *, wrong_mode_on_e8=False):
    args, native_reader, calls = native_fixture(native)
    closure, result_bytes, epoch_bytes, visits, _ = args
    phase = dict(expected_marker(), mask_resize_mode=mode)
    audits = [dict(json.loads(line), discarded_empty_mask_rows=0) for line in epoch_bytes.splitlines()]
    epochs = b"".join(map(encoded, audits))
    result = dict(json.loads(result_bytes), epoch_audit_sha256=sha256(epochs).hexdigest(),
                  filtered_empty_rows_total=0)
    if mode == "center":
        for ref, digit in zip(result["archive_refs"], "456", strict=True):
            ref["archive_sha256"] = digit * 64
    result_bytes = encoded(result)
    stdout = b"".join(encoded(dict(audit, event="HRSID_COCO_NATIVE_EPOCH_OBSERVED")) for audit in audits)
    stdout += result_bytes + phase_event(mode=mode)
    closure = dict(closure, stdout_sha256=sha256(stdout).hexdigest())
    sidecar = encoded(dict(phase, status=PHASE_SIDECAR_STATUS,
                           native_result_sha256=sha256(result_bytes).hexdigest()))

    def reader(ref):
        saved, size, _ = native_reader(ref)
        saved[native.MARKER]["filtered_empty_rows_total"] = 0
        saved[PHASE_MARKER] = phase.copy()
        if wrong_mode_on_e8 and ref["epoch"] == 8:
            saved[PHASE_MARKER]["mask_resize_mode"] = "native" if mode == "center" else "center"
        return saved, size, ref["archive_sha256"]

    return (closure, result_bytes, epochs, visits, stdout, sidecar, phase, reader, range(native.FIT)), calls


class PhaseAcceptanceTests(unittest.TestCase):
    def test_split_preserves_every_native_byte_with_lf_or_crlf(self):
        for ending in (b"\n", b"\r\n"):
            with self.subTest(ending=ending):
                prefix = b"runner log \xe4\xb8\xad" + ending + encoded({"filtered_empty_rows_total": 0}, ending)
                self.assertEqual(split_native_stdout(prefix + phase_event(ending=ending), "center"), prefix)

    def test_split_rejects_wrong_or_nonunique_terminal_event(self):
        prefix = encoded({"filtered_empty_rows_total": 0})
        rejected = (
            prefix,
            prefix + phase_event(mode="native"),
            prefix + phase_event(protocol="a" * 40),
            prefix + phase_event() + b"post-closure log\n",
            prefix + phase_event() + phase_event(),
            prefix + phase_event().rstrip(b"\n"),
        )
        for raw in rejected:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                split_native_stdout(raw, "center")

    def test_load_pins_source_bytes_and_gives_each_call_a_fresh_protocol_namespace(self):
        original = NATIVE_SOURCE.read_bytes()
        first = load_native_validator(NATIVE_SOURCE)
        self.assertEqual(first.PROTOCOL, PROTOCOL)
        first.PROTOCOL = "a" * 40
        self.assertEqual(load_native_validator(NATIVE_SOURCE).PROTOCOL, PROTOCOL)
        self.assertEqual(NATIVE_SOURCE.read_bytes(), original)
        with TemporaryDirectory() as directory:
            changed = Path(directory).resolve() / "acceptance.py"
            changed.write_bytes(original + b"\n# harmless comment still changes the pinned source\n")
            with self.assertRaises(ValueError):
                load_native_validator(changed)

    def test_sidecar_authenticates_mode_source_and_exact_result_bytes(self):
        marker = expected_marker()
        result = b'{"filtered_empty_rows_total":0}\n'
        actual = dict(marker, status=PHASE_SIDECAR_STATUS, native_result_sha256=sha256(result).hexdigest())
        self.assertEqual(validate_phase_sidecar(encoded(actual), result, marker), marker)
        for mutation in (
            {"mask_resize_mode": "native"},
            {"phase_patch_sha256": "e" * 64},
            {"native_result_sha256": sha256(result.rstrip()).hexdigest()},
            {"status": "COMPLETE"},
            {"seed": float(marker["seed"])},
            {"unapproved_field": True},
        ):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_phase_sidecar(encoded(dict(actual, **mutation)), result, marker)

    def test_order_comparison_keeps_padded_visits_and_separates_target_count_changes(self):
        baseline = visit_ledger()
        center = visit_ledger({(1, 0): 0, (8, 15): 2, (24, 2939): 164})
        summary = compare_ordered_visits(baseline, center)
        self.assertIs(summary["ordered_fit_visits_match"], True)
        self.assertEqual(summary["completed_epochs"], 24)
        self.assertEqual(summary["visits_per_arm"], 70560)
        self.assertEqual(summary["differing_target_count_visits"], 3)
        with self.assertRaises(ValueError):
            compare_ordered_visits(baseline, visit_ledger(swap_padding=True))
        with self.assertRaises(ValueError):
            compare_ordered_visits(baseline, baseline.splitlines(keepends=True)[0])

    def test_zero_filter_count_keeps_original_semantics_and_all_legacy_checks(self):
        native = load_native_validator(NATIVE_SOURCE)
        args, _, _ = native_fixture(native)
        _, result_bytes, epoch_bytes, visit_bytes, _ = args
        result = json.loads(result_bytes)
        self.assertEqual(validate_result(native, result_bytes, epoch_bytes, visit_bytes), result)
        with self.assertRaises(ValueError):
            native.validate_result(encoded(dict(result, filtered_empty_rows_total=0)), epoch_bytes, visit_bytes)
        zero = dict(result, filtered_empty_rows_total=0)
        self.assertEqual(validate_result(native, encoded(zero), epoch_bytes, visit_bytes), zero)
        for invalid in (False, -1, 164 * 2940 * 24 + 1):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_result(native, encoded(dict(result, filtered_empty_rows_total=invalid)),
                                epoch_bytes, visit_bytes)
        with self.assertRaises(ValueError):
            validate_result(native, encoded(zero), epoch_bytes, visit_bytes + b" ")


    def test_closed_zero_filter_runs_bind_both_arms_to_one_pair(self):
        native = load_native_validator(NATIVE_SOURCE)
        arms, ledgers = [], []
        for mode in ("native", "center"):
            args, calls = zero_phase_fixture(native, mode)
            raw, digest = validate_phase_run(native, *args)
            self.assertEqual(calls, [1, 8, 24])
            self.assertEqual(digest, sha256(raw).hexdigest())
            self.assertEqual(json.loads(raw)["filtered_empty_rows_total"], 0)
            self.assertEqual(json.loads(raw)["phase"], dict(expected_marker(), mask_resize_mode=mode))
            arms.append(raw)
            ledgers.append(args[3])
        raw, digest, stats = create_pair_receipt(*arms, *ledgers)
        self.assertEqual(digest, sha256(raw).hexdigest())
        self.assertEqual(json.loads(raw)["native_arm_receipt_sha256"], sha256(arms[0]).hexdigest())
        self.assertEqual(json.loads(raw)["center_arm_receipt_sha256"], sha256(arms[1]).hexdigest())
        self.assertEqual(stats["visits_per_arm"], 70560)

    def test_wrong_e8_arm_marker_blocks_reading_the_e24_archive(self):
        native = load_native_validator(NATIVE_SOURCE)
        args, calls = zero_phase_fixture(native, "center", wrong_mode_on_e8=True)
        with self.assertRaisesRegex(ValueError, "phase marker"):
            validate_phase_run(native, *args)
        self.assertEqual(calls, [1, 8])


if __name__ == "__main__":
    unittest.main()
