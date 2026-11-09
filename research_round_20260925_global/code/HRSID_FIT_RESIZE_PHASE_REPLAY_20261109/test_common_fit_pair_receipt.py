"""Synthetic closure and consumed-stream tests for a common FIT qualification pair."""
from hashlib import sha256
import json
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory
import unittest

from common_fit_train import FIT_INDEX_SHA, INITIAL_EXPANDED_SHA, SOURCE_PINS, SOURCES, verify_qualification
from common_fit_pair_receipt import ClosedArm, accept_short_pair, encode_receipt


PROTOCOL = "a" * 40
IMPLEMENTATION = "b" * 40
SOURCE_HASHES = {**SOURCE_PINS, "common_fit_train.py": "c" * 64, "common_fit_binding.py": "d" * 64}
assert set(SOURCE_HASHES) == SOURCES
PRIOR = runpy.run_path(str(Path(__file__).resolve().parent.parent /
                          "HRSID_MASK_PHASE_REPEAT_QUALIFICATION_20261109/test_repeat_qualification.py"))


def digest(raw):
    return sha256(raw).hexdigest()


def token(value):
    return digest(value.encode())


def pretty(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def line(value):
    return (json.dumps(value, separators=(",", ":"), sort_keys=True, allow_nan=False) + "\n").encode()


def make_arm(mode, *, edit_rows=None, edit_receipt=None, edit_native=None, edit_closure=None):
    old_closure, report_raw, visits_raw, witness_raw, stdout_raw = PRIOR["arm_fixture"](5202, mode)
    report, visits, witness = map(json.loads, (report_raw, visits_raw, witness_raw))
    report.update(protocol=PROTOCOL, phase_trainer_sha256=SOURCE_HASHES["common_fit_train.py"])
    witness["protocol"] = PROTOCOL
    if edit_native:
        edit_native(visits, report, witness)
    visits_raw, witness_raw = pretty(visits), pretty(witness)
    report["visit_ledger_sha256"] = digest(visits_raw)
    identity = [(row["image_id"], row["image_sha256"], row["cpu_rng_sha256"]) for row in visits]
    report["input_pair_sha256"] = digest(json.dumps(identity, separators=(",", ":")).encode())
    events = [json.loads(raw) for raw in stdout_raw.splitlines() if raw.startswith(b"{")]
    events[2] = dict(event=events[2]["event"], seed_witness_sha256=digest(witness_raw), **witness)
    events[3] = dict(event=events[3]["event"], **report)
    stdout = b"synthetic stdout\n" + b"".join(line(event) for event in events)
    phase = {name: report[name] for name in (
        "protocol", "mask_resize_mode", "stage", "seed", "phase_trainer_sha256",
        "phase_patch_sha256", "native_trainer_sha256", "initial_expanded_sha256",
    )}
    common_support = dict(policy="ordered_common_nonempty_prepared_rows_v1", training_protocol=PROTOCOL,
                          implementation_commit=IMPLEMENTATION, stage="qualification", mode=mode,
                          training_seed=5202, initialization_seed=5201, initial_expanded_sha256=INITIAL_EXPANDED_SHA,
                          fit_index_sha256=FIT_INDEX_SHA, source_manifest_sha256=digest(line(SOURCE_HASHES)[:-1]),
                          qualification_pair_sha256=None)
    rows = []
    for i, row in enumerate(visits, 1):
        rows.append(dict(ordinal=i, epoch=1, batch=i, image_id=row["image_id"], crop=i % 2 == 0, flipped=False,
                         raw_rows=1, native_present=1, center_present=1, native_nonempty=1, center_nonempty=1,
                         common_nonempty=1, native_only_nonempty=0, center_only_nonempty=0, common_changed_masks=1,
                         image_sha256=token(f"image-{i}"), trace_sha256=token(f"trace-{i}"),
                         rng_before_sha256=token(f"before-{i}"), rng_after_sha256=token(f"after-{i}"),
                         common_row_ids_sha256=token("id"), common_nonmask_sha256=token("nonmask"),
                         native_common_masks_sha256=token(f"mask-native-{i - 1}"),
                         center_common_masks_sha256=token(f"mask-center-{i - 1}")))
    if edit_rows:
        edit_rows(rows)
    ledger = b"".join(line(row) for row in rows)
    receipt = dict(status="HRSID_MASK_COMMON_FIT_ARM_CLOSED_PENDING_INDEPENDENT_ACCEPTANCE_V1",
                   common_support=common_support, phase=phase, native_result_sha256=digest(pretty(report)),
                   repeat_seed_witness_sha256=digest(witness_raw), consumed_ledger_sha256=digest(ledger),
                   consumed_ledger_bytes=len(ledger),
                   ledger=dict(fetched_visits=33, consumed_visits=32, prefetched_unconsumed=1,
                               unconsumed_prefetch_sha256=token("unconsumed"), common_targets_total=32,
                               common_empty_visits=0, changed_mask_visits=32),
                   optimizer_updates=8, ema_updates=8, completed_epochs=0, FIT=True, DEV=False, TEST=False, AP=False)
    if edit_receipt:
        edit_receipt(receipt)
    closure = dict(old_closure, stdout_sha256=digest(stdout), terminal_sha256=token(f"terminal-{mode}"))
    if edit_closure:
        edit_closure(closure)
    return ClosedArm(closure, pretty(receipt), ledger, pretty(report), visits_raw, witness_raw, stdout)


def accept(native=None, center=None, **scope):
    options = dict(training_protocol=PROTOCOL, implementation_commit=IMPLEMENTATION, source_sha256s=SOURCE_HASHES)
    options.update(scope)
    return accept_short_pair(native=native or make_arm("native"), center=center or make_arm("center"), **options)


class CommonFitPairReceiptTests(unittest.TestCase):
    def test_pair_emits_exact_e24_receipt(self):
        native, center = make_arm("native"), make_arm("center")
        receipt = accept(native, center)
        expected = dict(status="HRSID_MASK_COMMON_FIT_SHORT_PAIR_ACCEPTED_V1",
                        training_protocol=PROTOCOL, implementation_commit=IMPLEMENTATION, training_seed=5202,
                        initialization_seed=5201, initial_expanded_sha256=INITIAL_EXPANDED_SHA,
                        fit_index_sha256=FIT_INDEX_SHA, source_manifest_sha256=digest(line(SOURCE_HASHES)[:-1]),
                        ordered_common_fit_visits_match=True, visits_per_arm=32, optimizer_updates_per_arm=8,
                        ema_updates_per_arm=8, FIT=True, DEV=False, TEST=False, AP=False, changed_mask_visits=32,
                        native_arm_receipt_sha256=digest(native.common_receipt),
                        center_arm_receipt_sha256=digest(center.common_receipt),
                        shared_consumed_ledger_sha256=digest(native.consumed_ledger))
        self.assertEqual(receipt, expected)
        self.assertEqual(encode_receipt(receipt), pretty(expected))


    def test_e24_worker_accepts_the_exact_encoded_pair(self):
        raw = encode_receipt(accept())
        with TemporaryDirectory() as temp:
            path = Path(temp).resolve() / "PAIR_SAFE.json"
            path.write_bytes(raw)
            lock = dict(training_protocol=PROTOCOL, implementation_commit=IMPLEMENTATION,
                        sources={key: {"sha256": value} for key, value in SOURCE_HASHES.items()},
                        qualification_pair_ref=dict(path=path.as_posix(), bytes=len(raw), sha256=digest(raw)))
            self.assertEqual(verify_qualification(lock), digest(raw))

    def test_row_identity_and_random_stream_must_match_even_when_counts_match(self):
        for field in ("image_sha256", "trace_sha256", "rng_before_sha256", "rng_after_sha256",
                      "common_row_ids_sha256", "common_nonmask_sha256"):
            with self.subTest(field=field):
                center = make_arm("center", edit_rows=lambda rows: rows[7].update({field: token(field)}))
                with self.assertRaisesRegex(ValueError, "consumed streams must match"):
                    accept(center=center)

    def test_invalid_cursor_and_noncanonical_row_are_refused(self):
        shuffled = make_arm("center", edit_rows=lambda rows: rows.reverse())
        with self.assertRaisesRegex(ValueError, "Exact consumed FIT cursor"):
            accept(center=shuffled)
        canonical = make_arm("center")
        wrong = canonical._replace(consumed_ledger=canonical.consumed_ledger.replace(b"\n", b"\r\n", 1))
        with self.assertRaisesRegex(ValueError, "canonical LF-terminated"):
            accept(center=wrong)

    def test_prefetch_is_not_part_of_the_consumed_stream(self):
        def without_prefetch(receipt):
            receipt["ledger"].update(fetched_visits=32, prefetched_unconsumed=0, unconsumed_prefetch_sha256=None)
        no_lookahead = make_arm("center", edit_receipt=without_prefetch)
        self.assertEqual(accept(center=no_lookahead)["visits_per_arm"], 32)

        def false_prefetch(receipt):
            receipt["ledger"].update(fetched_visits=32, prefetched_unconsumed=1)
        bad_lookahead = make_arm("center", edit_receipt=false_prefetch)
        with self.assertRaisesRegex(ValueError, "lookahead accounting"):
            accept(center=bad_lookahead)

    def test_consumed_row_support_must_be_possible(self):
        def invalid_union(rows):
            rows[0].update(native_nonempty=2, native_present=2, native_only_nonempty=1,
                           center_nonempty=2, center_present=2, center_only_nonempty=1, raw_rows=2)
        bad = make_arm("center", edit_rows=invalid_union)
        with self.assertRaisesRegex(ValueError, "Invalid common support or mask changes"):
            accept(center=bad)

    def test_actual_native_batch_must_agree_with_the_consumed_identity(self):
        bad = make_arm("center", edit_rows=lambda rows: rows[3].update(image_id=999))
        with self.assertRaisesRegex(ValueError, "actual native batch image"):
            accept(center=bad)

    def test_actual_native_mask_changes_must_match_audited_changes(self):
        def repeat_native_mask(visits, report, witness):
            mask = token("mask-native-0")
            visits[0].update(masks_sha256=mask, postfilter_masks_sha256=mask)
        bad = make_arm("center", edit_native=repeat_native_mask)
        with self.assertRaisesRegex(ValueError, "reported mask changes"):
            accept(center=bad)

    def test_actual_native_masks_must_match_absolute_consumed_digests(self):
        def substitute_mask(visits, report, witness):
            mask = token("unrelated mask that preserves the change pattern")
            visits[0].update(masks_sha256=mask, postfilter_masks_sha256=mask)

        for mode in ("native", "center"):
            with self.subTest(mode=mode):
                bad = make_arm(mode, edit_native=substitute_mask)
                with self.assertRaisesRegex(ValueError, "Exact consumed mask fingerprints"):
                    accept(**{mode: bad})

    def test_arm_scope_and_worker_closure_cannot_be_forged_by_ledger(self):
        with self.assertRaisesRegex(ValueError, "worker closure"):
            accept(center=make_arm("center", edit_closure=lambda closure: closure.update(worker_closed=False)))
        with self.assertRaisesRegex(ValueError, "Distinct independently closed"):
            accept(center=make_arm("center", edit_closure=lambda closure: closure.update(
                terminal_sha256=token("terminal-native"))))
        with self.assertRaisesRegex(ValueError, "common-support arm marker"):
            accept(center=make_arm("center", edit_receipt=lambda receipt: receipt["common_support"].update(
                qualification_pair_sha256=token("unexpected qualification"))))


if __name__ == "__main__":
    unittest.main()

