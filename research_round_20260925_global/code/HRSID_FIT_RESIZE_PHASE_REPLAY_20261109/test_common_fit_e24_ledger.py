"""Synthetic streams for independent fixed E24 common FIT verification."""
from hashlib import sha256
from io import BytesIO
import json
import unittest
from unittest.mock import patch

import common_fit_e24_ledger as subject


ZERO = "0" * 64
ONE = "1" * 64


def encode(row):
    return (json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def shared_row(epoch, batch, epoch_visits):
    ordinal = (epoch - 1) * epoch_visits + batch
    return dict(
        ordinal=ordinal, epoch=epoch, batch=batch, image_id=100 + batch,
        crop=batch % 2 == 1, flipped=False,
        raw_rows=3, native_present=2, center_present=2, native_nonempty=2, center_nonempty=2,
        common_nonempty=1, native_only_nonempty=1, center_only_nonempty=1, common_changed_masks=1,
        image_sha256=ZERO, trace_sha256=ZERO, rng_before_sha256=ZERO, rng_after_sha256=ZERO,
        common_row_ids_sha256=ZERO, common_nonmask_sha256=ZERO,
        native_common_masks_sha256=ZERO, center_common_masks_sha256=ONE,
    )


def safe_receipt(mode, common, visits, epochs, updates, *, empty=0, changes=None):
    return dict(
        status="HRSID_MASK_COMMON_FIT_ARM_CLOSED_PENDING_INDEPENDENT_ACCEPTANCE_V1",
        common_support=dict(stage="e24", mode=mode),
        phase=dict(stage="e24", mask_resize_mode=mode),
        native_result_sha256=ZERO, repeat_seed_witness_sha256=ONE,
        consumed_ledger_sha256=sha256(common).hexdigest(), consumed_ledger_bytes=len(common),
        ledger=dict(fetched_visits=visits, consumed_visits=visits, prefetched_unconsumed=0,
                    unconsumed_prefetch_sha256=None, common_targets_total=visits-empty, common_empty_visits=empty,
                    changed_mask_visits=visits if changes is None else changes),
        optimizer_updates=updates, ema_updates=updates, completed_epochs=epochs,
        FIT=True, DEV=False, TEST=False, AP=False,
    )


def fixture(epochs=2, epoch_visits=4, *, edit_common=None, edit_native=None):
    rows = [shared_row(epoch, batch, epoch_visits) for epoch in range(1, epochs + 1)
            for batch in range(1, epoch_visits + 1)]
    if edit_common:
        edit_common(rows)
    native = [dict(epoch=epoch, visits=[[row["image_id"], row["common_nonempty"]] for row in rows
                                        if row["epoch"] == epoch]) for epoch in range(1, epochs + 1)]
    if edit_native:
        edit_native(native)
    common_raw = b"".join(encode(row) for row in rows)
    native_raw = b"".join(encode(row) for row in native)
    reports = {mode: safe_receipt(mode, common_raw, epochs*epoch_visits, epochs, epochs*epoch_visits//4,
                                  empty=sum(row["common_nonempty"] == 0 for row in rows),
                                  changes=sum(row["common_changed_masks"] > 0 for row in rows))
               for mode in ("native", "center")}
    return common_raw, native_raw, reports


def arguments():
    common, visits, reports = fixture()
    return dict(native_common=BytesIO(common), center_common=BytesIO(common), native_visits=BytesIO(visits),
                center_visits=BytesIO(visits), native_common_safe=reports["native"], center_common_safe=reports["center"])



class LazyLines:
    """Test that the verifier uses only bounded line reads."""

    def __init__(self, lines):
        self.lines = iter(lines)

    def readline(self, limit):
        assert 0 < limit <= 1024**2 + 1
        return next(self.lines, b"")[:limit]


class E24LedgerTests(unittest.TestCase):
    def verify_small(self, **options):
        args = arguments()
        args.update(options)
        with patch.multiple(subject, EPOCHS=2, VISITS_PER_EPOCH=4, UPDATES_PER_EPOCH=1):
            return subject.verify_e24_ledgers(**args)

    def test_synthetic_pair_recovers_shared_and_original_native_streams(self):
        common, visits, reports = fixture()
        result = self.verify_small()
        self.assertEqual(result, dict(
            ordered_common_fit_visits_match=True, completed_epochs=2, visits_per_arm=8,
            optimizer_updates_per_arm=2, ema_updates_per_arm=2,
            shared_consumed_ledger_sha256=sha256(common).hexdigest(), shared_consumed_ledger_bytes=len(common),
            native_visit_ledger_sha256=sha256(visits).hexdigest(), center_visit_ledger_sha256=sha256(visits).hexdigest(),
            ledger=reports["native"]["ledger"], native_only_nonempty_total=8, center_only_nonempty_total=8,
            common_changed_masks_total=8, support_disagreement_visits=8,
        ))


    def test_full_locked_shape_with_lazy_synthetic_streams(self):
        def shared():
            for epoch in range(1, 25):
                for batch in range(1, 2941):
                    yield encode(shared_row(epoch, batch, 2940))

        def native():
            visits = [[100 + batch, 1] for batch in range(1, 2941)]
            for epoch in range(1, 25):
                yield encode(dict(epoch=epoch, visits=visits))

        total = 0
        digest = sha256()
        for line in shared():
            digest.update(line)
            total += len(line)
        receipts = {mode: safe_receipt(mode, b"", 70560, 24, 17640) for mode in ("native", "center")}
        for receipt in receipts.values():
            receipt.update(consumed_ledger_sha256=digest.hexdigest(), consumed_ledger_bytes=total)
        result = subject.verify_e24_ledgers(
            native_common=LazyLines(shared()), center_common=LazyLines(shared()),
            native_visits=LazyLines(native()), center_visits=LazyLines(native()),
            native_common_safe=receipts["native"], center_common_safe=receipts["center"],
        )
        self.assertEqual(result["visits_per_arm"], 70560)
        self.assertEqual(result["optimizer_updates_per_arm"], 17640)
        self.assertEqual(result["ema_updates_per_arm"], 17640)
        self.assertEqual(result["ledger"]["changed_mask_visits"], 70560)
        self.assertEqual(result["shared_consumed_ledger_sha256"], digest.hexdigest())

    def test_same_counts_do_not_allow_different_common_rng_bytes(self):
        altered, _, reports = fixture(edit_common=lambda rows: rows[2].update(rng_before_sha256=ONE))
        with self.assertRaisesRegex(ValueError, "common FIT consumed streams must match"):
            self.verify_small(center_common=BytesIO(altered), center_common_safe=reports["center"])

    def test_both_original_native_streams_must_match_common_order_and_counts(self):
        for index, new in ((0, 112), (1, 0)):
            _, bad, _ = fixture(edit_native=lambda epochs: epochs[1]["visits"][1].__setitem__(index, new))
            with self.subTest(field=index):
                with self.assertRaisesRegex(ValueError, "Original native FIT visit streams differ"):
                    self.verify_small(center_visits=BytesIO(bad))
                with self.assertRaisesRegex(ValueError, "retained targets must match both"):
                    self.verify_small(native_visits=BytesIO(bad), center_visits=BytesIO(bad))

    def test_wrong_epoch_or_batch_cursor_is_refused(self):
        args = arguments()
        common = args["native_common"].getvalue().replace(b'"batch":4', b'"batch":5', 1)
        with self.assertRaisesRegex(ValueError, "common FIT visit cursor"):
            self.verify_small(native_common=BytesIO(common), center_common=BytesIO(common))
        visits = args["native_visits"].getvalue().replace(b'"epoch":2', b'"epoch":3', 1)
        with self.assertRaisesRegex(ValueError, "native FIT epoch cursor"):
            self.verify_small(native_visits=BytesIO(visits), center_visits=BytesIO(visits))

    def test_invalid_support_and_no_changed_masks_are_refused(self):
        raw, _, _ = fixture(edit_common=lambda rows: rows[0].update(raw_rows=2))
        with self.assertRaisesRegex(ValueError, "Invalid common support or mask change"):
            self.verify_small(native_common=BytesIO(raw), center_common=BytesIO(raw))

        def same_masks(rows):
            for row in rows:
                row.update(center_common_masks_sha256=ZERO, common_changed_masks=0)
        raw, _, reports = fixture(edit_common=same_masks)
        with self.assertRaisesRegex(ValueError, "At least one actual E24"):
            self.verify_small(native_common=BytesIO(raw), center_common=BytesIO(raw),
                              native_common_safe=reports["native"], center_common_safe=reports["center"])

    def test_empty_common_visit_keeps_its_place_and_counts(self):
        def zero_common(rows):
            rows[0].update(common_nonempty=0, native_nonempty=1, center_nonempty=1,
                           center_common_masks_sha256=ZERO, common_changed_masks=0)
        raw, native, reports = fixture(edit_common=zero_common)
        result = self.verify_small(native_common=BytesIO(raw), center_common=BytesIO(raw),
                                   native_visits=BytesIO(native), center_visits=BytesIO(native),
                                   native_common_safe=reports["native"], center_common_safe=reports["center"])
        self.assertEqual(result["visits_per_arm"], 8)
        self.assertEqual(result["ledger"]["common_empty_visits"], 1)
        self.assertEqual(result["ledger"]["common_targets_total"], 7)
        self.assertEqual(result["ledger"]["changed_mask_visits"], 7)

    def test_lookahead_and_false_summaries_or_digests_are_refused(self):
        for field, value, error in (
            ("prefetched_unconsumed", 1, "zero unconsumed"),
            ("common_targets_total", 7, "summary disagrees"),
            ("changed_mask_visits", 0, "summary disagrees"),
        ):
            receipt = arguments()["center_common_safe"]
            receipt["ledger"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, error):
                self.verify_small(center_common_safe=receipt)
        receipt = arguments()["center_common_safe"]
        receipt["consumed_ledger_sha256"] = ZERO
        with self.assertRaisesRegex(ValueError, "digest or byte count"):
            self.verify_small(center_common_safe=receipt)

    def test_stream_bounds_canonical_encoding_and_eof(self):
        good = arguments()["native_common"].getvalue()
        for raw, error in (
            (good[:-1], "unterminated"),
            (b"x" * (subject.COMMON_LINE_LIMIT + 10) + b"\n", "oversized"),
            (good.replace(b'{"batch":', b'{ "batch":', 1), "Expected canonical"),
            (good + b"unexpected", "trailing bytes"),
        ):
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                self.verify_small(native_common=BytesIO(raw), center_common=BytesIO(raw))


if __name__ == "__main__":
    unittest.main()

