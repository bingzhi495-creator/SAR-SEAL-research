import json
from pathlib import Path
import unittest

import one_pass_common_short_receipt as receipt
from test_common_fit_pair_receipt import line, make_arm, pretty, token


PROTOCOL = "1" * 40
IMPLEMENTATION = "2" * 40


def make_one_pass_arm(mode, *, change=None, edit_stats=None, edit_row=None):
    legacy_mode = "native" if mode == "stepwise_center" else "center"
    legacy = make_arm(legacy_mode)
    report, visits, witness = map(json.loads, (legacy.native_report, legacy.native_visits, legacy.seed_witness))
    rows = [json.loads(record) for record in legacy.consumed_ledger.splitlines()]
    for row in rows:
        if not row["crop"]:
            row["common_changed_masks"] = 0
            row["center_common_masks_sha256"] = row["native_common_masks_sha256"]
    if edit_row:
        edit_row(rows)
    field = "native_common_masks_sha256" if mode == "stepwise_center" else "center_common_masks_sha256"
    for row, visit in zip(rows, visits, strict=True):
        visit.update(masks_sha256=row[field], postfilter_masks_sha256=row[field])
    if change:
        change(visits)
    visits_raw = pretty(visits)
    report.update(protocol=PROTOCOL, mask_resize_mode=mode,
                  phase_trainer_sha256=receipt.SOURCE_PINS[receipt.SELF], visit_ledger_sha256=receipt.digest(visits_raw))
    witness.update(protocol=PROTOCOL, mask_resize_mode=mode)
    witness_raw = pretty(witness)
    report_raw = pretty(report)
    events = [json.loads(raw) for raw in legacy.stdout.splitlines() if raw.startswith(b"{")]
    events[2] = dict(event=events[2]["event"], seed_witness_sha256=receipt.digest(witness_raw), **witness)
    events[3] = dict(event=events[3]["event"], **report)
    stdout = b"synthetic stdout\n" + b"".join(line(event) for event in events)
    closure = dict(legacy.closure, stdout_sha256=receipt.digest(stdout))
    common = json.loads(legacy.common_receipt)
    ledger = b"".join(line(row) for row in rows)
    common.update(status=receipt.AUDIT_STATUS, native_result_sha256=receipt.digest(report_raw),
                  repeat_seed_witness_sha256=receipt.digest(witness_raw), consumed_ledger_sha256=receipt.digest(ledger),
                  consumed_ledger_bytes=len(ledger))
    common["phase"].update(protocol=PROTOCOL, mask_resize_mode=mode, phase_trainer_sha256=receipt.SOURCE_PINS[receipt.SELF])
    common["common_support"].update(policy=receipt.POLICY, training_protocol=PROTOCOL, implementation_commit=IMPLEMENTATION,
                                    mode=mode, source_manifest_sha256=receipt.digest(receipt.compact(receipt.SOURCE_PINS)))
    selected = b"".join(line(dict(ordinal=row["ordinal"], image_id=row["image_id"], epoch=row["epoch"],
                                  batch=row["batch"], mask_sha256=row[field])) for row in rows)
    crop_visits = sum(row["crop"] for row in rows)
    changes = sum(row["common_changed_masks"] > 0 for row in rows)
    common["ledger"].update(changed_mask_visits=changes, verified_selected_mask_visits=32,
                            verified_selected_mask_stream_sha256=receipt.digest(selected),
                            consumed_direct_visits=32 - crop_visits, consumed_crop_visits=crop_visits,
                            changed_crop_visits=sum(row["crop"] and row["common_changed_masks"] > 0 for row in rows))
    if edit_stats:
        edit_stats(common["ledger"])
    return receipt.ClosedArm(closure, pretty(common), ledger, report_raw, visits_raw, witness_raw, stdout)


def verify(stepwise=None, one_pass=None, **changes):
    inputs = dict(training_protocol=PROTOCOL, implementation_commit=IMPLEMENTATION,
                  source_sha256s=receipt.SOURCE_PINS, stepwise_center=stepwise or make_one_pass_arm("stepwise_center"),
                  one_pass=one_pass or make_one_pass_arm("one_pass"))
    return receipt.accept_short_pair(**dict(inputs, **changes))


class OnePassCommonShortReceiptTests(unittest.TestCase):
    def test_paired_short_receipt_links_actual_selected_streams_and_crop_changes(self):
        result = verify()
        self.assertEqual(result["status"], receipt.PAIR_STATUS)
        self.assertEqual((result["visits_per_arm"], result["optimizer_updates_per_arm"],
                          result["ema_updates_per_arm"], result["changed_crop_visits"]), (32, 8, 8, 16))
        self.assertEqual(result["consumed_direct_visits"], 16)
        self.assertNotEqual(result["stepwise_center_selected_mask_stream_sha256"],
                            result["one_pass_selected_mask_stream_sha256"])
        self.assertEqual(json.loads(receipt.encode_receipt(result)), result)

    def test_actual_mask_digest_must_match_the_selected_arm_even_with_rehashed_native_outputs(self):
        def wrong_actual(visits):
            visits[1].update(masks_sha256=token("misbound"), postfilter_masks_sha256=token("misbound"))

        with self.assertRaisesRegex(ValueError, "Actual selected native postfilter mask"):
            verify(one_pass=make_one_pass_arm("one_pass", change=wrong_actual))

    def test_selected_stream_is_recomputed_from_original_native_batch_visits(self):
        def wrong_stream(stats):
            stats["verified_selected_mask_stream_sha256"] = token("rehashed wrong selected stream")

        with self.assertRaisesRegex(ValueError, "Closed selected-mask stream"):
            verify(stepwise=make_one_pass_arm("stepwise_center", edit_stats=wrong_stream))

    def test_direct_mask_change_is_refused_and_crop_change_is_required(self):
        def changed_direct(rows):
            rows[0].update(common_changed_masks=1, center_common_masks_sha256=token("changed direct"))

        def unchanged_crop(rows):
            for row in rows:
                row.update(common_changed_masks=0, center_common_masks_sha256=row["native_common_masks_sha256"])

        with self.assertRaisesRegex(ValueError, "direct visits must have identical"):
            verify(one_pass=make_one_pass_arm("one_pass", edit_row=changed_direct))
        with self.assertRaisesRegex(ValueError, "actual common crop mask changes"):
            verify(one_pass=make_one_pass_arm("one_pass", edit_row=unchanged_crop))

    def test_both_consumed_ledgers_and_original_terminals_bind_the_pair(self):
        changed = make_one_pass_arm("one_pass", edit_row=lambda rows: rows[0].update(trace_sha256=token("other trace")))
        with self.assertRaisesRegex(ValueError, "ordered shared consumed ledger"):
            verify(one_pass=changed)
        stepwise, one_pass = make_one_pass_arm("stepwise_center"), make_one_pass_arm("one_pass")
        duplicate = one_pass._replace(closure=dict(one_pass.closure, terminal_sha256=stepwise.closure["terminal_sha256"]))
        with self.assertRaisesRegex(ValueError, "Distinct original OS-closed"):
            verify(stepwise=stepwise, one_pass=duplicate)

    def test_exact_prospective_source_map_and_mode_are_required(self):
        from one_pass_common_qualification import configured_engine

        here = Path(__file__).resolve().parent
        engine = configured_engine(here / "common_fit_train.py")
        self.assertEqual(receipt.SOURCE_PINS,
                         {**engine.SOURCE_PINS, engine.SELF: receipt.digest((here / engine.SELF).read_bytes())})
        with self.assertRaisesRegex(ValueError, "transitive one-pass qualification source pins"):
            verify(source_sha256s=dict(receipt.SOURCE_PINS, **{"one_pass_common_binding.py": token("other source")}))
        with self.assertRaisesRegex(ValueError, "separate prospective"):
            verify(training_protocol=receipt.OLD_PROTOCOLS[2])
        with self.assertRaisesRegex(ValueError, "fixed-B one-pass short policy and lineage"):
            verify(stepwise=make_one_pass_arm("one_pass"))


if __name__ == "__main__":
    unittest.main()
