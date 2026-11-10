import unittest

from common_fit_pair_receipt import SCOPE, compact, digest, encode_receipt
import one_pass_common_short_receipt as short

from one_pass_common_e24_acceptance import TRAINER_SOURCE, TRAINER_SOURCE_SHA256, admit_qualification


class QualificationTests(unittest.TestCase):
    @staticmethod
    def short_pair():
        short_protocol = "1" * 40
        short_commit = "2" * 40
        training_protocol = "3" * 40
        implementation_commit = "4" * 40
        trainer_source = TRAINER_SOURCE
        short_receipt = dict(
            status=short.PAIR_STATUS, stage="qualification", policy=short.POLICY,
            training_protocol=short_protocol, implementation_commit=short_commit,
            training_seed=5202, initialization_seed=5201,
            initial_expanded_sha256=short.INITIAL_EXPANDED_SHA, fit_index_sha256=short.FIT_INDEX_SHA,
            source_manifest_sha256=digest(compact(short.SOURCE_PINS)),
            ordered_common_fit_visits_match=True, visits_per_arm=32, optimizer_updates_per_arm=8,
            ema_updates_per_arm=8, verified_selected_mask_visits_per_arm=32,
            changed_mask_visits=5, consumed_direct_visits=11, consumed_crop_visits=21, changed_crop_visits=5,
            stepwise_center_arm_receipt_sha256="1" * 64, one_pass_arm_receipt_sha256="2" * 64,
            stepwise_center_terminal_sha256="3" * 64, one_pass_terminal_sha256="4" * 64,
            stepwise_center_selected_mask_stream_sha256="5" * 64, one_pass_selected_mask_stream_sha256="6" * 64,
            shared_consumed_ledger_sha256="7" * 64, **SCOPE,
        )
        source_sha256s = {**short.SOURCE_PINS, trainer_source: TRAINER_SOURCE_SHA256}
        raw = encode_receipt(short_receipt)
        args = dict(training_protocol=training_protocol, implementation_commit=implementation_commit,
                    source_sha256s=source_sha256s, trainer_source=trainer_source,
                    qualification_protocol=short_protocol, qualification_implementation_commit=short_commit,
                    qualification_pair_bytes=raw, qualification_pair_sha256=digest(raw))
        return args, short_receipt

    def test_prospective_pair_binds_exact_short_receipt_and_formal_source(self):
        args, short_receipt = self.short_pair()
        training_protocol, implementation_commit, source_sha256s, raw = (
            args[key] for key in ("training_protocol", "implementation_commit", "source_sha256s", "qualification_pair_bytes")
        )
        marker = admit_qualification(**args)
        self.assertEqual(marker, dict(training_protocol=training_protocol, implementation_commit=implementation_commit,
                                     training_seed=5202, initialization_seed=5201,
                                     initial_expanded_sha256=short.INITIAL_EXPANDED_SHA, fit_index_sha256=short.FIT_INDEX_SHA,
                                     source_manifest_sha256=digest(compact(source_sha256s)),
                                     qualification_pair_sha256=digest(raw)))
        bad_cases = (
            dict(status="HRSID_MASK_COMMON_FIT_SHORT_PAIR_ACCEPTED_V1"),
            dict(training_seed=5203),
            dict(initialization_seed=5202),
            dict(source_manifest_sha256="9" * 64),
            dict(changed_crop_visits=0, changed_mask_visits=0),
            dict(one_pass_terminal_sha256=short_receipt["stepwise_center_terminal_sha256"]),
            dict(one_pass_selected_mask_stream_sha256=short_receipt["stepwise_center_selected_mask_stream_sha256"]),
        )
        for patch in bad_cases:
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                edited = encode_receipt(dict(short_receipt, **patch))
                admit_qualification(**dict(args, qualification_pair_bytes=edited, qualification_pair_sha256=digest(edited)))
        for patch in (dict(qualification_pair_sha256="0" * 64), dict(qualification_protocol=training_protocol),
                      dict(source_sha256s=dict(source_sha256s, **{TRAINER_SOURCE: "0" * 64})),
                      dict(source_sha256s=dict(source_sha256s, **{"one_pass_common_binding.py": "0" * 64}))):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                admit_qualification(**dict(args, **patch))

    def test_scoped_legacy_validator_checks_exact_b_c_mode_and_stdout(self):
        from one_pass_common_e24_acceptance import scoped_bc_repeat

        verifier = scoped_bc_repeat("3" * 40, "8" * 64)
        phase = verifier._phase
        prefix = b'{"preserved":true}\n'
        for mode in short.MODES:
            marker = verifier.expected_phase(mode)
            self.assertEqual(marker["mask_resize_mode"], mode)
            tail = compact(dict(event=phase.TRAILER, mask_resize_mode=mode, protocol=marker["protocol"])) + b"\n"
            self.assertEqual(phase.split_native_stdout(prefix + tail, mode), prefix)
            with self.assertRaises(ValueError):
                phase.split_native_stdout(prefix + tail + tail, mode)
            with self.assertRaises(ValueError):
                phase.split_native_stdout(prefix + tail, "center")
        with self.assertRaises(ValueError):
            verifier.expected_phase("center")


class LedgerTests(unittest.TestCase):
    def test_formal_readback_reconciles_selected_streams_and_direct_equality(self):
        from io import BytesIO
        from unittest.mock import patch

        import common_fit_e24_ledger as original
        from one_pass_common_e24_acceptance import verify_bc_e24_ledgers

        rows = []
        for cursor, crop in enumerate((False, True), 1):
            row = dict.fromkeys(original.COUNTS, 0)
            row.update(dict.fromkeys(original.DIGESTS, "1" * 64))
            row.update(ordinal=cursor, epoch=1, batch=cursor, image_id=cursor, crop=crop, flipped=False,
                       raw_rows=1, native_present=1, center_present=1, native_nonempty=1, center_nonempty=1,
                       common_nonempty=1, common_changed_masks=int(crop))
            if crop:
                row["center_common_masks_sha256"] = "2" * 64
            rows.append(row)
        native = compact(dict(epoch=1, visits=[[1, 1], [2, 1]])) + b"\n"

        def check(selected_rows, *, corrupt=None):
            consumed = b"".join(compact(row) + b"\n" for row in selected_rows)
            receipts = []
            for mode in short.MODES:
                selected = b"".join(compact(dict(
                    ordinal=row["ordinal"], epoch=row["epoch"], batch=row["batch"], image_id=row["image_id"],
                    mask_sha256=row[short.SELECTED_MASK[mode]],
                )) + b"\n" for row in selected_rows)
                receipts.append(dict(
                    status=short.AUDIT_STATUS, completed_epochs=1, optimizer_updates=1, ema_updates=1, **SCOPE,
                    common_support=dict(stage="e24", mode=mode), phase=dict(stage="e24", mask_resize_mode=mode),
                    native_result_sha256="3" * 64, repeat_seed_witness_sha256="4" * 64,
                    consumed_ledger_sha256=digest(consumed), consumed_ledger_bytes=len(consumed),
                    ledger=dict(fetched_visits=2, consumed_visits=2, prefetched_unconsumed=0,
                                unconsumed_prefetch_sha256=None, common_targets_total=2, common_empty_visits=0,
                                changed_mask_visits=1, verified_selected_mask_visits=2,
                                verified_selected_mask_stream_sha256=digest(selected), consumed_direct_visits=1,
                                consumed_crop_visits=1, changed_crop_visits=1),
                ))
            if corrupt == "selected":
                receipts[1]["ledger"]["verified_selected_mask_stream_sha256"] = "9" * 64
            return verify_bc_e24_ledgers(
                stepwise_center_common=BytesIO(consumed), one_pass_common=BytesIO(consumed),
                stepwise_center_visits=BytesIO(native), one_pass_visits=BytesIO(native),
                stepwise_center_safe=receipts[0], one_pass_safe=receipts[1],
            )

        with patch.multiple(original, EPOCHS=1, VISITS_PER_EPOCH=2, UPDATES_PER_EPOCH=1):
            proof = check(rows)
            self.assertEqual(proof["changed_crop_visits"], 1)
            self.assertEqual(proof["consumed_direct_visits"], 1)
            self.assertNotEqual(proof["stepwise_center_selected_mask_stream_sha256"],
                                proof["one_pass_selected_mask_stream_sha256"])
            with self.assertRaisesRegex(ValueError, "selected-mask"):
                check(rows, corrupt="selected")
            with self.assertRaisesRegex(ValueError, "Direct"):
                check([dict(rows[0], common_changed_masks=1, center_common_masks_sha256="2" * 64), rows[1]])
            with self.assertRaisesRegex(ValueError, "same B crop roster"):
                check([rows[0], dict(rows[1], raw_rows=2, center_present=2)])


class EndpointTests(unittest.TestCase):
    def test_fixed_budget_accepts_both_original_arm_endpoints(self):
        from hashlib import sha256
        import json
        from unittest.mock import patch

        import test_common_fit_e24_acceptance as prior
        from one_pass_common_e24_acceptance import PAIR_STATUS, accept_e24_pair, scoped_bc_repeat

        arguments, _ = QualificationTests.short_pair()
        identity = admit_qualification(**arguments)
        source_sha = arguments["source_sha256s"][arguments.pop("trainer_source")]
        protocol = identity["training_protocol"]
        verifier = scoped_bc_repeat(protocol, source_sha)

        def stream(visits):
            for line in visits.splitlines():
                epoch = json.loads(line)
                for batch, (image_id, retained) in enumerate(epoch["visits"], 1):
                    row = prior.shared_row(epoch["epoch"], batch, 2940)
                    self.assertEqual(retained, row["common_nonempty"])
                    row.update(image_id=image_id, native_nonempty=1, center_nonempty=1,
                               native_only_nonempty=0, center_only_nonempty=0, common_changed_masks=int(row["crop"]))
                    if not row["crop"]:
                        row["center_common_masks_sha256"] = row["native_common_masks_sha256"]
                    yield row

        with patch.multiple(prior, PROTOCOL=protocol):
            old_b, calls_b = prior.synthetic_arm("native")
            old_c, calls_c = prior.synthetic_arm("center")
        mask_streams = {mode: sha256() for mode in short.MODES}
        common_hash = sha256()
        common_bytes = 0
        for row in stream(old_b.visits):
            line = compact(row) + b"\n"
            common_hash.update(line)
            common_bytes += len(line)
            for mode in short.MODES:
                keys = ("ordinal", "image_id", "epoch", "batch")
                mask = dict({key: row[key] for key in keys}, mask_sha256=row[short.SELECTED_MASK[mode]])
                mask_streams[mode].update(compact(mask) + b"\n")

        def translate(old, mode, terminal):
            marker = verifier.expected_phase(mode)
            common = dict(identity, policy=short.POLICY, stage="e24", mode=mode)
            witness = dict(json.loads(old.seed_witness), mask_resize_mode=mode)
            witness_raw = encode_receipt(witness)
            event = compact(dict(witness, event="HRSID_MASK_PHASE_REPEAT_SEED_OBSERVED",
                                 seed_witness_sha256=digest(witness_raw))) + b"\n"
            logs = old.stdout.splitlines(keepends=True)
            tail = compact(dict(json.loads(logs[-1]), mask_resize_mode=mode)) + b"\n"
            stdout = event + b"".join(logs[1:-1]) + tail
            audit = dict(json.loads(old.common_audit), status=short.AUDIT_STATUS, common_support=common, phase=marker,
                         consumed_ledger_sha256=common_hash.hexdigest(), consumed_ledger_bytes=common_bytes,
                         repeat_seed_witness_sha256=digest(witness_raw), ledger=dict(
                             fetched_visits=70560, consumed_visits=70560, prefetched_unconsumed=0,
                             unconsumed_prefetch_sha256=None, common_targets_total=70560, common_empty_visits=0,
                             changed_mask_visits=35280, verified_selected_mask_visits=70560,
                             verified_selected_mask_stream_sha256=mask_streams[mode].hexdigest(),
                             consumed_direct_visits=35280, consumed_crop_visits=35280, changed_crop_visits=35280))
            sidecar = encode_receipt(dict(json.loads(old.phase_provenance), **marker))

            def reader(ref):
                saved, size, sha = old.checkpoint_reader(ref)
                saved[prior.COMMON_MARKER] = common
                saved[verifier._phase.PHASE_MARKER] = marker
                return saved, size, sha

            return old._replace(
                closure=dict(old.closure, stdout_sha256=digest(stdout), terminal_sha256=terminal * 64),
                common_audit=encode_receipt(audit), stdout=stdout, phase_provenance=sidecar,
                seed_witness=witness_raw, checkpoint_reader=reader,
            )

        arm_b = translate(old_b, "stepwise_center", "a")
        arm_c = translate(old_c, "one_pass", "b")
        streams = dict(
            stepwise_center_common=prior.LazyLines(compact(row) + b"\n" for row in stream(arm_b.visits)),
            one_pass_common=prior.LazyLines(compact(row) + b"\n" for row in stream(arm_c.visits)),
        )
        closed = accept_e24_pair(**arguments, stepwise_center=arm_b, one_pass=arm_c, **streams,
                                 expected_fit_ids=range(2937))
        pair = json.loads(closed.pair_receipt)
        self.assertEqual((calls_b, calls_c), ([1, 8, 24], [1, 8, 24]))
        self.assertEqual((pair["status"], pair["visits_per_arm"], pair["optimizer_updates_per_arm"]),
                         (PAIR_STATUS, 70560, 17640))
        self.assertEqual(pair["qualification_pair_sha256"], arguments["qualification_pair_sha256"])
        self.assertEqual(pair["stepwise_center_arm_receipt_sha256"], digest(closed.stepwise_center_receipt))
        self.assertEqual(pair["one_pass_arm_receipt_sha256"], digest(closed.one_pass_receipt))

