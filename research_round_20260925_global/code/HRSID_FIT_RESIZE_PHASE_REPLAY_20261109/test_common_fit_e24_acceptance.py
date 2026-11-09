from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import runpy
import sys
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from common_fit_e24_acceptance import ClosedE24Arm, accept_e24_pair
from common_fit_eval_admission import load_phase, read_pair
from common_fit_pair_receipt import SCOPE, compact, encode_receipt
from common_fit_train import (
    AUDIT_STATUS, COMMON_MARKER, FIT_INDEX_SHA, INITIAL_EXPANDED_SHA, INITIALIZATION_SEED, NATIVE,
    POLICY, QUALIFIED_STATUS, SELF, SOURCE_PINS, TRAINING_SEED,
)


from test_common_fit_e24_ledger import LazyLines, shared_row


CODE = Path(__file__).resolve().parent.parent
PHASE_DIR = CODE / "HRSID_MASK_PHASE_E24_ACCEPTANCE_20261109"
PROTOCOL, IMPLEMENTATION = "b" * 40, "c" * 40
SOURCE_SHAS = {**SOURCE_PINS, SELF: "8" * 64, "common_fit_binding.py": "9" * 64}
IDENTITY = dict(training_protocol=PROTOCOL, implementation_commit=IMPLEMENTATION, training_seed=TRAINING_SEED,
                initialization_seed=INITIALIZATION_SEED, initial_expanded_sha256=INITIAL_EXPANDED_SHA,
                fit_index_sha256=FIT_INDEX_SHA, source_manifest_sha256=sha256(compact(SOURCE_SHAS)).hexdigest())
SHORT = encode_receipt(dict(
    status=QUALIFIED_STATUS, **IDENTITY, ordered_common_fit_visits_match=True, visits_per_arm=32,
    optimizer_updates_per_arm=8, ema_updates_per_arm=8, native_arm_receipt_sha256="d" * 64,
    center_arm_receipt_sha256="e" * 64, shared_consumed_ledger_sha256="1" * 64,
    changed_mask_visits=12, **SCOPE,
))
LEDGER_SHA = "2" * 64


def synthetic_arm(mode, *, bad_archive_epoch=None):
    with patch.object(sys, "path", [str(PHASE_DIR), *sys.path]):
        fixtures = runpy.run_path(str(PHASE_DIR / "test_phase_acceptance.py"))
    original = fixtures["load_native_validator"](fixtures["NATIVE_SOURCE"])
    original.PROTOCOL = PROTOCOL
    phase = dict(protocol=PROTOCOL, mask_resize_mode=mode, stage="e24", seed=TRAINING_SEED,
                 initial_expanded_sha256=INITIAL_EXPANDED_SHA, phase_trainer_sha256=SOURCE_SHAS[SELF],
                 phase_patch_sha256=SOURCE_SHAS["phase_patch.py"], native_trainer_sha256=SOURCE_SHAS[NATIVE])
    fixture = fixtures["zero_phase_fixture"]
    with patch.dict(fixture.__globals__, expected_marker=lambda: phase,
                    phase_event=lambda *, mode: encode_receipt(dict(
                        event="HRSID_MASK_PHASE_E24_OUTPUT_COMPLETE", mask_resize_mode=mode, protocol=PROTOCOL))):
        legacy, calls = fixture(original, mode)
    closure, result, epochs, visits, stdout, sidecar, _, reader, _ = legacy
    visit_lines = [compact(json.loads(line)) + b"\n" for line in visits.splitlines()]
    epoch_lines = [compact(dict(json.loads(line), visit_row_sha256=sha256(row).hexdigest())) + b"\n"
                   for line, row in zip(epochs.splitlines(), visit_lines, strict=True)]
    visits, epochs = b"".join(visit_lines), b"".join(epoch_lines)
    result = encode_receipt(dict(json.loads(result), epoch_audit_sha256=sha256(epochs).hexdigest(),
                                 fit_visit_ledger_sha256=sha256(visits).hexdigest()))
    sidecar = encode_receipt(dict(json.loads(sidecar), native_result_sha256=sha256(result).hexdigest()))
    common = dict(**IDENTITY, qualification_pair_sha256=sha256(SHORT).hexdigest(), policy=POLICY, stage="e24", mode=mode)
    witness = dict(status="HRSID_MASK_PHASE_REPEAT_SEED_WITNESS_V1", protocol=PROTOCOL,
                   mask_resize_mode=mode, stage="e24", initialization_seed=INITIALIZATION_SEED,
                   training_seed=TRAINING_SEED, global_rank=0, train_config_seed=TRAINING_SEED,
                   torch_cpu_initial_seed=TRAINING_SEED, torch_cuda_initial_seed=TRAINING_SEED,
                   lightning_global_seed=TRAINING_SEED, lightning_seed_workers=True)
    witness_raw = encode_receipt(witness)
    witness_sha = sha256(witness_raw).hexdigest()
    # The original worker emits one JSON object per stdout line.
    original_result = json.loads(result)
    actual_events = [json.dumps(dict(json.loads(line), event="HRSID_COCO_NATIVE_EPOCH_OBSERVED"),
                                sort_keys=True).encode() + b"\n" for line in epochs.splitlines()]
    prefix = json.dumps(dict(witness, event="HRSID_MASK_PHASE_REPEAT_SEED_OBSERVED",
                             seed_witness_sha256=witness_sha), sort_keys=True).encode() + b"\n"
    tail = dict(event="HRSID_MASK_PHASE_E24_OUTPUT_COMPLETE", mask_resize_mode=mode, protocol=PROTOCOL)
    stdout = (prefix + b"".join(actual_events) + json.dumps(original_result, sort_keys=True).encode() + b"\n"
              + json.dumps(tail, sort_keys=True).encode() + b"\n")
    closure = dict(closure, stdout_sha256=sha256(stdout).hexdigest(), terminal_sha256=("3" if mode == "native" else "4") * 64)
    common_audit = encode_receipt(dict(
        status=AUDIT_STATUS, common_support=common, phase=phase, native_result_sha256=sha256(result).hexdigest(),
        repeat_seed_witness_sha256=witness_sha, consumed_ledger_sha256=LEDGER_SHA, consumed_ledger_bytes=42,
        ledger={"changed_mask_visits": 50}, optimizer_updates=17640, ema_updates=17640, completed_epochs=24, **SCOPE,
    ))

    def checkpoint_reader(ref):
        saved, size, digest = reader(ref)
        saved["args"] = {"seed": TRAINING_SEED}
        saved[COMMON_MARKER] = dict(common)
        if ref["epoch"] == bad_archive_epoch:
            saved[COMMON_MARKER]["mode"] = "native" if mode == "center" else "center"
        return saved, size, digest

    return ClosedE24Arm(closure, common_audit, result, epochs, visits, stdout, sidecar, witness_raw, checkpoint_reader), calls


def shared_lines(visits):
    for line in visits.splitlines():
        epoch = json.loads(line)
        for batch, (image_id, count) in enumerate(epoch["visits"], 1):
            row = shared_row(epoch["epoch"], batch, 2940)
            self_changed = int(row["ordinal"] <= 50)
            assert count == row["common_nonempty"]
            row.update(image_id=image_id, common_changed_masks=self_changed)
            if not self_changed:
                row["center_common_masks_sha256"] = row["native_common_masks_sha256"]
            yield compact(row) + b"\n"


def synthetic_full_streams(native, center):
    h = sha256()
    byte_count = 0
    for line in shared_lines(native.visits):
        h.update(line)
        byte_count += len(line)
    stats = dict(fetched_visits=70560, consumed_visits=70560, prefetched_unconsumed=0,
                 unconsumed_prefetch_sha256=None, common_targets_total=70560, common_empty_visits=0,
                 changed_mask_visits=50)
    arms = []
    for arm in (native, center):
        safe = dict(json.loads(arm.common_audit), consumed_ledger_sha256=h.hexdigest(),
                    consumed_ledger_bytes=byte_count, ledger=stats)
        arms.append(arm._replace(common_audit=encode_receipt(safe)))
    return (*arms, LazyLines(shared_lines(native.visits)), LazyLines(shared_lines(center.visits)))


def checked_streams(**streams):
    return dict(ordered_common_fit_visits_match=True, completed_epochs=24, visits_per_arm=70560,
                optimizer_updates_per_arm=17640, ema_updates_per_arm=17640, shared_consumed_ledger_sha256=LEDGER_SHA,
                shared_consumed_ledger_bytes=42, ledger={"changed_mask_visits": 50},
                native_visit_ledger_sha256=sha256(streams["native_visits"].read()).hexdigest(),
                center_visit_ledger_sha256=sha256(streams["center_visits"].read()).hexdigest())


class CommonFitE24AcceptanceTest(TestCase):
    def run_pair(self, native, center, *, qualification=SHORT, qualification_sha=sha256(SHORT).hexdigest()):
        with patch("common_fit_e24_acceptance.verify_e24_ledgers", side_effect=checked_streams):
            return accept_e24_pair(
                training_protocol=PROTOCOL, implementation_commit=IMPLEMENTATION, source_sha256s=SOURCE_SHAS,
                qualification_pair_bytes=qualification, qualification_pair_sha256=qualification_sha,
                native=native, center=center, native_common=BytesIO(), center_common=BytesIO(),
                expected_fit_ids=range(2937),
            )

    def test_acceptance_reuses_original_archive_checks_at_all_three_endpoints(self):
        native, calls_a = synthetic_arm("native")
        center, calls_b = synthetic_arm("center")
        native, center, native_common, center_common = synthetic_full_streams(native, center)
        outcome = accept_e24_pair(
            training_protocol=PROTOCOL, implementation_commit=IMPLEMENTATION, source_sha256s=SOURCE_SHAS,
            qualification_pair_bytes=SHORT, qualification_pair_sha256=sha256(SHORT).hexdigest(),
            native=native, center=center, native_common=native_common, center_common=center_common,
            expected_fit_ids=range(2937),
        )
        a, b, pair = map(json.loads, (outcome.native_receipt, outcome.center_receipt, outcome.pair_receipt))
        self.assertEqual((calls_a, calls_b), ([1, 8, 24], [1, 8, 24]))
        self.assertEqual(pair["status"], "HRSID_MASK_COMMON_FIT_E24_PAIR_READY_V1")
        self.assertEqual(pair["native_arm_receipt_sha256"], sha256(outcome.native_receipt).hexdigest())
        self.assertEqual(pair["center_checkpoint_sha256"], b["checkpoint_sha256"])
        self.assertEqual(a["common_support"]["qualification_pair_sha256"], sha256(SHORT).hexdigest())
        self.assertEqual(pair["changed_mask_visits"], 50)
        source = CODE / "HRSID_MASK_PHASE_E24_DEV705_20261109/hrsid_mask_phase_dev705.py"
        with TemporaryDirectory() as directory:
            native_path, center_path, pair_path = (Path(directory).resolve() / name
                                                  for name in ("native.json", "center.json", "pair.json"))
            native_path.write_bytes(outcome.native_receipt)
            center_path.write_bytes(outcome.center_receipt)
            pair_path.write_bytes(outcome.pair_receipt)
            selected = read_pair(load_phase(source), pair_path, sha256(outcome.pair_receipt).hexdigest(),
                                 native_path=native_path, center_path=center_path, mode="center",
                                 training_protocol=PROTOCOL)
        self.assertEqual(selected, b)

    def test_wrong_middle_checkpoint_marker_stops_before_e24_read(self):
        native, calls_a = synthetic_arm("native", bad_archive_epoch=8)
        center, calls_b = synthetic_arm("center")
        with self.assertRaisesRegex(ValueError, "own common-support marker"):
            self.run_pair(native, center)
        self.assertEqual((calls_a, calls_b), ([1, 8], []))

    def test_short_pair_must_belong_to_same_implementation(self):
        native, calls_a = synthetic_arm("native")
        center, calls_b = synthetic_arm("center")
        wrong = encode_receipt(dict(json.loads(SHORT), implementation_commit="f" * 40))
        with self.assertRaisesRegex(ValueError, "same-implementation short pair"):
            self.run_pair(native, center, qualification=wrong, qualification_sha=sha256(wrong).hexdigest())
        self.assertEqual((calls_a, calls_b), ([], []))

    def test_unclosed_owner_refuses_ledger_streams_and_archive_access(self):
        native, calls_a = synthetic_arm("native")
        center, calls_b = synthetic_arm("center")
        native = native._replace(closure=dict(native.closure, owner_closed=False))
        with patch("common_fit_e24_acceptance.verify_e24_ledgers", side_effect=checked_streams) as verify:
            with self.assertRaisesRegex(ValueError, "closed original owner and worker"):
                accept_e24_pair(
                    training_protocol=PROTOCOL, implementation_commit=IMPLEMENTATION, source_sha256s=SOURCE_SHAS,
                    qualification_pair_bytes=SHORT, qualification_pair_sha256=sha256(SHORT).hexdigest(),
                    native=native, center=center, native_common=BytesIO(), center_common=BytesIO(),
                    expected_fit_ids=range(2937),
                )
        verify.assert_not_called()
        self.assertEqual((calls_a, calls_b), ([], []))

