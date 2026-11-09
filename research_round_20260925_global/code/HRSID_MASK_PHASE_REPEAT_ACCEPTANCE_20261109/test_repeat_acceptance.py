from hashlib import sha256
import json
from pathlib import Path
import runpy
import sys
import unittest

from repeat_acceptance import RepeatAcceptance


PROTOCOL = "a49361b92977a605267025b1e7798420936cbe53"
TRAINER_SHA = "a" * 64
CODE = Path(__file__).resolve().parent.parent
PHASE_SOURCE = CODE / "HRSID_MASK_PHASE_E24_ACCEPTANCE_20261109" / "phase_acceptance.py"
NATIVE_SOURCE = CODE / "HRSID_COCO_E24_FILTERED_TERMINAL_ACCEPTANCE_20261109" / "acceptance.py"


def encoded(obj):
    return (json.dumps(obj, sort_keys=True, indent=2) + "\n").encode()


def line(obj):
    return (json.dumps(obj, sort_keys=True) + "\n").encode()


def synthetic_arm(seed, mode, *, checkpoint_seed=None, witness_changes=None, log_changes=None):
    phase_api = runpy.run_path(str(PHASE_SOURCE))
    native = phase_api["load_native_validator"](NATIVE_SOURCE)
    native.PROTOCOL = PROTOCOL
    previous = sys.modules.get("acceptance")
    sys.modules["acceptance"] = native
    try:
        fixture = runpy.run_path(str(NATIVE_SOURCE.with_name("test_acceptance.py")))["synthetic_run"]
        args, native_reader, calls = fixture()
    finally:
        if previous is None:
            del sys.modules["acceptance"]
        else:
            sys.modules["acceptance"] = previous
    closure, result_bytes, epoch_bytes, visit_bytes, native_stdout = args
    result = json.loads(result_bytes)
    if mode == "center":
        for digit, ref in zip("456", result["archive_refs"], strict=True):
            ref["archive_sha256"] = digit * 64
    result_bytes = encoded(result)
    phase = dict(
        protocol=PROTOCOL, mask_resize_mode=mode, stage="e24", seed=seed,
        initial_expanded_sha256="63ff93afdd977ba79b5b5a46ba8b4839ef6c40eb0722b29e4667d4932404fbc2",
        phase_trainer_sha256=TRAINER_SHA,
        phase_patch_sha256="811eccf2b42319272aea8b3f50a71bc029734d006bba43260f66117e9421948e",
        native_trainer_sha256="11d17cab878d62576b8d19e6231caab90c9990c03909cb01e1d04468add452a6",
    )
    sidecar = encoded(dict(
        phase, status="NATIVE_HRSID_MASK_RESIZE_E24_CLOSED_PENDING_INDEPENDENT_ACCEPTANCE",
        native_result_sha256=sha256(result_bytes).hexdigest(),
    ))
    witness = dict(
        status="HRSID_MASK_PHASE_REPEAT_SEED_WITNESS_V1", protocol=PROTOCOL, mask_resize_mode=mode,
        stage="e24", initialization_seed=5201, training_seed=seed, global_rank=0, train_config_seed=seed,
        torch_cpu_initial_seed=seed, torch_cuda_initial_seed=seed, lightning_global_seed=seed,
        lightning_seed_workers=True,
    )
    witness.update(witness_changes or {})
    witness_bytes = encoded(witness)
    event = dict(witness, event="HRSID_MASK_PHASE_REPEAT_SEED_OBSERVED",
                 seed_witness_sha256=sha256(witness_bytes).hexdigest())
    event.update(log_changes or {})
    prefix = b"".join(native_stdout.splitlines(keepends=True)[:-1])
    trailer = line(dict(event="HRSID_MASK_PHASE_E24_OUTPUT_COMPLETE", mask_resize_mode=mode, protocol=PROTOCOL))
    stdout = line(event) + prefix + line(result) + trailer
    closure = dict(closure, stdout_sha256=sha256(stdout).hexdigest())

    def reader(ref):
        saved, size, _ = native_reader(ref)
        saved["sar_seal_hrsid_mask_resize_phase_v1"] = phase.copy()
        saved["args"] = {"seed": checkpoint_seed if ref["epoch"] == 8 and checkpoint_seed is not None else seed}
        return saved, size, ref["archive_sha256"]

    return dict(
        closure=closure, result_bytes=result_bytes, epoch_bytes=epoch_bytes, visit_bytes=visit_bytes,
        stdout_bytes=stdout, sidecar_bytes=sidecar, seed_witness_bytes=witness_bytes,
        checkpoint_reader=reader, expected_fit_ids=range(native.FIT),
    ), calls


class RepeatAcceptanceTest(unittest.TestCase):
    def test_independently_accepted_arms_bind_seed_and_original_receipts(self):
        bridge = RepeatAcceptance(PROTOCOL, 5202, TRAINER_SHA)
        receipts, ledgers = [], []
        for mode in ("native", "center"):
            args, calls = synthetic_arm(5202, mode)
            raw, digest = bridge.validate_arm(mode, **args)
            self.assertEqual(calls, [1, 8, 24])
            self.assertEqual(digest, sha256(raw).hexdigest())
            arm = json.loads(raw)
            self.assertEqual(arm["phase"]["seed"], 5202)
            self.assertEqual(arm["repeat_seed_witness_sha256"], sha256(args["seed_witness_bytes"]).hexdigest())
            receipts.append(raw)
            ledgers.append(args["visit_bytes"])
        raw, digest, stats = bridge.create_pair(*receipts, *ledgers)
        self.assertEqual(digest, sha256(raw).hexdigest())
        pair = json.loads(raw)
        self.assertEqual(pair["seed"], 5202)
        self.assertEqual(pair["training_protocol"], PROTOCOL)
        self.assertEqual(pair["center_arm_receipt_sha256"], sha256(receipts[1]).hexdigest())
        self.assertEqual(stats["visits_per_arm"], 70560)


    def test_seed_5203_uses_fresh_seed_scope(self):
        previous = RepeatAcceptance(PROTOCOL, 5202, TRAINER_SHA)
        current = RepeatAcceptance(PROTOCOL, 5203, TRAINER_SHA)
        args, calls = synthetic_arm(5203, "center")
        raw, _ = current.validate_arm("center", **args)
        self.assertEqual(json.loads(raw)["phase"]["seed"], 5203)
        self.assertEqual(calls, [1, 8, 24])
        self.assertEqual(previous.expected_phase("center")["seed"], 5202)

    def test_wrong_native_e8_args_seed_refuses_e24_archive(self):
        bridge = RepeatAcceptance(PROTOCOL, 5202, TRAINER_SHA)
        args, calls = synthetic_arm(5202, "center", checkpoint_seed=5201)
        with self.assertRaisesRegex(ValueError, "checkpoint args.seed"):
            bridge.validate_arm("center", **args)
        self.assertEqual(calls, [1, 8])

    def test_wrong_seed_witness_or_logged_digest_refuses_all_archives(self):
        bridge = RepeatAcceptance(PROTOCOL, 5202, TRAINER_SHA)
        for changes, message in (
            (dict(witness_changes={"torch_cuda_initial_seed": 5201}), "Exact repeat seed witness"),
            (dict(log_changes={"seed_witness_sha256": "0" * 64}), "witness digest must agree"),
        ):
            with self.subTest(changes=changes):
                args, calls = synthetic_arm(5202, "native", **changes)
                with self.assertRaisesRegex(ValueError, message):
                    bridge.validate_arm("native", **args)
                self.assertEqual(calls, [])

    def test_live_worker_is_rejected_before_archive_reads(self):
        bridge = RepeatAcceptance(PROTOCOL, 5202, TRAINER_SHA)
        args, calls = synthetic_arm(5202, "native")
        args["closure"]["worker_closed"] = False
        with self.assertRaisesRegex(ValueError, "closed original"):
            bridge.validate_arm("native", **args)
        self.assertEqual(calls, [])

    def test_unapproved_protocol_or_seed_cannot_create_a_bridge(self):
        for protocol, seed in (("a" * 40, 5202), (PROTOCOL, 5201), (PROTOCOL, 5202.0), (PROTOCOL, 5204)):
            with self.subTest(protocol=protocol, seed=seed):
                with self.assertRaisesRegex(ValueError, "Exact P216 protocol and repeat seed"):
                    RepeatAcceptance(protocol, seed, TRAINER_SHA)

    def test_pair_rejects_cross_seed_receipt_and_changed_padding_order(self):
        bridge = RepeatAcceptance(PROTOCOL, 5202, TRAINER_SHA)
        receipts, ledgers = [], []
        for mode in ("native", "center"):
            args, _ = synthetic_arm(5202, mode)
            raw, _ = bridge.validate_arm(mode, **args)
            receipts.append(raw)
            ledgers.append(args["visit_bytes"])

        changed = json.loads(receipts[1])
        changed["phase"]["seed"] = 5203
        with self.assertRaisesRegex(ValueError, "repeat arm phase marker"):
            bridge.create_pair(receipts[0], encoded(changed), *ledgers)

        lines = ledgers[1].splitlines(keepends=True)
        last = json.loads(lines[-1])
        last["visits"][-2:] = reversed(last["visits"][-2:])
        lines[-1] = line(last)
        center_visits = b"".join(lines)
        changed = json.loads(receipts[1])
        changed["visit_ledger_sha256"] = sha256(center_visits).hexdigest()
        with self.assertRaisesRegex(ValueError, "Ordered FIT image IDs differ"):
            bridge.create_pair(receipts[0], encoded(changed), ledgers[0], center_visits)


if __name__ == "__main__":
    unittest.main()
