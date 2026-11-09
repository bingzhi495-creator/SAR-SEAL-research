"""Independently admit closed HRSID mask-phase E24 repeats for seeds 5202 and 5203."""

from hashlib import sha256
from pathlib import Path
from types import ModuleType


REPEAT_PROTOCOL = "a49361b92977a605267025b1e7798420936cbe53"
REPEAT_SEEDS = (5202, 5203)
INITIALIZATION_SEED = 5201
PHASE_ACCEPTANCE_SHA = "fabf39e8a5fb2f7c3ed8613375bcbbb960a818dfe1c8ff1587c44c7e30b50ce6"
PHASE_PATCH_SHA = "811eccf2b42319272aea8b3f50a71bc029734d006bba43260f66117e9421948e"
NATIVE_TRAINER_SHA = "11d17cab878d62576b8d19e6231caab90c9990c03909cb01e1d04468add452a6"
WITNESS_STATUS = "HRSID_MASK_PHASE_REPEAT_SEED_WITNESS_V1"
WITNESS_EVENT = "HRSID_MASK_PHASE_REPEAT_SEED_OBSERVED"
WITNESS_RECEIPT_FIELD = "repeat_seed_witness_sha256"
CODE = Path(__file__).resolve().parent.parent
PHASE_SOURCE = CODE / "HRSID_MASK_PHASE_E24_ACCEPTANCE_20261109" / "phase_acceptance.py"
NATIVE_SOURCE = CODE / "HRSID_COCO_E24_FILTERED_TERMINAL_ACCEPTANCE_20261109" / "acceptance.py"


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def load_phase():
    """Load the verified historical companion into a fresh, mutable namespace."""
    path = PHASE_SOURCE
    require(path.resolve(strict=True) == path, "Canonical immutable phase acceptance path")
    raw = path.read_bytes()
    require(0 < len(raw) <= 64 * 1024 and sha256(raw).hexdigest() == PHASE_ACCEPTANCE_SHA,
            "Exact immutable phase acceptance dependency")
    phase = ModuleType("_repeat_scoped_phase_acceptance")
    phase.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), phase.__dict__)
    return phase


class RepeatAcceptance:
    """Scope the original arm and pair validators to one prelocked repeat."""

    def __init__(self, protocol, seed, repeat_trainer_sha256):
        require(type(protocol) is str and protocol == REPEAT_PROTOCOL
                and type(seed) is int and seed in REPEAT_SEEDS, "Exact P216 protocol and repeat seed required")
        phase = load_phase()
        require(phase.is_sha256(repeat_trainer_sha256), "Expected repeat trainer SHA-256 required")
        self.protocol = protocol
        self.seed = seed
        self.repeat_trainer_sha256 = repeat_trainer_sha256
        self._phase = phase
        phase.PROTOCOL = protocol
        phase.check_phase = self._check_phase
        self._native = phase.load_native_validator(NATIVE_SOURCE)

    def expected_phase(self, mode):
        require(mode in ("native", "center"), "Exact repeat arm mode")
        return dict(
            protocol=self.protocol, mask_resize_mode=mode, stage="e24", seed=self.seed,
            initial_expanded_sha256=self._phase.INITIAL_SHA, phase_trainer_sha256=self.repeat_trainer_sha256,
            phase_patch_sha256=PHASE_PATCH_SHA, native_trainer_sha256=NATIVE_TRAINER_SHA,
        )

    def _check_phase(self, value, mode):
        require(self._phase.same_fields(value, self.expected_phase(mode)), "Exact repeat arm phase marker required")

    def _check_witness(self, mode, raw, stdout):
        require(type(raw) is bytes and 0 < len(raw) <= 8192, "Bounded repeat seed witness required")
        seed = self.seed
        witness = self._phase.object_from(raw, "repeat seed witness")
        expected = dict(
            status=WITNESS_STATUS, protocol=self.protocol, mask_resize_mode=mode, stage="e24",
            initialization_seed=INITIALIZATION_SEED, training_seed=seed, global_rank=0, train_config_seed=seed,
            torch_cpu_initial_seed=seed, torch_cuda_initial_seed=seed, lightning_global_seed=seed,
            lightning_seed_workers=True,
        )
        require(self._phase.same_fields(witness, expected), "Exact repeat seed witness required")
        digest = sha256(raw).hexdigest()
        events = [self._phase.object_from(line, "repeat training stdout event") for line in stdout.splitlines()
                  if line.startswith(b"{")]
        witness_positions = [index for index, event in enumerate(events) if event.get("event") == WITNESS_EVENT]
        first_training_evidence = next(
            (index for index, event in enumerate(events) if event.get("event") in
             ("HRSID_COCO_NATIVE_PROGRESS", "HRSID_NATIVE_EMPTY_MASK_FILTER_APPLIED", "HRSID_COCO_NATIVE_EPOCH_OBSERVED")),
            len(events),
        )
        require(len(witness_positions) == 1 and witness_positions[0] < first_training_evidence,
                "Exactly one original repeat seed event before logged training progress required")
        require(
            self._phase.same_fields(events[witness_positions[0]],
                                    dict(expected, event=WITNESS_EVENT, seed_witness_sha256=digest)),
            "Original repeat seed event and witness digest must agree",
        )
        return digest

    def read_closed_checkpoint(self, path, ref):
        return self._native.read_closed_checkpoint(path, ref)

    def validate_arm(self, mode, *, closure, result_bytes, epoch_bytes, visit_bytes, stdout_bytes,
                     sidecar_bytes, seed_witness_bytes, checkpoint_reader, expected_fit_ids):
        """Return the accepted arm receipt and digest after all three archive checks."""
        expected = self.expected_phase(mode)
        self._native.validate_closure(closure, stdout_bytes)
        witness_sha = self._check_witness(mode, seed_witness_bytes, stdout_bytes)

        def checked_reader(ref):
            saved, size, digest = checkpoint_reader(ref)
            args = saved.get("args") if type(saved) is dict else None
            require(type(args) is dict and type(args.get("seed")) is int and args["seed"] == self.seed,
                    "Exact native checkpoint args.seed on every fixed repeat archive required")
            return saved, size, digest

        raw, _ = self._phase.validate_phase_run(
            self._native, closure, result_bytes, epoch_bytes, visit_bytes, stdout_bytes, sidecar_bytes,
            expected, checked_reader, expected_fit_ids,
        )
        receipt = self._phase.object_from(raw, "accepted phase receipt")
        receipt[WITNESS_RECEIPT_FIELD] = witness_sha
        raw = self._phase.encoded(receipt)
        return raw, sha256(raw).hexdigest()

    def create_pair(self, native_raw, center_raw, native_visits, center_visits):
        """Compare the ordered FIT ledgers and bind the original repeat arm receipt bytes."""
        phase = self._phase
        witnesses, base_receipts = [], []
        for raw in (native_raw, center_raw):
            receipt = phase.object_from(raw, "accepted repeat arm receipt")
            witness = receipt.pop(WITNESS_RECEIPT_FIELD, None)
            require(phase.is_sha256(witness), "Accepted repeat arm must bind its original seed witness")
            witnesses.append(witness)
            base_receipts.append(phase.encoded(receipt))
        require(witnesses[0] != witnesses[1], "Distinct original seed witnesses for native and center arms")
        raw, _, stats = phase.create_pair_receipt(*base_receipts, native_visits, center_visits)
        receipt = phase.object_from(raw, "accepted phase pair")
        receipt.update(
            seed=self.seed, native_arm_receipt_sha256=sha256(native_raw).hexdigest(),
            center_arm_receipt_sha256=sha256(center_raw).hexdigest(),
        )
        raw = phase.encoded(receipt)
        return raw, sha256(raw).hexdigest(), stats
