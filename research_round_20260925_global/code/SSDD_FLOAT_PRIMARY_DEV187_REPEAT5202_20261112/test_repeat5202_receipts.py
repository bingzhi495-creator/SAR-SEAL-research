"""Verify synthetic 5202 receipt admission and unchanged fixed-reference arithmetic."""

from argparse import ArgumentParser
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType

from repeat5202_source import probe_admission, require


def load_worker(path):
    worker = ModuleType("_generated_repeat_worker")
    worker.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), "exec"), worker.__dict__)
    return worker


def fixtures(worker, api):
    arm_values = {}
    for i, mode in enumerate(worker.ARMS):
        digests = {field: sha256(f"{i}/{field}".encode()).hexdigest() for field in worker.ARM_DIGESTS}
        digests["visit_ledger_sha256"] = "3" * 64
        phase = api.phase_marker(mode, worker.TRAINING_PROTOCOL, "7" * 64, worker.FIT_SOURCE_MANIFEST_SHA)
        arm_values[mode] = dict(
            status=worker.ARM_STATUS, phase=phase, qualification_pair_sha256=worker.FLOAT_SHORT_SHA,
            training_seed=5202, initialization_seed=5201, operation_commit="5"*40,
            original_os_wait=True, original_os_exit=0, resource_stop=False, worker_closed=True,
            completed_epochs=24, visits=17856, optimizer_updates=4464, ema_updates=4464, checkpoint_bytes=520,
            common_ledger_sha256="1"*64, identical_fields_sha256="2"*64, common_targets_total=32000,
            common_empty_visits=4, changed_mask_visits=12000, filtered_empty_rows_total=0,
            **worker.FIT_ONLY, **digests)
    pair = dict(
        status=worker.PAIR_STATUS, training_protocol=worker.TRAINING_PROTOCOL,
        source_manifest_sha256=worker.FIT_SOURCE_MANIFEST_SHA, qualification_pair_sha256=worker.FLOAT_SHORT_SHA,
        seed=5202, training_seed=5202, initialization_seed=5201, fit_images=741,
        completed_epochs_per_arm=24, visits_per_arm=17856, optimizer_updates_per_arm=4464, ema_updates_per_arm=4464,
        common_ledger_sha256="1"*64, identical_fields_sha256="2"*64, common_targets_total=32000,
        common_empty_visits=4, changed_mask_visits=12000,
        selected_mask_stream_sha256={mode:arm_values[mode]["selected_mask_stream_sha256"] for mode in worker.ARMS},
        identical_nonmask_inputs_and_rng=True, identical_common_support=True, **worker.FIT_ONLY)
    return pair, arm_values


def probe_receipts(worker, api):
    pair, arms = fixtures(worker, api)

    def run(pair, arms):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve(strict=True)
            def ref(name, value):
                path = root / (name + ".json")
                raw = json.dumps(value, allow_nan=False).encode()
                path.write_bytes(raw)
                return worker.file_ref(path, sha256(raw).hexdigest())
            refs = {mode:ref(mode, arm) for mode, arm in arms.items()}
            pair = dict(pair, arms={mode:{key:value[key] for key in ("sha256","bytes")} for mode,value in refs.items()})
            got_pair, got_arms = worker.read_training_pair(
                api, ref("pair", pair), native_arm_ref=refs["native"], center_arm_ref=refs["center"],
                training_protocol=worker.TRAINING_PROTOCOL, evaluation_protocol="8"*40)
            require((got_pair, got_arms) == (pair, arms), "Exact synthetic accepted receipt roundtrip")

    run(pair, arms)
    checks = [
        ("pair downstream seed", "pair", "seed", 5201),
        ("pair qualifier", "pair", "qualification_pair_sha256", "0"*64),
        ("pair source manifest", "pair", "source_manifest_sha256", "0"*64),
        ("pair initial seed", "pair", "initialization_seed", 5202),
        ("pair status", "pair", "status", "SSDD_WINDOWS_FLOAT_COMMON_E24_PAIR_INDEPENDENTLY_ACCEPTED_V1"),
        ("arm downstream seed", "native", "training_seed", 5201),
        ("arm natural wait", "center", "original_os_wait", False),
        ("arm EMA cursor", "native", "ema_updates", 4463),
        ("arm marker initialization seed", "native", "phase", dict(arms["native"]["phase"], initialization_seed=5202)),
    ]
    for label, role, field, value in checks:
        pair_bad, arms_bad = deepcopy(pair), deepcopy(arms)
        target = pair_bad if role == "pair" else arms_bad[role]
        target[field] = value
        try:
            run(pair_bad, arms_bad)
        except ValueError:
            continue
        raise ValueError("Invalid synthetic variant was admitted: " + label)
    return len(checks)


def main():
    parser = ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    args = parser.parse_args()
    worker = load_worker(args.worker)
    stage, modules = worker.load_stock(args.code_root)
    api = modules["ssdd_phase_dev_admission"]
    rejected = probe_receipts(worker, api)
    archive_rejected = probe_admission(api)
    reference = {"AP": worker.HISTORICAL_AP}
    first = worker.gate({"AP": .6746015427021742}, {"AP": .7069359097559995}, reference)
    require(worker.HISTORICAL_AP == .7047324561474607
            and first["status"] == "POSITIVE_BELOW_ONE_POINT_AT_THIS_SEED"
            and abs(first["controlling_margin_AP_points"] - .22034536085387968) < 1e-12,
            "Identical historical strong reference and gate arithmetic required")
    print(json.dumps(dict(status="SSDD_FLOAT_REPEAT5202_SYNTHETIC_RECEIPT_CHECK_PASSED_V1",
                          admitted_pairs=1, rejected_receipts=rejected, rejected_archive_metadata=archive_rejected,
                          FIT=False, DEV=False, TEST=False, GPU=False), sort_keys=True))


if __name__ == "__main__":
    main()

