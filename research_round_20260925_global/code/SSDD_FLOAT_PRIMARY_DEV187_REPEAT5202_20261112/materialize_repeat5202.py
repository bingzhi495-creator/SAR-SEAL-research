"""Materialize the fixed 5202 DEV187 float evaluator from the byte-pinned original sources."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

from repeat5202_source import FLOAT_OWNER, FLOAT_OWNER_SHA, FLOAT_WORKER, FLOAT_WORKER_SHA, exact_source, require


PROTOCOL_SHA = "bf0cace274aea7cc00c7bbfb104de9c1e83da2dae837e4f11dd63a88cd45a14c"
SUPPORT_SHA = "c3f80ce2f7cb3f444c5fa01775269e1889368ad078e66291b89c7c791bc08376"
SOURCE_MANIFEST_SHA = "ac34720d21833e81ee98eda9a8f4a1bc21a4d1cd71668cfd39803f71ae375239"
SHORT_ACCEPTANCE_SHA = "6bef2983c978f3614fb610089ce173713dfc3892641a1890139fe560517692f8"
ARM_STATUS = "SSDD_WINDOWS_FLOAT_COMMON_REPEAT5202_E24_ARM_INDEPENDENTLY_ACCEPTED_V1"
PAIR_STATUS = "SSDD_WINDOWS_FLOAT_COMMON_REPEAT5202_E24_PAIR_INDEPENDENTLY_ACCEPTED_V1"


def replace_once(code, old, new):
    require(code.count(old) == 1, "Expected one byte-pinned seed adaptation site: " + old)
    return code.replace(old, new, 1)


def change(code, sites):
    for old, new in sites:
        code = replace_once(code, old, new)
    return code


def worker_bytes(code):
    code = change(code, (
        ('ARMS = ("native", "center")', 'ARMS = ("native", "center")\nINITIALIZATION_SEED, TRAINING_SEED = 5201, 5202'),
        ("4db9c71edda60dfa89bf2e9065ae5fcecdf23dbb4199e0aceb1a3556855d7ffc", PROTOCOL_SHA),
        ("56fdd4be8d19b4e2e42ff736e89775f32b1e6ce2a9c91d2a90afad509640bc14", SOURCE_MANIFEST_SHA),
        ("78d2c6d1b99a1b05755324e061c1ad1bd4c5a460578e06e73720fa8c748ea8e0", SHORT_ACCEPTANCE_SHA),
        ("SSDD_WINDOWS_FLOAT_COMMON_E24_PAIR_INDEPENDENTLY_ACCEPTED_V1", PAIR_STATUS),
        ("SSDD_WINDOWS_FLOAT_COMMON_E24_ARM_INDEPENDENTLY_ACCEPTED_V1", ARM_STATUS),
        ("seed=5201, fit_images=741", "seed=TRAINING_SEED, training_seed=TRAINING_SEED, initialization_seed=INITIALIZATION_SEED,\n                         fit_images=741"),
        ("original_os_wait=True, original_os_exit=0, resource_stop=False, worker_closed=True,\n                            completed_epochs=24", "original_os_wait=True, original_os_exit=0, resource_stop=False, worker_closed=True,\n                            training_seed=TRAINING_SEED, initialization_seed=INITIALIZATION_SEED, completed_epochs=24"),
        ("mode=mode, training_seed=5201, training_pair_sha256=pair_ref", "mode=mode, training_seed=TRAINING_SEED, initialization_seed=INITIALIZATION_SEED,\n                training_pair_sha256=pair_ref"),
        ("evaluation_protocol=evaluation_protocol, mode=mode, training_seed=5201,", "evaluation_protocol=evaluation_protocol, mode=mode, training_seed=TRAINING_SEED,\n                                initialization_seed=INITIALIZATION_SEED,"),
        ("max_dets=[1,10,100], seed=5201,", "max_dets=[1,10,100], seed=TRAINING_SEED, initialization_seed=INITIALIZATION_SEED,"),
        ("first_seed_gate=gate(a, b, reference)", "replication_seed_gate=gate(a, b, reference)"),
        ("QUALIFIES_NEXT_FIXED_SEED", "AT_LEAST_ONE_POINT_AT_THIS_SEED"),
        ("AT_MOST_ONE_PRELOCKED_REPLICATION", "POSITIVE_BELOW_ONE_POINT_AT_THIS_SEED"),
        ("NO_PROMOTION_NONPOSITIVE", "NONPOSITIVE_AT_THIS_SEED"),
        ("return stage, stage.load_published(code_root, scoring=True)", """
    support_path = Path(__file__).with_name("repeat5202_source.py")
    support = ModuleType("_float_5202_admission_support")
    support.__file__ = str(support_path)
    raw_support = support_path.read_bytes()
    need(len(raw_support) <= 128 * 1024 and sha256(raw_support).hexdigest() == "c3f80ce2f7cb3f444c5fa01775269e1889368ad078e66291b89c7c791bc08376",
         "Exact archived seed-5202 admission support required")
    exec(compile(raw_support, str(support_path), "exec"), support.__dict__)
    return support.configure_admission(code_root, stage, stage.load_published(code_root, scoring=True))""".strip()),
    ))
    return code.encode("utf-8")


def owner_bytes(code, worker_sha):
    return change(code, (
        (FLOAT_WORKER_SHA, worker_sha),
        ("SSDD_FLOAT_COMMON_DEV187_WINDOWS_OWNER_CONFIG_V1", "SSDD_FLOAT_COMMON_REPEAT5202_DEV187_WINDOWS_OWNER_CONFIG_V1"),
        ("seed=5201, primary_GT=", "seed=worker.TRAINING_SEED, initialization_seed=worker.INITIALIZATION_SEED, primary_GT="),
        ('report["first_seed_gate"]', 'report["replication_seed_gate"]'),
        ("SSDD_FLOAT_COMMON_DEV187_WINDOWS_OWNER_SOURCE_CHECK_PASSED_V1",
         "SSDD_FLOAT_COMMON_REPEAT5202_DEV187_WINDOWS_OWNER_SOURCE_CHECK_PASSED_V1"),
        ("phase_data_opened=False,\n                  image_attestation", "phase_data_opened=False,\n                  training_seed=ctx.worker.TRAINING_SEED, initialization_seed=ctx.worker.INITIALIZATION_SEED,\n                  image_attestation"),
    )).encode("utf-8")


def write(output, name, data):
    with (output / name).open("xb") as stream:
        stream.write(data)
    return sha256(data).hexdigest()


def main():
    parser = ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.code_root, args.output
    bundle = Path(__file__).resolve(strict=True).parent
    support = exact_source(bundle / "repeat5202_source.py", SUPPORT_SHA)
    protocol = exact_source(bundle / "PROTOCOL.md", PROTOCOL_SHA)
    original_worker = exact_source(root / FLOAT_WORKER, FLOAT_WORKER_SHA).decode().replace("\r\n", "\n")
    original_owner = exact_source(root / FLOAT_OWNER, FLOAT_OWNER_SHA).decode().replace("\r\n", "\n")
    worker = worker_bytes(original_worker)
    owner = owner_bytes(original_owner, sha256(worker).hexdigest())
    compile(worker, "float_dev187.py", "exec")
    compile(owner, "float_dev187_owner.py", "exec")
    require(out.is_absolute() and out.parent.resolve(strict=True) == out.parent and not out.exists(),
            "Fresh canonical materialization directory required")
    out.mkdir()
    files = {name: write(out, name, data) for name, data in
             (("float_dev187.py", worker), ("float_dev187_owner.py", owner),
              ("repeat5202_source.py", support), ("PROTOCOL.md", protocol))}
    print(json.dumps(dict(status="SSDD_FLOAT_REPEAT5202_DEV187_SOURCE_MATERIALIZED_V1", files=files,
                          phase_data_opened=False, DEV=False, GPU=False, TEST=False), sort_keys=True))


if __name__ == "__main__":
    main()

