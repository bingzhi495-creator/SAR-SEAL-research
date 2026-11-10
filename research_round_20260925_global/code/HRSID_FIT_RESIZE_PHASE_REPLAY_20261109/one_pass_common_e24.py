"""Run one independently admitted HRSID fixed-support B/C E24 FIT arm."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from types import ModuleType


SELF = "one_pass_common_e24.py"
SHORT = "one_pass_common_qualification.py"
SHORT_SHA = "4b2bb3bbe00aaf28c748f66a540182993589039304ea1d966d1dde7e7c1980d8"
LOCK_STATUS = "HRSID_ONE_PASS_COMMON_FIT_E24_LOCK_V1"
PAIR_STATUS = "HRSID_ONE_PASS_COMMON_FIT_SHORT_PAIR_ACCEPTED_V1"
MODES = ("stepwise_center", "one_pass")
TRAINING_SEED = 5202
OTHER_PROTOCOLS = ("04308c52bd2b4730f1474335c6316f6b8a11762e",
                   "3ee0dc2f5f6508c238b760b9b234edc016e89f54",
                   "91728d38a359e4a207e3f741490f09060c46663f")
ARM_DIGEST_SUFFIXES = ("arm_receipt_sha256", "terminal_sha256", "selected_mask_stream_sha256")
BRANCH_COUNTS = ("changed_mask_visits", "consumed_direct_visits", "consumed_crop_visits", "changed_crop_visits")


def configured_engine(engine_source):
    qualifier_path = engine_source.with_name(SHORT)
    if not qualifier_path.is_absolute() or qualifier_path.resolve(strict=True) != qualifier_path:
        raise RuntimeError("Exact canonical short qualification source required beside the engine")
    with qualifier_path.open("rb") as stream:
        raw = stream.read(64 * 1024 + 1)
    if not 0 < len(raw) <= 64 * 1024 or sha256(raw).hexdigest() != SHORT_SHA:
        raise RuntimeError("Exact original B/C qualification source required")
    qualifier = ModuleType("_one_pass_common_e24_original_qualifier")
    qualifier.__file__ = str(qualifier_path)
    exec(compile(raw, str(qualifier_path), "exec"), qualifier.__dict__)
    engine = qualifier.configured_engine(engine_source)
    short_pins = {**engine.SOURCE_PINS, SHORT: SHORT_SHA}
    engine.SELF = SELF
    engine.LOCK_STATUS = LOCK_STATUS
    engine.SOURCE_PINS = short_pins.copy()
    engine.SOURCES = set(short_pins) | {SELF}
    return engine, qualifier, short_pins


def verify_short_pair(engine, qualifier, args, lock, short_pins):
    reference = lock["qualification_pair_ref"]
    require = engine.require
    blocked = (*engine.ORIGINAL_PROTOCOLS, *OTHER_PROTOCOLS)
    identities = (args.qualification_protocol, args.qualification_implementation_commit,
                  lock["training_protocol"], lock["implementation_commit"])
    require(all(engine.is_hex(value, 40) and value not in blocked for value in identities)
            and identities[0] != identities[2] and identities[1] != identities[3],
            "Short qualification and prospective B/C E24 require distinct protocol and implementation identities")
    require(engine.is_hex(args.qualification_pair_sha256, 64)
            and reference["sha256"] == args.qualification_pair_sha256,
            "Lock must bind the independently preaccepted B/C short pair SHA-256")
    raw = engine.read_ref(reference)
    receipt = engine.object_from(raw)
    short_manifest = engine.manifest_digest(dict(sources={name: dict(sha256=digest)
                                                         for name, digest in short_pins.items()}))
    fixed = dict(
        status=PAIR_STATUS, stage="qualification", policy=qualifier.POLICY, training_protocol=identities[0],
        implementation_commit=identities[1], training_seed=TRAINING_SEED,
        initialization_seed=engine.INITIALIZATION_SEED, initial_expanded_sha256=engine.INITIAL_EXPANDED_SHA,
        fit_index_sha256=engine.FIT_INDEX_SHA, source_manifest_sha256=short_manifest,
        ordered_common_fit_visits_match=True, visits_per_arm=32, optimizer_updates_per_arm=8,
        ema_updates_per_arm=8, verified_selected_mask_visits_per_arm=32,
        FIT=True, DEV=False, TEST=False, AP=False,
    )
    digest_fields = {f"{mode}_{suffix}" for mode in MODES for suffix in ARM_DIGEST_SUFFIXES}
    digest_fields.add("shared_consumed_ledger_sha256")
    require(set(receipt) == fixed.keys() | digest_fields | set(BRANCH_COUNTS)
            and engine.same_fields(receipt, fixed)
            and all(engine.is_hex(receipt[name], 64) for name in digest_fields)
            and all(receipt[f"{MODES[0]}_{suffix}"] != receipt[f"{MODES[1]}_{suffix}"]
                    for suffix in ARM_DIGEST_SUFFIXES)
            and all(type(receipt[name]) is int and 0 < receipt[name] <= 32 for name in BRANCH_COUNTS),
            "Exact independently verified short B/C source, arm, selected-mask and scope evidence required")
    require(receipt["consumed_direct_visits"] + receipt["consumed_crop_visits"] == 32
            and receipt["changed_mask_visits"] == receipt["changed_crop_visits"] <= receipt["consumed_crop_visits"],
            "Short pair must bind actual direct and changed crop supervision")
    require(raw == (json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8"),
            "Canonical independent short pair receipt bytes required")
    return reference["sha256"]


def admit_training(args):
    """Verify the pure closed-source and short-pair gate before runtime imports."""
    engine, qualifier, short_pins = configured_engine(args.engine_source)
    require = engine.require
    require(args.stage == "e24" and args.mode in MODES and args.training_seed == TRAINING_SEED,
            "Only prospective fixed-support B/C E24 arms are admitted")
    lock = engine.read_training_lock(args)
    refs = lock["sources"]
    require(Path(refs[SELF]["path"]) == Path(__file__).resolve(strict=True)
            and Path(refs[SHORT]["path"]) == Path(qualifier.__file__)
            and Path(refs[qualifier.ENGINE]["path"]) == args.engine_source
            and Path(refs[engine.NATIVE]["path"]) == args.native_source,
            "Exact running E24 wrapper, short qualifier, original engine and native source required")
    for name in (SELF, SHORT, qualifier.ENGINE):
        engine.read_ref(refs[name])
    qualification_sha = verify_short_pair(engine, qualifier, args, lock, short_pins)
    return engine, qualifier, lock, qualification_sha


def run_locked(args):
    engine, qualifier, lock, qualification_sha = admit_training(args)
    require = engine.require
    require(Path(sys.executable).resolve() == engine.PYTHON.resolve(strict=True)
            and sys.version_info[:3] == (3, 10, 13) and sys.flags.isolated == 1
            and sys.dont_write_bytecode and sys.flags.utf8_mode == 1
            and os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
            and os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1",
            "Exact original isolated offline Windows GPU runtime required")
    engine.isolated_outputs(args.output, args.audit_output)
    return _run_training(args, engine, qualifier, lock, qualification_sha)


def _run_training(args, engine, qualifier, lock, qualification_sha):
    # Mirror the original qualifier's dependency admission and runtime pins.
    # Its public worker remains qualification-only.
    refs = lock["sources"]
    native = engine.load_source(refs[engine.NATIVE], "_one_pass_common_e24_original_native")
    sys.addaudithook(native.offline)
    base = native.pinned_module(native.E1, native.E1_SHA, "_one_pass_common_e24_original_e1")
    base.admit_dependency_overlay()
    engine.load_source(refs["trace_resize.py"], "trace_resize")
    engine.load_source(refs["phase_patch.py"], "phase_patch")
    phase = engine.load_source(refs["phase_train.py"], "_one_pass_common_e24_original_phase")
    repeat = engine.load_source(refs["phase_repeat_train.py"], "_one_pass_common_e24_original_repeat")
    modules = {}
    for filename in qualifier.RUNTIME_FILES:
        name = Path(filename).stem
        modules[name] = engine.load_source(refs[filename], name)
    binding = modules["one_pass_common_binding"]
    engine.require(modules["one_pass_common_pair_transform"].MODES == MODES, "Exact selected B/C modes")

    from importlib.metadata import version
    from pytorch_lightning import seed_everything
    from rfdetr.datasets import _torchvision, coco
    from rfdetr.training import module_data, trainer as trainer_api
    import torch

    engine.require(version("torchvision") == "0.25.0+cu130", "Exact paired FIT torchvision runtime required")
    runtime_sources = (
        (coco, "be64b20a178d9cde41db69e01e17e1296e36eee791e837294c3f930ae56f03b9"),
        (_torchvision, "03c12e881f733d4f6b469f784187e461753b8775b7e35e57c1d8a6c35c46ef56"),
    )
    for module, expected in runtime_sources:
        ref, _ = engine.stable_closed(Path(module.__file__).resolve(strict=True), 256 * 1024, capture=False)
        engine.require(ref["sha256"] == expected, "Exact qualified FIT converter and augmentation source required")
    return engine.run_engine(
        args, lock, qualification_sha, phase=phase, repeat=repeat, native=native, base=base, binding=binding,
        module_data=module_data, trainer_api=trainer_api, coco=coco, native_transforms=_torchvision,
        seed_everything=seed_everything, torch=torch,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("native-source", "engine-source", "output", "audit-output", "training-lock"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("protocol", "expected-initial-sha256", "training-lock-sha256", "qualification-protocol",
                 "qualification-implementation-commit", "qualification-pair-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--stage", choices=("e24",), required=True)
    parser.add_argument("--training-seed", choices=(TRAINING_SEED,), type=int, required=True)
    return run_locked(parser.parse_args(argv))


if __name__ == "__main__":
    main()
