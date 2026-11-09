"""Train one locked HRSID arm with fixed common nonempty FIT supervision."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
import sys
from types import ModuleType
from unittest.mock import patch


TRAINING_SEED = 5202
INITIALIZATION_SEED = 5201
INITIAL_EXPANDED_SHA = "63ff93afdd977ba79b5b5a46ba8b4839ef6c40eb0722b29e4667d4932404fbc2"
FIT_INDEX_SHA = "e95cf26cc60a2c6fb651e907541380a256f2df7b1e153244a6c0f803b0b19f8f"
ORIGINAL_PROTOCOLS = ("d66baf3c57fafdba4242ad1b55e08cf76bfd77f7", "a49361b92977a605267025b1e7798420936cbe53")
LOCK_STATUS = "HRSID_MASK_COMMON_FIT_TRAIN_LOCK_V1"
QUALIFIED_STATUS = "HRSID_MASK_COMMON_FIT_SHORT_PAIR_ACCEPTED_V1"
AUDIT_STATUS = "HRSID_MASK_COMMON_FIT_ARM_CLOSED_PENDING_INDEPENDENT_ACCEPTANCE_V1"
COMMON_MARKER = "sar_seal_hrsid_mask_common_support_v1"
POLICY = "ordered_common_nonempty_prepared_rows_v1"
AUDIT_LEDGER = "COMMON_FIT_VISITS_PRIVATE.jsonl"
AUDIT_RESULT = "COMMON_FIT_TRAIN_SAFE.json"
SELF = "common_fit_train.py"
NATIVE = "train_hrsid_coco_native_e24_filtered.py"
SOURCE_PINS = {
    NATIVE: "11d17cab878d62576b8d19e6231caab90c9990c03909cb01e1d04468add452a6",
    "trace_resize.py": "902a0e3640a319ba729172847f49ce582f3c0e3954593862ed94971727dc309f",
    "phase_patch.py": "811eccf2b42319272aea8b3f50a71bc029734d006bba43260f66117e9421948e",
    "phase_train.py": "8550b43791a6cb6f53736dc7e64b9c5c6204723cf4c07c3b112491ba1bd74043",
    "phase_repeat_train.py": "42a174573a04c529dfbae943bcc88ef4ed5d793d7eda6766511396034fda321f",
    "common_mask_supervision.py": "59ea82e24353eeb4a14115058d5115333a4e7ea517a0abef581bee137a9a3a57",
    "paired_fit_transform.py": "64f3f63c6ac89b66578f0ada42fa3eb8b4db3b2f1ae2ea9113debe6ab9b270e0",
}
SOURCES = set(SOURCE_PINS) | {SELF, "common_fit_binding.py"}
PRIVATE = Path("D:/SAR_SEAL_private_execution_20261021_01")
RELEASE = PRIVATE / "github_release_20261011"
FIT_LAYOUT = PRIVATE / "HRSID_FIT_NATIVE_COCO_LAYOUT_ACTUAL_PRIVATE_20261109_01"
PYTHON = PRIVATE / "RFDETR_WINDOWS_112_RUNTIME_PRIVATE_20261108_01/PYTHON_HOST_31013/python.exe"
LIMIT = 64 * 1024


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def is_hex(value, count):
    return type(value) is str and re.fullmatch(rf"[0-9a-f]{{{count}}}", value) is not None


def same_fields(value, expected):
    return type(value) is dict and all(key in value and type(value[key]) is type(item) and value[key] == item
                                       for key, item in expected.items())


def object_from(raw):
    def unique(pairs):
        value = dict(pairs)
        require(len(value) == len(pairs), "Duplicate training lock or receipt field")
        return value

    result = json.loads(raw, object_pairs_hook=unique)
    require(type(result) is dict, "Expected a training lock or receipt JSON object")
    return result


def stable_closed(path, limit=LIMIT, *, capture=True):
    require(path.is_absolute() and path.resolve(strict=True) == path, "Exact canonical closed file required")

    def signature(value):
        return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns

    digest = sha256()
    with path.open("rb") as stream:
        first = os.fstat(stream.fileno())
        require(stat.S_ISREG(first.st_mode) and 0 < first.st_size <= limit, "Bounded closed regular file required")
        raw = stream.read(first.st_size) if capture else None
        if raw is not None:
            require(len(raw) == first.st_size, "Short closed file read")
            digest.update(raw)
        else:
            remaining = first.st_size
            while remaining:
                chunk = stream.read(min(remaining, 1024 * 1024))
                require(bool(chunk), "Short closed file read")
                digest.update(chunk)
                remaining -= len(chunk)
        require(signature(first) == signature(os.fstat(stream.fileno())) == signature(path.stat()),
                "Closed file identity changed during readback")
    return dict(path=path.as_posix(), bytes=first.st_size, sha256=digest.hexdigest()), raw


def valid_ref(value):
    return (type(value) is dict and set(value) == {"path", "bytes", "sha256"} and type(value["path"]) is str
            and Path(value["path"]).is_absolute() and type(value["bytes"]) is int and 0 < value["bytes"] <= LIMIT
            and is_hex(value["sha256"], 64))


def read_ref(value):
    require(valid_ref(value), "Exact bounded training reference required")
    actual, raw = stable_closed(Path(value["path"]))
    require(actual == value, "Exact closed training reference bytes required")
    return raw


def read_training_lock(args):
    require(is_hex(args.training_lock_sha256, 64) and is_hex(args.protocol, 40)
            and args.protocol not in ORIGINAL_PROTOCOLS and args.expected_initial_sha256 == INITIAL_EXPANDED_SHA,
            "Exact prospectively locked common-support protocol and COCO initialization required")
    actual, raw = stable_closed(args.training_lock)
    require(actual["sha256"] == args.training_lock_sha256, "Exact training lock digest required")
    lock = object_from(raw)
    expected = dict(status=LOCK_STATUS, training_protocol=args.protocol, stage=args.stage, mode=args.mode,
                    training_seed=TRAINING_SEED, initialization_seed=INITIALIZATION_SEED,
                    initial_expanded_sha256=INITIAL_EXPANDED_SHA, fit_index_sha256=FIT_INDEX_SHA,
                    native_output=args.output.as_posix(), audit_output=args.audit_output.as_posix())
    require(set(lock) == set(expected) | {"implementation_commit", "sources", "qualification_pair_ref"}
            and same_fields(lock, expected) and args.training_seed == TRAINING_SEED
            and is_hex(lock["implementation_commit"], 40) and lock["implementation_commit"] not in ORIGINAL_PROTOCOLS,
            "Exact per-arm stage, mode, seed, implementation and output lock required")
    if args.stage == "e24":
        require(valid_ref(lock["qualification_pair_ref"]), "E24 requires an independently accepted short pair")
    else:
        require(lock["qualification_pair_ref"] is None, "A fresh short window cannot carry an earlier acceptance")
    refs = lock["sources"]
    require(type(refs) is dict and set(refs) == SOURCES and all(valid_ref(ref) for ref in refs.values())
            and all(refs[name]["sha256"] == digest for name, digest in SOURCE_PINS.items()),
            "Exact closed source pins for the common FIT worker required")
    return lock


def manifest_digest(lock):
    pins = {name: ref["sha256"] for name, ref in lock["sources"].items()}
    return sha256(json.dumps(pins, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()


def verify_qualification(lock):
    reference = lock["qualification_pair_ref"]
    if reference is None:
        return None
    receipt = object_from(read_ref(reference))
    expected = dict(status=QUALIFIED_STATUS, training_protocol=lock["training_protocol"],
                    implementation_commit=lock["implementation_commit"], training_seed=TRAINING_SEED,
                    initialization_seed=INITIALIZATION_SEED, initial_expanded_sha256=INITIAL_EXPANDED_SHA,
                    fit_index_sha256=FIT_INDEX_SHA, source_manifest_sha256=manifest_digest(lock),
                    ordered_common_fit_visits_match=True, visits_per_arm=32, optimizer_updates_per_arm=8,
                    ema_updates_per_arm=8, FIT=True, DEV=False, TEST=False, AP=False)
    digests = {"native_arm_receipt_sha256", "center_arm_receipt_sha256", "shared_consumed_ledger_sha256"}
    require(set(receipt) == set(expected) | digests | {"changed_mask_visits"} and same_fields(receipt, expected)
            and all(is_hex(receipt[name], 64) for name in digests)
            and receipt["native_arm_receipt_sha256"] != receipt["center_arm_receipt_sha256"]
            and type(receipt["changed_mask_visits"]) is int and 0 < receipt["changed_mask_visits"] <= 32,
            "Exact independently accepted short common FIT pair required before E24")
    return reference["sha256"]


def isolated_outputs(native_output, audit_output):
    for path in (native_output, audit_output):
        require(path.is_absolute() and path.parent.resolve(strict=True) == path.parent
                and PRIVATE in path.parents and not path.is_relative_to(RELEASE)
                and not path.is_relative_to(FIT_LAYOUT) and path not in FIT_LAYOUT.parents
                and not path.exists() and not path.is_symlink(),
                "Fresh canonical private output outside FIT and ordinary source required")
    require(not native_output.is_relative_to(audit_output) and not audit_output.is_relative_to(native_output),
            "Separate fresh native and audit output directories required")


def load_source(reference, name):
    require(name not in sys.modules, "Exact fresh pinned source namespace required")
    raw = read_ref(reference)
    path = reference["path"]
    module = ModuleType(name)
    module.__file__ = path
    sys.modules[name] = module
    try:
        exec(compile(raw, path, "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def verify_native_result(args, native_output, visits, updates, ledger_stats):
    qualification = args.stage == "qualification"
    filename = "QUALIFICATION_SAFE.json" if qualification else "NATIVE_HRSID_COCO_FIT_E24_SAFE.json"
    reference, raw = stable_closed(native_output / filename)
    result = object_from(raw)
    if qualification:
        expected = dict(status="NATIVE_HRSID_MASK_RESIZE_QUALIFICATION_COMPLETE", visits=visits,
                        optimizer_updates=updates, ema_updates=updates, completed_epochs=0,
                        removed_empty_rows=0, targets_after_filter=ledger_stats["common_targets_total"],
                        checkpoint_saved=False, FIT=True, seed=TRAINING_SEED, stage=args.stage,
                        mask_resize_mode=args.mode, initial_online_sha256=INITIAL_EXPANDED_SHA)
    else:
        expected = dict(status="NATIVE_HRSID_COCO_FIT_E24_COMPLETE_PENDING_INDEPENDENT_ACCEPTANCE",
                        total_visits=visits, total_updates=updates, ema_updates=updates,
                        completed_epochs=24, filtered_empty_rows_total=0, FIT_only=True,
                        native_index_sha256=FIT_INDEX_SHA, expanded_online_sha256=INITIAL_EXPANDED_SHA)
    require(same_fields(result, dict(expected, protocol=args.protocol, DEV=False, TEST=False, AP=False)),
            "Exact native FIT scope, common counts and training clocks required")
    return reference["sha256"]


def run_engine(args, lock, qualification_sha, *, phase, repeat, native, base, binding, module_data,
               trainer_api, coco, native_transforms, seed_everything, torch):
    require(native.SEED == repeat.INITIALIZATION_SEED == INITIALIZATION_SEED
            and repeat.INITIAL_EXPANDED_SHA == INITIAL_EXPANDED_SHA and base.LAYOUT == FIT_LAYOUT
            and base.INDEX_SHA == FIT_INDEX_SHA and (native.FIT, native.VISITS, native.STEPS) == (2937, 2940, 735)
            and native.ENDPOINTS == (1, 8, 24) and phase.WINDOW_UPDATES == 8 and native.ACCUM == 4,
            "Exact original FIT population, initialization and endpoint clocks required")
    short = args.stage == "qualification"
    visits, updates = (32, 8) if short else (70560, 17640)
    pins = lock["sources"]
    phase_marker = dict(protocol=args.protocol, mask_resize_mode=args.mode, stage=args.stage, seed=TRAINING_SEED,
                        phase_trainer_sha256=pins[SELF]["sha256"], phase_patch_sha256=pins["phase_patch.py"]["sha256"],
                        native_trainer_sha256=pins[NATIVE]["sha256"], initial_expanded_sha256=INITIAL_EXPANDED_SHA)
    common_marker = dict(policy=POLICY, training_protocol=args.protocol, implementation_commit=lock["implementation_commit"],
                         mode=args.mode, stage=args.stage, training_seed=TRAINING_SEED, initialization_seed=INITIALIZATION_SEED,
                         initial_expanded_sha256=INITIAL_EXPANDED_SHA, fit_index_sha256=FIT_INDEX_SHA,
                         source_manifest_sha256=manifest_digest(lock), qualification_pair_sha256=qualification_sha)
    original_build = trainer_api.build_trainer
    witness = None

    def observe_witness(value):
        nonlocal witness
        require(witness is None, "Exactly one actual common FIT seed witness required")
        witness = value
        native.event(repeat.WITNESS_EVENT, **value, seed_witness_sha256=repeat.witness_digest(value))

    args.audit_output.mkdir()
    ledger_path = args.audit_output / AUDIT_LEDGER
    with ledger_path.open("x", encoding="utf8", newline="\n") as stream:
        ledger = binding.ConsumedFitLedger(stream)

        def build(train, model, *, extra_callbacks, **kwargs):
            observer, = extra_callbacks
            trainer = original_build(train, model, extra_callbacks=extra_callbacks, **kwargs)
            if short:
                phase.attach_short_window(observer, trainer, native, base, args, phase_marker)
            else:
                original_save = observer.on_save_checkpoint

                def save_checkpoint(active, module, checkpoint):
                    original_save(active, module, checkpoint)
                    require(phase.PHASE_MARKER not in checkpoint and COMMON_MARKER not in checkpoint,
                            "Fresh original phase and common-support checkpoint markers required")
                    checkpoint[phase.PHASE_MARKER] = phase_marker.copy()
                    checkpoint[COMMON_MARKER] = common_marker.copy()

                observer.on_save_checkpoint = save_checkpoint
            binding.bind_common_observer(observer, ledger)
            repeat.bind_repeat_training(trainer, observer, train, args, seed_everything, torch, observe_witness)
            return trainer

        stopped_short = False
        with (binding.common_fit_dataset(module_data, coco.CocoDetection, base.LAYOUT / "train", native.FIT,
                                         native_transforms, args.mode, ledger),
              patch.object(trainer_api, "build_trainer", build)):
            try:
                native.run(base, args.output, args.protocol, args.expected_initial_sha256)
            except phase.QualificationComplete:
                require(short, "Only the independently admitted qualification may stop at eight updates")
                stopped_short = True
        require(witness is not None and stopped_short is short, "Exact training stage and actual seed witness required")
        stats = ledger.finish(visits)
        require((short or stats["prefetched_unconsumed"] == 0) and 0 < stats["changed_mask_visits"] <= visits,
                "Exact consumed common FIT comparison with actual mask changes required")
        stream.flush()
        os.fsync(stream.fileno())
    ledger_ref, _ = stable_closed(ledger_path, 128 * 1024**2, capture=False)
    native_result_sha = verify_native_result(args, args.output, visits, updates, stats)
    witness_path = args.output / repeat.WITNESS_NAME
    phase.exclusive_json(witness_path, witness)
    witness_ref, _ = stable_closed(witness_path)
    require(witness_ref["sha256"] == repeat.witness_digest(witness), "Exact actual seed witness bytes required")
    if not short:
        phase.exclusive_json(args.output / "PHASE_PROVENANCE_SAFE.json", dict(
            status="NATIVE_HRSID_MASK_RESIZE_E24_CLOSED_PENDING_INDEPENDENT_ACCEPTANCE",
            native_result_sha256=native_result_sha, **phase_marker))
    phase.exclusive_json(args.audit_output / AUDIT_RESULT, dict(
        status=AUDIT_STATUS, common_support=common_marker, phase=phase_marker, native_result_sha256=native_result_sha,
        repeat_seed_witness_sha256=witness_ref["sha256"], consumed_ledger_sha256=ledger_ref["sha256"],
        consumed_ledger_bytes=ledger_ref["bytes"], ledger=stats, optimizer_updates=updates, ema_updates=updates,
        completed_epochs=0 if short else 24, FIT=True, DEV=False, TEST=False, AP=False))
    if not short:
        native.event("HRSID_MASK_PHASE_E24_OUTPUT_COMPLETE", mask_resize_mode=args.mode, protocol=args.protocol)


def run_locked(args):
    lock = read_training_lock(args)
    require(Path(sys.executable).resolve() == PYTHON.resolve(strict=True) and sys.version_info[:3] == (3, 10, 13)
            and sys.flags.isolated == 1 and sys.dont_write_bytecode and sys.flags.utf8_mode == 1
            and os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
            and os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1",
            "Exact original isolated offline Windows GPU runtime required")
    isolated_outputs(args.output, args.audit_output)
    qualification_sha = verify_qualification(lock)
    refs = lock["sources"]
    require(Path(refs[SELF]["path"]) == Path(__file__).resolve(strict=True)
            and Path(refs[NATIVE]["path"]) == args.native_source,
            "Exact running wrapper and locked native trainer paths required")
    read_ref(refs[SELF])
    native = load_source(refs[NATIVE], "_common_fit_original_native")
    sys.addaudithook(native.offline)
    base = native.pinned_module(native.E1, native.E1_SHA, "_common_fit_original_e1")
    base.admit_dependency_overlay()
    load_source(refs["trace_resize.py"], "trace_resize")
    load_source(refs["phase_patch.py"], "phase_patch")
    phase = load_source(refs["phase_train.py"], "_common_fit_original_phase")
    repeat = load_source(refs["phase_repeat_train.py"], "_common_fit_original_repeat")
    load_source(refs["common_mask_supervision.py"], "common_mask_supervision")
    load_source(refs["paired_fit_transform.py"], "paired_fit_transform")
    binding = load_source(refs["common_fit_binding.py"], "common_fit_binding")

    from importlib.metadata import version
    from pytorch_lightning import seed_everything
    from rfdetr.datasets import _torchvision, coco
    from rfdetr.training import module_data, trainer as trainer_api
    import torch

    require(version("torchvision") == "0.25.0+cu130", "Exact paired FIT torchvision runtime required")
    runtime_sources = ((coco, "be64b20a178d9cde41db69e01e17e1296e36eee791e837294c3f930ae56f03b9"),
                       (_torchvision, "03c12e881f733d4f6b469f784187e461753b8775b7e35e57c1d8a6c35c46ef56"))
    for module, expected in runtime_sources:
        ref, _ = stable_closed(Path(module.__file__).resolve(strict=True), 256 * 1024, capture=False)
        require(ref["sha256"] == expected, "Exact qualified FIT converter and augmentation source required")
    run_engine(args, lock, qualification_sha, phase=phase, repeat=repeat, native=native, base=base, binding=binding,
               module_data=module_data, trainer_api=trainer_api, coco=coco, native_transforms=_torchvision,
               seed_everything=seed_everything, torch=torch)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("native-source", "output", "audit-output", "training-lock"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    for name in ("protocol", "expected-initial-sha256", "training-lock-sha256"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--mode", choices=("native", "center"), required=True)
    parser.add_argument("--stage", choices=("qualification", "e24"), required=True)
    parser.add_argument("--training-seed", choices=(TRAINING_SEED,), type=int, required=True)
    args = parser.parse_args(argv)
    return run_locked(args)


if __name__ == "__main__":
    main()
