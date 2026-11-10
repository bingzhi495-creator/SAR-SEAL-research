"""Run one prospective HRSID fixed-B one-pass FIT qualification arm."""
import argparse
from hashlib import sha256
import os
from pathlib import Path
import sys
from types import ModuleType

SELF = "one_pass_common_qualification.py"
ENGINE = "common_fit_train.py"
ENGINE_SHA = "b52f8dc7a8d5bd06c0bda3eb839f31cbed3a036befa709c8fc718c47706a959b"
MODES = ("stepwise_center", "one_pass")
TRAINING_SEED = 5202
LOCK_STATUS = "HRSID_ONE_PASS_COMMON_FIT_SHORT_LOCK_V1"
AUDIT_STATUS = "HRSID_ONE_PASS_COMMON_FIT_ARM_CLOSED_PENDING_INDEPENDENT_ACCEPTANCE_V1"
POLICY = "one_pass_on_stepwise_common_nonempty_rows_v1"
ADDED_PINS = {
    ENGINE: ENGINE_SHA,
    "fit_query_export.py": "6a2d7e38d69b82ef3df790a5c359c408b1e74778c39672b47a62c4dc86cabbf5",
    "reference_resize.py": "3ca1d4a6400324c13c777cb4d5f7627713bc625580195d8e05363950872cc8df",
    "replay_fit.py": "4f95f1b5b84fcf89b005f999cfb8c5fd647a7b262aedd2622860cf8785a14157",
    "one_pass_fit_transform.py": "05cad566f9b5cc761e91b5283585a5392a7450c42c3133349e6dc21e8bf7496d",
    "one_pass_common_pair_transform.py": "5db1d9024a7ffa03608488b07669e5692d378f166f8773b1a5475089519d08b7",
    "common_fit_binding.py": "2bfc4d6039b28a632693131126120cc2502c144b7cf6564de5dc4d765a9e4fba",
    "one_pass_common_binding.py": "987f95263c5dd685403afc8c7c96150ee2f3e1f911ce7ba6b2f43734632f028f",
}
RUNTIME_FILES = (
    "common_mask_supervision.py", "paired_fit_transform.py", "fit_query_export.py", "reference_resize.py",
    "replay_fit.py", "one_pass_fit_transform.py", "one_pass_common_pair_transform.py", "common_fit_binding.py",
    "one_pass_common_binding.py",
)


def configured_engine(path):
    if not path.is_absolute() or path.resolve(strict=True) != path:
        raise RuntimeError("Exact canonical original engine path required")
    with path.open("rb") as stream:
        raw = stream.read(64 * 1024 + 1)
    if not 0 < len(raw) <= 64 * 1024 or sha256(raw).hexdigest() != ENGINE_SHA:
        raise RuntimeError("Exact original common FIT engine bytes required")
    engine = ModuleType("_one_pass_common_fit_engine")
    engine.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), engine.__dict__)
    # Configure only the private module instance. The original A/B engine,
    # its approved inputs, and its public entry are independent.
    engine.SELF = SELF
    engine.LOCK_STATUS = LOCK_STATUS
    engine.AUDIT_STATUS = AUDIT_STATUS
    engine.POLICY = POLICY
    engine.SOURCE_PINS = {**engine.SOURCE_PINS, **ADDED_PINS}
    engine.SOURCES = set(engine.SOURCE_PINS) | {SELF}
    return engine


def run_locked(args):
    engine = configured_engine(args.engine_source)
    require = engine.require
    require(args.stage == "qualification" and args.mode in MODES and args.training_seed == TRAINING_SEED,
            "Only the new prospective B/C qualification arms are allowed")
    lock = engine.read_training_lock(args)
    require(Path(sys.executable).resolve() == engine.PYTHON.resolve(strict=True)
            and sys.version_info[:3] == (3, 10, 13) and sys.flags.isolated == 1
            and sys.dont_write_bytecode and sys.flags.utf8_mode == 1
            and os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
            and os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1",
            "Exact original isolated offline Windows GPU runtime required")
    engine.isolated_outputs(args.output, args.audit_output)
    refs = lock["sources"]
    require(Path(refs[SELF]["path"]) == Path(__file__).resolve(strict=True)
            and Path(refs[ENGINE]["path"]) == args.engine_source
            and Path(refs[engine.NATIVE]["path"]) == args.native_source,
            "Exact running qualifier, original engine and native trainer paths required")
    engine.read_ref(refs[SELF])
    engine.read_ref(refs[ENGINE])
    native = engine.load_source(refs[engine.NATIVE], "_one_pass_common_original_native")
    sys.addaudithook(native.offline)
    base = native.pinned_module(native.E1, native.E1_SHA, "_one_pass_common_original_e1")
    base.admit_dependency_overlay()
    engine.load_source(refs["trace_resize.py"], "trace_resize")
    engine.load_source(refs["phase_patch.py"], "phase_patch")
    phase = engine.load_source(refs["phase_train.py"], "_one_pass_common_original_phase")
    repeat = engine.load_source(refs["phase_repeat_train.py"], "_one_pass_common_original_repeat")
    modules = {}
    for filename in RUNTIME_FILES:
        name = Path(filename).stem
        modules[name] = engine.load_source(refs[filename], name)
    binding = modules["one_pass_common_binding"]
    require(modules["one_pass_common_pair_transform"].MODES == MODES, "Exact selected B/C modes")

    from importlib.metadata import version
    from pytorch_lightning import seed_everything
    from rfdetr.datasets import _torchvision, coco
    from rfdetr.training import module_data, trainer as trainer_api
    import torch

    require(version("torchvision") == "0.25.0+cu130", "Exact paired FIT torchvision runtime required")
    runtime_sources = (
        (coco, "be64b20a178d9cde41db69e01e17e1296e36eee791e837294c3f930ae56f03b9"),
        (_torchvision, "03c12e881f733d4f6b469f784187e461753b8775b7e35e57c1d8a6c35c46ef56"),
    )
    for module, expected in runtime_sources:
        ref, _ = engine.stable_closed(Path(module.__file__).resolve(strict=True), 256 * 1024, capture=False)
        require(ref["sha256"] == expected, "Exact qualified FIT converter and augmentation source required")
    return engine.run_engine(
        args, lock, None, phase=phase, repeat=repeat, native=native, base=base, binding=binding, module_data=module_data,
        trainer_api=trainer_api, coco=coco, native_transforms=_torchvision, seed_everything=seed_everything, torch=torch,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("native-source", "engine-source", "output", "audit-output", "training-lock"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("protocol", "expected-initial-sha256", "training-lock-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--stage", choices=("qualification",), required=True)
    parser.add_argument("--training-seed", choices=(TRAINING_SEED,), type=int, required=True)
    return run_locked(parser.parse_args(argv))


if __name__ == "__main__":
    main()
