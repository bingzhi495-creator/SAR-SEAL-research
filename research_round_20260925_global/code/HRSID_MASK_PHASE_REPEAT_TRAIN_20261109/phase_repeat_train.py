"""Train one prospectively selected HRSID repeat arm with the original COCO initialization."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys
from types import ModuleType
from unittest.mock import patch


INITIALIZATION_SEED = 5201
INITIAL_EXPANDED_SHA = "63ff93afdd977ba79b5b5a46ba8b4839ef6c40eb0722b29e4667d4932404fbc2"
PHASE_TRAIN_SHA = "8550b43791a6cb6f53736dc7e64b9c5c6204723cf4c07c3b112491ba1bd74043"
PHASE_PATCH_SHA = "811eccf2b42319272aea8b3f50a71bc029734d006bba43260f66117e9421948e"
TRACE_RESIZE_SHA = "902a0e3640a319ba729172847f49ce582f3c0e3954593862ed94971727dc309f"
TRAINING_SEEDS = (5202, 5203)
WITNESS_NAME = "REPEAT_SEED_WITNESS_SAFE.json"
WITNESS_EVENT = "HRSID_MASK_PHASE_REPEAT_SEED_OBSERVED"
WITNESS_STATUS = "HRSID_MASK_PHASE_REPEAT_SEED_WITNESS_V1"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def bind_repeat_training(trainer, observer, train, args, seed_everything, torch, record_witness):
    """Set the downstream seed at the fit boundary after native initialization."""
    seed = args.training_seed
    require(seed in TRAINING_SEEDS and train.seed == INITIALIZATION_SEED,
            "Expected a locked repeat seed and the original TrainConfig seed")
    train.seed = seed
    fit = trainer.fit
    on_train_start = observer.on_train_start

    def fit_with_seed(module, *, datamodule, ckpt_path):
        require(ckpt_path is None and module.train_config is train and datamodule.train_config is train
                and train.seed == seed, "Fresh fit must share the effective repeat TrainConfig")
        # The native trainer has just reset the global RNG to 5201. Seed
        # before Lightning setup and let RF-DETR's on_fit_start reseed again
        # from the same updated TrainConfig.
        seed_everything(seed, workers=True, verbose=False)
        if args.stage == "qualification":
            return fit(module, datamodule=datamodule, ckpt_path=ckpt_path)

        on_save_checkpoint = module.on_save_checkpoint

        def save_with_seed(checkpoint):
            on_save_checkpoint(checkpoint)
            metadata = checkpoint.get("args")
            require(type(metadata) is dict and type(metadata.get("seed")) is int and metadata["seed"] == seed,
                    "Native checkpoint does not record the locked training seed")

        # Lightning calls the model hook after callback checkpoint hooks.
        with patch.object(module, "on_save_checkpoint", save_with_seed):
            return fit(module, datamodule=datamodule, ckpt_path=ckpt_path)

    def observe_seed(active, module):
        on_train_start(active, module)
        cpu_seed = torch.initial_seed()
        cuda_seed = torch.cuda.initial_seed()
        lightning_seed = os.environ.get("PL_GLOBAL_SEED")
        seed_workers = os.environ.get("PL_SEED_WORKERS") == "1"
        require(active.global_rank == 0 and module.train_config is train and train.seed == cpu_seed == cuda_seed == seed
                and lightning_seed == str(seed) and seed_workers,
                "Actual single-device fit did not use the locked repeat seed")
        record_witness(dict(
            status=WITNESS_STATUS, protocol=args.protocol, mask_resize_mode=args.mode, stage=args.stage,
            initialization_seed=INITIALIZATION_SEED, training_seed=seed, global_rank=active.global_rank,
            train_config_seed=train.seed, torch_cpu_initial_seed=cpu_seed, torch_cuda_initial_seed=cuda_seed,
            lightning_global_seed=int(lightning_seed), lightning_seed_workers=seed_workers,
        ))

    trainer.fit = fit_with_seed
    observer.on_train_start = observe_seed


def pinned_module(path, expected_sha256, name):
    require(path.resolve(strict=True) == path and name not in sys.modules, "Exact fresh source module")
    raw = path.read_bytes()
    require(sha256(raw).hexdigest() == expected_sha256, f"Pinned source bytes differ for {path.name}")
    module = ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def witness_digest(witness):
    raw = (json.dumps(witness, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf8")
    return sha256(raw).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--expected-initial-sha256", required=True)
    parser.add_argument("--mode", required=True, choices=("native", "center"))
    parser.add_argument("--stage", required=True, choices=("qualification", "e24"))
    parser.add_argument("--training-seed", required=True, type=int, choices=TRAINING_SEEDS)
    args = parser.parse_args(argv)
    require(sys.version_info[:3] == (3, 10, 13) and os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
            and os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1"
            and bool(re.fullmatch(r"[0-9a-f]{40}", args.protocol))
            and args.expected_initial_sha256 == INITIAL_EXPANDED_SHA
            and args.native_source.is_absolute() and args.native_source.resolve(strict=True) == args.native_source
            and args.output.is_absolute() and args.output.parent.resolve(strict=True) == args.output.parent
            and not args.output.exists(), "Pinned repeat request, original initialization and fresh private output")

    source = Path(__file__).resolve(strict=True).parent
    pinned_module(source / "trace_resize.py", TRACE_RESIZE_SHA, "trace_resize")
    pinned_module(source / "phase_patch.py", PHASE_PATCH_SHA, "phase_patch")
    phase = pinned_module(source / "phase_train.py", PHASE_TRAIN_SHA, "_closed_mask_phase_trainer")
    native = pinned_module(args.native_source, phase.NATIVE_SHA, "_pinned_native_filtered_trainer")
    require(native.SEED == INITIALIZATION_SEED, "Exact original initialization seed")
    base = native.pinned_module(native.E1, native.E1_SHA, "_closed_hrsid_e1_environment")
    require(not args.output.is_relative_to(base.LAYOUT) and args.output not in base.LAYOUT.parents,
            "Output must be isolated from FIT layout")
    sys.addaudithook(native.offline)
    base.admit_dependency_overlay()
    from pytorch_lightning import seed_everything
    from rfdetr.datasets import _torchvision
    from rfdetr.training import trainer as trainer_api
    import torch

    marker = dict(protocol=args.protocol, mask_resize_mode=args.mode, stage=args.stage, seed=args.training_seed,
                  phase_trainer_sha256=phase.digest(Path(__file__)), phase_patch_sha256=PHASE_PATCH_SHA,
                  native_trainer_sha256=phase.NATIVE_SHA, initial_expanded_sha256=INITIAL_EXPANDED_SHA)
    build_original = trainer_api.build_trainer
    witness = None

    def record_witness(observed):
        nonlocal witness
        require(witness is None, "Repeat seed was observed more than once")
        witness = observed
        native.event(WITNESS_EVENT, **witness, seed_witness_sha256=witness_digest(witness))

    def build(train, model, *, extra_callbacks, **kwargs):
        observer, = extra_callbacks
        trainer = build_original(train, model, extra_callbacks=extra_callbacks, **kwargs)
        if args.stage == "qualification":
            phase.attach_short_window(observer, trainer, native, base, args, marker)
        else:
            save_original = observer.on_save_checkpoint

            def save_checkpoint(active, module, checkpoint):
                save_original(active, module, checkpoint)
                require(phase.PHASE_MARKER not in checkpoint, "Fresh exact mask phase marker")
                checkpoint[phase.PHASE_MARKER] = marker.copy()

            observer.on_save_checkpoint = save_checkpoint
        bind_repeat_training(trainer, observer, train, args, seed_everything, torch, record_witness)
        return trainer

    qualification_completed = False
    with phase.mask_resize_phase(_torchvision, args.mode), patch.object(trainer_api, "build_trainer", build):
        try:
            native.run(base, args.output, args.protocol, args.expected_initial_sha256)
        except phase.QualificationComplete:
            require(args.stage == "qualification", "Only the qualification route may stop short")
            qualification_completed = True
    require(witness is not None and (args.stage == "e24" or qualification_completed),
            "Exact successful training stage and actual repeat seed witness")
    witness_path = args.output / WITNESS_NAME
    phase.exclusive_json(witness_path, witness)
    require(phase.digest(witness_path) == witness_digest(witness), "Repeat seed witness bytes differ from stdout digest")
    if args.stage == "e24":
        result = args.output / "NATIVE_HRSID_COCO_FIT_E24_SAFE.json"
        phase.exclusive_json(args.output / "PHASE_PROVENANCE_SAFE.json",
                             dict(status="NATIVE_HRSID_MASK_RESIZE_E24_CLOSED_PENDING_INDEPENDENT_ACCEPTANCE",
                                  native_result_sha256=phase.digest(result), **marker))
        native.event("HRSID_MASK_PHASE_E24_OUTPUT_COMPLETE", mask_resize_mode=args.mode, protocol=args.protocol)


if __name__ == "__main__":
    main()
