"""Reuse the exact SSDD float evaluator while keeping training and initialization seeds distinct."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace


FLOAT_WORKER = "SSDD_FLOAT_PRIMARY_DEV187_20261112/float_dev187.py"
FLOAT_WORKER_SHA = "5463f294191929bceaac3b04fb6c17179c05066eaa29b6d859ef4943cca686c7"
FLOAT_OWNER = "SSDD_FLOAT_PRIMARY_DEV187_WINDOWS_OWNER_20261112/float_dev187_owner.py"
FLOAT_OWNER_SHA = "63258b34f546d0b7b87f14365c17890896a48c695b3d115e4fbc9f83e9ce713f"
REPEAT_ADAPTER = "SSDD_DEV187_ALIAS_AND_COMMON_FIT_20261111/ssdd_repeat5202_dev187_stage.py"
REPEAT_ADAPTER_SHA = "ff381da5c53dd8e44abf42f5168f724b3732b57187b46daf3864827d9c450f2c"
INITIALIZATION_SEED, TRAINING_SEED = 5201, 5202
SOURCE_LIMIT = 128 * 1024


def require(condition, message):
    if not condition:
        raise ValueError(message)


def exact_source(path, expected_sha):
    require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file(),
            "Exact canonical ordinary evaluator source required")
    with path.open("rb") as stream:
        raw = stream.read(SOURCE_LIMIT + 1)
    require(0 < len(raw) <= SOURCE_LIMIT and sha256(raw).hexdigest() == expected_sha,
            "Original ordinary evaluator bytes changed: " + path.name)
    return raw


def exact_module(path, expected_sha, name):
    module = ModuleType(name)
    module.__file__ = str(path)
    exec(compile(exact_source(path, expected_sha), str(path), "exec"), module.__dict__)
    return module


def configure_admission(code_root, stage, modules):
    """Apply the existing seed-5202 archive check only to this float admission instance."""
    repeat = exact_module(code_root / REPEAT_ADAPTER, REPEAT_ADAPTER_SHA, "_float_repeat_seed_contract")
    admission = modules["ssdd_phase_dev_admission"]
    require((repeat.INITIALIZATION_SEED, repeat.TRAINING_SEED, admission.SEED) ==
            (INITIALIZATION_SEED, TRAINING_SEED, INITIALIZATION_SEED), "Unexpected original SSDD seed scope")
    phase_marker = admission.phase_marker
    admission.SEED = TRAINING_SEED
    admission._repeat_original_sha = stage.PINS["ssdd_phase_dev_admission"]
    repeat.replace_literal(admission, "bind_archive_validator",
                           old="Exact saved RF-DETR seed 5201 required",
                           new="Exact saved RF-DETR training seed 5202 required")

    def repeat_phase_marker(*args, **kwargs):
        phase = phase_marker(*args, **kwargs)
        require((phase["seed"], phase["initialization_seed"]) == (TRAINING_SEED, TRAINING_SEED),
                "Unexpected stock seed marker layout")
        return dict(phase, initialization_seed=INITIALIZATION_SEED)

    admission.phase_marker = repeat_phase_marker
    return stage, modules


def probe_admission(admission):
    calls = []
    boundary = {"format": "synthetic archive proof only"}
    long_worker = SimpleNamespace(
        BOUNDARY_KEY="boundary",
        endpoint_marker=lambda epoch, recipe: boundary,
        validate_archive_progress=lambda checkpoint, epoch, **kwargs: calls.append(epoch),
    )
    phase = admission.phase_marker("native", "1" * 40, "2" * 64, "3" * 64)
    require(phase["seed"] == TRAINING_SEED and phase["initialization_seed"] == INITIALIZATION_SEED,
            "Repeat must preserve the initialization seed")
    validate = admission.bind_archive_validator({"phase": phase}, long_worker=long_worker)
    archive = {
        "epoch": 23, "global_step": 4464, "args": {"seed": TRAINING_SEED},
        "model_config": dict(num_queries=100, num_select=100, group_detr=13,
                             resolution=576, num_classes=1, segmentation_head=True),
        admission.PHASE_MARKER: phase, "boundary": boundary,
    }
    def check(checkpoint):
        validate(checkpoint, 24, boundary_api=None, recipe_sha256=admission.RECIPE_SHA)
    check(archive)
    invalid = (
        dict(archive, args={"seed": INITIALIZATION_SEED}),
        dict(archive, global_step=4463),
        dict(archive, boundary={}),
        dict(archive, **{admission.PHASE_MARKER: dict(phase, initialization_seed=TRAINING_SEED)}),
        dict(archive, **{admission.PHASE_MARKER: dict(phase, mask_resize_mode="center")}),
    )
    for candidate in invalid:
        try:
            check(candidate)
        except ValueError:
            continue
        raise ValueError("Invalid synthetic E24 checkpoint was accepted")
    require(calls == [24], "The historical full archive validator must run only after exact repeat checks")
    return len(invalid)


def main():
    parser = ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--code-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.code_root
    exact_source(root / FLOAT_OWNER, FLOAT_OWNER_SHA)
    original = exact_module(root / FLOAT_WORKER, FLOAT_WORKER_SHA, "_stock_float_primary")
    stage, modules = original.load_stock(root)
    score = modules["ssdd_phase_dev_score"]
    score_one = score.score_one
    _, modules = configure_admission(root, stage, modules)
    count = probe_admission(modules["ssdd_phase_dev_admission"])
    require(modules["ssdd_phase_dev_score"].score_one is score_one
            and original.HISTORICAL_AP == 0.7047324561474607
            and original.FLOAT_GT_SHA == "8d98e6f49200d5390c17bee66697de1f582133d08063259a561ed5f710be7b6c",
            "Exact original scorer and frozen comparator must remain intact")
    print(json.dumps(dict(status="SSDD_FLOAT_REPEAT5202_SOURCE_AND_ARCHIVE_SEED_PROBE_PASSED_V1",
                          training_seed=TRAINING_SEED, initialization_seed=INITIALIZATION_SEED,
                          synthetic_rejections=count, historical_mask_AP_points=100 * original.HISTORICAL_AP,
                          scorer_unchanged=True, future_training_receipt_schema_bound=False,
                          phase_data_opened=False, GPU=False, DEV=False, TEST=False), sort_keys=True))


if __name__ == "__main__":
    main()

