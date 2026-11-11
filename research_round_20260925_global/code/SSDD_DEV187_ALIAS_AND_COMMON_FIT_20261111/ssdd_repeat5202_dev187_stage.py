"""Run the fixed SSDD DEV187 stages for the independently accepted 5202 repeat."""

import argparse
import ast
from hashlib import sha256
from pathlib import Path
from types import ModuleType


STAGE_PATH = "SSDD_DEV187_STAGE_CUDA_INIT_20261111/ssdd_dev187_stage_worker.py"
STAGE_SHA = "a491f847fab5411240967172792c73cc47574120be07367a3cf931b998aa0be0"
INITIALIZATION_SEED = 5201
TRAINING_SEED = 5202


def pinned_source(path, expected):
    if path.resolve(strict=True) != path or not path.is_file():
        raise ValueError("Canonical original DEV187 source required")
    with path.open("rb") as stream:
        data = stream.read(128 * 1024 + 1)
    if len(data) > 128 * 1024 or sha256(data).hexdigest() != expected:
        raise ValueError("Original DEV187 source SHA-256 differs")
    return data


def load_stage(code_root):
    """Reuse the byte-pinned stage, file scopes, CUDA setup, and result format."""
    root = Path(code_root)
    path = root / STAGE_PATH
    source = pinned_source(path, STAGE_SHA)
    stage = ModuleType("_ssdd_repeat5202_original_dev_stage")
    stage.__file__ = str(path)
    exec(compile(source, str(path), "exec"), stage.__dict__)
    return stage


def replace_literal(module, name, *, old, new, keyword=None):
    """Rebind one audited literal inside one original function."""
    path = Path(module.__file__)
    tree = ast.parse(pinned_source(path, module._repeat_original_sha), filename=str(path))
    candidates = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(candidates) != 1:
        raise ValueError(f"Missing unique original function {name}")
    function = candidates[0]
    if keyword is None:
        sites = [
            node for node in ast.walk(function)
            if isinstance(node, ast.Constant) and type(node.value) is type(old) and node.value == old
        ]
    else:
        sites = [
            node.value for node in ast.walk(function)
            if isinstance(node, ast.keyword) and node.arg == keyword
            and isinstance(node.value, ast.Constant) and type(node.value.value) is type(old)
            and node.value.value == old
        ]
    if len(sites) != 1:
        raise ValueError(f"Expected one audited original literal in {name}")
    sites[0].value = new
    code = ast.Module(body=[function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(code), str(path), "exec"), module.__dict__)


def configure_repeat_modules(original_load, pins, code_root, *, scoring):
    modules = original_load(code_root, scoring=scoring)
    admission = modules["ssdd_phase_dev_admission"]
    if admission.SEED != INITIALIZATION_SEED:
        raise ValueError("Original initialization and training seed must be 5201")
    original_phase_marker = admission.phase_marker
    admission.SEED = TRAINING_SEED
    admission._repeat_original_sha = pins["ssdd_phase_dev_admission"]
    replace_literal(
        admission,
        "bind_archive_validator",
        old="Exact saved RF-DETR seed 5201 required",
        new="Exact saved RF-DETR training seed 5202 required",
    )

    def phase_marker(*args, **kwargs):
        marker = original_phase_marker(*args, **kwargs)
        if marker["seed"] != TRAINING_SEED or marker["initialization_seed"] != TRAINING_SEED:
            raise ValueError("Unexpected original phase marker seed layout")
        marker["initialization_seed"] = INITIALIZATION_SEED
        return marker

    admission.phase_marker = phase_marker
    if scoring:
        score = modules["ssdd_phase_dev_score"]
        score._repeat_original_sha = pins["ssdd_phase_dev_score"]
        replace_literal(score, "fixed_pair", old=INITIALIZATION_SEED, new=TRAINING_SEED, keyword="seed")
        replace_literal(score, "check_arm", old=INITIALIZATION_SEED, new=TRAINING_SEED, keyword="training_seed")
    return modules


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--code-root", type=Path, required=True)
    options, _ = parser.parse_known_args(argv)
    stage = load_stage(options.code_root)
    original_load = stage.load_published

    def load_repeat(code_root, *, scoring):
        return configure_repeat_modules(original_load, stage.PINS, code_root, scoring=scoring)

    stage.load_published = load_repeat
    return stage.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

