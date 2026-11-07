"""Bind the native SSDD sanity probe to five admitted files and the frozen recipe."""

import argparse
from contextlib import redirect_stdout
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys


ROLES = ("train", "valid")
DATASET_PLACEHOLDER = "/data/REQUIRES_QUALIFIED_SAR_COCO_SPLITS"


def _require_keys(value, keys, label):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError(f"unexpected {label} fields")


def _absolute_path(value):
    path = Path(value)
    if not path.is_absolute():
        raise ValueError("expected an explicit absolute path")
    return path


def _references(binding, root):
    annotations = binding["annotation_refs"]
    sample_refs = binding["sample_refs"]
    samples = binding["samples"]
    for value, label in ((annotations, "annotation refs"), (sample_refs, "sample refs"), (samples, "samples")):
        _require_keys(value, ROLES, label)

    references = []
    for role in ROLES:
        _require_keys(samples[role], ("image_id", "image_path"), f"{role} sample")
        references.extend((
            (f"{role} annotations", annotations[role], root / role / "_annotations.coco.json"),
            (f"{role} sample", sample_refs[role], _absolute_path(samples[role]["image_path"])),
        ))
    references.append(("layout receipt", binding["layout_receipt_ref"], None))

    paths = set()
    for label, ref, expected_path in references:
        _require_keys(ref, ("path", "bytes", "sha256"), label)
        path = _absolute_path(ref["path"])
        if path in paths or (expected_path is not None and path != expected_path):
            raise ValueError(f"unexpected or repeated {label} path")
        if type(ref["bytes"]) is not int or ref["bytes"] <= 0:
            raise ValueError(f"invalid {label} byte count")
        if type(ref["sha256"]) is not str or re.fullmatch(r"[0-9a-f]{64}", ref["sha256"]) is None:
            raise ValueError(f"invalid {label} SHA-256")
        paths.add(path)
    return references


def _verify_references(references):
    for label, ref, _ in references:
        path = Path(ref["path"])
        if path.resolve(strict=True) != path or not path.is_file():
            raise ValueError(f"expected a canonical regular file for {label}")
        with path.open("rb") as stream:
            if os.fstat(stream.fileno()).st_size != ref["bytes"]:
                raise ValueError(f"byte count differs for {label}")
            if hashlib.file_digest(stream, "sha256").hexdigest() != ref["sha256"]:
                raise ValueError(f"SHA-256 differs for {label}")


def _load_configs(config_path, root):
    import yaml
    from rfdetr.config import RFDETRSegNanoConfig, SegmentationTrainConfig

    recipe = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _require_keys(recipe, ("seed_everything", "model", "trainer"), "recipe")
    _require_keys(recipe["model"], ("model_config", "train_config"), "recipe model")
    model = recipe["model"]["model_config"]
    train = recipe["model"]["train_config"]
    for section, cls in ((model, RFDETRSegNanoConfig), (train, SegmentationTrainConfig)):
        _require_keys(section, ("class_path", "init_args"), "recipe class")
        if section["class_path"] != f"rfdetr.config.{cls.__name__}" or type(section["init_args"]) is not dict:
            raise ValueError("unexpected recipe class or constructor arguments")
    train_args = dict(train["init_args"])
    if recipe["seed_everything"] != 5201 or train_args.get("seed") != 5201:
        raise ValueError("expected the frozen data probe seed")
    if train_args.get("dataset_dir") not in (DATASET_PLACEHOLDER, str(root)):
        raise ValueError("unexpected recipe dataset path")
    train_args["dataset_dir"] = str(root)
    return RFDETRSegNanoConfig(**model["init_args"]), SegmentationTrainConfig(**train_args)


def run(binding_path, config_path):
    """Return an unqualified observation for two caller-bound positive samples."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("expected a CPU-only worker environment")
    config_path = _absolute_path(config_path)
    binding = json.loads(_absolute_path(binding_path).read_text(encoding="utf-8"))
    _require_keys(binding, ("dataset_root", "expected_ids", "samples", "annotation_refs", "sample_refs",
                            "layout_receipt_ref"), "data binding")
    root = _absolute_path(binding["dataset_root"])
    if root.resolve(strict=True) != root or not root.is_dir():
        raise ValueError("expected the canonical admitted data root")
    references = _references(binding, root)
    _verify_references(references)
    try:
        model_config, train_config = _load_configs(config_path, root)
        spec = importlib.util.spec_from_file_location("_ssdd_native_probe", Path(__file__).with_name("probe.py"))
        if spec is None or spec.loader is None:
            raise ImportError("could not load the sibling SSDD probe")
        probe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(probe)
        result = probe.run_probe(root, model_config, train_config, binding["expected_ids"], binding["samples"])
    finally:
        _verify_references(references)
    return {**result, "bound_file_count": len(references), "bound_files_verified_before_and_after": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binding_path")
    parser.add_argument("config_path")
    args = parser.parse_args()
    with redirect_stdout(sys.stderr):
        result = run(args.binding_path, args.config_path)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
