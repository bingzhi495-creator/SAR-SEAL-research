"""Run the fixed SSDD FIT-only mask replay with the qualified DGX CPU runtime."""
import argparse
from hashlib import sha256
from importlib.metadata import version
from importlib.util import find_spec
import json
import os
from pathlib import Path
import platform
import sys


HERE = Path(__file__).resolve().parent
SOURCE_PINS = {
    "ssdd_phase_init.py": "444e91009bce7a56e65a9bd2110c3479a86609d29b5ccdc925b06ec013b7894a",
    "ssdd_fit_stage.py": "5623d554d918ae2dc678d52dce96d86aaa407d6fb0b03fb4cf478d07a4676a88",
    "ssdd_fit_stage_readback.py": "4ca757bcfdd630ce8a58fab54d27e14631420f54711db185e5d408f46adc5dc4",
    "ssdd_fit_only.py": "e421295d4956e09ae42c776e7f7d68f821ab136f1c6ddcca004e97cf0435b646",
    "ssdd_phase_fit_replay.py": "14b097a19c8b951e7d8e246e75e855b2eb574bb1149aa3945b9cf15b91a2cacf",
}
PACKAGES = {"torch": "2.10.0+cu130", "torchvision": "0.25.0+cu130", "rfdetr": "1.12.0",
            "pytorch-lightning": "2.6.6", "numpy": "1.26.4", "Pillow": "12.3.0", "pycocotools": "2.0.11"}
RUNTIME_FILES = {
    ("rfdetr", "datasets/_torchvision.py"): "03c12e881f733d4f6b469f784187e461753b8775b7e35e57c1d8a6c35c46ef56",
    ("rfdetr", "datasets/coco.py"): "be64b20a178d9cde41db69e01e17e1296e36eee791e837294c3f930ae56f03b9",
    ("rfdetr", "training/module_data.py"): "624bc77f7028bc53e0ca57dcbf5be1a7d334685147dbfab65e9acaa063d715b5",
    ("PIL", "_imaging.*.so"): "2bdbf62e38af4c59c6563b9391f11d411e42f5ced5dc0991aaf31bdda72928a8",
    ("pycocotools", "_mask.*.so"): "53d99dd5a70f01aed23bcbfa59aa4f82f0dc68c48cbc4a676a9d5df9e31f1597",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def source_digest(path):
    require(path.is_file() and path.resolve(strict=True) == path, "Canonical replay source required")
    return sha256(path.read_bytes()).hexdigest()


def runtime():
    require(platform.system() == "Linux" and platform.machine() == "aarch64"
            and sys.version_info[:3] == (3, 12, 14) and sys.flags.isolated == 1 and sys.flags.utf8_mode == 1
            and sys.dont_write_bytecode and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
            and all(os.environ.get(name) == "1" for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"))
            and os.sched_getaffinity(0) == {2, 3},
            "Expected isolated offline DGX CPU affinity and interpreter")
    require({key: version(key) for key in PACKAGES} == PACKAGES, "Expected locked DGX package versions")
    for (module, name), expected in RUNTIME_FILES.items():
        root = Path(next(iter(find_spec(module).submodule_search_locations))).resolve(strict=True)
        candidates = list(root.glob(name))
        require(len(candidates) == 1 and source_digest(candidates[0]) == expected,
                "Locked stock loader or codec changed")
    require(all(source_digest(HERE / name) == digest for name, digest in SOURCE_PINS.items()),
            "Expected pinned ordinary replay sources")


def write_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def replay(archive, work):
    require(work.is_absolute() and work.parent.resolve(strict=True) == work.parent and not work.exists()
            and not work.is_symlink(), "Fresh canonical replay output required")
    sys.path.insert(0, str(HERE))
    from ssdd_phase_init import CONFIG, CONFIG_SHA, E1, E1_SHA, RECIPE, RECIPE_SHA, load_api, pinned_bytes
    from ssdd_fit_stage import stage_fit_layout
    from ssdd_fit_stage_readback import readback_fit_layout
    from ssdd_fit_only import FitOnlyData
    from ssdd_phase_fit_replay import ReplayViewError, replay_fit_population
    from rfdetr import config
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    require(not torch.cuda.is_initialized(), "CPU diagnostic must leave CUDA unused")
    work.mkdir()
    layout = work / "LAYOUT"
    stage = stage_fit_layout(archive, layout)
    readback = readback_fit_layout(archive, layout)
    ids = tuple(sorted(entry["id"] for entry in json.loads((layout / "train/_annotations.coco.json").read_bytes())["images"]))
    ids_sha = sha256(json.dumps(ids, separators=(",", ":")).encode("ascii")).hexdigest()
    require(ids_sha == stage["native_sorted_id_order_sha256"] and len(ids) == len(set(ids)) == 741,
            "Expected exact stock logical FIT order")
    pinned_bytes(RECIPE, RECIPE_SHA)
    model, train = load_api(CONFIG, CONFIG_SHA)._load_configs(RECIPE, layout)
    load_api(E1, E1_SHA).native_recipe(model, train, config)
    data = FitOnlyData(model, train, layout)
    data.setup("fit")
    try:
        result = replay_fit_population(data._dataset_train, ids)
    except ReplayViewError as error:
        write_json(work / "FIRST_REPLAY_FAILURE_PRIVATE.json", error.diagnostic)
        raise
    require(not torch.cuda.is_initialized() and torch.get_num_threads() == torch.get_num_interop_threads() == 1,
            "Expected one-thread CPU-only replay")
    require(readback == readback_fit_layout(archive, layout), "Isolated FIT input changed across replay")
    write_json(work / "FIT_REPLAY_SAFE.json",
               dict(**result, platform="Linux_aarch64_DGX_CPU", packages=PACKAGES, cpu_affinity=[2, 3],
                    torch_threads=1, cuda_initialized=False, fit_index_sha256=stage["fit_index_sha256"],
                    fit_archive_sha256=stage["fit_archive_sha256"], ordered_fit_ids_sha256=ids_sha))


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--check-runtime", action="store_true")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.check_runtime:
        require(args.archive is args.output_dir is None, "Runtime check takes no scientific paths")
    else:
        require(args.archive is not None and args.output_dir is not None, "Closed archive and fresh output required")
    runtime()
    if args.check_runtime:
        print("SSDD_DGX_CPU_RUNTIME_CHECK_PASSED", flush=True)
    else:
        replay(args.archive, args.output_dir)


if __name__ == "__main__":
    main()

