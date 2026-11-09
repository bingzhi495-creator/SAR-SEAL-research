"""Verify the fresh official COCO Seg Nano initialization at native SSDD Q100."""

import argparse
from hashlib import md5, sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import sys
from types import ModuleType


RESEARCH = Path(__file__).resolve().parents[2]
CODE = RESEARCH / "code"
RECIPE = RESEARCH / "configs/RFDETR_SEG_NANO_BASELINE_576_E100_20261107.yaml"
RECIPE_SHA = "ba9508d9acab5ff68b10db8a1aff5d9001868d6dda00e39f299d6f5cc1b8011b"
CONFIG = RESEARCH / "RFDETR_REAL_SSDD_DATA_QUALIFICATION_DRAFT_20261107_01/entry.py"
CONFIG_SHA = "f0642e8f33e75a2440eac0968dd76f270d445498ba88e812543902e7d0ad8453"
E1 = CODE / "RFDETR_NATIVE_E1_CANDIDATE_20261107/e1_worker.py"
E1_SHA = "3bfeba308487140efeaa1c67f15638bdf72e3cb7cb97dd4e7a26163c455be46b"
STATE = CODE / "RFDETR_E24_OPTIMIZER_PRESERVATION_20261108/state_preservation.py"
STATE_SHA = "8dda41131a2930aacf00c51dbe8cfa4817c0a3cd4b02c6bb606948db77341678"
COCO_BYTES = 134545398
COCO_SHA = "a44613a4ecd6b5ba61a62002c600b0b6cb7a9da2936a45317ec4b62c635fb99b"
COCO_MD5 = "9995497791d0ff1664a1d9ddee9cfd20"
EXPECTED_Q100_SHA = "40524fc63797d2fcab7b50b6ee1f041f9429df73e505cb9e647a5a8cdf6f2e2f"
SEED = 5201


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def pinned_bytes(path, expected, limit=64*1024):
    require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file(),
            "Canonical pinned source required")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    require(0 < len(data) <= limit and sha256(data).hexdigest() == expected,
            "Exact pinned source bytes required for " + path.name)
    return data


def load_api(path, expected):
    module = ModuleType("_ssdd_phase_" + path.stem)
    module.__file__ = str(path)
    exec(compile(pinned_bytes(path, expected), str(path), "exec"), module.__dict__)
    return module


def token(stat):
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def verify_coco_source(path):
    require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file()
            and path.stat().st_size == COCO_BYTES, "Exact official COCO Seg Nano size and canonical path required")
    strong, upstream = sha256(), md5()
    with path.open("rb") as stream:
        before = token(os.fstat(stream.fileno()))
        for block in iter(lambda: stream.read(1024**2), b""):
            strong.update(block)
            upstream.update(block)
        after = token(os.fstat(stream.fileno()))
    require(before == after == token(path.stat()) and strong.hexdigest() == COCO_SHA
            and upstream.hexdigest() == COCO_MD5, "Exact verified official COCO Seg Nano bytes required")
    return before


def describe_q100(module, tensor_digest):
    config, model = module.model_config, module.model
    values = model.state_dict()
    require(config.num_queries == config.num_select == model.num_queries == 100
            and config.num_classes == 1 and config.group_detr == 13
            and model.class_embed.weight.shape[0] == 2
            and all(values[key].shape[0] == 1300 for key in ("refpoint_embed.weight", "query_feat.weight")),
            "Expected original Q100 query layout and two-slot native head")
    digest = tensor_digest(values)
    require(type(digest) is str and re.fullmatch("[a-f0-9]{64}", digest), "Full native tensor digest required")
    return dict(online_tensor_sha256=digest, state_tensors=len(values), num_queries=100, num_select=100,
                query_groups=13, training_queries_per_image=1300, class_head_slots=2)


def initialize_q100(dataset_root, output_root, coco_path):
    """Create a fresh native module. Dataset construction stays outside initialization."""
    import torch
    from pytorch_lightning import seed_everything
    from rfdetr import config
    from rfdetr.training.module_model import RFDETRModelModule

    require(torch.__version__ == "2.10.0+cu130" and version("rfdetr") == "1.12.0"
            and version("pytorch-lightning") == "2.6.6", "Exact native phase numeric runtime required")
    config_api, e1_api, state_api = load_api(CONFIG, CONFIG_SHA), load_api(E1, E1_SHA), load_api(STATE, STATE_SHA)
    pinned_bytes(RECIPE, RECIPE_SHA)
    model, train = config_api._load_configs(RECIPE, dataset_root)
    model = model.model_copy(update={"pretrain_weights": str(coco_path)})
    train = train.model_copy(update={"output_dir": str(output_root), "compute_train_metrics": False})
    e1_api.native_recipe(model, train, config)
    before = verify_coco_source(coco_path)
    seed_everything(SEED, workers=True, verbose=False)
    module = RFDETRModelModule(model, train, load_encoder_weights=False)
    require(before == token(coco_path.stat()), "Official COCO asset changed during native module load")
    witness = describe_q100(module, state_api.tensor_dict_sha256)
    require(witness["online_tensor_sha256"] == EXPECTED_Q100_SHA and witness["state_tensors"] == 545,
            "Fresh Q100 tensor state must reproduce the independent native reference")
    return model, train, module, dict(**witness, initialization_seed=SEED, coco_asset_sha256=COCO_SHA,
                                     coco_asset_bytes=COCO_BYTES, recipe_sha256=RECIPE_SHA)


def offline(event, args):
    if event in ("socket.connect", "socket.getaddrinfo"):
        raise RuntimeError("Offline fresh COCO initialization required")


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--coco", type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and sys.version_info[:3] == (3, 10, 13)
            and sys.flags.isolated == 1 and sys.dont_write_bytecode and sys.flags.utf8_mode == 1,
            "Expected exact isolated CPU-only Q100 probe")
    sys.addaudithook(offline)
    import torch

    require(not torch.cuda.is_available(), "Q100 initialization probe has no visible GPU")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    unused = Path(__file__).resolve().parent / "UNUSED_DATASET_FOR_INITIALIZATION"
    require(not unused.exists(), "Synthetic placeholder must not name an existing dataset")
    _, _, _, report = initialize_q100(unused, unused, args.coco)
    print(json.dumps(dict(status="SSDD_Q100_OFFICIAL_COCO_CPU_INITIALIZATION_VERIFIED",
                          FIT=False, DEV=False, TEST=False, training=False, **report), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

