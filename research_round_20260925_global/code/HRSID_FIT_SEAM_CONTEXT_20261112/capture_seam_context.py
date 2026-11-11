"""Image-only same-carrier native seam diagnostic prediction capture."""

from argparse import ArgumentParser
from hashlib import sha256
from importlib import import_module
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import sys
from time import perf_counter
from types import ModuleType

import numpy as np


PROTOCOL = "6b0955af80a701175f2a5b1ebdb4f9d82d239df2e18586e5037dc414c850eba2"
MANIFEST = "ec6a168757ebcfc2915b837e8853c286dd0df6f6750b6f8476fde210fc44450b"
PRIOR_CAPTURE = "17a81104ab91a1570aea21e9caccc763855aa9242690f96ddec141ffeb42f680"
CENTER_5202 = "84402a6cf29a6ae428853d3049a7759f97e342defa31ab719b74667ce932ef3a"
SEED = 5202
IMAGE_FIELDS = {"ordinal", "group", "basename", "encoded_sha256", "decoded_rgb_sha256", "encoded_bytes"}


def require(valid, reason):
    if not valid:
        raise ValueError(reason)


def exact(path, digest, limit=2**20):
    require(path.resolve(strict=True) == path and 0 < path.stat().st_size <= limit,
            "Exact bounded input: " + path.name)
    data = path.read_bytes()
    require(sha256(data).hexdigest() == digest, "Unexpected digest: " + path.name)
    return data


def load_prior(path):
    module = ModuleType("_seam_exact_prior_capture")
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec(compile(exact(path, PRIOR_CAPTURE), str(path), "exec"), module.__dict__)
    return module


def read_manifest(path):
    manifest = json.loads(exact(path, MANIFEST))
    require(manifest["status"] == "HRSID_FIT_SEAM_CONTEXT_IMAGE_MANIFEST_V1" and
            manifest["FIT"] is True and all(manifest[k] is False for k in ("DEV", "TEST", "GT_fields_present")) and
            manifest["protocol_sha256"] == PROTOCOL, "Exact image-only seam cohort")
    images = manifest["images"]
    require(len(images) == len({r["basename"] for r in images}) == 144
            and len({r["group"] for r in images}) == 40 and
            [r["ordinal"] for r in images] == list(range(1, 145)), "Exact deduplicated ordered manifest")
    for row in images:
        name = row["basename"]
        require(set(row) == IMAGE_FIELDS and re.fullmatch(r"P\d+_\d+_\d+_\d+_\d+\.png", name)
                and name.split("_")[0] == row["group"], "Image fields and native source identity only")
    return images


def native_rows(raw, image_ordinal, *, post, processor, torch, codec):
    retained = []

    class RecordNative(processor):
        def forward(self, *args, **kwargs):
            out = super().forward(*args, **kwargs)
            if len(out[0]["scores"]) == 100:
                retained.append({k: out[0][k].tolist() for k in ("boxes", "labels", "scores")})
            return out

    observed = post.aligned_query_rows(raw, (800, 800), torch=torch, np=np, codec=codec, processor=RecordNative)
    require(len(retained) == 1, "One original stock top-100 call")
    native = retained[0]
    probs = raw["pred_logits"].sigmoid().flatten()
    chosen = torch.argsort(probs, descending=True, stable=True)[:100].tolist()
    areas = codec.area(observed["all_rles"])
    rows = []
    for rank, (flat, score, label, box) in enumerate(zip(chosen, native["scores"], native["labels"],
                                                         native["boxes"], strict=True), start=1):
        query, slot = divmod(flat, 2)
        require(label == slot and score == probs[flat].item() and len(box) == 4 and np.isfinite(box).all(),
                "Original stock query, class, score and box")
        rows.append(dict(uid=f"{image_ordinal}:{rank}", rank=rank, query=query, class_slot=slot,
                         score=score, box=box, area=int(areas[query]), rle=observed["all_rles"][query]))
    ship = [row for row in rows if row["class_slot"] == 0]
    require([r["rank"] for r in ship] == observed["native_ranks"] and
            [r["query"] for r in ship] == observed["native_ids"], "Native ship-slot mapping")
    return rows, ship


def run(args):
    import torch
    from pycocotools import mask as codec
    from rfdetr.datasets.coco import make_coco_transforms_square_div_64
    from rfdetr.datasets.io_utils import decode_pil_image_bytes
    from rfdetr.models.postprocess import PostProcess
    from rfdetr.utilities.tensors import nested_tensor_from_tensor_list

    started = perf_counter()
    exact(Path(__file__).resolve().parent / "PROTOCOL.md", PROTOCOL)
    require(sys.flags.isolated and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and not torch.cuda.is_initialized(),
            "Exact isolated CPU runtime")
    require(torch.__version__ == "2.10.0+cu130" and version("rfdetr") == "1.12.0" and
            version("Pillow") == "12.3.0" and version("pycocotools") == "2.0.11" and np.__version__ == "1.26.4",
            "Exact earlier native package versions")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    require(args.output.parent.resolve(strict=True) == args.output.parent and not args.output.exists(),
            "Fresh canonical output")

    rows = read_manifest(args.manifest)
    prior = load_prior(args.code / "HRSID_FIT_SEED_CONSENSUS_20261112/capture_consensus.py")
    admission = prior.module(args.code / "HRSID_TO_SSDD_FROZEN_DEV187_20261110/export_frozen_hrsid_ssdd.py",
                             prior.ADMISSION_SHA, "_seam_source_admission")
    post = prior.module(args.code / "HRSID_FIT_NATIVE_QUERY_RETENTION_20261109/fit_query_export.py",
                        prior.POST_SHA, "_seam_native_post")
    plan = json.loads(exact(args.source_plan, prior.SOURCE_PLAN_SHA))["pairs"][str(SEED)]
    center = plan["arms"]["center"]
    checkpoint = center["checkpoint"]
    require(checkpoint["sha256"] == CENTER_5202, "Exact natural-center seed5202 E24 endpoint")
    lock = dict(seed=SEED, mode="center", checkpoint=checkpoint, pair_receipt=plan["pair_receipt"],
                native_receipt=plan["arms"]["native"]["receipt"], center_receipt=center["receipt"],
                sources=prior.source_refs(args.code, admission))
    base = admission.admit_source(lock)
    for name, digest in base.MODEL_SOURCES.items():
        exact(Path(import_module(name).__file__), digest)
    args.output.mkdir()
    model, config, train, _ = base.load_model(Path(checkpoint["path"]), checkpoint["sha256"], torch,
                                              args.output, admission.TRAINING_PROTOCOLS[SEED])
    model.eval()
    require(model.num_queries == 164 and model.group_detr == 13 and config.patch_size == 12 and
            config.num_windows == 1 and all(v.device.type == "cpu" for v in model.parameters()),
            "Exact admitted Q164 CPU stock network")
    transform = make_coco_transforms_square_div_64(
        "val", 576, multi_scale=False, expanded_scales=False, skip_random_resize=True,
        patch_size=config.patch_size, num_windows=config.num_windows, aug_config=None,
        scale_jitter=train.scale_jitter, gpu_postprocess=False,
    )

    mapped = nonempty = 0
    digest = sha256()
    with (args.output / "CANDIDATES_PRIVATE.jsonl").open("xb") as stream, torch.inference_mode():
        for row in rows:
            image_ordinal = row["ordinal"]
            raw_png = exact(args.fit_images / row["basename"], row["encoded_sha256"], 4 * 2**20)
            image, scales = decode_pil_image_bytes(raw_png, draft_size=None)
            require(image.mode == "RGB" and image.size == (800, 800) and scales == (1., 1.) and
                    len(raw_png) == row["encoded_bytes"] and
                    sha256(np.asarray(image).tobytes(order="C")).hexdigest() == row["decoded_rgb_sha256"],
                    "Exact encoded PNG and decoded RGB identity")
            tensor, target = transform(image, None)
            require(target is None and tensor.shape == (3, 576, 576) and tensor.dtype == torch.float32,
                    "Frozen image-only val576 transform")
            sample = nested_tensor_from_tensor_list([tensor], block_size=config.patch_size * config.num_windows)
            require(sample.mask is not None and not bool(sample.mask.any()), "No native padding")
            raw = model(sample)
            native, ship = native_rows(raw, image_ordinal, post=post, processor=PostProcess, torch=torch, codec=codec)
            candidates = [r for r in ship if r["area"] > 0]
            mapped += len(ship)
            nonempty += len(candidates)
            block = dict(ordinal=image_ordinal, group=row["group"], basename=row["basename"],
                         rgb_sha256=row["decoded_rgb_sha256"], mapped=len(ship), native=native, candidates=candidates)
            line = prior.encode(block)
            require(stream.write(line) == len(line), "Exact stream write")
            digest.update(line)
            if image_ordinal % 12 == 0:
                print(json.dumps(dict(event="HRSID_FIT_SEAM_CONTEXT_CPU_PROGRESS", images=image_ordinal)), flush=True)
        stream.flush()
        os.fsync(stream.fileno())

    count = len(rows)
    result = dict(status="HRSID_FIT_SEAM_CONTEXT_CAPTURE_ACTUAL", protocol_sha256=PROTOCOL,
                  manifest_sha256=MANIFEST, checkpoint_sha256=CENTER_5202, prediction_sha256=digest.hexdigest(),
                  images=count, source_groups=40, model_forwards=count, raw_queries=164*count, native_rows=100*count,
                  mapped=mapped, nonempty=nonempty, FIT=True, DEV=False, TEST=False, GT_opened=False, AP=False,
                  optimizer_updates=0, worker_seconds=perf_counter() - started)
    result_sha = prior.write(args.output / "RESULT_SAFE.json", prior.encode(result))
    print(json.dumps(dict(status=result["status"], result_sha256=result_sha)), flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("code", "source-plan", "manifest", "fit-images", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    def offline(event, _args):
        if event.startswith("socket."):
            raise PermissionError("Closed local image-only inference")
    sys.addaudithook(offline)
    run(args)

